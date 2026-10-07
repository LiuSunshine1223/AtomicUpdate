"""Scientific contracts and interruption/resumption for the bounded fifth study."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from atomic_memory import validation as v
from atomic_memory.backend import LLMResponse
from atomic_memory.reanalysis import check_analysis
from scripts.exp5_validation.build_validation import build
from scripts.exp5_validation.run_validation import run
from scripts.exp5_validation.analyze_validation import analyze, summarize


class ConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = v.source_context(v.paths.OUTPUT_ROOT)
        cls.cases, cls.plan, cls.overlap = v.build_design(cls.context)
        cls.rows = {(r["case_id"], r["variant"]): r for r in cls.plan}

    def test_new_cases_and_complete_pairs(self):
        self.assertEqual(len(self.cases), 50)
        self.assertEqual(len(self.plan), 300)
        self.assertEqual(len({r["input_sha256"] for r in self.plan}), 300)
        self.assertFalse(any(self.overlap["overlap_counts"].values()))
        for c in self.cases:
            rows = [self.rows[c["case_id"], arm] for arm in v.VARIANTS]
            self.assertTrue(all(r["expected_values"] == c["expected_values"] for r in rows))
            self.assertTrue(all(r["attributes"] == c["attributes"] for r in rows))
            self.assertTrue(all("evidence_text" not in r["payload"]["user_prompt"] for r in rows))
            self.assertTrue(all(r["input_sha256"] == v.sha256_json(r["payload"]) for r in rows))

    def test_surface_changes_one_sentence_preserving_independent_truth(self):
        for c in self.cases:
            clean, surface = (self.rows[c["case_id"], a] for a in ("clean", "surface"))
            self.assertTrue(clean["strict_evaluation"]["end_to_end_success"])
            self.assertFalse(surface["strict_evaluation"]["contract_valid"])
            self.assertIsNone(surface["strict_evaluation"]["target_new_current"])
            self.assertEqual(clean["state_properties"], surface["state_properties"])
            self.assertEqual(sum(a != b for a, b in zip(clean["constructed_assertions"],
                                                       surface["constructed_assertions"])), 1)
            self.assertEqual(sum(u["text"].count("At present,") for u in surface["memory"]), 1)

    def test_intervention_scope_and_nonunique_support(self):
        for c in self.cases:
            contradiction = self.rows[c["case_id"], "contradiction"]
            state = contradiction["state_properties"]
            self.assertEqual(state["assertion_count"], 7)
            self.assertTrue(state["oracle_profile_supported"])
            self.assertFalse(state["queried_profile_unique_and_noncontradictory"])
            self.assertTrue(state["all_reference_assertions_present"])
            self.assertFalse(state["global_reference_exact"])
            for arm in ("deletion", "misbinding"):
                row = self.rows[c["case_id"], arm]
                self.assertFalse(row["state_properties"]["oracle_profile_supported"])
                self.assertFalse(row["state_properties"]["queried_profile_complete"])
                self.assertIsNone(row["state_properties"]["queried_profile_unique_and_noncontradictory"])
                target = c["update"]["new_fact"]["attribute"]
                self.assertEqual(row["state_properties"]["queried_current_values"][target], [])
                self.assertEqual(row["state_properties"]["assertion_count"], 5 if arm == "deletion" else 6)
            other = self.rows[c["case_id"], "other_damage"]["state_properties"]
            self.assertTrue(other["oracle_profile_supported"])
            self.assertTrue(other["queried_profile_unique_and_noncontradictory"])
            self.assertFalse(other["global_reference_exact"])
            self.assertEqual(other["per_unit_counts"], {"m00": 3, "m01": 3})
            self.assertTrue(other["constructed_membership_preserved"])
            source = {a["fact_id"]: a for a in c["reference_assertions"]}
            changed = [a for a in self.rows[c["case_id"], "other_damage"]["constructed_assertions"]
                       if a != source[a["fact_id"]]]
            self.assertEqual(len(changed), 1)
            self.assertNotEqual(changed[0]["subject"], c["subject"])

    def test_exact_match_can_be_unsupported_and_unknown_is_not_success(self):
        c = self.cases[0]
        row = self.rows[c["case_id"], "deletion"]
        gold = c["expected_values"]
        values = row["state_properties"]["queried_current_values"]
        self.assertFalse(v.answer_support(gold, values)["answer_supported"])
        abstain = dict(gold)
        abstain[c["update"]["new_fact"]["attribute"]] = "UNKNOWN"
        self.assertIsNone(v.answer_support(abstain, values)["answer_supported"])
        self.assertFalse(v.exact_profile_match(abstain, gold))
        abstain[next(a for a in abstain if abstain[a] != "UNKNOWN")] = "unsupported test value"
        self.assertFalse(v.answer_support(abstain, values)["answer_supported"])
        self.assertIsNone(v.answer_support(None, values)["answer_supported"])

    def test_duplicate_truth_conventions_do_not_hide_global_redundancy(self):
        c = self.cases[0]
        repeated = c["reference_assertions"] + [deepcopy(c["reference_assertions"][0])]
        state = v.state_properties(repeated, c["reference_assertions"], c["subject"],
                                   c["attributes"], c["expected_values"])
        self.assertTrue(state["queried_profile_unique_and_noncontradictory"])
        self.assertFalse(state["global_reference_exact"])


class PipelineTests(unittest.TestCase):
    def test_interruption_invalid_response_no_retries_raw_rescoring_and_freeze_guards(self):
        # Fixture outputs are kept in a temporary directory, never in outputs/.
        fixture_parent = os.environ.get("ATOMICUPDATE_TEST_TMP")
        with tempfile.TemporaryDirectory(prefix="exp5_test_fixture_", dir=fixture_parent) as name:
            root = Path(name)
            for source in v.SOURCE_FILES:
                destination = root / source
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(v.paths.OUTPUT_ROOT / source, destination)
            context = v.source_context(root)
            stage = root / "exp5_validation"

            class FixtureBackend:
                calls = 0
                stop_after = 7

                def describe(self, model):
                    return context["qualified_snapshot"]

                def generate(self, **kwargs):
                    if self.stop_after is not None and self.calls == self.stop_after:
                        raise ConnectionError("Synthetic interruption; no live model call")
                    self.calls += 1
                    # Deliberately invalid JSON tests retention without regeneration.
                    return LLMResponse(text="{invalid fixture JSON", model=v.CONFIG["model"],
                                       metadata={**context["qualified_snapshot"], "test_fixture": True})

            backend = FixtureBackend()
            with patch.object(v.paths, "OUTPUT_ROOT", root), redirect_stdout(io.StringIO()):
                freeze = build(stage, backend)
                self.assertEqual(backend.calls, 0)
                self.assertEqual(freeze["unique_inputs"], 300)
                with self.assertRaises(ConnectionError):
                    run(stage, backend)
                self.assertEqual(len(v.records(stage / "responses.jsonl")), 7)
                wrong_snapshot = dict(context["qualified_snapshot"], ollama_version="changed-fixture-runtime")
                with patch.object(backend, "describe", return_value=wrong_snapshot):
                    with self.assertRaisesRegex(ValueError, "Model/runtime changed"):
                        run(stage, backend)
                self.assertEqual(backend.calls, 7)
                self.assertEqual(len(v.records(stage / "responses.jsonl")), 7)
                backend.stop_after = None
                run(stage, backend)
                self.assertEqual(backend.calls, 300)
                run(stage, backend)
                self.assertEqual(backend.calls, 300)
                results = v.records(stage / "reader_results.jsonl")
                self.assertEqual(len(results), 300)
                self.assertTrue(all(not r["valid_json"] for r in results))
                self.assertTrue(all(r["answer_supported"] is None for r in results))
                analyze(stage)
                data = v.load(stage / "analysis.json")
                self.assertTrue(all(r["Z_0_0"] == 50 for r in data["paired_transitions"]))
                self.assertTrue(all(r["n"] == 50 for r in data["arm_summary"]))
                before = {p.name: v.digest(p) for p in stage.iterdir() if p.is_file()}
                check_analysis(lambda: analyze(stage), analyze.__globals__)
                self.assertEqual(before, {p.name: v.digest(p) for p in stage.iterdir() if p.is_file()})
                freeze_before = v.digest(stage / "freeze_manifest.json")
                build(stage, backend)
                self.assertEqual(freeze_before, v.digest(stage / "freeze_manifest.json"))
                raw = v.records(stage / "responses.jsonl")
                _, plan = v.verify(stage)
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    v.validate_raw(raw + [raw[0]], plan, freeze["model_snapshot"])
                modified = deepcopy(raw)
                modified[0]["response_text"] = "changed"
                with self.assertRaisesRegex(ValueError, "text hash"):
                    v.validate_raw(modified, plan, freeze["model_snapshot"])
                plan_path = stage / "reader_plan.jsonl"
                plan_bytes = plan_path.read_bytes()
                plan_path.write_bytes(plan_bytes + b"\n")
                with self.assertRaisesRegex(ValueError, "Frozen artifact"):
                    v.verify(stage)
                plan_path.write_bytes(plan_bytes)
                results[0]["profile_exact_match"] = True
                v.write_jsonl_atomic(stage / "reader_results.jsonl", results)
                with self.assertRaisesRegex(ValueError, "raw-response rescoring"):
                    analyze(stage)

    def test_paired_bootstrap_retains_clean_failures(self):
        context = v.source_context(v.paths.OUTPUT_ROOT)
        _, plan, _ = v.build_design(context)
        cache = {}
        for row in plan:
            answer = dict(row["expected_values"])
            # Every clean case fails; every surface case passes. A success-only
            # comparison would discard every case and is therefore detectable.
            if row["variant"] == "clean":
                answer = {a: "UNKNOWN" for a in row["attributes"]}
            cache[row["input_sha256"]] = {"response_text": json.dumps(answer)}
        data = summarize(v.score(plan, cache))
        surface = next(r for r in data["paired_transitions"] if r["variant"] == "surface")
        self.assertEqual(surface["n"], 50)
        self.assertEqual(surface["Z_0_1"], 50)
        self.assertEqual(surface["profile_diff_pp"], 100)
        self.assertEqual(surface["profile_diff_ci_low_pp"], 100)
        self.assertEqual(surface["profile_diff_ci_high_pp"], 100)
        self.assertEqual(data["cluster_unit"], "case")


if __name__ == "__main__":
    unittest.main()
