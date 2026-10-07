"""Adjacent-swap isolation, raw-response integrity and resumable new-only calls."""
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
from scripts.exp5_validation import order_check as o


class AdjacentSwapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freeze, cls.original, cls.scores, cls.cases = o.parent_records(check_code=False)
        _, cls.plan, cls.audit = o.construct(check_code=False)
        cls.by_id = {r["row_id"]: r for r in cls.original}
        cls.case_by_id = {c["case_id"]: c for c in cls.cases}

    def test_only_adjacent_pair_changes_not_content_scope_or_oracle(self):
        for row in self.plan:
            before = self.by_id[row["original_row_id"]]
            first, second = row["swapped_positions"]
            self.assertEqual(second, first + 1)
            expected = deepcopy(before["constructed_assertions"])
            self.assertEqual(expected[first]["memory_id"], expected[second]["memory_id"])
            expected[first], expected[second] = expected[second], expected[first]
            self.assertEqual(row["constructed_assertions"], expected)
            self.assertEqual([m["memory_id"] for m in row["memory"]],
                             [m["memory_id"] for m in before["memory"]])
            for key in ("state_properties", "strict_evaluation", "expected_values", "subject", "attributes"):
                self.assertEqual(row[key], before[key])
            for key in ("system_prompt", "generation_config", "response_schema"):
                self.assertEqual(row["payload"][key], before["payload"][key])

    def test_all_cases_and_original_schedule_retained_without_cache_overlap(self):
        original = [r for r in self.original if r["variant"] == "contradiction"]
        self.assertEqual([r["case_id"] for r in self.plan], [r["case_id"] for r in original])
        self.assertEqual(len(self.plan), 50)
        self.assertEqual(len({r["case_id"] for r in self.plan}), 50)
        self.assertEqual(len({r["input_sha256"] for r in self.plan}), 50)
        self.assertFalse({r["input_sha256"] for r in self.plan} &
                         {r["input_sha256"] for r in self.original})
        self.assertEqual(sum(self.scores[r["original_row_id"]]["profile_exact_match"]
                             for r in self.plan), 6)
        self.assertEqual(self.audit["additional_independent_cases"], 0)

    def test_nonadjacent_and_different_unit_pairs_are_rejected(self):
        row = deepcopy(self.by_id[self.plan[0]["original_row_id"]])
        case = self.case_by_id[row["case_id"]]
        first, second = self.plan[0]["swapped_positions"]
        row["constructed_assertions"].insert(second, deepcopy(case["reference_assertions"][0]))
        with self.assertRaisesRegex(ValueError, "not adjacent"):
            o.reverse_target_pair(row, case)
        row = deepcopy(self.by_id[self.plan[0]["original_row_id"]])
        row["constructed_assertions"][second]["memory_id"] = "different_unit"
        with self.assertRaisesRegex(ValueError, "different units"):
            o.reverse_target_pair(row, case)

    def test_misbinding_provenance_checks_the_two_nonexact_city_answers(self):
        self.assertEqual(self.audit["misbinding_concrete"], 9)
        self.assertEqual(self.audit["misbinding_unknown"], 41)
        self.assertTrue(self.audit["all_concrete_values_supported_only_by_other_subject"])
        other = [r for r in self.audit["misbinding_value_provenance"] if not r["exact_match"]]
        self.assertEqual({(r["queried_subject"], r["other_subject"], r["attribute"], r["returned_value"])
                          for r in other},
                         {("Nora", "Lucas", "city", "Portland"), ("Noah", "Liam", "city", "Seattle")})
        for row in self.audit["misbinding_value_provenance"]:
            self.assertFalse(row["support_under_queried_subject"])
            self.assertTrue(row["matching_other_assertions"])
            self.assertTrue(all(a["subject"] == row["other_subject"] and
                                a["attribute"] == row["attribute"] and
                                a["value"] == row["returned_value"]
                                for a in row["matching_other_assertions"]))

    def test_equal_rates_do_not_hide_opposing_paired_transitions(self):
        original = [r for r in self.scores.values() if r["variant"] == "contradiction"]
        passing = next(r["case_id"] for r in original if r["profile_exact_match"])
        failing = next(r["case_id"] for r in original if not r["profile_exact_match"])
        old = {r["case_id"]: r for r in original}
        cache = {}
        for row in self.plan:
            answer = old[row["case_id"]]["parsed_response"]
            if row["case_id"] == passing:
                answer = {a: "UNKNOWN" for a in row["attributes"]}
            elif row["case_id"] == failing:
                answer = row["expected_values"]
            cache[row["input_sha256"]] = {"response_text": json.dumps(answer)}
        data = o.summarize(original, v.score(self.plan, cache))
        self.assertEqual(data["profile_difference_pp"], 0)
        self.assertEqual(data["exact_match_transitions"], {"0->0": 43, "0->1": 1, "1->0": 1, "1->1": 5})
        self.assertEqual(data["changed_answers"], 2)
        self.assertEqual(data["paired_cases"], 50)
        self.assertLess(data["profile_difference_ci_low_pp"], 0)
        self.assertGreater(data["profile_difference_ci_high_pp"], 0)


class OrderPipelineTests(unittest.TestCase):
    def test_generation_checks_producer_code_while_offline_checks_saved_evidence(self):
        digest = v.digest
        changed = v.paths.PROJECT_ROOT / "atomic_memory/validation.py"
        def modified_code(path):
            return "0" * 64 if Path(path) == changed else digest(path)
        with patch.object(v, "digest", side_effect=modified_code):
            with self.assertRaisesRegex(ValueError, "Original scientific code changed"):
                o.parent_records()
            _, plan, scores, _ = o.parent_records(check_code=False)
            self.assertEqual(len(plan), 300)
            self.assertEqual(len(scores), 300)

    def test_new_only_resume_retains_invalid_outputs_and_validates_sources(self):
        fixture_parent = os.environ.get("ATOMICUPDATE_TEST_TMP")
        with tempfile.TemporaryDirectory(prefix="order_check_fixture_", dir=fixture_parent) as name:
            root = Path(name)
            sources = set(v.SOURCE_FILES) | set(o.EXTRA_SOURCES)
            sources |= {"exp5_validation/" + n for n in o.PARENT_FILES}
            for source in sources:
                destination = root / source
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(v.paths.OUTPUT_ROOT / source, destination)
            parent = root / "exp5_validation"
            out = parent / "order_check"
            # This private fixture represents a new batch under current source;
            # the historical experiment manifests in outputs are never changed.
            parent_freeze = v.load(parent / "freeze_manifest.json")
            parent_freeze["code_sha256"] = {
                n: v.digest(v.paths.PROJECT_ROOT / n) for n in v.CODE_FILES}
            v.write_json_atomic(parent / "freeze_manifest.json", parent_freeze)
            summary = v.load(parent / "run_summary.json")
            summary["freeze_manifest_sha256"] = v.digest(parent / "freeze_manifest.json")
            v.write_json_atomic(parent / "run_summary.json", summary)
            analysis = v.load(parent / "analysis.json")
            for n in analysis["source_hashes"]:
                analysis["source_hashes"][n] = v.digest(parent / n)
            v.write_json_atomic(parent / "analysis.json", analysis)
            before = {n: v.digest(parent / n) for n in o.PARENT_FILES}
            snapshot = v.load(parent / "freeze_manifest.json")["model_snapshot"]

            class FixtureBackend:
                calls = 0
                stop_after = 7

                def describe(self, model):
                    return snapshot

                def generate(self, **kwargs):
                    if self.stop_after is not None and self.calls == self.stop_after:
                        raise ConnectionError("Synthetic interruption; no live generation")
                    self.calls += 1
                    return LLMResponse(text="{invalid fixture JSON", model=v.CONFIG["model"],
                                       metadata={**snapshot, "test_fixture": True})

            backend = FixtureBackend()
            with patch.object(v.paths, "OUTPUT_ROOT", root), redirect_stdout(io.StringIO()):
                freeze = o.prepare(out, parent, backend)
                self.assertEqual(backend.calls, 0)
                self.assertEqual(freeze["calls_at_lock"], 0)
                self.assertFalse(freeze["originally_planned_in_300"])
                self.assertTrue(freeze["locked_at_utc"])
                with self.assertRaises(ConnectionError):
                    o.run(out, parent, backend)
                self.assertEqual(len(v.records(out / "responses.jsonl")), 7)
                with patch.object(backend, "describe", return_value={**snapshot, "ollama_version": "changed"}):
                    with self.assertRaisesRegex(ValueError, "Model/runtime changed"):
                        o.run(out, parent, backend)
                self.assertEqual(backend.calls, 7)
                backend.stop_after = None
                o.run(out, parent, backend)
                o.run(out, parent, backend)
                self.assertEqual(backend.calls, 50)
                results = v.records(out / "reader_results.jsonl")
                self.assertEqual(len(results), 50)
                self.assertTrue(all(not r["valid_json"] for r in results))
                o.analyze(out, parent)
                data = v.load(out / "analysis.json")
                self.assertEqual(data["exact_match_transitions"], {"0->0": 44, "0->1": 0, "1->0": 6, "1->1": 0})
                self.assertEqual(data["accounting"]["constructed_physical_calls"], 350)
                self.assertEqual(data["accounting"]["supplemental_new_calls"], 955)
                hashes = {p.name: v.digest(p) for p in out.iterdir() if p.is_file()}
                check_analysis(lambda: o.analyze(out, parent), o.__dict__)
                o.prepare(out, parent, backend)
                self.assertEqual(hashes, {p.name: v.digest(p) for p in out.iterdir() if p.is_file()})
                self.assertEqual(before, {n: v.digest(parent / n) for n in o.PARENT_FILES})
                audit_path = out / "source_audit.json"
                audit_bytes = audit_path.read_bytes()
                audit_path.write_bytes(audit_bytes + b"\n")
                with self.assertRaisesRegex(ValueError, "frozen artifact changed"):
                    o.verify(out, parent)
                audit_path.write_bytes(audit_bytes)
                results[0]["profile_exact_match"] = True
                v.write_jsonl_atomic(out / "reader_results.jsonl", results)
                with self.assertRaisesRegex(ValueError, "raw-response rescoring differs"):
                    o.analyze(out, parent)


if __name__ == "__main__":
    unittest.main()
