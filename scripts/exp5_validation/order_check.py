"""Exploratory adjacent-swap check after the 300-row constructed-state study.

PyCharm Parameters: empty (verify/freeze, run 50 new inputs, then analyze).
--freeze-only makes no generation calls; --check verifies saved analysis offline.
Original New->Old responses are read, never regenerated or appended to their batch.
"""
from collections import Counter
from copy import deepcopy
import argparse
import csv
import io
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from atomic_memory import paths
from atomic_memory import validation as v
from atomic_memory.backend import GenerationConfig, OllamaBackend
from atomic_memory.core import evaluate_final_memory
from atomic_memory.reanalysis import check_analysis
from atomic_memory.validation import write_json_atomic, write_jsonl_atomic, write_text_atomic

OUT = v.OUT / "order_check"
VERSION = "constructed-adjacent-swap-exploratory-v1"
BOOTSTRAP_SEED = 20261016
REPLICATES = 20000
PARENT_FILES = (
    "freeze_manifest.json", "cases.jsonl", "reader_plan.jsonl", "protocol.md",
    "responses.jsonl", "reader_results.jsonl", "run_summary.json", "analysis.json",
)
CODE_FILES = (
    "scripts/exp5_validation/order_check.py", "atomic_memory/validation.py",
    "atomic_memory/core.py", "atomic_memory/backend.py", "atomic_memory/paths.py",
    "atomic_memory/reanalysis.py", "atomic_memory/reader.py",
)
EXTRA_SOURCES = (
    "exp4_profile/followup/run_summary.json", "exp4_profile/followup/reader_results.jsonl",
    "exp4_profile/followup/responses.jsonl", "exp1_baseline/formal_results.jsonl",
    "exp2_wording/robustness_results.jsonl", "exp3_unit_load/unit_load_results.jsonl",
    "exp4_profile/updater_results.jsonl", "exp1_baseline/formal_cases.jsonl",
    "exp3_unit_load/unit_load_cases.jsonl", "exp4_profile/formal_cases.jsonl",
)


def row_count(path):
    with Path(path).open("rb") as stream:
        return sum(bool(line.strip()) for line in stream)


def call_accounting():
    followup = v.load(paths.OUTPUT_ROOT / EXTRA_SOURCES[0])
    old_rows = row_count(paths.OUTPUT_ROOT / EXTRA_SOURCES[1])
    old_calls = row_count(paths.OUTPUT_ROOT / EXTRA_SOURCES[2])
    v.require(followup["status"] == "COMPLETE" and followup["logical_rows"] == old_rows
              and followup["new_generation_calls"] == old_calls, "Existing follow-up accounting mismatch")
    updater_rows = sum(row_count(paths.OUTPUT_ROOT / name) for name in EXTRA_SOURCES[3:7])
    signatures = {v.initial_signature(case) for name in EXTRA_SOURCES[7:]
                  for case in v.records(paths.OUTPUT_ROOT / name)}
    return dict(original_followup_logical_rows=old_rows, original_followup_new_calls=old_calls,
                updater_calls=updater_rows, updater_cases=len(signatures))


def parent_records(parent=v.OUT, check_code=True):
    parent = Path(parent)
    freeze, plan = v.verify(parent, check_code=False)
    # Generation requires the original producer code. Offline analysis instead
    # verifies frozen inputs, reconstructs the plan and rescores every response.
    if check_code:
        for name, expected in freeze["code_sha256"].items():
            if not name.startswith("plot/"):
                v.require(v.digest(paths.PROJECT_ROOT / name) == expected,
                          "Original scientific code changed: " + name)
    cache = v.validate_raw(v.records(parent / "responses.jsonl"), plan,
                           freeze["model_snapshot"], complete=True)
    results = v.score(plan, cache)
    v.require(results == v.records(parent / "reader_results.jsonl"), "Original raw scores differ")
    summary = v.load(parent / "run_summary.json")
    v.require(summary["status"] == "COMPLETE" and summary["logical_rows"] == 300,
              "Complete the original 300 reader evaluations first")
    for field, name in (("freeze_manifest_sha256", "freeze_manifest.json"),
                        ("responses_sha256", "responses.jsonl"),
                        ("reader_results_sha256", "reader_results.jsonl")):
        v.require(summary[field] == v.digest(parent / name), "Original run summary mismatch")
    analysis = v.load(parent / "analysis.json")
    for name, expected in analysis["source_hashes"].items():
        v.require(v.digest(parent / name) == expected, "Original analysis source changed: " + name)
    return freeze, plan, {r["row_id"]: r for r in results}, v.records(parent / "cases.jsonl")


def reverse_target_pair(row, case):
    """Swap the complete adjacent current assertions; preserve all other content."""
    new = deepcopy(row)
    target = case["update"]["new_fact"]
    old_value = case["update"]["old_fact"]["value"]
    assertions = new["constructed_assertions"]
    indices = [i for i, a in enumerate(assertions)
               if (a["subject"], a["attribute"]) == (target["subject"], target["attribute"])]
    v.require(len(indices) == 2 and indices[1] == indices[0] + 1,
              "Original target assertions are not adjacent")
    first, second = indices
    v.require(assertions[first]["memory_id"] == assertions[second]["memory_id"],
              "Original target assertions occupy different units")
    v.require((assertions[first]["value"], assertions[second]["value"]) ==
              (target["value"], old_value), "Original order must be New->Old")
    assertions[first], assertions[second] = assertions[second], assertions[first]
    v.require(Counter(v.canonical_json(a) for a in assertions) ==
              Counter(v.canonical_json(a) for a in row["constructed_assertions"]),
              "Assertion multiset changed")
    new["memory"] = v.memory_from_assertions(assertions)
    truth = v.state_properties(assertions, case["reference_assertions"], case["subject"],
                               case["attributes"], case["expected_values"])
    v.require(truth == row["state_properties"], "Construction truth changed after swap")
    membership = {mid: sorted(a["fact_id"] for a in case["reference_assertions"]
                              if a["memory_id"] == mid) for mid in ("m00", "m01")}
    evaluation = evaluate_final_memory(new["memory"], case["update"], True, True,
        expected_unit_fact_counts={"m00": 3, "m01": 3}, expected_unit_membership=membership)
    v.require(evaluation == row["strict_evaluation"], "Strict checks changed after swap")
    new["payload"]["user_prompt"] = v.build_reader_user_prompt(memory=new["memory"],
        subject=row["subject"], attributes=row["attributes"],
        labels={a: v.ATTRIBUTE_SPECS[a]["label"] for a in row["attributes"]})
    new["input_sha256"] = v.sha256_json(new["payload"])
    v.require(new["input_sha256"] != row["input_sha256"], "Swapped input key did not change")
    new["original_row_id"] = row["row_id"]
    new["original_input_sha256"] = row["input_sha256"]
    new["row_id"] = row["case_id"] + "/contradiction_old_new"
    new["variant"] = "contradiction_old_new"
    new["swapped_positions"] = indices
    return new


def construct(parent=v.OUT, check_code=True):
    freeze, original_plan, original_scores, cases = parent_records(parent, check_code=check_code)
    case_by_id = {c["case_id"]: c for c in cases}
    original = [r for r in original_plan if r["variant"] == "contradiction"]
    v.require(len(original) == 50 and len({r["case_id"] for r in original}) == 50,
              "Expected all 50 original contradiction cases")
    plan = [reverse_target_pair(r, case_by_id[r["case_id"]]) for r in original]
    original_keys = {r["input_sha256"] for r in original_plan}
    v.require(len({r["input_sha256"] for r in plan}) == 50, "Repeated swapped input")
    v.require(not ({r["input_sha256"] for r in plan} & original_keys),
              "A swapped input overlaps the original response cache")
    for i, row in enumerate(plan):
        row["original_execution_index"] = row["execution_index"]
        row["execution_index"] = i
    misbinding = []
    unknown = 0
    for row in original_plan:
        if row["variant"] != "misbinding":
            continue
        scored = original_scores[row["row_id"]]
        case = case_by_id[row["case_id"]]
        attr = case["update"]["new_fact"]["attribute"]
        if scored["any_unknown"]:
            v.require(scored["parsed_response"][attr] == "UNKNOWN", "Abstention is on a different slot")
            unknown += 1
            continue
        v.require(scored["valid_json"], "Invalid JSON requires separate provenance reporting")
        value = scored["parsed_response"][attr]
        supports = [a for a in row["constructed_assertions"]
                    if (a["attribute"], a["value"]) == (attr, value)]
        queried = [a for a in supports if a["subject"] == case["subject"]]
        other = [a for a in supports if a["subject"] == case["other_subject"]]
        misbinding.append(dict(case_id=row["case_id"], queried_subject=case["subject"],
            other_subject=case["other_subject"], attribute=attr, returned_value=value,
            oracle_value=case["expected_values"][attr], exact_match=scored["profile_exact_match"],
            support_under_queried_subject=bool(queried), matching_other_assertions=other))
    audit = dict(parent_cases=len(cases), additional_independent_cases=0,
        adjacent_same_unit_pairs_verified=len(plan), original_order="New->Old",
        added_order="Old->New", new_unique_inputs=len(plan), overlap_with_original_cache=0,
        original_exact_match=sum(original_scores[r["row_id"]]["profile_exact_match"] for r in original),
        misbinding_unknown=unknown, misbinding_concrete=len(misbinding),
        misbinding_exact_match=sum(r["exact_match"] for r in misbinding),
        all_concrete_values_supported_only_by_other_subject=all(
            not r["support_under_queried_subject"] and r["matching_other_assertions"] for r in misbinding),
        misbinding_value_provenance=misbinding)
    return freeze, plan, audit


def protocol():
    return """# Exploratory adjacent-swap check after the original 300 responses

This check was selected after seeing the original constructed-state results
(including 6/50 exact matches on New->Old contradictions). It was not part of
the initial 300-input plan and is not publicly preregistered. The input list,
source/code hashes, actual model snapshot and analysis rules are locally locked
before the added calls; freeze_manifest.json supplies the timestamp and hashes.

## Fixed design

Use all 50 original exp5 contradiction cases, irrespective of original success.
They are the same new base cases used by the original six-arm constructed study,
not 50 additional independent cases and not the main updater-study case batch.
Each original new/old pair must be adjacent in the same memory unit. Exchange
only these two complete sentences. Preserve unit IDs and unit order, all other
sentences and their positions, assertion multiset, state-property labels, strict
scores, query, oracle, system instructions, JSON schema and generation settings.
The original pair is New->Old; the added pair is Old->New. Both remain contradictory.
Project the original shuffled schedule onto its 50 contradiction rows. No new
case selection, target selection, updater, reader or prompt tuning is allowed.

## Calls and caching

Read the original New->Old responses as the comparison baseline; never regenerate
them. Added inputs have different complete-prompt hashes, and all 50 must be
unique and absent from the original 300-input cache. Make at most one response
per added input under the original qualified Llama 3.1 8B Instruct Q4_K_M model
and runtime (temperature 0, seed 42, context 8192, output 64). A transport failure
stops; resume only unfinished inputs. Keep invalid JSON and unexpected answers;
do not retry based on the output. No old file, endpoint or score is overwritten.

## Analysis committed before added responses

Pair by case, retaining all 50 cases including original failures. Report counts
out of 50 for both orders, parsed-answer changes, 1->0, 0->1, 1->1 and 0->0 exact
match transitions, valid JSON, UNKNOWN and answer support. Changed means a change
in parsed three-field response, as in the existing supplement order-probe table.
Estimate Old->New minus New->Old exact-match rate, with a descriptive 95% paired
percentile interval from 20,000 whole-case resamples, numpy RandomState seed
20261016. No significance test, multiplicity claim or equivalence claim is added.
Zero/full outcomes can produce degenerate empirical bootstrap intervals; present
counts and do not treat such intervals as certainty about population probabilities.

## Interpretation written before added responses

A larger recovery rate under Old->New, with failures becoming passes, shows that
presentation order affects this fixed reader's oracle recovery while the same
contradiction remains. It does not identify the reader's internal mechanism.
A small net rate change with many opposing transitions can still conceal strong
case-level order sensitivity. A small change with few transitions/answer changes
provides only limited sensitivity in this reader and construction; it establishes
neither equivalence nor general order robustness. If recovery falls, interpret
the observed direction without changing the oracle or selecting a favorable order.
This check estimates neither naturally generated contradiction prevalence nor
performance on arbitrary natural language or other readers/models.

## Accounting and stopping

The original 300 evaluations stay intact. Add 50 reader inputs/calls, using the
same 50 case clusters; the constructed study then has 350 logical evaluations
and 350 physical calls, but the order comparison has 50 pairs (100 assessments,
50 of which were already among the 300). The existing profile follow-up remains
2,060 logical rows and 605 new calls. Together these two supplemental batches
have 2,410 logical rows and 955 new calls; their case sets and dependence differ.
The 3,720 updater calls over 90 updater cases are unchanged. After this check,
update the figure, manuscript and supplement once, and stop adding experiments.
"""


def verify(out=OUT, parent=v.OUT, check_code=True):
    out = Path(out)
    v.require((out / "freeze_manifest.json").is_file(), "Prepare the order-check manifest first")
    freeze = v.load(out / "freeze_manifest.json")
    v.require(freeze["protocol_version"] == VERSION, "Unknown adjacent-swap protocol")
    for name, expected in freeze["source_sha256"].items():
        v.require(v.digest(Path(parent) / name) == expected, "Original source changed: " + name)
    for name, expected in freeze["external_evidence_sha256"].items():
        v.require(v.digest(paths.OUTPUT_ROOT / name) == expected, "Accounting evidence changed: " + name)
    for name, expected in freeze["artifact_sha256"].items():
        v.require(v.digest(out / name) == expected, "Added frozen artifact changed: " + name)
    if check_code:
        for name, expected in freeze["code_sha256"].items():
            v.require(v.digest(paths.PROJECT_ROOT / name) == expected, "Added frozen code changed: " + name)
    original, plan, audit = construct(parent, check_code=check_code)
    v.require(freeze["original_accounting"] == call_accounting(), "Original reader/updater counts changed")
    v.require(plan == v.records(out / "reader_plan.jsonl"), "Swap plan differs from construction")
    v.require(audit == v.load(out / "source_audit.json"), "Original-source audit differs")
    v.require(freeze["model_snapshot"] == original["model_snapshot"], "Qualified model snapshot changed")
    v.require(freeze["generation_config"] == v.CONFIG, "Added generation configuration changed")
    v.require(freeze["bootstrap_seed"] == BOOTSTRAP_SEED and freeze["bootstrap_replicates"] == REPLICATES,
              "Added analysis settings differ")
    v.require(freeze["logical_new_rows"] == freeze["unique_new_inputs"] == len(plan) == 50,
              "Expected exactly 50 added inputs")
    return freeze, plan


def prepare(out=OUT, parent=v.OUT, backend=None):
    out, parent = Path(out), Path(parent)
    paths.require_output_stage(out, Path(__file__))
    if (out / "freeze_manifest.json").is_file():
        freeze, _ = verify(out, parent)
        print("VERIFIED_EXISTING | 50 paired cases | no generation calls")
        return freeze
    allowed = {"reader_plan.jsonl", "source_audit.json", "protocol.md"}
    if out.exists():
        v.require(all(p.name in allowed or p.name.endswith(".tmp") for p in out.iterdir()),
                  "Preserve unexpected unfrozen files before preparing")
    original, plan, audit = construct(parent)
    accounting = call_accounting()
    backend = backend or OllamaBackend(paths.OLLAMA_URL, timeout_seconds=30)
    snapshot = v.normalize_snapshot(backend.describe(v.CONFIG["model"]))
    v.require(snapshot == original["model_snapshot"], "Model/runtime changed since the original study")
    write_jsonl_atomic(out / "reader_plan.jsonl", plan)
    write_json_atomic(out / "source_audit.json", audit)
    write_text_atomic(out / "protocol.md", protocol())
    freeze = dict(protocol_version=VERSION, locked_at_utc=v.now(),
        motivation="Exploratory check selected after inspecting the original 300 responses",
        originally_planned_in_300=False, public_preregistration=False, calls_at_lock=0,
        logical_new_rows=50, unique_new_inputs=50, paired_cases=50, additional_independent_cases=0,
        original_order="New->Old", added_order="Old->New", model_snapshot=snapshot,
        generation_config=v.CONFIG, bootstrap_seed=BOOTSTRAP_SEED, bootstrap_replicates=REPLICATES,
        source_sha256={name: v.digest(parent / name) for name in PARENT_FILES},
        original_accounting=accounting,
        external_evidence_sha256={name: v.digest(paths.OUTPUT_ROOT / name) for name in EXTRA_SOURCES},
        code_sha256={name: v.digest(paths.PROJECT_ROOT / name) for name in CODE_FILES},
        artifact_sha256={name: v.digest(out / name) for name in sorted(allowed)})
    write_json_atomic(out / "freeze_manifest.json", freeze)
    verify(out, parent)
    print("LOCKED | exploratory addition | 50 new inputs | original responses retained | generation calls: 0")
    return freeze


def run(out=OUT, parent=v.OUT, backend=None):
    out = Path(out)
    paths.require_output_stage(out, Path(__file__))
    freeze, plan = verify(out, parent)
    path = out / "responses.jsonl"
    saved = v.records(path) if path.is_file() else []
    cache = v.validate_raw(saved, plan, freeze["model_snapshot"])
    pending = [r for r in plan if r["input_sha256"] not in cache]
    print("Added rows: 50 | saved: {} | pending new calls: {}".format(len(cache), len(pending)))
    if pending:
        backend = backend or OllamaBackend(paths.OLLAMA_URL, timeout_seconds=600)
        snapshot = v.normalize_snapshot(backend.describe(v.CONFIG["model"]))
        v.require(snapshot == freeze["model_snapshot"], "Model/runtime changed after the added lock")
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            for i, row in enumerate(pending, 1):
                p = row["payload"]
                start = time.monotonic()
                response = backend.generate(request_id=row["row_id"], system_prompt=p["system_prompt"],
                    user_prompt=p["user_prompt"], config=GenerationConfig(**p["generation_config"]),
                    response_schema=p["response_schema"])
                raw = dict(input_sha256=row["input_sha256"], payload=p, first_row_id=row["row_id"],
                    model=response.model, model_snapshot=snapshot, metadata=dict(response.metadata),
                    response_text=response.text, response_sha256=v.sha256_json(response.text),
                    response_id=response.response_id, usage=dict(response.usage),
                    latency_seconds=time.monotonic()-start, saved_at_utc=v.now())
                stream.write(v.canonical_json(raw)+"\n")
                stream.flush()
                os.fsync(stream.fileno())
                v.validate_raw([raw], plan, snapshot)
                print("[{}/{}] {}".format(i, len(pending), row["row_id"]), flush=True)
    cache = v.validate_raw(v.records(path), plan, freeze["model_snapshot"], complete=True)
    results = v.score(plan, cache)
    write_jsonl_atomic(out / "reader_results.jsonl", results)
    write_json_atomic(out / "run_summary.json", dict(status="COMPLETE", completed_at_utc=v.now(),
        logical_new_rows=len(results), unique_new_responses=len(cache), new_calls_this_invocation=len(pending),
        original_new_old_calls_regenerated=0, freeze_manifest_sha256=v.digest(out / "freeze_manifest.json"),
        responses_sha256=v.digest(path), reader_results_sha256=v.digest(out / "reader_results.jsonl")))


def summarize(original, reversed_rows):
    by_case = {r["case_id"]: r for r in original if r["variant"] == "contradiction"}
    rev_case = {r["case_id"]: r for r in reversed_rows}
    v.require(len(by_case) == len(rev_case) == len(reversed_rows) == 50 and set(by_case) == set(rev_case),
              "Expected 50 complete unique pairs")
    case_ids = sorted(by_case)
    old = [by_case[c] for c in case_ids]
    rev = [rev_case[c] for c in case_ids]
    transitions = {"{}->{}".format(a,b): sum(int(x["profile_exact_match"]) == a and
        int(y["profile_exact_match"]) == b for x,y in zip(old,rev)) for a in (0,1) for b in (0,1)}
    delta = np.array([int(y["profile_exact_match"])-int(x["profile_exact_match"])
                      for x,y in zip(old,rev)], dtype=float)
    index = np.random.RandomState(BOOTSTRAP_SEED).randint(50, size=(REPLICATES,50))
    low, high = np.percentile(delta[index].mean(axis=1)*100,[2.5,97.5])
    arm = lambda rows: dict(n=len(rows), exact_match=sum(r["profile_exact_match"] for r in rows),
        valid_json=sum(r["valid_json"] for r in rows), any_unknown=sum(r["any_unknown"] for r in rows),
        supported=sum(r["answer_supported"] is True for r in rows),
        unsupported=sum(r["answer_supported"] is False for r in rows),
        support_undetermined=sum(r["answer_supported"] is None for r in rows))
    return dict(paired_cases=50, additional_independent_cases=0, new_old=arm(old), old_new=arm(rev),
        changed_answers=sum(x["parsed_response"] != y["parsed_response"] for x,y in zip(old,rev)),
        exact_match_transitions=transitions, profile_difference_pp=float(delta.mean()*100),
        profile_difference_ci_low_pp=float(low), profile_difference_ci_high_pp=float(high),
        difference_direction="Old->New minus New->Old", bootstrap_replicates=REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED, cluster_unit="case", scope="Exploratory adjacent swap with one reader",
        original_response_regenerations=0)


def write_csv_atomic(path, rows, fieldnames):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    write_text_atomic(path, stream.getvalue())


def analyze(out=OUT, parent=v.OUT):
    out = Path(out)
    freeze, plan = verify(out, parent, check_code=False)
    v.require((out / "reader_results.jsonl").is_file(), "Run the 50 added inputs to completion first")
    cache = v.validate_raw(v.records(out / "responses.jsonl"), plan, freeze["model_snapshot"], complete=True)
    results = v.score(plan, cache)
    v.require(results == v.records(out / "reader_results.jsonl"), "Added raw-response rescoring differs")
    summary = v.load(out / "run_summary.json")
    v.require(summary["status"] == "COMPLETE" and summary["logical_new_rows"] == 50,
              "Incomplete added-run summary")
    for field, name in (("freeze_manifest_sha256", "freeze_manifest.json"),
                        ("responses_sha256", "responses.jsonl"), ("reader_results_sha256", "reader_results.jsonl")):
        v.require(summary[field] == v.digest(out / name), "Added-run summary hash mismatch")
    _, _, original, _ = parent_records(parent, check_code=False)
    data = summarize(list(original.values()), results)
    data.update(protocol_version=VERSION, created_at_utc=v.now(), analysis_script_sha256=v.digest(Path(__file__)),
        model_snapshot=freeze["model_snapshot"],
        source_hashes={name: v.digest(out / name) for name in
                      ("freeze_manifest.json", "reader_results.jsonl", "responses.jsonl", "source_audit.json")},
        accounting=dict(constructed_reader_rows=300+len(results), constructed_physical_calls=300+len(cache),
            constructed_cases=50, **freeze["original_accounting"],
            supplemental_logical_rows=freeze["original_accounting"]["original_followup_logical_rows"]+300+len(results),
        supplemental_new_calls=freeze["original_accounting"]["original_followup_new_calls"]+300+len(cache)))
    write_json_atomic(out / "analysis.json", data)
    row = dict(reader="Llama", state="Adjacent current-value pair", n=50,
        original=data["new_old"]["exact_match"], reversed=data["old_new"]["exact_match"],
        changed=data["changed_answers"], pass_to_fail=data["exact_match_transitions"]["1->0"],
        fail_to_pass=data["exact_match_transitions"]["0->1"],
        pass_to_pass=data["exact_match_transitions"]["1->1"], fail_to_fail=data["exact_match_transitions"]["0->0"],
        difference_pp=data["profile_difference_pp"], ci_low_pp=data["profile_difference_ci_low_pp"],
        ci_high_pp=data["profile_difference_ci_high_pp"])
    write_csv_atomic(out / "paired_transitions.csv", [row], list(row))
    print("ANALYZED | 50 pairs | New->Old {}/50 | Old->New {}/50 | 1->0 {} | 0->1 {}".format(
        row["original"], row["reversed"], row["pass_to_fail"], row["fail_to_pass"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--freeze-only", action="store_true")
    group.add_argument("--analyze-only", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check_analysis(analyze, globals())
    elif args.analyze_only:
        paths.require_output_stage(OUT, Path(__file__))
        analyze()
    else:
        prepare()
        if not args.freeze_only:
            run()
            analyze()


if __name__ == "__main__":
    main()
