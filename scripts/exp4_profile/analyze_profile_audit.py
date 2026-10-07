#!/usr/bin/env python3
"""Reproduce the post-hoc disagreement partition and paired endpoint changes.

Use --check to verify saved outputs without writing files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "atomic_memory").is_dir())
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from atomic_memory import paths
MODELS = (
    "llama3.1:8b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "gemma3:12b-it-q4_K_M",
)
INTERFACES = ("Patch", "Select")
CONDITIONS = ("L1_select", "L1_patch", "L3_select", "L3_patch")
CLASSIFICATION_PRIORITY = (
    "grammar_unscorable",
    "old_and_new_current",
    "outside_subject_loss",
    "unsupported_target_hit",
    "duplicate_fact",
)
BOOTSTRAP_REPLICATES = 20_000
BASE_SEED = 20260927
EXPECTED_CASES = 50
EXPECTED_ROWS = 1200
MANIFEST_NAME = "explanatory_analysis_manifest.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def input_path(source, relative):
    path = (source / relative).resolve()
    require(path.is_relative_to(source), "Source path escapes results directory")
    return path


def verify_inputs(source):
    analysis = read_json(source / "formal_analysis.json")
    freeze = read_json(source / "freeze_manifest.json")
    sources = {}
    for recorded in (analysis["source_hashes"], freeze["frozen_file_sha256"]):
        for relative, expected in recorded.items():
            actual = sha(input_path(source, relative))
            require(actual == expected, "Frozen input changed: " + relative)
            sources[relative] = actual
    for relative in ("formal_analysis.json", "freeze_manifest.json"):
        sources[relative] = sha(source / relative)

    updaters = read_jsonl(source / "updater_results.jsonl")
    readers = read_jsonl(source / "actual_reader_results.jsonl")
    by_request = {row["request_id"]: row for row in updaters}
    require(len(by_request) == len(updaters) == len(readers) == EXPECTED_ROWS,
            "Expected 1,200 unique updater records and 1,200 reader records")
    reader_ids = {(row["updater_model"], row["case_id"], row["update_id"],
                   row["condition"]) for row in readers}
    require(len(reader_ids) == EXPECTED_ROWS, "Duplicate reader row identity")
    require({row["source_updater_request_id"] for row in readers} == set(by_request),
            "Reader rows do not cover the updater records exactly once")
    cases = sorted({row["case_id"] for row in readers})
    require(len(cases) == EXPECTED_CASES, "Expected 50 case clusters")
    cells = defaultdict(list)
    for row in readers:
        updater = by_request[row["source_updater_request_id"]]
        require(type(row["strict_state_success"]) is bool and
                type(row["functional_task_pass"]) is bool,
                "Endpoint scores must be boolean")
        require(row["strict_state_success"] ==
                updater["evaluation"]["end_to_end_success"],
                "Reader/updater strict score mismatch")
        require(row["functional_task_pass"] ==
                (row["parsed_response"] == row["oracle_expected_values"]),
                "Profile score differs from exact oracle equality")
        require(row["updater_model"] == updater["model"] and
                (row["case_id"], row["update_id"], row["condition"]) ==
                (updater["case_id"], updater["update_id"], updater["condition"]),
                "Reader/updater identity mismatch")
        require(row["condition"] ==
                "L{}_{}".format(row["unit_load"], row["interface"]),
                "Reader condition disagrees with grouping/interface")
        cells[row["updater_model"], row["condition"]].append(row)
    expected_cells = {(model, condition) for model in MODELS for condition in CONDITIONS}
    require(set(cells) == expected_cells, "Incomplete configuration coverage")
    summaries = {(row["model"], row["condition"]): row
                 for row in analysis["condition_summary"]}
    require(len(analysis["condition_summary"]) == len(summaries) and
            set(summaries) == expected_cells, "Invalid frozen condition summaries")
    for identity, rows in cells.items():
        summary = summaries[identity]
        require(len(rows) == 100 and {row["case_id"] for row in rows} == set(cases),
                "Expected 100 rows over the same 50 cases per configuration")
        require(sum(row["strict_state_success"] for row in rows) ==
                summary["strict_success_count"] and
                sum(row["functional_task_pass"] for row in rows) ==
                summary["task_success_count"],
                "Raw scores disagree with frozen analysis")
    return analysis, by_request, readers, cases, sources


def classify_disagreement(evaluation, subject):
    """Observed-set priority partition, not a semantic annotation procedure.

    Keep null fields on grammar failures. Categories may conceal overlapping
    defects; in particular this partition is not a count of all unsupported
    answers. The separate query-scoped audit reports those properties.
    """
    e = evaluation
    if not e["contract_valid"]:
        require(e["target_new_current"] is None and
                e["protected_facts_preserved"] is None,
                "Grammar-unscorable semantics must remain missing")
        return "grammar_unscorable"
    if not e["target_old_not_current"]:
        require(e["target_new_current"] is True,
                "Old/new-current category requires the new value")
        return "old_and_new_current"
    if e["protected_facts_preserved"] is False:
        missing = e["missing_protected_facts"]
        require(missing and all(fact["subject"] != subject for fact in missing),
                "Protected-fact loss is not restricted to the other subject")
        return "outside_subject_loss"
    if e["target_new_current"] is False:
        return "unsupported_target_hit"
    require(e["duplicate_fact_count"] > 0 and e["target_new_current"] is True and
            e["protected_facts_preserved"] is True,
            "Disagreement has no category in the documented observed-set partition")
    return "duplicate_fact"


def partition_disagreements(by_request, readers):
    disagreements = []
    for row in readers:
        if row["strict_state_success"] or not row["functional_task_pass"]:
            continue
        updater = by_request[row["source_updater_request_id"]]
        evaluation = updater["evaluation"]
        disagreements.append({
            "category": classify_disagreement(evaluation, row["subject"]),
            "updater_request_id": updater["request_id"],
            "case_id": row["case_id"], "update_id": row["update_id"],
            "model": updater["model"], "condition": updater["condition"],
            "reader_input_sha256": row["reader_input_sha256"],
            "subject": row["subject"], "original_evaluation": evaluation,
            "failure_reasons": updater["failure_reasons"],
            "reader_visible_memory": [
                {"memory_id": unit["memory_id"], "text": unit["text"]}
                for unit in updater["execution"]["final_memory"]],
            "parsed_response": row["parsed_response"],
            "oracle_expected_values": row["oracle_expected_values"],
        })
    return disagreements


def endpoint_changes(readers, cases):
    # Cluster on case, preserving the two targets, both groupings and endpoints.
    cells = defaultdict(list)
    for row in readers:
        cells[row["updater_model"], row["case_id"], row["interface"]].append(row)
    results = []
    for model_index, model in enumerate(MODELS):
        for interface_index, interface in enumerate(INTERFACES):
            differences = []
            for case in cases:
                rows = cells[model, case, interface.lower()]
                groups = {g: [row for row in rows if row["unit_load"] == g]
                          for g in (1, 3)}
                require(len(rows) == 4 and all(len(group) == 2 for group in groups.values()),
                        "Expected two targets in both groupings per case/interface")
                require({row["update_id"] for row in groups[1]} ==
                        {row["update_id"] for row in groups[3]} and
                        len({row["update_id"] for row in groups[1]}) == 2,
                        "Target pairing differs between groupings")
                by_group = {
                    g: np.mean([int(row["functional_task_pass"]) -
                                int(row["strict_state_success"]) for row in group])
                    for g, group in groups.items()}
                differences.append(by_group[1] - by_group[3])
            values = np.asarray(differences)
            seed = BASE_SEED + 10 * model_index + interface_index
            rng = np.random.default_rng(seed)
            bootstrap = values[rng.integers(
                0, len(cases), (BOOTSTRAP_REPLICATES, len(cases)))].mean(axis=1)
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            results.append({
                "model": model, "interface": interface, "definition": "F_j - G_j",
                "estimate_pp": 100 * float(values.mean()),
                "ci_low_pp": 100 * float(low), "ci_high_pp": 100 * float(high),
                "n_cases": len(cases), "seed": seed, "status": "post_hoc_exploratory",
            })
    return results


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def csv_bytes(fields, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def build_outputs(source):
    analysis, by_request, readers, cases, sources = verify_inputs(source)
    disagreements = partition_disagreements(by_request, readers)
    counts = Counter(row["category"] for row in disagreements)
    changes = endpoint_changes(readers, cases)
    # Same scientific payload as the original paper audit; provenance goes in
    # its own manifest so this migration can be checked field by field.
    audit = {
        "status": "post_hoc_descriptive_not_semantic_rescoring", "model_calls": 0,
        "sources": analysis["source_hashes"],
        "classification_priority": list(CLASSIFICATION_PRIORITY),
        "counts": dict(counts),
        "cases": len({row["case_id"] for row in disagreements}),
        "unique_reader_inputs": len({row["reader_input_sha256"] for row in disagreements}),
        "rows": disagreements, "exploratory_endpoint_changes": changes,
    }
    artifacts = {
        "profile_content_audit.json": json_bytes(audit),
        "profile_audit.csv": csv_bytes(["category", "rows"], [
            {"category": category, "rows": counts[category]}
            for category in CLASSIFICATION_PRIORITY]),
        "profile_endpoint_changes.csv": csv_bytes(list(changes[0]), changes),
    }
    manifest = {
        "analysis": "profile-post-hoc-explanatory-v1", "model_calls": 0,
        "primary_analysis_modified": False,
        "script": Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix(),
        "script_sha256": sha(Path(__file__).resolve()),
        "numpy_version": np.__version__, "source_sha256": sources,
        "updater_rows": len(by_request), "reader_rows": len(readers),
        "case_clusters": len(cases),
        "classification_priority": list(CLASSIFICATION_PRIORITY),
        "bootstrap": {
            "status": "post_hoc_exploratory", "replicates": BOOTSTRAP_REPLICATES,
            "cluster_unit": "case", "case_order": "sorted case_id",
            "pairing": "both endpoints, both targets and both groupings within case",
            "model_order": list(MODELS), "interface_order": list(INTERFACES),
            "seed_formula": "20260927 + 10 * model_index + interface_index",
            "ci_quantiles": [0.025, 0.975], "quantile_method": "linear",
            "multiplicity_adjusted": False,
        },
        "output_sha256": {name: hashlib.sha256(content).hexdigest()
                          for name, content in artifacts.items()},
    }
    artifacts[MANIFEST_NAME] = json_bytes(manifest)
    return audit, artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=paths.PROFILE,
                        help="The existing profile experiment directory")
    parser.add_argument("--check", action="store_true",
                        help="Recompute and verify saved outputs without writing")
    args = parser.parse_args()
    source = args.results_dir.resolve()
    audit, artifacts = build_outputs(source)
    if args.check:
        for name, content in artifacts.items():
            path = source / name
            if name == MANIFEST_NAME:
                # NumPy's installed version is provenance rather than an
                # enforced environment: exact output hashes still must match.
                saved = read_json(path)
                expected = json.loads(content)
                # These fields identify the saved producer release, not an
                # offline reanalysis. Input/output hashes remain mandatory.
                for field in ("script", "script_sha256", "numpy_version"):
                    expected[field] = saved[field]
                require(saved == expected, "Manifest verification failed: " + name)
            else:
                require(path.read_bytes() == content, "Output verification failed: " + name)
    else:
        paths.require_output_stage(source, Path(__file__).with_name("build_profile.py"))
        for name, content in artifacts.items():
            (source / name).write_bytes(content)
    print(json.dumps({
        "mode": "check" if args.check else "write", "model_calls": 0,
        "disagreement_rows": len(audit["rows"]), "counts": audit["counts"],
        "exploratory_endpoint_changes": audit["exploratory_endpoint_changes"],
        "outputs": list(artifacts),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
