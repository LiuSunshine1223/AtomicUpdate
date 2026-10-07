#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit candidate-selection reader errors without changing the fixed gate."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "atomic_memory").is_dir())
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from atomic_memory import paths
OUTPUT_DIR = paths.SELECTION

TASKS_PATH = OUTPUT_DIR / "profile_tasks.jsonl"
CONTROL_PLAN_PATH = OUTPUT_DIR / "reader_control_plan.jsonl"
RESULTS_PATH = OUTPUT_DIR / "reader_control_results.jsonl"
RUN_SUMMARY_PATH = OUTPUT_DIR / "reader_control_run_summary.json"
FREEZE_PATH = OUTPUT_DIR / "freeze_manifest.json"

AUDIT_JSON_PATH = OUTPUT_DIR / "reader_control_error_audit.json"
ERROR_CASES_PATH = OUTPUT_DIR / "reader_control_error_cases.jsonl"
REPORT_MD_PATH = OUTPUT_DIR / "reader_control_error_report.md"

EXPECTED_TASKS = 100
EXPECTED_RESULTS = 500


# ============================================================================
# Generic helpers
# ============================================================================

def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(
                    "{}:{} must contain a JSON object".format(
                        path,
                        line_number,
                    )
                )
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def nested_counter_to_dict(
    mapping: Mapping[str, Counter],
) -> dict:
    return {
        str(key): dict(sorted(counter.items()))
        for key, counter in sorted(mapping.items())
    }


# ============================================================================
# Candidate/task helpers
# ============================================================================

def candidate_by_id(
    task: Mapping[str, Any],
) -> Dict[str, dict]:
    return {
        str(candidate["profile_id"]): dict(candidate)
        for candidate in task["candidates"]
    }


def selected_role(
    task: Mapping[str, Any],
    selected_profile_id: str,
) -> Tuple[str, List[str]]:
    candidates = candidate_by_id(task)
    candidate = candidates.get(str(selected_profile_id))
    if candidate is None:
        return "invalid_profile_id", []
    return (
        str(candidate.get("semantic_role", "unknown")),
        [
            str(value)
            for value in candidate.get("wrong_attributes", [])
        ],
    )


def subject_schema_label(
    task: Mapping[str, Any],
) -> str:
    attrs = tuple(str(x) for x in task["attributes"])
    if set(attrs) == {"city", "job", "drink"}:
        return "city_job_drink"
    if set(attrs) == {"city", "device", "hobby"}:
        return "city_device_hobby"
    return "other:" + ",".join(sorted(attrs))


# ============================================================================
# Integrity
# ============================================================================

def verify_inputs() -> Tuple[
    dict,
    dict,
    List[dict],
    List[dict],
    Dict[str, dict],
]:
    for path in (
        TASKS_PATH,
        CONTROL_PLAN_PATH,
        RESULTS_PATH,
        RUN_SUMMARY_PATH,
        FREEZE_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(
                "Missing required audit input: {}".format(path)
            )

    freeze = read_json(FREEZE_PATH)
    run_summary = read_json(RUN_SUMMARY_PATH)
    tasks = read_jsonl(TASKS_PATH)
    plan = read_jsonl(CONTROL_PLAN_PATH)
    results = read_jsonl(RESULTS_PATH)

    if len(tasks) != EXPECTED_TASKS:
        raise RuntimeError(
            "Expected {} tasks, found {}.".format(
                EXPECTED_TASKS,
                len(tasks),
            )
        )

    if len(results) != EXPECTED_RESULTS:
        raise RuntimeError(
            "Expected {} control results, found {}.".format(
                EXPECTED_RESULTS,
                len(results),
            )
        )

    if len(plan) != EXPECTED_RESULTS:
        raise RuntimeError(
            "Expected {} control-plan rows, found {}.".format(
                EXPECTED_RESULTS,
                len(plan),
            )
        )

    if str(run_summary.get("results_sha256", "")) != sha256_file(
        RESULTS_PATH
    ):
        raise RuntimeError(
            "reader_control_results.jsonl hash differs from run summary."
        )

    if freeze.get("status") != "FORMAL_PROTOCOL_FROZEN":
        raise RuntimeError(
            "freeze_manifest.json is not FORMAL_PROTOCOL_FROZEN."
        )

    task_by_id = {
        str(task["task_id"]): task
        for task in tasks
    }
    if len(task_by_id) != EXPECTED_TASKS:
        raise RuntimeError("Duplicate task_id detected.")

    result_by_control_id = {
        str(row["control_id"]): row
        for row in results
    }
    if len(result_by_control_id) != EXPECTED_RESULTS:
        raise RuntimeError("Duplicate control_id detected.")

    plan_by_control_id = {
        str(row["control_id"]): row
        for row in plan
    }
    if set(result_by_control_id) != set(plan_by_control_id):
        raise RuntimeError(
            "Control result IDs do not exactly match the frozen plan."
        )

    for control_id, result in result_by_control_id.items():
        planned = plan_by_control_id[control_id]
        for key in (
            "task_id",
            "case_id",
            "update_id",
            "arm",
            "grouping",
            "gold_profile_id",
            "stale_profile_id",
            "reader_input_sha256",
        ):
            if result.get(key) != planned.get(key):
                raise RuntimeError(
                    "{} differs from frozen plan on field {}.".format(
                        control_id,
                        key,
                    )
                )

    return (
        freeze,
        run_summary,
        tasks,
        results,
        task_by_id,
    )


# ============================================================================
# Audit
# ============================================================================

def build_audit() -> Tuple[dict, List[dict]]:
    (
        freeze,
        run_summary,
        tasks,
        results,
        task_by_id,
    ) = verify_inputs()

    rows_by_task_arm_grouping = {}
    for row in results:
        key = (
            str(row["task_id"]),
            str(row["arm"]),
            str(row["grouping"]),
        )
        rows_by_task_arm_grouping[key] = row

    # ------------------------------------------------------------------------
    # Basic arm summaries
    # ------------------------------------------------------------------------

    arm_summary = {}
    for arm, grouping in (
        ("oracle", "g1"),
        ("oracle", "g3"),
        ("stale", "g1"),
        ("stale", "g3"),
        ("no_memory", "None"),
    ):
        subset = [
            row
            for row in results
            if str(row["arm"]) == arm
            and str(row["grouping"]) == grouping
        ]
        arm_summary["{}__{}".format(arm, grouping)] = {
            "n": len(subset),
            "gold_correct": sum(
                1 for row in subset
                if bool(row["gold_correct"])
            ),
            "stale_profile_selected": sum(
                1 for row in subset
                if bool(row["stale_profile_selected"])
            ),
            "valid_json": sum(
                1 for row in subset
                if bool(row["response_json_valid"])
            ),
            "selected_profile_counts": dict(
                sorted(
                    Counter(
                        str(row["selected_profile_id"])
                        for row in subset
                    ).items()
                )
            ),
        }

    # ------------------------------------------------------------------------
    # Oracle / stale error roles
    # ------------------------------------------------------------------------

    error_cases = []
    selected_role_counts = defaultdict(Counter)
    wrong_attribute_counts = defaultdict(Counter)
    target_attribute_stats = defaultdict(
        lambda: {
            "n": 0,
            "correct": 0,
            "expected_stale": 0,
        }
    )
    schema_stats = defaultdict(
        lambda: {
            "n": 0,
            "correct": 0,
            "expected_stale": 0,
        }
    )
    gold_position_stats = defaultdict(
        lambda: {
            "n": 0,
            "correct": 0,
            "expected_stale": 0,
        }
    )

    for row in results:
        arm = str(row["arm"])
        grouping = str(row["grouping"])
        task = task_by_id[str(row["task_id"])]
        target_attr = str(task["target_attribute"])
        schema = subject_schema_label(task)
        gold_id = str(row["gold_profile_id"])
        selected_id = str(row["selected_profile_id"])

        stat_key = "{}__{}".format(arm, grouping)
        target_attribute_stats[(stat_key, target_attr)]["n"] += 1
        schema_stats[(stat_key, schema)]["n"] += 1
        gold_position_stats[(stat_key, gold_id)]["n"] += 1

        if bool(row["gold_correct"]):
            target_attribute_stats[(stat_key, target_attr)][
                "correct"
            ] += 1
            schema_stats[(stat_key, schema)]["correct"] += 1
            gold_position_stats[(stat_key, gold_id)]["correct"] += 1

        if bool(row["stale_profile_selected"]):
            target_attribute_stats[(stat_key, target_attr)][
                "expected_stale"
            ] += 1
            schema_stats[(stat_key, schema)][
                "expected_stale"
            ] += 1
            gold_position_stats[(stat_key, gold_id)][
                "expected_stale"
            ] += 1

        if arm not in ("oracle", "stale"):
            continue

        expected_ok = (
            bool(row["gold_correct"])
            if arm == "oracle"
            else bool(row["stale_profile_selected"])
        )

        if expected_ok:
            continue

        role, wrong_attrs = selected_role(
            task,
            selected_id,
        )
        selected_role_counts[stat_key][role] += 1
        for attr in wrong_attrs:
            wrong_attribute_counts[stat_key][attr] += 1

        error_cases.append(
            {
                "task_id": str(row["task_id"]),
                "case_id": str(row["case_id"]),
                "update_id": str(row["update_id"]),
                "arm": arm,
                "grouping": (
                    None
                    if row["grouping"] is None
                    else str(row["grouping"])
                ),
                "target_attribute": target_attr,
                "subject_schema": schema,
                "gold_profile_id": gold_id,
                "stale_profile_id": str(
                    row["stale_profile_id"]
                ),
                "selected_profile_id": selected_id,
                "selected_semantic_role": role,
                "selected_wrong_attributes": wrong_attrs,
                "gold_correct": bool(row["gold_correct"]),
                "stale_profile_selected": bool(
                    row["stale_profile_selected"]
                ),
                "subject": str(task["subject"]),
                "correct_values": dict(task["correct_values"]),
                "alternative_values": dict(
                    task["alternative_values"]
                ),
            }
        )

    # ------------------------------------------------------------------------
    # Paired grouping diagnostics
    # ------------------------------------------------------------------------

    def paired_grouping_summary(arm: str) -> dict:
        counts = Counter()
        same_selected = 0
        selected_pairs = Counter()

        for task in tasks:
            task_id = str(task["task_id"])
            g1 = rows_by_task_arm_grouping[
                (task_id, arm, "g1")
            ]
            g3 = rows_by_task_arm_grouping[
                (task_id, arm, "g3")
            ]

            if arm == "oracle":
                g1_ok = bool(g1["gold_correct"])
                g3_ok = bool(g3["gold_correct"])
            elif arm == "stale":
                g1_ok = bool(g1["stale_profile_selected"])
                g3_ok = bool(g3["stale_profile_selected"])
            else:
                raise ValueError("Unsupported paired arm.")

            if g1_ok and g3_ok:
                category = "both_expected"
            elif g1_ok and not g3_ok:
                category = "g1_only_expected"
            elif not g1_ok and g3_ok:
                category = "g3_only_expected"
            else:
                category = "neither_expected"

            counts[category] += 1

            s1 = str(g1["selected_profile_id"])
            s3 = str(g3["selected_profile_id"])
            if s1 == s3:
                same_selected += 1
            selected_pairs["{}|{}".format(s1, s3)] += 1

        return {
            "n_tasks": EXPECTED_TASKS,
            "outcome_pairs": dict(
                sorted(counts.items())
            ),
            "same_selected_profile": same_selected,
            "different_selected_profile": (
                EXPECTED_TASKS - same_selected
            ),
            "same_selected_profile_rate": (
                float(same_selected) / EXPECTED_TASKS
            ),
            "selected_profile_pair_counts": dict(
                selected_pairs.most_common()
            ),
        }

    paired_oracle = paired_grouping_summary("oracle")
    paired_stale = paired_grouping_summary("stale")

    # ------------------------------------------------------------------------
    # Shared-vs-grouping-specific error sets
    # ------------------------------------------------------------------------

    def failure_set(arm: str, grouping: str) -> set:
        result = set()
        for task in tasks:
            row = rows_by_task_arm_grouping[
                (str(task["task_id"]), arm, grouping)
            ]
            if arm == "oracle":
                ok = bool(row["gold_correct"])
            else:
                ok = bool(row["stale_profile_selected"])
            if not ok:
                result.add(str(task["task_id"]))
        return result

    oracle_g1_fail = failure_set("oracle", "g1")
    oracle_g3_fail = failure_set("oracle", "g3")
    stale_g1_fail = failure_set("stale", "g1")
    stale_g3_fail = failure_set("stale", "g3")

    overlap = {
        "oracle": {
            "g1_fail_count": len(oracle_g1_fail),
            "g3_fail_count": len(oracle_g3_fail),
            "fail_both_groupings": len(
                oracle_g1_fail & oracle_g3_fail
            ),
            "g1_only_fail": len(
                oracle_g1_fail - oracle_g3_fail
            ),
            "g3_only_fail": len(
                oracle_g3_fail - oracle_g1_fail
            ),
            "fail_either_grouping": len(
                oracle_g1_fail | oracle_g3_fail
            ),
        },
        "stale": {
            "g1_fail_count": len(stale_g1_fail),
            "g3_fail_count": len(stale_g3_fail),
            "fail_both_groupings": len(
                stale_g1_fail & stale_g3_fail
            ),
            "g1_only_fail": len(
                stale_g1_fail - stale_g3_fail
            ),
            "g3_only_fail": len(
                stale_g3_fail - stale_g1_fail
            ),
            "fail_either_grouping": len(
                stale_g1_fail | stale_g3_fail
            ),
        },
    }

    # ------------------------------------------------------------------------
    # No-memory diagnostics
    # ------------------------------------------------------------------------

    no_memory_rows = [
        row
        for row in results
        if str(row["arm"]) == "no_memory"
    ]
    no_memory_selected = Counter(
        str(row["selected_profile_id"])
        for row in no_memory_rows
    )
    no_memory_gold_by_position = Counter()
    no_memory_n_by_gold_position = Counter()

    for row in no_memory_rows:
        gold = str(row["gold_profile_id"])
        no_memory_n_by_gold_position[gold] += 1
        if bool(row["gold_correct"]):
            no_memory_gold_by_position[gold] += 1

    no_memory = {
        "n": len(no_memory_rows),
        "gold_correct": sum(
            1 for row in no_memory_rows
            if bool(row["gold_correct"])
        ),
        "gold_accuracy": (
            sum(
                1 for row in no_memory_rows
                if bool(row["gold_correct"])
            )
            / float(len(no_memory_rows))
        ),
        "selected_profile_counts": dict(
            sorted(no_memory_selected.items())
        ),
        "gold_hits_by_gold_position": {
            profile_id: {
                "n": int(
                    no_memory_n_by_gold_position[
                        profile_id
                    ]
                ),
                "gold_hits": int(
                    no_memory_gold_by_position[
                        profile_id
                    ]
                ),
            }
            for profile_id in sorted(
                no_memory_n_by_gold_position
            )
        },
    }

    # ------------------------------------------------------------------------
    # Compact tables
    # ------------------------------------------------------------------------

    def flatten_stats(
        stats: Mapping[Tuple[str, str], Mapping[str, int]],
    ) -> List[dict]:
        rows = []
        for (arm_grouping, category), values in sorted(
            stats.items()
        ):
            rows.append(
                {
                    "arm_grouping": arm_grouping,
                    "category": category,
                    "n": int(values["n"]),
                    "gold_correct": int(values["correct"]),
                    "gold_accuracy": (
                        float(values["correct"])
                        / float(values["n"])
                        if values["n"]
                        else 0.0
                    ),
                    "stale_expected_selected": int(
                        values["expected_stale"]
                    ),
                    "stale_expected_rate": (
                        float(values["expected_stale"])
                        / float(values["n"])
                        if values["n"]
                        else 0.0
                    ),
                }
            )
        return rows

    audit = {
        "status": "OFFLINE_READER_CONTROL_ERROR_AUDIT_COMPLETE",
        "model_calls": 0,
        "source_results_sha256": sha256_file(RESULTS_PATH),
        "source_run_summary_sha256": sha256_file(
            RUN_SUMMARY_PATH
        ),
        "source_gate_status": str(
            run_summary.get("status", "")
        ),
        "frozen_reader_gates": dict(
            freeze["reader_gates"]
        ),
        "arm_summary": arm_summary,
        "paired_grouping_diagnostics": {
            "oracle": paired_oracle,
            "stale": paired_stale,
        },
        "failure_overlap_between_groupings": overlap,
        "unexpected_selection_roles": (
            nested_counter_to_dict(
                selected_role_counts
            )
        ),
        "wrong_attribute_counts_in_unexpected_selections": (
            nested_counter_to_dict(
                wrong_attribute_counts
            )
        ),
        "by_target_attribute": flatten_stats(
            target_attribute_stats
        ),
        "by_subject_schema": flatten_stats(
            schema_stats
        ),
        "by_gold_position": flatten_stats(
            gold_position_stats
        ),
        "no_memory_diagnostics": no_memory,
        "error_case_count": len(error_cases),
        "interpretation_guardrails": [
            (
                "This audit is descriptive. It must not be used to "
                "lower the frozen reader gate after observing failures."
            ),
            (
                "Shared failures across g1 and g3 suggest task/reader "
                "difficulty; grouping-discordant failures suggest reader "
                "sensitivity to memory layout. Neither pattern alone "
                "establishes a causal mechanism."
            ),
            (
                "No prospective updater calls should begin while the "
                "frozen reader-suitability gate remains failed."
            ),
        ],
    }

    return audit, error_cases


# ============================================================================
# Markdown report
# ============================================================================

def render_report(
    audit: Mapping[str, Any],
    error_cases: Sequence[Mapping[str, Any]],
) -> str:
    arm = audit["arm_summary"]
    paired = audit["paired_grouping_diagnostics"]
    overlap = audit["failure_overlap_between_groupings"]
    roles = audit["unexpected_selection_roles"]
    no_mem = audit["no_memory_diagnostics"]

    lines = [
        "# Layer-2 Reader-Control Offline Error Audit",
        "",
        "**Mode:** ZERO MODEL CALLS  ",
        "**Purpose:** diagnose the failed frozen reader-suitability gate; "
        "do not tune the gate from these results.",
        "",
        "## 1. Control outcomes",
        "",
        "| Control | Correct/expected | n | Rate |",
        "|---|---:|---:|---:|",
    ]

    for key, label, metric in (
        ("oracle__g1", "Oracle g1", "gold_correct"),
        ("oracle__g3", "Oracle g3", "gold_correct"),
        ("stale__g1", "Stale g1", "stale_profile_selected"),
        ("stale__g3", "Stale g3", "stale_profile_selected"),
        ("no_memory__None", "No-memory gold", "gold_correct"),
    ):
        row = arm[key]
        value = int(row[metric])
        n = int(row["n"])
        lines.append(
            "| {} | {} | {} | {:.1f}% |".format(
                label,
                value,
                n,
                pct(value, n),
            )
        )

    lines.extend(
        [
            "",
            "## 2. Are failures shared across g1 and g3?",
            "",
            "### Oracle",
            "",
            "| Pattern | Tasks |",
            "|---|---:|",
        ]
    )
    for key in (
        "both_expected",
        "g1_only_expected",
        "g3_only_expected",
        "neither_expected",
    ):
        lines.append(
            "| {} | {} |".format(
                key,
                paired["oracle"]["outcome_pairs"].get(
                    key,
                    0,
                ),
            )
        )

    lines.extend(
        [
            "",
            "Oracle failure overlap:",
            "",
            "- fail both groupings: **{}**".format(
                overlap["oracle"]["fail_both_groupings"]
            ),
            "- g1-only failure: **{}**".format(
                overlap["oracle"]["g1_only_fail"]
            ),
            "- g3-only failure: **{}**".format(
                overlap["oracle"]["g3_only_fail"]
            ),
            "",
            "### Stale",
            "",
            "| Pattern | Tasks |",
            "|---|---:|",
        ]
    )
    for key in (
        "both_expected",
        "g1_only_expected",
        "g3_only_expected",
        "neither_expected",
    ):
        lines.append(
            "| {} | {} |".format(
                key,
                paired["stale"]["outcome_pairs"].get(
                    key,
                    0,
                ),
            )
        )

    lines.extend(
        [
            "",
            "Stale failure overlap:",
            "",
            "- fail both groupings: **{}**".format(
                overlap["stale"]["fail_both_groupings"]
            ),
            "- g1-only failure: **{}**".format(
                overlap["stale"]["g1_only_fail"]
            ),
            "- g3-only failure: **{}**".format(
                overlap["stale"]["g3_only_fail"]
            ),
            "",
            "## 3. What wrong profile does the reader choose?",
            "",
        ]
    )

    for key in (
        "oracle__g1",
        "oracle__g3",
        "stale__g1",
        "stale__g3",
    ):
        lines.append("### {}".format(key))
        counts = roles.get(key, {})
        if not counts:
            lines.append("")
            lines.append("No unexpected selections.")
            lines.append("")
        else:
            lines.append("")
            lines.append("| Selected semantic role | Count |")
            lines.append("|---|---:|")
            for role, count in sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0]),
            ):
                lines.append(
                    "| {} | {} |".format(role, count)
                )
            lines.append("")

    lines.extend(
        [
            "## 4. No-memory position behavior",
            "",
            "- Gold hits: **{}/{} ({:.1f}%)**".format(
                no_mem["gold_correct"],
                no_mem["n"],
                100.0 * no_mem["gold_accuracy"],
            ),
            "",
            "| Selected profile | Count |",
            "|---|---:|",
        ]
    )
    for profile_id, count in sorted(
        no_mem["selected_profile_counts"].items()
    ):
        lines.append(
            "| {} | {} |".format(profile_id, count)
        )

    # Most common category rates for target attribute.
    lines.extend(
        [
            "",
            "## 5. Accuracy by target attribute",
            "",
            "| Arm/grouping | Attribute | n | Gold acc. | Stale-expected rate |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in audit["by_target_attribute"]:
        lines.append(
            "| {} | {} | {} | {:.1f}% | {:.1f}% |".format(
                row["arm_grouping"],
                row["category"],
                row["n"],
                100.0 * row["gold_accuracy"],
                100.0 * row["stale_expected_rate"],
            )
        )

    lines.extend(
        [
            "",
            "## 6. Interpretation guardrails",
            "",
        ]
    )
    for item in audit["interpretation_guardrails"]:
        lines.append("- " + str(item))

    lines.extend(
        [
            "",
            "## 7. Error-row inventory",
            "",
            "Detailed unexpected Oracle/Stale rows are saved in "
            "`reader_control_error_cases.jsonl`.",
            "",
            "Total unexpected Oracle/Stale rows: **{}**.".format(
                len(error_cases)
            ),
            "",
        ]
    )

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    paths.require_output_stage(paths.SELECTION, Path(__file__).with_name("build_selection.py"))
    print("=" * 78)
    print("LAYER-2 READER-CONTROL OFFLINE ERROR AUDIT")
    print("=" * 78)
    print("Mode: ZERO MODEL CALLS")
    print()

    audit, error_cases = build_audit()

    write_json(AUDIT_JSON_PATH, audit)
    write_jsonl(ERROR_CASES_PATH, error_cases)
    REPORT_MD_PATH.write_text(
        render_report(audit, error_cases),
        encoding="utf-8",
    )

    arm = audit["arm_summary"]
    paired = audit["paired_grouping_diagnostics"]
    overlap = audit["failure_overlap_between_groupings"]
    no_mem = audit["no_memory_diagnostics"]

    print("Integrity: PASS")
    print()
    print(
        "Oracle g1:         {}/100".format(
            arm["oracle__g1"]["gold_correct"]
        )
    )
    print(
        "Oracle g3:         {}/100".format(
            arm["oracle__g3"]["gold_correct"]
        )
    )
    print(
        "Stale g1 expected: {}/100".format(
            arm["stale__g1"]["stale_profile_selected"]
        )
    )
    print(
        "Stale g3 expected: {}/100".format(
            arm["stale__g3"]["stale_profile_selected"]
        )
    )
    print(
        "No-memory gold:    {}/100".format(
            no_mem["gold_correct"]
        )
    )
    print()
    print("Oracle paired grouping outcomes:")
    print(
        "  both expected:    {}".format(
            paired["oracle"]["outcome_pairs"].get(
                "both_expected",
                0,
            )
        )
    )
    print(
        "  g1 only expected: {}".format(
            paired["oracle"]["outcome_pairs"].get(
                "g1_only_expected",
                0,
            )
        )
    )
    print(
        "  g3 only expected: {}".format(
            paired["oracle"]["outcome_pairs"].get(
                "g3_only_expected",
                0,
            )
        )
    )
    print(
        "  neither expected: {}".format(
            paired["oracle"]["outcome_pairs"].get(
                "neither_expected",
                0,
            )
        )
    )
    print(
        "  fail both:        {}".format(
            overlap["oracle"]["fail_both_groupings"]
        )
    )
    print()
    print("Stale paired grouping outcomes:")
    print(
        "  both expected:    {}".format(
            paired["stale"]["outcome_pairs"].get(
                "both_expected",
                0,
            )
        )
    )
    print(
        "  g1 only expected: {}".format(
            paired["stale"]["outcome_pairs"].get(
                "g1_only_expected",
                0,
            )
        )
    )
    print(
        "  g3 only expected: {}".format(
            paired["stale"]["outcome_pairs"].get(
                "g3_only_expected",
                0,
            )
        )
    )
    print(
        "  neither expected: {}".format(
            paired["stale"]["outcome_pairs"].get(
                "neither_expected",
                0,
            )
        )
    )
    print(
        "  fail both:        {}".format(
            overlap["stale"]["fail_both_groupings"]
        )
    )
    print()
    print(
        "Unexpected Oracle/Stale rows: {}".format(
            audit["error_case_count"]
        )
    )
    print("Model calls made: 0")
    print()
    print("Wrote:")
    print("  {}".format(AUDIT_JSON_PATH))
    print("  {}".format(ERROR_CASES_PATH))
    print("  {}".format(REPORT_MD_PATH))


if __name__ == "__main__":
    main()
