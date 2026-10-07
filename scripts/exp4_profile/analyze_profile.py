#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze paired strict-state and profile-readout effects in experiment 4.

Use --check for read-only verification of saved analyses."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "atomic_memory").is_dir())
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from atomic_memory import paths
FORMAL_DIR = paths.PROFILE

PREFLIGHT_PATH = FORMAL_DIR / "formal_preflight.json"
FREEZE_PATH = FORMAL_DIR / "freeze_manifest.json"
TASKS_PATH = FORMAL_DIR / "extraction_tasks.jsonl"

UPDATER_RESULTS_PATH = FORMAL_DIR / "updater_results.jsonl"
UPDATER_SUMMARY_PATH = FORMAL_DIR / "updater_run_summary.json"

ACTUAL_READER_PLAN_PATH = FORMAL_DIR / "actual_reader_plan.jsonl"
ACTUAL_READER_PLAN_MANIFEST_PATH = (
    FORMAL_DIR / "actual_reader_plan_manifest.json"
)
ACTUAL_READER_RESULTS_PATH = FORMAL_DIR / "actual_reader_results.jsonl"
ACTUAL_READER_SUMMARY_PATH = (
    FORMAL_DIR / "actual_reader_run_summary.json"
)

# Root marker reserved by the formal protocol.
FORMAL_ANALYSIS_JSON_PATH = FORMAL_DIR / "formal_analysis.json"

ANALYSIS_DIR = FORMAL_DIR / "formal_analysis"
ANALYSIS_MANIFEST_PATH = ANALYSIS_DIR / "analysis_manifest.json"
CONDITION_SUMMARY_PATH = ANALYSIS_DIR / "condition_summary.csv"
CONTRAST_SUMMARY_PATH = ANALYSIS_DIR / "contrast_summary.csv"
STATE_FUNCTION_GAP_PATH = (
    ANALYSIS_DIR / "state_function_gap_summary.csv"
)
JOINT_OUTCOME_PATH = ANALYSIS_DIR / "joint_outcome_summary.csv"
FAILURE_REASON_PATH = (
    ANALYSIS_DIR / "failure_reason_functional_summary.csv"
)
FAILURE_STAGE_PATH = (
    ANALYSIS_DIR / "failure_stage_functional_summary.csv"
)
STRICT_FAILURE_PROFILE_CLASS_PATH = (
    ANALYSIS_DIR / "strict_failure_profile_class_summary.csv"
)
REPORT_PATH = ANALYSIS_DIR / "report.md"

PROTOCOL_VERSION = "profile-extraction-layer2-formal-v1"

EXPECTED_ROWS = 1200
EXPECTED_CASES = 50
EXPECTED_TARGETS_PER_CASE = 2

MODELS = (
    "llama3.1:8b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "gemma3:12b-it-q4_K_M",
)

CONDITIONS = (
    "L1_select",
    "L1_patch",
    "L3_select",
    "L3_patch",
)

CONDITION_LABELS = {
    "L1_select": "g1 / Select",
    "L1_patch": "g1 / Patch",
    "L3_select": "g3 / Select",
    "L3_patch": "g3 / Patch",
}

# Fixed now for the frozen offline analysis; changing it should not change
# point estimates and should only change bootstrap Monte Carlo noise.
BOOTSTRAP_SEED = 20261012
BOOTSTRAP_REPS = 20000
CI_LEVEL = 0.95

# All contrasts are LEFT - RIGHT, matching the frozen formal estimands.
CONTRAST_SPECS = (
    {
        "contrast_id": "F_Patch",
        "metric": "functional_task_pass",
        "role": "PRIMARY_FUNCTIONAL",
        "interface": "Patch",
        "left": "L1_patch",
        "right": "L3_patch",
        "definition": "TaskAcc(g1,Patch)-TaskAcc(g3,Patch)",
    },
    {
        "contrast_id": "G_Patch",
        "metric": "strict_state_success",
        "role": "MATCHING_STATE",
        "interface": "Patch",
        "left": "L1_patch",
        "right": "L3_patch",
        "definition": "StrictAcc(g1,Patch)-StrictAcc(g3,Patch)",
    },
    {
        "contrast_id": "F_Select",
        "metric": "functional_task_pass",
        "role": "SUPPORTING_FUNCTIONAL",
        "interface": "Select",
        "left": "L1_select",
        "right": "L3_select",
        "definition": "TaskAcc(g1,Select)-TaskAcc(g3,Select)",
    },
    {
        "contrast_id": "G_Select",
        "metric": "strict_state_success",
        "role": "SUPPORTING_STATE",
        "interface": "Select",
        "left": "L1_select",
        "right": "L3_select",
        "definition": "StrictAcc(g1,Select)-StrictAcc(g3,Select)",
    },
)


# ============================================================================
# IO / hashing
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
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(
                    "{}:{} is not a JSON object.".format(
                        path,
                        line_number,
                    )
                )
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


# ============================================================================
# Statistics
# ============================================================================

def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean() requires non-empty values")
    return float(sum(float(x) for x in values)) / float(len(values))


def percentile(
    sorted_values: Sequence[float],
    q: float,
) -> float:
    if not sorted_values:
        raise ValueError("percentile() requires non-empty values")
    if q <= 0.0:
        return float(sorted_values[0])
    if q >= 1.0:
        return float(sorted_values[-1])

    position = (len(sorted_values) - 1) * q
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return float(sorted_values[lo])

    weight = position - lo
    return (
        float(sorted_values[lo]) * (1.0 - weight)
        + float(sorted_values[hi]) * weight
    )


def percentile_ci(
    values: Sequence[float],
    level: float = CI_LEVEL,
) -> Tuple[float, float]:
    ordered = sorted(float(x) for x in values)
    alpha = 1.0 - float(level)
    return (
        percentile(ordered, alpha / 2.0),
        percentile(ordered, 1.0 - alpha / 2.0),
    )


def case_cluster_bootstrap_mean(
    case_values: Mapping[str, float],
    rng: random.Random,
    reps: int = BOOTSTRAP_REPS,
) -> Tuple[float, float, float]:
    """
    Bootstrap whole case clusters.

    Each scalar in case_values is already the mean of that case's two target
    outcomes for the estimand. Resampling these case values therefore retains
    both targets as one cluster.
    """
    case_ids = sorted(case_values)
    if len(case_ids) != EXPECTED_CASES:
        raise RuntimeError(
            "Expected {} case clusters, found {}.".format(
                EXPECTED_CASES,
                len(case_ids),
            )
        )

    observed = mean(
        [float(case_values[case_id]) for case_id in case_ids]
    )

    bootstrap_values: List[float] = []
    n_cases = len(case_ids)

    for _ in range(reps):
        total = 0.0
        for _sample_index in range(n_cases):
            sampled_case = case_ids[rng.randrange(n_cases)]
            total += float(case_values[sampled_case])
        bootstrap_values.append(total / float(n_cases))

    ci_low, ci_high = percentile_ci(bootstrap_values)
    return observed, ci_low, ci_high


def fmt_pct(value: float) -> str:
    return "{:.1f}%".format(float(value) * 100.0)


def fmt_pp(value: float) -> str:
    return "{:+.1f} pp".format(float(value) * 100.0)


# ============================================================================
# Integrity audit
# ============================================================================

def audit_sources() -> dict:
    required = (
        PREFLIGHT_PATH,
        FREEZE_PATH,
        TASKS_PATH,
        UPDATER_RESULTS_PATH,
        UPDATER_SUMMARY_PATH,
        ACTUAL_READER_PLAN_PATH,
        ACTUAL_READER_PLAN_MANIFEST_PATH,
        ACTUAL_READER_RESULTS_PATH,
        ACTUAL_READER_SUMMARY_PATH,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                "Missing required formal artifact: {}".format(path)
            )

    preflight = read_json(PREFLIGHT_PATH)
    freeze = read_json(FREEZE_PATH)
    updater_summary = read_json(UPDATER_SUMMARY_PATH)
    reader_plan_manifest = read_json(
        ACTUAL_READER_PLAN_MANIFEST_PATH
    )
    reader_summary = read_json(ACTUAL_READER_SUMMARY_PATH)

    tasks = read_jsonl(TASKS_PATH)
    updater_rows = read_jsonl(UPDATER_RESULTS_PATH)
    reader_plan = read_jsonl(ACTUAL_READER_PLAN_PATH)
    rows = read_jsonl(ACTUAL_READER_RESULTS_PATH)

    if preflight.get("status") != "PASS":
        raise RuntimeError("Formal preflight is not PASS.")
    if freeze.get("status") != "FORMAL_PROTOCOL_FROZEN":
        raise RuntimeError("Formal protocol is not frozen.")
    if freeze.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Protocol version mismatch.")

    if updater_summary.get("status") != "FORMAL_UPDATER_STAGE_COMPLETE":
        raise RuntimeError("Updater stage is not COMPLETE.")
    if int(updater_summary.get("completed_calls", -1)) != EXPECTED_ROWS:
        raise RuntimeError("Updater summary is not 1200/1200 complete.")

    if reader_plan_manifest.get("status") != "ACTUAL_READER_PLAN_FROZEN":
        raise RuntimeError("Actual-reader plan is not frozen.")

    if reader_summary.get("status") != (
        "FORMAL_ACTUAL_READER_STAGE_COMPLETE"
    ):
        raise RuntimeError(
            "Actual-reader stage is not COMPLETE: {!r}".format(
                reader_summary.get("status")
            )
        )

    if len(tasks) != 100:
        raise RuntimeError(
            "Expected 100 extraction tasks, found {}.".format(len(tasks))
        )
    if len(updater_rows) != EXPECTED_ROWS:
        raise RuntimeError("Updater results are not 1,200 rows.")
    if len(reader_plan) != EXPECTED_ROWS:
        raise RuntimeError("Actual-reader plan is not 1,200 rows.")
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError("Actual-reader results are not 1,200 rows.")

    if sha256_file(UPDATER_RESULTS_PATH) != str(
        updater_summary["results_sha256"]
    ):
        raise RuntimeError("Updater-results hash mismatch.")

    if sha256_file(ACTUAL_READER_PLAN_PATH) != str(
        reader_plan_manifest["actual_reader_plan_sha256"]
    ):
        raise RuntimeError("Actual-reader-plan hash mismatch.")

    if sha256_file(ACTUAL_READER_RESULTS_PATH) != str(
        reader_summary["results_sha256"]
    ):
        raise RuntimeError("Actual-reader-results hash mismatch.")

    if sha256_file(UPDATER_RESULTS_PATH) != str(
        reader_plan_manifest["source_updater_results_sha256"]
    ):
        raise RuntimeError(
            "Actual-reader plan is not bound to current updater results."
        )

    if sha256_file(ACTUAL_READER_PLAN_PATH) != str(
        reader_summary["actual_reader_plan_sha256"]
    ):
        raise RuntimeError(
            "Actual-reader summary is not bound to current reader plan."
        )

    if tuple(freeze.get("updater_models", [])) != MODELS:
        raise RuntimeError("Frozen updater-model list changed.")
    if tuple(freeze.get("conditions", [])) != CONDITIONS:
        raise RuntimeError("Frozen condition list changed.")

    commitments = freeze.get("analysis_commitments", {})
    if not bool(commitments.get("models_analyzed_separately")):
        raise RuntimeError(
            "Frozen protocol does not preserve model-separated analysis."
        )
    if not bool(commitments.get("whole_case_bootstrap")):
        raise RuntimeError(
            "Frozen protocol does not specify whole-case bootstrap."
        )
    if int(commitments.get("bootstrap_replicates", -1)) != BOOTSTRAP_REPS:
        raise RuntimeError(
            "Frozen bootstrap replicate count differs from 20,000."
        )

    if str(commitments.get("primary_functional_contrast")) != (
        "TaskAcc(g1,Patch)-TaskAcc(g3,Patch)"
    ):
        raise RuntimeError("Primary functional contrast changed.")
    if str(commitments.get("matching_state_contrast")) != (
        "StrictAcc(g1,Patch)-StrictAcc(g3,Patch)"
    ):
        raise RuntimeError("Matching state contrast changed.")
    if str(commitments.get("supporting_select_contrast")) != (
        "TaskAcc(g1,Select)-TaskAcc(g3,Select)"
    ):
        raise RuntimeError("Supporting Select contrast changed.")

    case_ids = sorted({str(row["case_id"]) for row in rows})
    if len(case_ids) != EXPECTED_CASES:
        raise RuntimeError(
            "Expected 50 case clusters, found {}.".format(len(case_ids))
        )

    # Identity and cell checks.
    identities = set()
    by_cell = defaultdict(list)
    by_model_case_update_condition = {}

    for expected_index, row in enumerate(rows):
        if int(row["logical_reader_index"]) != expected_index:
            raise RuntimeError(
                "Actual-reader result order is not 0..1199."
            )

        identity = (
            str(row["updater_model"]),
            str(row["case_id"]),
            str(row["update_id"]),
            str(row["condition"]),
        )
        if identity in identities:
            raise RuntimeError(
                "Duplicate analysis identity: {}".format(identity)
            )
        identities.add(identity)
        by_model_case_update_condition[identity] = row

        model = str(row["updater_model"])
        condition = str(row["condition"])
        if model not in MODELS:
            raise RuntimeError("Unexpected model: {}".format(model))
        if condition not in CONDITIONS:
            raise RuntimeError(
                "Unexpected condition: {}".format(condition)
            )
        by_cell[(model, condition)].append(row)

        strict_ok = bool(row["strict_state_success"])
        task_ok = bool(row["functional_task_pass"])
        expected_joint = (
            "strict_pass_task_pass"
            if strict_ok and task_ok
            else "strict_pass_task_fail"
            if strict_ok and not task_ok
            else "strict_fail_task_pass"
            if not strict_ok and task_ok
            else "strict_fail_task_fail"
        )
        if str(row["joint_strict_task"]) != expected_joint:
            raise RuntimeError(
                "Joint outcome label mismatch at row {}.".format(
                    expected_index
                )
            )

    if len(identities) != EXPECTED_ROWS:
        raise RuntimeError("Expected 1,200 unique analysis identities.")

    for model in MODELS:
        for condition in CONDITIONS:
            cell = by_cell[(model, condition)]
            if len(cell) != 100:
                raise RuntimeError(
                    "{} / {} expected 100 rows, found {}.".format(
                        model,
                        condition,
                        len(cell),
                    )
                )
            per_case = Counter(str(row["case_id"]) for row in cell)
            if set(per_case.values()) != {EXPECTED_TARGETS_PER_CASE}:
                raise RuntimeError(
                    "{} / {} does not have exactly two targets per case."
                    .format(model, condition)
                )

    # Verify paired target completeness across four conditions.
    for model in MODELS:
        model_rows = [
            row
            for row in rows
            if str(row["updater_model"]) == model
        ]
        case_update_pairs = {
            (str(row["case_id"]), str(row["update_id"]))
            for row in model_rows
        }
        if len(case_update_pairs) != 100:
            raise RuntimeError(
                "{} expected 100 case/update pairs.".format(model)
            )
        for case_id, update_id in case_update_pairs:
            present = {
                condition
                for condition in CONDITIONS
                if (
                    model,
                    case_id,
                    update_id,
                    condition,
                ) in by_model_case_update_condition
            }
            if present != set(CONDITIONS):
                raise RuntimeError(
                    "Missing paired conditions for {} / {} / {}.".format(
                        model,
                        case_id,
                        update_id,
                    )
                )

    strict_total = sum(
        1 for row in rows if bool(row["strict_state_success"])
    )
    task_total = sum(
        1 for row in rows if bool(row["functional_task_pass"])
    )

    # Bind to completed run summary, without using it as analysis input.
    if int(reader_summary.get("strict_success_total", -1)) != strict_total:
        raise RuntimeError(
            "Strict-success total differs from completed reader summary."
        )
    if int(reader_summary.get("task_success_total", -1)) != task_total:
        raise RuntimeError(
            "Task-success total differs from completed reader summary."
        )

    source_hashes = {
        "formal_preflight.json": sha256_file(PREFLIGHT_PATH),
        "freeze_manifest.json": sha256_file(FREEZE_PATH),
        "extraction_tasks.jsonl": sha256_file(TASKS_PATH),
        "updater_results.jsonl": sha256_file(UPDATER_RESULTS_PATH),
        "updater_run_summary.json": sha256_file(UPDATER_SUMMARY_PATH),
        "actual_reader_plan.jsonl": sha256_file(ACTUAL_READER_PLAN_PATH),
        "actual_reader_plan_manifest.json": sha256_file(
            ACTUAL_READER_PLAN_MANIFEST_PATH
        ),
        "actual_reader_results.jsonl": sha256_file(
            ACTUAL_READER_RESULTS_PATH
        ),
        "actual_reader_run_summary.json": sha256_file(
            ACTUAL_READER_SUMMARY_PATH
        ),
    }

    return {
        "preflight": preflight,
        "freeze": freeze,
        "rows": rows,
        "by_cell": by_cell,
        "paired": by_model_case_update_condition,
        "case_ids": case_ids,
        "source_hashes": source_hashes,
        "strict_success_total": strict_total,
        "task_success_total": task_total,
    }


# ============================================================================
# Case-level helpers
# ============================================================================

def metric_case_means(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> Dict[str, float]:
    # Both targets belong to one case; inference resamples these case means.
    by_case: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        value = bool(row[metric])
        by_case[str(row["case_id"])].append(
            1.0 if value else 0.0
        )

    result: Dict[str, float] = {}
    for case_id, values in by_case.items():
        if len(values) != EXPECTED_TARGETS_PER_CASE:
            raise RuntimeError(
                "{} expected two target outcomes, found {}.".format(
                    case_id,
                    len(values),
                )
            )
        result[case_id] = mean(values)
    return result


def paired_case_differences(
    *,
    audit: Mapping[str, Any],
    model: str,
    left: str,
    right: str,
    metric: str,
) -> Tuple[Dict[str, float], dict]:
    paired = audit["paired"]
    by_case_diffs: Dict[str, List[float]] = defaultdict(list)

    target_left_only = 0
    target_right_only = 0
    target_tie = 0

    # Iterate exact paired target identities.
    model_pairs = sorted(
        {
            (case_id, update_id)
            for (
                row_model,
                case_id,
                update_id,
                _condition,
            ) in paired.keys()
            if row_model == model
        }
    )

    for case_id, update_id in model_pairs:
        left_row = paired[
            (model, case_id, update_id, left)
        ]
        right_row = paired[
            (model, case_id, update_id, right)
        ]

        left_value = (
            1.0 if bool(left_row[metric]) else 0.0
        )
        right_value = (
            1.0 if bool(right_row[metric]) else 0.0
        )

        # Contrast is left - right.
        by_case_diffs[case_id].append(
            left_value - right_value
        )

        if left_value > right_value:
            target_left_only += 1
        elif right_value > left_value:
            target_right_only += 1
        else:
            target_tie += 1

    case_diffs: Dict[str, float] = {}
    for case_id, values in by_case_diffs.items():
        if len(values) != EXPECTED_TARGETS_PER_CASE:
            raise RuntimeError(
                "{} paired contrast expected two target differences."
                .format(case_id)
            )
        case_diffs[case_id] = mean(values)

    case_signs = Counter()
    for value in case_diffs.values():
        if value > 0.0:
            case_signs["left_higher"] += 1
        elif value < 0.0:
            case_signs["right_higher"] += 1
        else:
            case_signs["tie"] += 1

    diagnostics = {
        "paired_target_left_only_success": target_left_only,
        "paired_target_right_only_success": target_right_only,
        "paired_target_tie": target_tie,
        "n_paired_targets": len(model_pairs),
        "case_left_higher": int(case_signs["left_higher"]),
        "case_right_higher": int(case_signs["right_higher"]),
        "case_tie": int(case_signs["tie"]),
    }
    return case_diffs, diagnostics


# ============================================================================
# Condition + joint summaries
# ============================================================================

def build_condition_summary(
    audit: Mapping[str, Any],
) -> List[dict]:
    rows_out: List[dict] = []
    rng = random.Random(BOOTSTRAP_SEED)

    for model in MODELS:
        for condition in CONDITIONS:
            rows = list(audit["by_cell"][(model, condition)])

            strict_case = metric_case_means(
                rows,
                "strict_state_success",
            )
            task_case = metric_case_means(
                rows,
                "functional_task_pass",
            )

            (
                strict_rate,
                strict_ci_low,
                strict_ci_high,
            ) = case_cluster_bootstrap_mean(
                strict_case,
                rng,
            )
            (
                task_rate,
                task_ci_low,
                task_ci_high,
            ) = case_cluster_bootstrap_mean(
                task_case,
                rng,
            )

            joint = Counter(
                str(row["joint_strict_task"])
                for row in rows
            )
            profile = Counter(
                str(row["profile_class"])
                for row in rows
            )

            strict_fail = sum(
                1
                for row in rows
                if not bool(row["strict_state_success"])
            )
            strict_fail_task_pass = int(
                joint["strict_fail_task_pass"]
            )

            rows_out.append(
                {
                    "model": model,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "n_rows": len(rows),
                    "n_cases": EXPECTED_CASES,
                    "strict_success_count": sum(
                        1
                        for row in rows
                        if bool(row["strict_state_success"])
                    ),
                    "strict_success_rate": strict_rate,
                    "strict_ci_low": strict_ci_low,
                    "strict_ci_high": strict_ci_high,
                    "task_success_count": sum(
                        1
                        for row in rows
                        if bool(row["functional_task_pass"])
                    ),
                    "task_success_rate": task_rate,
                    "task_ci_low": task_ci_low,
                    "task_ci_high": task_ci_high,
                    "strict_pass_task_pass": int(
                        joint["strict_pass_task_pass"]
                    ),
                    "strict_pass_task_fail": int(
                        joint["strict_pass_task_fail"]
                    ),
                    "strict_fail_task_pass": strict_fail_task_pass,
                    "strict_fail_task_fail": int(
                        joint["strict_fail_task_fail"]
                    ),
                    "strict_failure_count": strict_fail,
                    "strict_failure_functional_pass_rate": (
                        (
                            float(strict_fail_task_pass)
                            / float(strict_fail)
                        )
                        if strict_fail > 0
                        else None
                    ),
                    "oracle_current_count": int(
                        profile["oracle_current"]
                    ),
                    "stale_profile_count": int(
                        profile["stale_profile"]
                    ),
                    "other_profile_count": int(
                        profile["other_profile"]
                    ),
                    "all_unknown_count": int(
                        profile["all_unknown"]
                    ),
                    "invalid_json_profile_count": int(
                        profile["invalid_json"]
                    ),
                }
            )

    return rows_out


def build_joint_outcome_summary(
    audit: Mapping[str, Any],
) -> List[dict]:
    rows_out: List[dict] = []

    for model in MODELS:
        for condition in CONDITIONS:
            rows = list(audit["by_cell"][(model, condition)])
            counts = Counter(
                str(row["joint_strict_task"])
                for row in rows
            )
            for outcome in (
                "strict_pass_task_pass",
                "strict_pass_task_fail",
                "strict_fail_task_pass",
                "strict_fail_task_fail",
            ):
                count = int(counts[outcome])
                rows_out.append(
                    {
                        "model": model,
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "joint_outcome": outcome,
                        "count": count,
                        "rate_over_cell": (
                            float(count) / float(len(rows))
                        ),
                        "n_rows": len(rows),
                    }
                )
    return rows_out


# ============================================================================
# Frozen contrasts
# ============================================================================

def build_contrast_summary(
    audit: Mapping[str, Any],
) -> List[dict]:
    rows_out: List[dict] = []
    rng = random.Random(BOOTSTRAP_SEED + 1)

    for model in MODELS:
        for spec in CONTRAST_SPECS:
            case_diffs, diagnostics = paired_case_differences(
                audit=audit,
                model=model,
                left=str(spec["left"]),
                right=str(spec["right"]),
                metric=str(spec["metric"]),
            )

            (
                estimate,
                ci_low,
                ci_high,
            ) = case_cluster_bootstrap_mean(
                case_diffs,
                rng,
            )

            rows_out.append(
                {
                    "model": model,
                    "contrast_id": str(spec["contrast_id"]),
                    "role": str(spec["role"]),
                    "metric": str(spec["metric"]),
                    "interface": str(spec["interface"]),
                    "left_condition": str(spec["left"]),
                    "right_condition": str(spec["right"]),
                    "difference_definition": str(spec["definition"]),
                    "estimate_rate_difference": estimate,
                    "estimate_percentage_points": estimate * 100.0,
                    "cluster_bootstrap_ci_low": ci_low,
                    "cluster_bootstrap_ci_high": ci_high,
                    "cluster_bootstrap_ci_low_pp": ci_low * 100.0,
                    "cluster_bootstrap_ci_high_pp": ci_high * 100.0,
                    "n_case_clusters": EXPECTED_CASES,
                    **diagnostics,
                }
            )

    return rows_out


def build_state_function_gap_summary(
    audit: Mapping[str, Any],
) -> List[dict]:
    """
    Secondary descriptive quantity:
        TaskSuccess - StrictSuccess
    within the same model/configuration.
    """
    rows_out: List[dict] = []
    rng = random.Random(BOOTSTRAP_SEED + 2)

    for model in MODELS:
        for condition in CONDITIONS:
            rows = list(audit["by_cell"][(model, condition)])
            by_case: Dict[str, List[float]] = defaultdict(list)

            for row in rows:
                task = 1.0 if bool(
                    row["functional_task_pass"]
                ) else 0.0
                strict = 1.0 if bool(
                    row["strict_state_success"]
                ) else 0.0
                by_case[str(row["case_id"])].append(
                    task - strict
                )

            case_gap: Dict[str, float] = {}
            for case_id, values in by_case.items():
                if len(values) != EXPECTED_TARGETS_PER_CASE:
                    raise RuntimeError(
                        "{} gap summary expected two target rows."
                        .format(case_id)
                    )
                case_gap[case_id] = mean(values)

            estimate, ci_low, ci_high = (
                case_cluster_bootstrap_mean(
                    case_gap,
                    rng,
                )
            )

            rows_out.append(
                {
                    "model": model,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "definition": "TaskSuccessRate-StrictSuccessRate",
                    "estimate_rate_difference": estimate,
                    "estimate_percentage_points": estimate * 100.0,
                    "cluster_bootstrap_ci_low": ci_low,
                    "cluster_bootstrap_ci_high": ci_high,
                    "cluster_bootstrap_ci_low_pp": ci_low * 100.0,
                    "cluster_bootstrap_ci_high_pp": ci_high * 100.0,
                    "n_case_clusters": EXPECTED_CASES,
                    "interpretation": (
                        "Descriptive state-function disagreement gap; "
                        "not a causal effect."
                    ),
                }
            )

    return rows_out


# ============================================================================
# Failure-class functional audit
# ============================================================================

def build_failure_reason_summary(
    audit: Mapping[str, Any],
) -> List[dict]:
    rows_out: List[dict] = []

    for model in MODELS:
        for condition in CONDITIONS:
            failures = [
                row
                for row in audit["by_cell"][(model, condition)]
                if not bool(row["strict_state_success"])
            ]

            all_reasons = sorted(
                {
                    str(reason)
                    for row in failures
                    for reason in row.get(
                        "strict_failure_reasons",
                        [],
                    )
                }
            )

            for reason in all_reasons:
                subset = [
                    row
                    for row in failures
                    if reason
                    in {
                        str(value)
                        for value in row.get(
                            "strict_failure_reasons",
                            [],
                        )
                    }
                ]
                pass_count = sum(
                    1
                    for row in subset
                    if bool(row["functional_task_pass"])
                )
                fail_count = len(subset) - pass_count

                rows_out.append(
                    {
                        "model": model,
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "failure_reason": reason,
                        "rows_with_reason": len(subset),
                        "distinct_cases": len(
                            {
                                str(row["case_id"])
                                for row in subset
                            }
                        ),
                        "functional_task_pass_count": pass_count,
                        "functional_task_fail_count": fail_count,
                        "functional_pass_rate_among_rows_with_reason": (
                            float(pass_count) / float(len(subset))
                        ),
                        "failure_reason_labels_overlap": True,
                        "counts_are_not_additive_across_reasons": True,
                        "inference_status": "DESCRIPTIVE_ONLY",
                    }
                )

    return rows_out


def build_failure_stage_summary(
    audit: Mapping[str, Any],
) -> List[dict]:
    """
    Primary failure stage is mutually exclusive because it comes from the
    frozen updater runner's first diagnostic stage label.
    """
    rows_out: List[dict] = []

    for model in MODELS:
        for condition in CONDITIONS:
            failures = [
                row
                for row in audit["by_cell"][(model, condition)]
                if not bool(row["strict_state_success"])
            ]

            stages = sorted(
                {
                    str(row["strict_failure_stage"])
                    for row in failures
                }
            )
            for stage in stages:
                subset = [
                    row
                    for row in failures
                    if str(row["strict_failure_stage"]) == stage
                ]
                pass_count = sum(
                    1
                    for row in subset
                    if bool(row["functional_task_pass"])
                )

                rows_out.append(
                    {
                        "model": model,
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "primary_failure_stage": stage,
                        "rows": len(subset),
                        "distinct_cases": len(
                            {
                                str(row["case_id"])
                                for row in subset
                            }
                        ),
                        "functional_task_pass_count": pass_count,
                        "functional_task_fail_count": (
                            len(subset) - pass_count
                        ),
                        "functional_pass_rate": (
                            float(pass_count) / float(len(subset))
                        ),
                        "mutually_exclusive_stage_label": True,
                        "inference_status": "DESCRIPTIVE_ONLY",
                    }
                )

    return rows_out


def build_strict_failure_profile_class_summary(
    audit: Mapping[str, Any],
) -> List[dict]:
    rows_out: List[dict] = []

    for model in MODELS:
        for condition in CONDITIONS:
            failures = [
                row
                for row in audit["by_cell"][(model, condition)]
                if not bool(row["strict_state_success"])
            ]
            counts = Counter(
                str(row["profile_class"])
                for row in failures
            )
            for profile_class in (
                "oracle_current",
                "stale_profile",
                "other_profile",
                "all_unknown",
                "invalid_json",
            ):
                count = int(counts[profile_class])
                rows_out.append(
                    {
                        "model": model,
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "profile_class": profile_class,
                        "count_among_strict_failures": count,
                        "strict_failure_count": len(failures),
                        "rate_among_strict_failures": (
                            (
                                float(count) / float(len(failures))
                            )
                            if failures
                            else None
                        ),
                    }
                )

    return rows_out


# ============================================================================
# Report
# ============================================================================

def row_lookup(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> Dict[Tuple[Any, ...], Mapping[str, Any]]:
    return {
        tuple(row[key] for key in keys): row
        for row in rows
    }


def render_report(
    *,
    audit: Mapping[str, Any],
    condition_rows: Sequence[Mapping[str, Any]],
    contrast_rows: Sequence[Mapping[str, Any]],
    gap_rows: Sequence[Mapping[str, Any]],
    failure_reason_rows: Sequence[Mapping[str, Any]],
    failure_stage_rows: Sequence[Mapping[str, Any]],
) -> str:
    condition = row_lookup(
        condition_rows,
        ("model", "condition"),
    )
    contrast = row_lookup(
        contrast_rows,
        ("model", "contrast_id"),
    )

    strict_total = int(audit["strict_success_total"])
    task_total = int(audit["task_success_total"])
    strict_fail_total = EXPECTED_ROWS - strict_total

    all_rows = audit["rows"]
    joint = Counter(
        str(row["joint_strict_task"])
        for row in all_rows
    )
    disagreement = int(joint["strict_fail_task_pass"])

    lines: List[str] = []
    lines.append("# Profile Extraction Layer-2 — Frozen Offline Analysis")
    lines.append("")
    lines.append("This report is generated offline from the frozen prospective formal batch. It makes **zero model calls**.")
    lines.append("")
    lines.append("## Integrity")
    lines.append("")
    lines.append("- Formal cases: **50**; two targets per case.")
    lines.append("- Logical updater/downstream rows: **1,200/1,200**.")
    lines.append("- Cluster unit: **case**; both targets remain inside each resampled case.")
    lines.append("- Whole-case bootstrap: **20,000** replicates.")
    lines.append("- Updater models are analyzed separately; no pooled model-average estimand is used.")
    lines.append("- Failure-reason labels overlap; their row counts are not additive.")
    lines.append("- Failure-class/task relations are descriptive and do not establish causality.")
    lines.append("")
    lines.append("## Headline state–function result")
    lines.append("")
    lines.append(
        "- Strict state success: **{} / 1200 ({})**.".format(
            strict_total,
            fmt_pct(float(strict_total) / 1200.0),
        )
    )
    lines.append(
        "- Functional task success: **{} / 1200 ({})**.".format(
            task_total,
            fmt_pct(float(task_total) / 1200.0),
        )
    )
    lines.append(
        "- Strict-fail / task-pass: **{} / {} strict failures ({})**.".format(
            disagreement,
            strict_fail_total,
            fmt_pct(
                float(disagreement) / float(strict_fail_total)
            ),
        )
    )
    lines.append(
        "- Strict-pass / task-fail: **{}**.".format(
            int(joint["strict_pass_task_fail"])
        )
    )
    lines.append("")
    lines.append(
        "These are observed joint outcomes under the fixed Llama Profile Extraction reader. They do not imply that strict evaluation is unnecessary or that a strict failure is harmless for other downstream tasks."
    )
    lines.append("")
    lines.append("## Condition-level strict vs functional success")
    lines.append("")
    lines.append("| Updater | Condition | Strict | Strict 95% case-bootstrap CI | Functional | Functional 95% case-bootstrap CI | Strict-fail / Task-pass |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for model in MODELS:
        for cond in CONDITIONS:
            row = condition[(model, cond)]
            lines.append(
                "| {} | {} | {}/100 ({}) | [{} , {}] | {}/100 ({}) | [{} , {}] | {} |".format(
                    model,
                    CONDITION_LABELS[cond],
                    int(row["strict_success_count"]),
                    fmt_pct(float(row["strict_success_rate"])),
                    fmt_pct(float(row["strict_ci_low"])),
                    fmt_pct(float(row["strict_ci_high"])),
                    int(row["task_success_count"]),
                    fmt_pct(float(row["task_success_rate"])),
                    fmt_pct(float(row["task_ci_low"])),
                    fmt_pct(float(row["task_ci_high"])),
                    int(row["strict_fail_task_pass"]),
                )
            )
    lines.append("")
    lines.append("## Frozen g1 − g3 contrasts")
    lines.append("")
    lines.append("| Updater | Contrast | Role | Estimate | 95% whole-case bootstrap CI |")
    lines.append("|---|---|---|---:|---:|")
    for model in MODELS:
        for contrast_id in (
            "F_Patch",
            "G_Patch",
            "F_Select",
            "G_Select",
        ):
            row = contrast[(model, contrast_id)]
            lines.append(
                "| {} | {} | {} | {} | [{}, {}] |".format(
                    model,
                    contrast_id,
                    row["role"],
                    fmt_pp(float(row["estimate_rate_difference"])),
                    fmt_pp(float(row["cluster_bootstrap_ci_low"])),
                    fmt_pp(float(row["cluster_bootstrap_ci_high"])),
                )
            )
    lines.append("")
    lines.append(
        "`F_*` uses functional task success; `G_*` uses strict state success. Positive values mean the observed g1 rate is higher than g3 for that interface. These are model-specific paired contrasts, not model rankings."
    )
    lines.append("")
    lines.append("## Failure-class functional audit")
    lines.append("")
    lines.append(
        "Failure-reason summaries are descriptive because labels overlap. The CSV outputs retain event count, distinct-case count, task-pass count, task-fail count, and pass proportion for every observed model/condition/reason combination."
    )
    lines.append("")

    # Surface a few directly relevant descriptive rows without selecting a
    # "winner": memory_contract_failure and major stale/execution patterns.
    notable_names = (
        "memory_contract_failure",
        "execution_failure",
        "target_new_missing",
        "stale_old_retained",
        "protected_fact_loss",
        "structure_failure",
        "locator_failure",
        "unsupported_change",
    )
    notable = [
        row
        for row in failure_reason_rows
        if str(row["failure_reason"]) in notable_names
        and int(row["rows_with_reason"]) > 0
    ]
    lines.append("| Updater | Condition | Failure reason | Rows | Cases | Task pass | Task fail | Functional pass rate |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for row in notable:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["model"],
                row["condition_label"],
                row["failure_reason"],
                int(row["rows_with_reason"]),
                int(row["distinct_cases"]),
                int(row["functional_task_pass_count"]),
                int(row["functional_task_fail_count"]),
                fmt_pct(
                    float(
                        row[
                            "functional_pass_rate_among_rows_with_reason"
                        ]
                    )
                ),
            )
        )
    lines.append("")
    lines.append("## Interpretation guardrails")
    lines.append("")
    lines.append("- `strict-fail / task-pass` is state–function disagreement for this fixed local Profile Extraction task, not proof that the underlying memory state is globally correct.")
    lines.append("- `strict-fail / task-fail` is an observed co-occurrence, not proof that the strict failure caused the downstream failure.")
    lines.append("- Functional outcomes are conditional on the fixed reader and target-subject three-attribute scope.")
    lines.append("- Corruption outside the target subject can be functionally irrelevant to this task while still being a strict-state failure.")
    lines.append("- The study uses natural-rate failures under this controlled synthetic regime; it does not estimate deployment prevalence.")
    lines.append("- g1 vs g3 is an endpoint comparison only; no monotonicity claim is made.")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append("- `condition_summary.csv` — condition rates and case-bootstrap CIs.")
    lines.append("- `contrast_summary.csv` — frozen F/G contrasts and whole-case bootstrap CIs.")
    lines.append("- `state_function_gap_summary.csv` — descriptive Task − Strict gaps.")
    lines.append("- `joint_outcome_summary.csv` — four-way Strict × Task counts.")
    lines.append("- `failure_reason_functional_summary.csv` — overlapping failure reasons × task outcomes.")
    lines.append("- `failure_stage_functional_summary.csv` — mutually exclusive primary failure stage × task outcomes.")
    lines.append("- `strict_failure_profile_class_summary.csv` — downstream profile classes among strict failures.")
    lines.append("")
    return "\n".join(lines) + "\n"


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 78)
    print("PROFILE EXTRACTION LAYER-2 — FROZEN OFFLINE ANALYSIS")
    print("=" * 78)
    print("Model calls:          0")
    print("Bootstrap clusters:   case")
    print("Bootstrap replicates: {:,}".format(BOOTSTRAP_REPS))
    print("Bootstrap seed:       {}".format(BOOTSTRAP_SEED))
    print()

    audit = audit_sources()
    print("Integrity audit:      PASS")
    print("Cases:                50")
    print("Logical rows:         1200")
    print(
        "Strict success:       {}/1200".format(
            audit["strict_success_total"]
        )
    )
    print(
        "Task success:         {}/1200".format(
            audit["task_success_total"]
        )
    )
    print()

    condition_rows = build_condition_summary(audit)
    contrast_rows = build_contrast_summary(audit)
    gap_rows = build_state_function_gap_summary(audit)
    joint_rows = build_joint_outcome_summary(audit)
    failure_reason_rows = build_failure_reason_summary(audit)
    failure_stage_rows = build_failure_stage_summary(audit)
    profile_class_rows = (
        build_strict_failure_profile_class_summary(audit)
    )

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(
        CONDITION_SUMMARY_PATH,
        condition_rows,
        (
            "model",
            "condition",
            "condition_label",
            "n_rows",
            "n_cases",
            "strict_success_count",
            "strict_success_rate",
            "strict_ci_low",
            "strict_ci_high",
            "task_success_count",
            "task_success_rate",
            "task_ci_low",
            "task_ci_high",
            "strict_pass_task_pass",
            "strict_pass_task_fail",
            "strict_fail_task_pass",
            "strict_fail_task_fail",
            "strict_failure_count",
            "strict_failure_functional_pass_rate",
            "oracle_current_count",
            "stale_profile_count",
            "other_profile_count",
            "all_unknown_count",
            "invalid_json_profile_count",
        ),
    )

    write_csv(
        CONTRAST_SUMMARY_PATH,
        contrast_rows,
        (
            "model",
            "contrast_id",
            "role",
            "metric",
            "interface",
            "left_condition",
            "right_condition",
            "difference_definition",
            "estimate_rate_difference",
            "estimate_percentage_points",
            "cluster_bootstrap_ci_low",
            "cluster_bootstrap_ci_high",
            "cluster_bootstrap_ci_low_pp",
            "cluster_bootstrap_ci_high_pp",
            "paired_target_left_only_success",
            "paired_target_right_only_success",
            "paired_target_tie",
            "n_paired_targets",
            "case_left_higher",
            "case_right_higher",
            "case_tie",
            "n_case_clusters",
        ),
    )

    write_csv(
        STATE_FUNCTION_GAP_PATH,
        gap_rows,
        (
            "model",
            "condition",
            "condition_label",
            "definition",
            "estimate_rate_difference",
            "estimate_percentage_points",
            "cluster_bootstrap_ci_low",
            "cluster_bootstrap_ci_high",
            "cluster_bootstrap_ci_low_pp",
            "cluster_bootstrap_ci_high_pp",
            "n_case_clusters",
            "interpretation",
        ),
    )

    write_csv(
        JOINT_OUTCOME_PATH,
        joint_rows,
        (
            "model",
            "condition",
            "condition_label",
            "joint_outcome",
            "count",
            "rate_over_cell",
            "n_rows",
        ),
    )

    write_csv(
        FAILURE_REASON_PATH,
        failure_reason_rows,
        (
            "model",
            "condition",
            "condition_label",
            "failure_reason",
            "rows_with_reason",
            "distinct_cases",
            "functional_task_pass_count",
            "functional_task_fail_count",
            "functional_pass_rate_among_rows_with_reason",
            "failure_reason_labels_overlap",
            "counts_are_not_additive_across_reasons",
            "inference_status",
        ),
    )

    write_csv(
        FAILURE_STAGE_PATH,
        failure_stage_rows,
        (
            "model",
            "condition",
            "condition_label",
            "primary_failure_stage",
            "rows",
            "distinct_cases",
            "functional_task_pass_count",
            "functional_task_fail_count",
            "functional_pass_rate",
            "mutually_exclusive_stage_label",
            "inference_status",
        ),
    )

    write_csv(
        STRICT_FAILURE_PROFILE_CLASS_PATH,
        profile_class_rows,
        (
            "model",
            "condition",
            "condition_label",
            "profile_class",
            "count_among_strict_failures",
            "strict_failure_count",
            "rate_among_strict_failures",
        ),
    )

    report = render_report(
        audit=audit,
        condition_rows=condition_rows,
        contrast_rows=contrast_rows,
        gap_rows=gap_rows,
        failure_reason_rows=failure_reason_rows,
        failure_stage_rows=failure_stage_rows,
    )
    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(report)

    overall_joint = Counter(
        str(row["joint_strict_task"])
        for row in audit["rows"]
    )

    analysis_json = {
        "status": "FROZEN_OFFLINE_LAYER2_ANALYSIS_COMPLETE",
        "model_calls": 0,
        "protocol_version": PROTOCOL_VERSION,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "ci_level": CI_LEVEL,
        "cluster_unit": "case",
        "analysis_commitments": {
            "models_analyzed_separately": True,
            "primary_functional_contrast": (
                "TaskAcc(g1,Patch)-TaskAcc(g3,Patch)"
            ),
            "matching_state_contrast": (
                "StrictAcc(g1,Patch)-StrictAcc(g3,Patch)"
            ),
            "supporting_select_contrast": (
                "TaskAcc(g1,Select)-TaskAcc(g3,Select)"
            ),
            "failure_class_analysis": "descriptive",
            "failure_reasons_overlap": True,
            "natural_rate_failures_only": True,
            "no_causal_mediation_claim": True,
            "no_monotonicity_claim": True,
        },
        "source_hashes": audit["source_hashes"],
        "overall": {
            "strict_success": audit["strict_success_total"],
            "strict_failure": (
                EXPECTED_ROWS - audit["strict_success_total"]
            ),
            "functional_task_success": audit["task_success_total"],
            "functional_task_failure": (
                EXPECTED_ROWS - audit["task_success_total"]
            ),
            "joint_strict_task": {
                key: int(overall_joint[key])
                for key in (
                    "strict_pass_task_pass",
                    "strict_pass_task_fail",
                    "strict_fail_task_pass",
                    "strict_fail_task_fail",
                )
            },
        },
        "condition_summary": condition_rows,
        "contrast_summary": contrast_rows,
        "state_function_gap_summary": gap_rows,
        "joint_outcome_summary": joint_rows,
        "failure_reason_functional_summary": failure_reason_rows,
        "failure_stage_functional_summary": failure_stage_rows,
        "strict_failure_profile_class_summary": profile_class_rows,
    }

    write_json(FORMAL_ANALYSIS_JSON_PATH, analysis_json)

    manifest = {
        "status": "FROZEN_OFFLINE_LAYER2_ANALYSIS_COMPLETE",
        "model_calls": 0,
        "analysis_script_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        "source_hashes": audit["source_hashes"],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "ci_level": CI_LEVEL,
        "cluster_unit": "case",
        "outputs": {
            "formal_analysis_json": str(
                FORMAL_ANALYSIS_JSON_PATH
            ),
            "condition_summary": str(CONDITION_SUMMARY_PATH),
            "contrast_summary": str(CONTRAST_SUMMARY_PATH),
            "state_function_gap_summary": str(
                STATE_FUNCTION_GAP_PATH
            ),
            "joint_outcome_summary": str(JOINT_OUTCOME_PATH),
            "failure_reason_functional_summary": str(
                FAILURE_REASON_PATH
            ),
            "failure_stage_functional_summary": str(
                FAILURE_STAGE_PATH
            ),
            "strict_failure_profile_class_summary": str(
                STRICT_FAILURE_PROFILE_CLASS_PATH
            ),
            "report": str(REPORT_PATH),
        },
    }
    write_json(ANALYSIS_MANIFEST_PATH, manifest)

    contrast_map = row_lookup(
        contrast_rows,
        ("model", "contrast_id"),
    )

    print("=" * 78)
    print("FROZEN OFFLINE LAYER-2 ANALYSIS COMPLETE")
    print("=" * 78)
    print(
        "Overall Strict x Task: pass/pass={} | pass/fail={} | "
        "fail/pass={} | fail/fail={}".format(
            int(overall_joint["strict_pass_task_pass"]),
            int(overall_joint["strict_pass_task_fail"]),
            int(overall_joint["strict_fail_task_pass"]),
            int(overall_joint["strict_fail_task_fail"]),
        )
    )
    print()
    print("Frozen g1 - g3 contrasts (estimate [95% case-bootstrap CI]):")
    for model in MODELS:
        print("  {}".format(model))
        for contrast_id in (
            "F_Patch",
            "G_Patch",
            "F_Select",
            "G_Select",
        ):
            row = contrast_map[(model, contrast_id)]
            print(
                "    {:8s} {:>8s} [{:>8s}, {:>8s}]".format(
                    contrast_id,
                    fmt_pp(
                        float(row["estimate_rate_difference"])
                    ),
                    fmt_pp(
                        float(row["cluster_bootstrap_ci_low"])
                    ),
                    fmt_pp(
                        float(row["cluster_bootstrap_ci_high"])
                    ),
                )
            )

    print()
    print("Wrote:")
    for path in (
        FORMAL_ANALYSIS_JSON_PATH,
        CONDITION_SUMMARY_PATH,
        CONTRAST_SUMMARY_PATH,
        STATE_FUNCTION_GAP_PATH,
        JOINT_OUTCOME_PATH,
        FAILURE_REASON_PATH,
        FAILURE_STAGE_PATH,
        STRICT_FAILURE_PROFILE_CLASS_PATH,
        REPORT_PATH,
        ANALYSIS_MANIFEST_PATH,
    ):
        print("  {}".format(path))

    print()
    print("NO MODEL CALLS WERE MADE. Raw formal outputs were not modified.")


if __name__ == "__main__":
    from atomic_memory.reanalysis import run_analysis
    run_analysis(main, globals(), paths.PROFILE, Path(__file__).with_name("build_profile.py"))
