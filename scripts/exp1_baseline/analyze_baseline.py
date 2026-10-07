#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute baseline case-cluster summaries and paired contrasts.

Use --check to verify saved analyses without file writes or model calls."""

import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "atomic_memory").is_dir())
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from atomic_memory import paths

FORMAL_DIR = paths.BASELINE

CASES_PATH = FORMAL_DIR / "formal_cases.jsonl"
PREFLIGHT_PATH = FORMAL_DIR / "formal_preflight.json"
FREEZE_PATH = FORMAL_DIR / "formal_freeze_manifest.json"
RESULTS_PATH = FORMAL_DIR / "formal_results.jsonl"
RUN_SUMMARY_PATH = FORMAL_DIR / "formal_run_summary.json"

ANALYSIS_DIR = FORMAL_DIR / "formal_analysis"
ANALYSIS_MANIFEST_PATH = ANALYSIS_DIR / "analysis_manifest.json"
CONDITION_SUMMARY_PATH = ANALYSIS_DIR / "condition_summary.csv"
CONTRAST_SUMMARY_PATH = ANALYSIS_DIR / "contrast_summary.csv"
INTERACTION_SUMMARY_PATH = ANALYSIS_DIR / "interaction_summary.csv"
FAILURE_SUMMARY_PATH = ANALYSIS_DIR / "failure_reason_summary.csv"
COST_SUMMARY_PATH = ANALYSIS_DIR / "cost_summary.csv"
ANALYSIS_JSON_PATH = ANALYSIS_DIR / "formal_analysis.json"
REPORT_PATH = ANALYSIS_DIR / "report.md"

BOOTSTRAP_SEED = 20261002
BOOTSTRAP_REPS = 20000
CI_LEVEL = 0.95

EXPECTED_CALLS = 600
EXPECTED_CASES = 20
EXPECTED_TARGETS_PER_CASE = 2

CONDITION_LABELS = {
    "A_compound_global": "A Compound + Global rewrite",
    "B_compound_select": "B Compound + Select-unit rewrite",
    "C_atomic_global": "C Atomic + Global rewrite",
    "D_atomic_select": "D Atomic + Select-unit rewrite",
    "E_compound_exact_patch": "E Compound + Exact local patch",
}

# right - left. These are design contrasts, not post-hoc rankings.
PREPLANNED_CONTRASTS = (
    {
        "contrast_id": "C_minus_A",
        "label": "Atomic vs Compound under Global rewrite",
        "left": "A_compound_global",
        "right": "C_atomic_global",
    },
    {
        "contrast_id": "D_minus_B",
        "label": "Atomic vs Compound under Select-unit rewrite",
        "left": "B_compound_select",
        "right": "D_atomic_select",
    },
    {
        "contrast_id": "B_minus_A",
        "label": "Select-unit vs Global rewrite within Compound",
        "left": "A_compound_global",
        "right": "B_compound_select",
    },
    {
        "contrast_id": "D_minus_C",
        "label": "Select-unit vs Global rewrite within Atomic",
        "left": "C_atomic_global",
        "right": "D_atomic_select",
    },
    {
        "contrast_id": "E_minus_B",
        "label": "Exact patch vs Select-unit rewrite within Compound",
        "left": "B_compound_select",
        "right": "E_compound_exact_patch",
    },
)

BOOLEAN_METRICS = (
    "end_to_end_success",
    "legal_action",
    "execution_success",
    "contract_valid",
    "structure_valid",
    "target_new_current",
    "target_old_not_current",
    "protected_facts_preserved",
    "no_unsupported_changes",
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


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise RuntimeError(
                    "{} line {} is not a JSON object".format(
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
# Small statistics
# ============================================================================

def mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values)) / float(len(values))


def median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def percentile(
    sorted_values: Sequence[float],
    q: float,
) -> Optional[float]:
    if not sorted_values:
        return None
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
) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    ordered = sorted(float(x) for x in values)
    alpha = 1.0 - float(level)
    return (
        percentile(ordered, alpha / 2.0),
        percentile(ordered, 1.0 - alpha / 2.0),
    )


def cluster_bootstrap_mean_ci(
    values_by_case: Mapping[str, Sequence[float]],
    rng: random.Random,
    reps: int = BOOTSTRAP_REPS,
) -> Tuple[float, float, float]:
    # Resample complete cases so their two targets stay in one cluster.
    case_ids = sorted(values_by_case)
    if not case_ids:
        raise ValueError("No cases for bootstrap")

    observed_values = []
    for case_id in case_ids:
        observed_values.extend(
            float(x)
            for x in values_by_case[case_id]
        )
    observed = float(sum(observed_values)) / float(
        len(observed_values)
    )

    bootstrap_means = []
    n_cases = len(case_ids)

    for _ in range(reps):
        total = 0.0
        count = 0
        for _sample_index in range(n_cases):
            sampled_case = case_ids[
                rng.randrange(n_cases)
            ]
            case_values = values_by_case[
                sampled_case
            ]
            total += sum(
                float(x)
                for x in case_values
            )
            count += len(case_values)

        bootstrap_means.append(
            total / float(count)
        )

    ci_low, ci_high = percentile_ci(
        bootstrap_means
    )
    return (
        observed,
        float(ci_low),
        float(ci_high),
    )


def exact_case_signflip_pvalue(
    case_differences: Sequence[float],
) -> float:
    """Exact two-sided sign-flip p-value for 20 paired case clusters.

    With exactly two targets per case, case-level differences are multiples of
    0.5. Scaling by 2 therefore yields exact integers and lets us compute the
    sign-flip distribution via dynamic programming.
    """
    scaled = []
    for value in case_differences:
        scaled_value = int(
            round(float(value) * 2.0)
        )
        if scaled_value != 0:
            scaled.append(
                abs(scaled_value)
            )

    if not scaled:
        return 1.0

    observed = abs(
        sum(
            int(round(float(value) * 2.0))
            for value in case_differences
        )
    )

    distribution = {
        0: 1,
    }
    for magnitude in scaled:
        updated = defaultdict(int)
        for partial_sum, count in distribution.items():
            updated[
                partial_sum + magnitude
            ] += count
            updated[
                partial_sum - magnitude
            ] += count
        distribution = dict(updated)

    total_assignments = float(
        2 ** len(scaled)
    )
    extreme = sum(
        count
        for signed_sum, count in distribution.items()
        if abs(signed_sum) >= observed
    )
    return float(extreme) / total_assignments


def fmt_rate(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return "{:.1f}%".format(
        float(value) * 100.0
    )


def fmt_float(
    value: Optional[float],
    digits: int = 3,
) -> str:
    if value is None:
        return "NA"
    return (
        ("{0:." + str(digits) + "f}")
        .format(float(value))
    )


# ============================================================================
# Frozen-data audit
# ============================================================================

def result_key(
    row: Mapping[str, Any],
) -> Tuple[str, str, str, str]:
    return (
        str(row["case_id"]),
        str(row["update_id"]),
        str(row["condition"]),
        str(row["model"]),
    )


def audit_inputs() -> dict:
    required_paths = (
        CASES_PATH,
        PREFLIGHT_PATH,
        FREEZE_PATH,
        RESULTS_PATH,
        RUN_SUMMARY_PATH,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                "Missing required formal artifact: {}".format(
                    path
                )
            )

    cases = read_jsonl(CASES_PATH)
    preflight = read_json(PREFLIGHT_PATH)
    freeze = read_json(FREEZE_PATH)
    results = read_jsonl(RESULTS_PATH)
    run_summary = read_json(RUN_SUMMARY_PATH)

    if len(cases) != EXPECTED_CASES:
        raise RuntimeError(
            "Expected {} cases, found {}".format(
                EXPECTED_CASES,
                len(cases),
            )
        )

    if preflight.get("status") != "PASS":
        raise RuntimeError(
            "formal_preflight.json is not PASS"
        )

    if freeze.get("status") != "FORMAL_PROTOCOL_FROZEN":
        raise RuntimeError(
            "Freeze manifest status is not FORMAL_PROTOCOL_FROZEN"
        )

    immutable = freeze.get("immutable_protocol")
    if not isinstance(immutable, dict):
        raise RuntimeError(
            "Freeze manifest missing immutable_protocol"
        )

    if immutable.get("formal_pilot_frozen") is not True:
        raise RuntimeError(
            "formal_pilot_frozen is not true"
        )

    if int(
        immutable.get("expected_total_calls", -1)
    ) != EXPECTED_CALLS:
        raise RuntimeError(
            "Freeze manifest expected_total_calls mismatch"
        )

    models = tuple(
        immutable.get("models", [])
    )
    conditions = tuple(
        immutable.get("conditions", [])
    )

    if len(models) != 3:
        raise RuntimeError(
            "Expected 3 frozen models"
        )
    if len(conditions) != 5:
        raise RuntimeError(
            "Expected 5 frozen conditions"
        )

    if len(results) != EXPECTED_CALLS:
        raise RuntimeError(
            "Expected {} formal rows, found {}".format(
                EXPECTED_CALLS,
                len(results),
            )
        )

    if run_summary.get("status") != "FORMAL_PILOT_COMPLETE":
        raise RuntimeError(
            "formal_run_summary.json is not FORMAL_PILOT_COMPLETE"
        )
    if int(
        run_summary.get("completed_calls", -1)
    ) != EXPECTED_CALLS:
        raise RuntimeError(
            "formal_run_summary completed_calls mismatch"
        )
    if int(
        run_summary.get("remaining_calls", -1)
    ) != 0:
        raise RuntimeError(
            "formal_run_summary says calls remain"
        )

    cases_file_sha = sha256_file(
        CASES_PATH
    )
    results_file_sha = sha256_file(
        RESULTS_PATH
    )
    freeze_file_sha = sha256_file(
        FREEZE_PATH
    )
    run_summary_file_sha = sha256_file(
        RUN_SUMMARY_PATH
    )

    expected_cases_file_sha = immutable.get(
        "formal_cases_file_sha256"
    )
    if (
        expected_cases_file_sha
        and expected_cases_file_sha != cases_file_sha
    ):
        raise RuntimeError(
            "formal_cases.jsonl hash differs from frozen manifest"
        )

    expected_results_file_sha = run_summary.get(
        "formal_results_file_sha256"
    )
    if (
        expected_results_file_sha
        and expected_results_file_sha != results_file_sha
    ):
        raise RuntimeError(
            "formal_results.jsonl hash differs from final run summary"
        )

    case_by_id = {
        str(case["case_id"]): case
        for case in cases
    }

    expected_keys = set()
    for case in cases:
        updates = list(
            case.get("updates", [])
        )
        if len(updates) != EXPECTED_TARGETS_PER_CASE:
            raise RuntimeError(
                "{} does not contain exactly {} updates".format(
                    case["case_id"],
                    EXPECTED_TARGETS_PER_CASE,
                )
            )

        for update in updates:
            for condition in conditions:
                for model in models:
                    expected_keys.add(
                        (
                            str(case["case_id"]),
                            str(update["update_id"]),
                            str(condition),
                            str(model),
                        )
                    )

    if len(expected_keys) != EXPECTED_CALLS:
        raise RuntimeError(
            "Expected-key construction did not produce 600 unique keys"
        )

    actual_keys = set()
    planned_indices = set()

    for row in results:
        key = result_key(row)
        if key in actual_keys:
            raise RuntimeError(
                "Duplicate formal result key: {}".format(
                    key
                )
            )
        actual_keys.add(key)

        if key not in expected_keys:
            raise RuntimeError(
                "Unexpected formal result key: {}".format(
                    key
                )
            )

        case = case_by_id[
            key[0]
        ]
        if row.get("case_sha256") != case.get("case_sha256"):
            raise RuntimeError(
                "case_sha256 mismatch: {}".format(
                    row.get("request_id")
                )
            )

        if row.get("system_prompt_sha256") != immutable.get(
            "system_prompt_sha256"
        ):
            raise RuntimeError(
                "system_prompt_sha256 mismatch: {}".format(
                    row.get("request_id")
                )
            )

        schema_hashes = immutable.get(
            "response_schema_sha256_by_condition",
            {},
        )
        if row.get("response_schema_sha256") != schema_hashes.get(
            key[2]
        ):
            raise RuntimeError(
                "response schema hash mismatch: {}".format(
                    row.get("request_id")
                )
            )

        planned_index = int(
            row.get("planned_call_index", -1)
        )
        if planned_index in planned_indices:
            raise RuntimeError(
                "Duplicate planned_call_index {}".format(
                    planned_index
                )
            )
        planned_indices.add(planned_index)

    if actual_keys != expected_keys:
        missing = sorted(
            expected_keys - actual_keys
        )
        extra = sorted(
            actual_keys - expected_keys
        )
        raise RuntimeError(
            "Formal key coverage mismatch. missing={} extra={}".format(
                missing[:5],
                extra[:5],
            )
        )

    if planned_indices != set(
        range(1, EXPECTED_CALLS + 1)
    ):
        raise RuntimeError(
            "planned_call_index does not cover 1..600 exactly"
        )

    return {
        "cases": cases,
        "preflight": preflight,
        "freeze": freeze,
        "immutable": immutable,
        "results": results,
        "run_summary": run_summary,
        "models": models,
        "conditions": conditions,
        "source_hashes": {
            "formal_cases_sha256": cases_file_sha,
            "formal_preflight_sha256": sha256_file(
                PREFLIGHT_PATH
            ),
            "formal_freeze_manifest_sha256": freeze_file_sha,
            "formal_results_sha256": results_file_sha,
            "formal_run_summary_sha256": run_summary_file_sha,
        },
        "audit": {
            "formal_cases": len(cases),
            "formal_rows": len(results),
            "unique_result_keys": len(actual_keys),
            "planned_call_indices_complete": True,
            "preflight_pass": True,
            "protocol_frozen": True,
            "run_complete": True,
        },
    }


# ============================================================================
# Row indexing
# ============================================================================

def index_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict:
    by_model_condition = defaultdict(list)
    by_model_case_update_condition = {}
    first_call_index_by_model = {}

    for row in rows:
        model = str(row["model"])
        condition = str(row["condition"])
        case_id = str(row["case_id"])
        update_id = str(row["update_id"])

        by_model_condition[
            (model, condition)
        ].append(row)

        paired_key = (
            model,
            case_id,
            update_id,
            condition,
        )
        if paired_key in by_model_case_update_condition:
            raise RuntimeError(
                "Duplicate paired analysis key: {}".format(
                    paired_key
                )
            )
        by_model_case_update_condition[
            paired_key
        ] = row

        planned_index = int(
            row["planned_call_index"]
        )
        previous = first_call_index_by_model.get(
            model
        )
        if (
            previous is None
            or planned_index < previous
        ):
            first_call_index_by_model[
                model
            ] = planned_index

    return {
        "by_model_condition": by_model_condition,
        "paired": by_model_case_update_condition,
        "first_call_index_by_model": first_call_index_by_model,
    }


# ============================================================================
# Condition summaries
# ============================================================================

def bool_rate(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> Optional[float]:
    values = []
    for row in rows:
        value = row.get(
            "evaluation",
            {},
        ).get(metric)
        if isinstance(value, bool):
            values.append(
                1.0 if value else 0.0
            )

    return mean(values)


def response_json_rate(
    rows: Sequence[Mapping[str, Any]],
) -> Optional[float]:
    values = [
        1.0 if bool(
            row.get("response_json_valid")
        ) else 0.0
        for row in rows
    ]
    return mean(values)


def locator_rate(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[Optional[float], int]:
    values = []
    for row in rows:
        value = row.get("locator_success")
        if isinstance(value, bool):
            values.append(
                1.0 if value else 0.0
            )
    return (
        mean(values),
        len(values),
    )


def build_condition_summary(
    audit: Mapping[str, Any],
    index: Mapping[str, Any],
) -> List[dict]:
    rows_out = []
    rng = random.Random(
        BOOTSTRAP_SEED
    )

    for model in audit["models"]:
        for condition in audit["conditions"]:
            rows = list(
                index[
                    "by_model_condition"
                ][
                    (model, condition)
                ]
            )
            if len(rows) != 40:
                raise RuntimeError(
                    "{} {} expected 40 rows, found {}".format(
                        model,
                        condition,
                        len(rows),
                    )
                )

            by_case = defaultdict(list)
            for row in rows:
                by_case[
                    str(row["case_id"])
                ].append(
                    1.0
                    if bool(
                        row["evaluation"][
                            "end_to_end_success"
                        ]
                    )
                    else 0.0
                )

            for case_id, values in by_case.items():
                if len(values) != 2:
                    raise RuntimeError(
                        "{} {} {} expected 2 target rows".format(
                            model,
                            condition,
                            case_id,
                        )
                    )

            (
                e2e_rate,
                e2e_ci_low,
                e2e_ci_high,
            ) = cluster_bootstrap_mean_ci(
                by_case,
                rng,
            )

            locator_value, locator_n = locator_rate(
                rows
            )

            summary = {
                "model": model,
                "condition": condition,
                "condition_label": CONDITION_LABELS.get(
                    condition,
                    condition,
                ),
                "n_rows": len(rows),
                "n_cases": len(by_case),
                "e2e_success_count": sum(
                    1
                    for row in rows
                    if bool(
                        row["evaluation"][
                            "end_to_end_success"
                        ]
                    )
                ),
                "e2e_rate": e2e_rate,
                "e2e_cluster_bootstrap_ci_low": e2e_ci_low,
                "e2e_cluster_bootstrap_ci_high": e2e_ci_high,
                "response_json_valid_rate": response_json_rate(
                    rows
                ),
                "locator_rate": locator_value,
                "locator_n": locator_n,
            }

            for metric in BOOLEAN_METRICS:
                summary[
                    metric + "_rate"
                ] = bool_rate(
                    rows,
                    metric,
                )

            rows_out.append(
                summary
            )

    return rows_out


# ============================================================================
# Preplanned paired contrasts
# ============================================================================

def build_contrast_summary(
    audit: Mapping[str, Any],
    index: Mapping[str, Any],
) -> List[dict]:
    paired = index["paired"]
    cases = audit["cases"]
    rng = random.Random(
        BOOTSTRAP_SEED + 1
    )
    rows_out = []

    case_updates = []
    for case in cases:
        case_id = str(
            case["case_id"]
        )
        for update in case["updates"]:
            case_updates.append(
                (
                    case_id,
                    str(
                        update["update_id"]
                    ),
                )
            )

    for model in audit["models"]:
        for spec in PREPLANNED_CONTRASTS:
            left = spec["left"]
            right = spec["right"]

            by_case = defaultdict(list)
            paired_win = 0
            paired_loss = 0
            paired_tie = 0

            for case_id, update_id in case_updates:
                left_row = paired[
                    (
                        model,
                        case_id,
                        update_id,
                        left,
                    )
                ]
                right_row = paired[
                    (
                        model,
                        case_id,
                        update_id,
                        right,
                    )
                ]

                left_value = (
                    1.0
                    if bool(
                        left_row[
                            "evaluation"
                        ][
                            "end_to_end_success"
                        ]
                    )
                    else 0.0
                )
                right_value = (
                    1.0
                    if bool(
                        right_row[
                            "evaluation"
                        ][
                            "end_to_end_success"
                        ]
                    )
                    else 0.0
                )

                difference = (
                    right_value
                    - left_value
                )
                by_case[
                    case_id
                ].append(
                    difference
                )

                if difference > 0:
                    paired_win += 1
                elif difference < 0:
                    paired_loss += 1
                else:
                    paired_tie += 1

            for case_id, values in by_case.items():
                if len(values) != 2:
                    raise RuntimeError(
                        "{} {} expected 2 paired target differences".format(
                            model,
                            case_id,
                        )
                    )

            (
                effect,
                ci_low,
                ci_high,
            ) = cluster_bootstrap_mean_ci(
                by_case,
                rng,
            )

            case_differences = [
                float(sum(values)) / float(
                    len(values)
                )
                for _case_id, values in sorted(
                    by_case.items()
                )
            ]

            p_value = exact_case_signflip_pvalue(
                case_differences
            )

            rows_out.append(
                {
                    "model": model,
                    "contrast_id": spec[
                        "contrast_id"
                    ],
                    "contrast_label": spec[
                        "label"
                    ],
                    "left_condition": left,
                    "right_condition": right,
                    "difference_definition": (
                        "right_minus_left"
                    ),
                    "effect_e2e_rate_difference": effect,
                    "cluster_bootstrap_ci_low": ci_low,
                    "cluster_bootstrap_ci_high": ci_high,
                    "paired_target_right_only_success": paired_win,
                    "paired_target_left_only_success": paired_loss,
                    "paired_target_tie": paired_tie,
                    "n_paired_targets": (
                        paired_win
                        + paired_loss
                        + paired_tie
                    ),
                    "n_case_clusters": len(
                        by_case
                    ),
                    "case_signflip_p_value_exploratory": p_value,
                }
            )

    return rows_out


# ============================================================================
# 2 x 2 Representation x Edit-Scope interaction
# ============================================================================

def build_interaction_summary(
    audit: Mapping[str, Any],
    index: Mapping[str, Any],
) -> List[dict]:
    """Estimate (D-B) - (C-A) separately for each model.

    A = Compound + Global
    B = Compound + Select-unit
    C = Atomic + Global
    D = Atomic + Select-unit

    The interaction is computed at each paired case/update:
        (D - B) - (C - A)
      = D - B - C + A

    Uncertainty is obtained by resampling whole cases so the two update targets
    within one case remain clustered.
    """
    paired = index["paired"]
    cases = audit["cases"]
    rng = random.Random(
        BOOTSTRAP_SEED + 2
    )
    rows_out = []

    for model in audit["models"]:
        by_case = defaultdict(list)
        positive_targets = 0
        negative_targets = 0
        zero_targets = 0

        for case in cases:
            case_id = str(
                case["case_id"]
            )

            for update in case["updates"]:
                update_id = str(
                    update["update_id"]
                )

                values = {}
                for condition in (
                    "A_compound_global",
                    "B_compound_select",
                    "C_atomic_global",
                    "D_atomic_select",
                ):
                    row = paired[
                        (
                            model,
                            case_id,
                            update_id,
                            condition,
                        )
                    ]
                    values[
                        condition
                    ] = (
                        1.0
                        if bool(
                            row[
                                "evaluation"
                            ][
                                "end_to_end_success"
                            ]
                        )
                        else 0.0
                    )

                interaction = (
                    values[
                        "D_atomic_select"
                    ]
                    - values[
                        "B_compound_select"
                    ]
                    - values[
                        "C_atomic_global"
                    ]
                    + values[
                        "A_compound_global"
                    ]
                )

                by_case[
                    case_id
                ].append(
                    interaction
                )

                if interaction > 0:
                    positive_targets += 1
                elif interaction < 0:
                    negative_targets += 1
                else:
                    zero_targets += 1

        for case_id, values in by_case.items():
            if len(values) != EXPECTED_TARGETS_PER_CASE:
                raise RuntimeError(
                    "{} {} expected {} interaction values".format(
                        model,
                        case_id,
                        EXPECTED_TARGETS_PER_CASE,
                    )
                )

        (
            effect,
            ci_low,
            ci_high,
        ) = cluster_bootstrap_mean_ci(
            by_case,
            rng,
        )

        case_level_interactions = [
            float(sum(values))
            / float(len(values))
            for _case_id, values in sorted(
                by_case.items()
            )
        ]

        rows_out.append(
            {
                "model": model,
                "interaction_id": (
                    "representation_x_edit_scope"
                ),
                "formula": (
                    "(D-B)-(C-A)"
                ),
                "effect_e2e_rate_difference_in_differences": (
                    effect
                ),
                "cluster_bootstrap_ci_low": ci_low,
                "cluster_bootstrap_ci_high": ci_high,
                "n_case_clusters": len(
                    by_case
                ),
                "n_paired_targets": (
                    positive_targets
                    + negative_targets
                    + zero_targets
                ),
                "positive_target_interactions": (
                    positive_targets
                ),
                "negative_target_interactions": (
                    negative_targets
                ),
                "zero_target_interactions": (
                    zero_targets
                ),
                "mean_case_level_interaction": mean(
                    case_level_interactions
                ),
                "interpretation_guardrail": (
                    "Descriptive difference-in-differences under the "
                    "controlled sample. Do not claim a stable interaction "
                    "when the case-cluster bootstrap interval includes zero."
                ),
            }
        )

    return rows_out


# ============================================================================
# Failure summaries
# ============================================================================

def build_failure_summary(
    audit: Mapping[str, Any],
    index: Mapping[str, Any],
) -> List[dict]:
    rows_out = []

    all_reasons = sorted(
        {
            str(reason)
            for row in audit["results"]
            for reason in row.get(
                "failure_reasons",
                [],
            )
        }
    )

    for model in audit["models"]:
        for condition in audit["conditions"]:
            rows = index[
                "by_model_condition"
            ][
                (model, condition)
            ]
            denominator = len(rows)

            counter = Counter()
            for row in rows:
                for reason in set(
                    str(x)
                    for x in row.get(
                        "failure_reasons",
                        [],
                    )
                ):
                    counter[
                        reason
                    ] += 1

            for reason in all_reasons:
                count = int(
                    counter.get(
                        reason,
                        0,
                    )
                )
                rows_out.append(
                    {
                        "model": model,
                        "condition": condition,
                        "condition_label": CONDITION_LABELS.get(
                            condition,
                            condition,
                        ),
                        "failure_reason": reason,
                        "rows_with_reason": count,
                        "n_rows": denominator,
                        "row_rate": (
                            float(count)
                            / float(denominator)
                        ),
                        "multi_label_overlap": True,
                    }
                )

    return rows_out


# ============================================================================
# Cost summaries
# ============================================================================

def number_or_none(
    value: Any,
) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_cost_summary(
    audit: Mapping[str, Any],
    index: Mapping[str, Any],
) -> List[dict]:
    # The first call can include model loading; warm summaries exclude it.
    rows_out = []

    first_by_model = index[
        "first_call_index_by_model"
    ]

    for model in audit["models"]:
        for condition in audit["conditions"]:
            rows = list(
                index[
                    "by_model_condition"
                ][
                    (model, condition)
                ]
            )

            input_tokens = []
            output_tokens = []
            latencies = []
            warm_latencies = []

            first_index = int(
                first_by_model[
                    model
                ]
            )

            for row in rows:
                usage = row.get(
                    "usage",
                    {},
                )
                input_value = number_or_none(
                    usage.get(
                        "input_tokens"
                    )
                )
                output_value = number_or_none(
                    usage.get(
                        "output_tokens"
                    )
                )
                latency_value = number_or_none(
                    row.get(
                        "latency_seconds"
                    )
                )

                if input_value is not None:
                    input_tokens.append(
                        input_value
                    )
                if output_value is not None:
                    output_tokens.append(
                        output_value
                    )
                if latency_value is not None:
                    latencies.append(
                        latency_value
                    )
                    if int(
                        row[
                            "planned_call_index"
                        ]
                    ) != first_index:
                        warm_latencies.append(
                            latency_value
                        )

            ordered_latencies = sorted(
                latencies
            )
            ordered_warm = sorted(
                warm_latencies
            )

            rows_out.append(
                {
                    "model": model,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS.get(
                        condition,
                        condition,
                    ),
                    "n_rows": len(rows),
                    "mean_input_tokens": mean(
                        input_tokens
                    ),
                    "median_input_tokens": median(
                        input_tokens
                    ),
                    "mean_output_tokens": mean(
                        output_tokens
                    ),
                    "median_output_tokens": median(
                        output_tokens
                    ),
                    "mean_latency_seconds_raw": mean(
                        latencies
                    ),
                    "median_latency_seconds_raw": median(
                        latencies
                    ),
                    "p95_latency_seconds_raw": percentile(
                        ordered_latencies,
                        0.95,
                    ),
                    "mean_latency_seconds_excluding_first_model_call": mean(
                        warm_latencies
                    ),
                    "median_latency_seconds_excluding_first_model_call": median(
                        warm_latencies
                    ),
                    "p95_latency_seconds_excluding_first_model_call": percentile(
                        ordered_warm,
                        0.95,
                    ),
                    "excluded_first_model_call_count": (
                        len(latencies)
                        - len(warm_latencies)
                    ),
                }
            )

    return rows_out


# ============================================================================
# Markdown report
# ============================================================================

def condition_lookup(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    return {
        (
            str(row["model"]),
            str(row["condition"]),
        ): row
        for row in rows
    }


def cost_lookup(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    return {
        (
            str(row["model"]),
            str(row["condition"]),
        ): row
        for row in rows
    }


def render_report(
    audit: Mapping[str, Any],
    condition_rows: Sequence[Mapping[str, Any]],
    contrast_rows: Sequence[Mapping[str, Any]],
    interaction_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
    cost_rows: Sequence[Mapping[str, Any]],
) -> str:
    condition_map = condition_lookup(
        condition_rows
    )
    cost_map = cost_lookup(
        cost_rows
    )

    lines = []
    lines.append(
        "# Representation × Edit-Scope — Frozen Formal Pilot Analysis"
    )
    lines.append("")
    lines.append(
        "This report is generated entirely offline from the frozen 600-call "
        "formal pilot. It makes no model calls."
    )
    lines.append("")
    lines.append("## Integrity audit")
    lines.append("")
    lines.append(
        "- Formal cases: **20**; two update targets per case."
    )
    lines.append(
        "- Formal calls: **600/600**, with 600 unique result keys and planned "
        "call indices 1–600 complete."
    )
    lines.append(
        "- Formal preflight: **PASS**; frozen protocol manifest verified."
    )
    lines.append(
        "- Statistical clustering unit: **case**; the two targets within one "
        "case remain clustered."
    )
    lines.append(
        "- Failure-reason labels are multi-label and overlapping; counts are "
        "not additive."
    )
    lines.append(
        "- Cost analysis covers input tokens, output tokens, and generation "
        "latency. **Retrieval cost is not measured.**"
    )
    lines.append("")

    lines.append(
        "## End-to-end success by model and condition"
    )
    lines.append("")
    lines.append(
        "| Model | A Compound Global | B Compound Select | "
        "C Atomic Global | D Atomic Select | E Exact Patch |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|"
    )

    for model in audit["models"]:
        cells = []
        for condition in audit["conditions"]:
            row = condition_map[
                (model, condition)
            ]
            cells.append(
                "{} / 40 ({})".format(
                    int(
                        row[
                            "e2e_success_count"
                        ]
                    ),
                    fmt_rate(
                        row["e2e_rate"]
                    ),
                )
            )

        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                model,
                cells[0],
                cells[1],
                cells[2],
                cells[3],
                cells[4],
            )
        )

    lines.append("")
    lines.append(
        "Cluster-bootstrap 95% confidence intervals for these rates are stored "
        "in `condition_summary.csv`."
    )
    lines.append("")

    lines.append(
        "## Preplanned paired contrasts"
    )
    lines.append("")
    lines.append(
        "Effects are **right condition minus left condition** in E2E success "
        "rate. Positive and negative values are descriptive effect directions, "
        "not model rankings."
    )
    lines.append("")
    lines.append(
        "| Model | Contrast | Difference | 95% case-cluster bootstrap CI | "
        "Right-only / Left-only / Tie | Exploratory case sign-flip p |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|"
    )

    for row in contrast_rows:
        lines.append(
            "| {} | {} | {} | [{}, {}] | {} / {} / {} | {} |".format(
                row["model"],
                row["contrast_id"],
                fmt_float(
                    row[
                        "effect_e2e_rate_difference"
                    ],
                    3,
                ),
                fmt_float(
                    row[
                        "cluster_bootstrap_ci_low"
                    ],
                    3,
                ),
                fmt_float(
                    row[
                        "cluster_bootstrap_ci_high"
                    ],
                    3,
                ),
                int(
                    row[
                        "paired_target_right_only_success"
                    ]
                ),
                int(
                    row[
                        "paired_target_left_only_success"
                    ]
                ),
                int(
                    row[
                        "paired_target_tie"
                    ]
                ),
                fmt_float(
                    row[
                        "case_signflip_p_value_exploratory"
                    ],
                    4,
                ),
            )
        )

    lines.append("")
    lines.append(
        "The sign-flip p-value is an exploratory paired case-level diagnostic; "
        "the effect size and clustered confidence interval should remain the "
        "main reported quantities."
    )
    lines.append("")

    lines.append(
        "## 2 x 2 Representation x Edit-Scope interaction"
    )
    lines.append("")
    lines.append(
        "For A-D, the descriptive interaction is defined as "
        "`(D-B)-(C-A)`: the Atomic-minus-Compound difference under Select-unit "
        "editing minus the same representation difference under Global rewrite."
    )
    lines.append("")
    lines.append(
        "| Model | Interaction | 95% case-cluster bootstrap CI | "
        "Positive / Negative / Zero target interactions |"
    )
    lines.append(
        "|---|---:|---:|---:|"
    )
    for row in interaction_rows:
        lines.append(
            "| {} | {} | [{}, {}] | {} / {} / {} |".format(
                row["model"],
                fmt_float(
                    row[
                        "effect_e2e_rate_difference_in_differences"
                    ],
                    3,
                ),
                fmt_float(
                    row[
                        "cluster_bootstrap_ci_low"
                    ],
                    3,
                ),
                fmt_float(
                    row[
                        "cluster_bootstrap_ci_high"
                    ],
                    3,
                ),
                int(
                    row[
                        "positive_target_interactions"
                    ]
                ),
                int(
                    row[
                        "negative_target_interactions"
                    ]
                ),
                int(
                    row[
                        "zero_target_interactions"
                    ]
                ),
            )
        )
    lines.append("")
    lines.append(
        "These interaction estimates are descriptive under 20 independent case "
        "clusters. A positive point estimate means the Atomic-minus-Compound "
        "difference is larger under Select-unit editing than under Global "
        "rewrite. It should not be described as a stable interaction when the "
        "case-cluster bootstrap interval includes zero."
    )
    lines.append("")

    lines.append(
        "## Locator success where localization is required"
    )
    lines.append("")
    lines.append(
        "| Model | B Compound Select | D Atomic Select | E Exact Patch |"
    )
    lines.append(
        "|---|---:|---:|---:|"
    )
    for model in audit["models"]:
        values = []
        for condition in (
            "B_compound_select",
            "D_atomic_select",
            "E_compound_exact_patch",
        ):
            row = condition_map[
                (model, condition)
            ]
            values.append(
                "{} (n={})".format(
                    fmt_rate(
                        row[
                            "locator_rate"
                        ]
                    ),
                    int(
                        row[
                            "locator_n"
                        ]
                    ),
                )
            )
        lines.append(
            "| {} | {} | {} | {} |".format(
                model,
                values[0],
                values[1],
                values[2],
            )
        )

    lines.append("")
    lines.append(
        "## Cost / overhead"
    )
    lines.append("")
    lines.append(
        "Input/output token counts are overhead measures. They should not be "
        "described as retrieval cost."
    )
    lines.append("")
    lines.append(
        "| Model | Condition | Mean input tokens | Mean output tokens | "
        "Raw mean latency (s) | Warm mean latency (s) |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|"
    )

    for model in audit["models"]:
        for condition in audit["conditions"]:
            row = cost_map[
                (model, condition)
            ]
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    model,
                    condition,
                    fmt_float(
                        row[
                            "mean_input_tokens"
                        ],
                        1,
                    ),
                    fmt_float(
                        row[
                            "mean_output_tokens"
                        ],
                        1,
                    ),
                    fmt_float(
                        row[
                            "mean_latency_seconds_raw"
                        ],
                        3,
                    ),
                    fmt_float(
                        row[
                            "mean_latency_seconds_excluding_first_model_call"
                        ],
                        3,
                    ),
                )
            )

    lines.append("")
    lines.append(
        "Warm latency excludes the first formal generation call for each model "
        "because that call can contain a model-loading/cold-start penalty."
    )
    lines.append("")

    total_reason_counts = Counter()
    for row in failure_rows:
        total_reason_counts[
            str(
                row[
                    "failure_reason"
                ]
            )
        ] += int(
            row[
                "rows_with_reason"
            ]
        )

    lines.append(
        "## Multi-label failure diagnostics"
    )
    lines.append("")
    lines.append(
        "Across all 600 rows, the number of rows carrying each diagnostic "
        "label is:"
    )
    lines.append("")
    for reason, count in sorted(
        total_reason_counts.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    ):
        lines.append(
            "- `{}`: **{}** rows".format(
                reason,
                count,
            )
        )
    lines.append("")
    lines.append(
        "These counts overlap. For example, one failed update can carry both "
        "`locator_failure` and `stale_old_retained`; therefore the counts must "
        "not be summed to infer a total number of failures."
    )
    lines.append("")

    lines.append(
        "## Reporting guardrails"
    )
    lines.append("")
    lines.append(
        "- Analyze the three model families separately as the primary view; "
        "do not let a pooled average hide model heterogeneity."
    )
    lines.append(
        "- Do not tune prompts, regenerate formal cases, or rerun failed "
        "outputs after seeing these results."
    )
    lines.append(
        "- The formal study supports controlled synthetic evidence about "
        "representation granularity and edit scope; it does not by itself "
        "establish universal superiority of Atomic memory."
    )
    lines.append("")

    return "\n".join(
        lines
    ) + "\n"


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 78)
    print(
        "REPRESENTATION x EDIT-SCOPE — FROZEN OFFLINE ANALYSIS"
    )
    print("=" * 78)
    print("Mode: ZERO MODEL CALLS")
    print(
        "Bootstrap: {} case-cluster replicates".format(
            BOOTSTRAP_REPS
        )
    )
    print()

    print("[1/5] Auditing frozen formal artifacts...")
    audit = audit_inputs()
    print(
        "  PASS | 20 cases | 600/600 rows | 600 unique keys | protocol frozen"
    )

    print("[2/5] Computing model x condition summaries...")
    index = index_rows(
        audit["results"]
    )
    condition_rows = build_condition_summary(
        audit,
        index,
    )
    print(
        "  PASS | {} model-condition cells".format(
            len(condition_rows)
        )
    )

    print(
        "[3/5] Computing paired contrasts + 2x2 case-cluster interaction..."
    )
    contrast_rows = build_contrast_summary(
        audit,
        index,
    )
    interaction_rows = build_interaction_summary(
        audit,
        index,
    )
    print(
        "  PASS | {} within-model contrasts | {} interaction estimates".format(
            len(contrast_rows),
            len(interaction_rows),
        )
    )

    print("[4/5] Computing failure and cost summaries...")
    failure_rows = build_failure_summary(
        audit,
        index,
    )
    cost_rows = build_cost_summary(
        audit,
        index,
    )
    print(
        "  PASS | multi-label failures kept overlapping | cold-start sensitivity recorded"
    )

    print("[5/5] Writing frozen offline analysis outputs...")
    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        CONDITION_SUMMARY_PATH,
        condition_rows,
        (
            "model",
            "condition",
            "condition_label",
            "n_rows",
            "n_cases",
            "e2e_success_count",
            "e2e_rate",
            "e2e_cluster_bootstrap_ci_low",
            "e2e_cluster_bootstrap_ci_high",
            "response_json_valid_rate",
            "locator_rate",
            "locator_n",
            "end_to_end_success_rate",
            "legal_action_rate",
            "execution_success_rate",
            "contract_valid_rate",
            "structure_valid_rate",
            "target_new_current_rate",
            "target_old_not_current_rate",
            "protected_facts_preserved_rate",
            "no_unsupported_changes_rate",
        ),
    )

    write_csv(
        CONTRAST_SUMMARY_PATH,
        contrast_rows,
        (
            "model",
            "contrast_id",
            "contrast_label",
            "left_condition",
            "right_condition",
            "difference_definition",
            "effect_e2e_rate_difference",
            "cluster_bootstrap_ci_low",
            "cluster_bootstrap_ci_high",
            "paired_target_right_only_success",
            "paired_target_left_only_success",
            "paired_target_tie",
            "n_paired_targets",
            "n_case_clusters",
            "case_signflip_p_value_exploratory",
        ),
    )

    write_csv(
        INTERACTION_SUMMARY_PATH,
        interaction_rows,
        (
            "model",
            "interaction_id",
            "formula",
            "effect_e2e_rate_difference_in_differences",
            "cluster_bootstrap_ci_low",
            "cluster_bootstrap_ci_high",
            "n_case_clusters",
            "n_paired_targets",
            "positive_target_interactions",
            "negative_target_interactions",
            "zero_target_interactions",
            "mean_case_level_interaction",
            "interpretation_guardrail",
        ),
    )

    write_csv(
        FAILURE_SUMMARY_PATH,
        failure_rows,
        (
            "model",
            "condition",
            "condition_label",
            "failure_reason",
            "rows_with_reason",
            "n_rows",
            "row_rate",
            "multi_label_overlap",
        ),
    )

    write_csv(
        COST_SUMMARY_PATH,
        cost_rows,
        (
            "model",
            "condition",
            "condition_label",
            "n_rows",
            "mean_input_tokens",
            "median_input_tokens",
            "mean_output_tokens",
            "median_output_tokens",
            "mean_latency_seconds_raw",
            "median_latency_seconds_raw",
            "p95_latency_seconds_raw",
            "mean_latency_seconds_excluding_first_model_call",
            "median_latency_seconds_excluding_first_model_call",
            "p95_latency_seconds_excluding_first_model_call",
            "excluded_first_model_call_count",
        ),
    )

    analysis_json = {
        "status": "FROZEN_OFFLINE_ANALYSIS_COMPLETE",
        "model_calls": 0,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "ci_level": CI_LEVEL,
        "cluster_unit": "case",
        "analysis_rules": {
            "models_analyzed_separately": True,
            "primary_endpoint": "end_to_end_success",
            "failure_reasons_overlap": True,
            "failure_reason_counts_are_not_additive": True,
            "retrieval_cost_measured": False,
            "cost_metrics": [
                "input_tokens",
                "output_tokens",
                "generation_latency_seconds",
            ],
            "warm_latency_excludes_first_formal_call_per_model": True,
        },
        "audit": audit["audit"],
        "source_hashes": audit[
            "source_hashes"
        ],
        "condition_summary": condition_rows,
        "contrast_summary": contrast_rows,
        "interaction_summary": interaction_rows,
        "failure_reason_summary": failure_rows,
        "cost_summary": cost_rows,
    }
    write_json(
        ANALYSIS_JSON_PATH,
        analysis_json,
    )

    report = render_report(
        audit,
        condition_rows,
        contrast_rows,
        interaction_rows,
        failure_rows,
        cost_rows,
    )
    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(report)

    manifest = {
        "status": "FROZEN_OFFLINE_ANALYSIS_COMPLETE",
        "model_calls": 0,
        "analysis_script_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        "source_hashes": audit[
            "source_hashes"
        ],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "ci_level": CI_LEVEL,
        "outputs": {
            "condition_summary": str(
                CONDITION_SUMMARY_PATH
            ),
            "contrast_summary": str(
                CONTRAST_SUMMARY_PATH
            ),
            "interaction_summary": str(
                INTERACTION_SUMMARY_PATH
            ),
            "failure_reason_summary": str(
                FAILURE_SUMMARY_PATH
            ),
            "cost_summary": str(
                COST_SUMMARY_PATH
            ),
            "formal_analysis_json": str(
                ANALYSIS_JSON_PATH
            ),
            "report": str(
                REPORT_PATH
            ),
        },
    }
    write_json(
        ANALYSIS_MANIFEST_PATH,
        manifest,
    )

    condition_map = condition_lookup(
        condition_rows
    )

    print()
    print("=" * 78)
    print("FROZEN OFFLINE ANALYSIS COMPLETE")
    print("=" * 78)

    for model in audit["models"]:
        values = []
        for condition in audit["conditions"]:
            row = condition_map[
                (model, condition)
            ]
            values.append(
                "{}={}/40 ({})".format(
                    condition.split("_")[0],
                    int(
                        row[
                            "e2e_success_count"
                        ]
                    ),
                    fmt_rate(
                        row["e2e_rate"]
                    ),
                )
            )
        print(
            "{}: {}".format(
                model,
                " | ".join(values),
            )
        )

    print()
    print("Wrote:")
    print(
        "  {}".format(
            CONDITION_SUMMARY_PATH
        )
    )
    print(
        "  {}".format(
            CONTRAST_SUMMARY_PATH
        )
    )
    print(
        "  {}".format(
            INTERACTION_SUMMARY_PATH
        )
    )
    print(
        "  {}".format(
            FAILURE_SUMMARY_PATH
        )
    )
    print(
        "  {}".format(
            COST_SUMMARY_PATH
        )
    )
    print(
        "  {}".format(
            ANALYSIS_JSON_PATH
        )
    )
    print(
        "  {}".format(
            REPORT_PATH
        )
    )
    print(
        "  {}".format(
            ANALYSIS_MANIFEST_PATH
        )
    )
    print()
    print(
        "NO MODEL CALLS WERE MADE. Formal raw outputs were not modified."
    )


if __name__ == "__main__":
    from atomic_memory.reanalysis import run_analysis
    run_analysis(main, globals(), paths.BASELINE, Path(__file__).with_name("run_baseline.py"))
