#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze paired instruction-wording sensitivity on the shared baseline cases.

Use --check for read-only verification of saved analyses."""

import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "atomic_memory").is_dir())
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from atomic_memory import paths

BASELINE_DIR = paths.BASELINE
ROBUSTNESS_DIR = paths.WORDING

BASELINE_RESULTS_PATH = BASELINE_DIR / "formal_results.jsonl"
BASELINE_FREEZE_PATH = BASELINE_DIR / "formal_freeze_manifest.json"
BASELINE_RUN_SUMMARY_PATH = BASELINE_DIR / "formal_run_summary.json"
FORMAL_CASES_PATH = BASELINE_DIR / "formal_cases.jsonl"

ROBUSTNESS_RESULTS_PATH = ROBUSTNESS_DIR / "robustness_results.jsonl"
ROBUSTNESS_FREEZE_PATH = ROBUSTNESS_DIR / "robustness_freeze_manifest.json"
ROBUSTNESS_RUN_SUMMARY_PATH = ROBUSTNESS_DIR / "robustness_run_summary.json"

ANALYSIS_DIR = ROBUSTNESS_DIR / "robustness_analysis"
CELL_SUMMARY_PATH = ANALYSIS_DIR / "wording_condition_summary.csv"
DELTA_SUMMARY_PATH = ANALYSIS_DIR / "wording_delta_vs_baseline.csv"
RANGE_SUMMARY_PATH = ANALYSIS_DIR / "wording_range_summary.csv"
INTERACTION_SUMMARY_PATH = ANALYSIS_DIR / "interaction_by_wording.csv"
D_VS_E_SUMMARY_PATH = ANALYSIS_DIR / "atomic_select_vs_exact_patch.csv"
LOCATOR_SUMMARY_PATH = ANALYSIS_DIR / "locator_summary.csv"
FAILURE_SUMMARY_PATH = ANALYSIS_DIR / "failure_reason_summary.csv"
ANALYSIS_JSON_PATH = ANALYSIS_DIR / "robustness_analysis.json"
REPORT_PATH = ANALYSIS_DIR / "report.md"
MANIFEST_PATH = ANALYSIS_DIR / "analysis_manifest.json"

BOOTSTRAP_SEED = 20261003
BOOTSTRAP_REPS = 20000
CI_LEVEL = 0.95

EXPECTED_CASES = 20
EXPECTED_TARGETS_PER_CASE = 2
EXPECTED_BASELINE_ROWS = 600
EXPECTED_ROBUSTNESS_ROWS = 1200

WORDINGS = (
    "baseline",
    "instruction_paraphrase_1",
    "instruction_paraphrase_2",
)

CONDITIONS = (
    "A_compound_global",
    "B_compound_select",
    "C_atomic_global",
    "D_atomic_select",
    "E_compound_exact_patch",
)

CONDITION_LABELS = {
    "A_compound_global": "A Compound + Global rewrite",
    "B_compound_select": "B Compound + Select-unit rewrite",
    "C_atomic_global": "C Atomic + Global rewrite",
    "D_atomic_select": "D Atomic + Select-unit rewrite",
    "E_compound_exact_patch": "E Compound + Exact local patch",
}


# ============================================================================
# IO / helpers
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
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
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
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values)) / float(len(values))


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
) -> Tuple[float, float]:
    ordered = sorted(float(x) for x in values)
    alpha = 1.0 - float(level)
    low = percentile(
        ordered,
        alpha / 2.0,
    )
    high = percentile(
        ordered,
        1.0 - alpha / 2.0,
    )
    if low is None or high is None:
        raise RuntimeError("Cannot compute empty CI")
    return float(low), float(high)


def cluster_bootstrap_mean_ci(
    values_by_case: Mapping[str, Sequence[float]],
    rng: random.Random,
) -> Tuple[float, float, float]:
    case_ids = sorted(values_by_case)
    if len(case_ids) != EXPECTED_CASES:
        raise RuntimeError(
            "Expected {} case clusters, found {}".format(
                EXPECTED_CASES,
                len(case_ids),
            )
        )

    observed_values = []
    for case_id in case_ids:
        values = list(
            values_by_case[case_id]
        )
        if len(values) != EXPECTED_TARGETS_PER_CASE:
            raise RuntimeError(
                "{} expected {} target values, found {}".format(
                    case_id,
                    EXPECTED_TARGETS_PER_CASE,
                    len(values),
                )
            )
        observed_values.extend(
            float(x)
            for x in values
        )

    observed = float(
        sum(observed_values)
    ) / float(
        len(observed_values)
    )

    bootstrap_means = []
    n_cases = len(case_ids)

    for _ in range(BOOTSTRAP_REPS):
        total = 0.0
        count = 0

        for _sample_index in range(n_cases):
            sampled_case = case_ids[
                rng.randrange(n_cases)
            ]
            values = values_by_case[
                sampled_case
            ]
            total += sum(
                float(x)
                for x in values
            )
            count += len(values)

        bootstrap_means.append(
            total / float(count)
        )

    ci_low, ci_high = percentile_ci(
        bootstrap_means
    )
    return (
        observed,
        ci_low,
        ci_high,
    )


def fmt_rate(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return "{:.1f}%".format(
        float(value) * 100.0
    )


def fmt_effect(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return "{:+.1f} pp".format(
        float(value) * 100.0
    )


def bool_e2e(row: Mapping[str, Any]) -> float:
    return (
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


# ============================================================================
# Audit and normalization
# ============================================================================

def audit_and_load() -> dict:
    required = (
        FORMAL_CASES_PATH,
        BASELINE_RESULTS_PATH,
        BASELINE_FREEZE_PATH,
        BASELINE_RUN_SUMMARY_PATH,
        ROBUSTNESS_RESULTS_PATH,
        ROBUSTNESS_FREEZE_PATH,
        ROBUSTNESS_RUN_SUMMARY_PATH,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                "Missing required artifact: {}".format(
                    path
                )
            )

    cases = read_jsonl(
        FORMAL_CASES_PATH
    )
    baseline = read_jsonl(
        BASELINE_RESULTS_PATH
    )
    robustness = read_jsonl(
        ROBUSTNESS_RESULTS_PATH
    )
    baseline_freeze = read_json(
        BASELINE_FREEZE_PATH
    )
    baseline_summary = read_json(
        BASELINE_RUN_SUMMARY_PATH
    )
    robustness_freeze = read_json(
        ROBUSTNESS_FREEZE_PATH
    )
    robustness_summary = read_json(
        ROBUSTNESS_RUN_SUMMARY_PATH
    )

    if len(cases) != EXPECTED_CASES:
        raise RuntimeError(
            "Expected 20 formal cases"
        )
    if len(baseline) != EXPECTED_BASELINE_ROWS:
        raise RuntimeError(
            "Expected 600 baseline rows"
        )
    if len(robustness) != EXPECTED_ROBUSTNESS_ROWS:
        raise RuntimeError(
            "Expected 1200 robustness rows"
        )

    if baseline_summary.get(
        "status"
    ) != "FORMAL_PILOT_COMPLETE":
        raise RuntimeError(
            "Baseline run is not marked complete"
        )

    if robustness_summary.get(
        "status"
    ) != "INSTRUCTION_ROBUSTNESS_COMPLETE":
        raise RuntimeError(
            "Robustness run is not marked complete"
        )

    baseline_immutable = baseline_freeze.get(
        "immutable_protocol",
        {},
    )
    robustness_immutable = robustness_freeze.get(
        "immutable_protocol",
        {},
    )

    if robustness_immutable.get(
        "independent_validation"
    ) is not False:
        raise RuntimeError(
            "Robustness manifest does not mark same-case sensitivity"
        )

    if robustness_immutable.get(
        "only_system_instruction_wording_changes"
    ) is not True:
        raise RuntimeError(
            "Robustness manifest does not freeze instruction-only change"
        )

    if robustness_immutable.get(
        "formal_results_sha256"
    ) != sha256_file(
        BASELINE_RESULTS_PATH
    ):
        raise RuntimeError(
            "Robustness manifest is not bound to current baseline results"
        )

    models = tuple(
        baseline_immutable.get(
            "models",
            [],
        )
    )
    if len(models) != 3:
        raise RuntimeError(
            "Expected 3 baseline models"
        )

    if tuple(
        baseline_immutable.get(
            "conditions",
            [],
        )
    ) != CONDITIONS:
        raise RuntimeError(
            "Baseline condition order differs"
        )

    case_ids = {
        str(case["case_id"])
        for case in cases
    }

    normalized = []

    baseline_seen = set()
    for row in baseline:
        key = (
            "baseline",
            str(row["case_id"]),
            str(row["update_id"]),
            str(row["condition"]),
            str(row["model"]),
        )
        if key in baseline_seen:
            raise RuntimeError(
                "Duplicate baseline key: {}".format(
                    key
                )
            )
        baseline_seen.add(
            key
        )
        if key[1] not in case_ids:
            raise RuntimeError(
                "Unknown baseline case: {}".format(
                    key[1]
                )
            )

        copy = dict(row)
        copy[
            "instruction_wording"
        ] = "baseline"
        normalized.append(
            copy
        )

    robustness_seen = set()
    for row in robustness:
        wording = str(
            row["instruction_variant"]
        )
        if wording not in WORDINGS[1:]:
            raise RuntimeError(
                "Unexpected robustness wording: {}".format(
                    wording
                )
            )

        key = (
            wording,
            str(row["case_id"]),
            str(row["update_id"]),
            str(row["condition"]),
            str(row["model"]),
        )
        if key in robustness_seen:
            raise RuntimeError(
                "Duplicate robustness key: {}".format(
                    key
                )
            )
        robustness_seen.add(
            key
        )
        if key[1] not in case_ids:
            raise RuntimeError(
                "Unknown robustness case: {}".format(
                    key[1]
                )
            )

        copy = dict(row)
        copy[
            "instruction_wording"
        ] = wording
        normalized.append(
            copy
        )

    expected_baseline_keys = (
        EXPECTED_CASES
        * EXPECTED_TARGETS_PER_CASE
        * len(CONDITIONS)
        * len(models)
    )
    if len(baseline_seen) != expected_baseline_keys:
        raise RuntimeError(
            "Baseline key coverage mismatch"
        )

    expected_robustness_keys = (
        expected_baseline_keys
        * 2
    )
    if len(robustness_seen) != expected_robustness_keys:
        raise RuntimeError(
            "Robustness key coverage mismatch"
        )

    # All 1800 rows should use direct JSON successfully if the console run is
    # consistent; keep this descriptive rather than a hard requirement.
    json_valid_count = sum(
        1
        for row in normalized
        if bool(
            row.get(
                "response_json_valid"
            )
        )
    )

    return {
        "cases": cases,
        "models": models,
        "baseline_rows": baseline,
        "robustness_rows": robustness,
        "rows": normalized,
        "json_valid_count": json_valid_count,
        "source_hashes": {
            "formal_cases_sha256": sha256_file(
                FORMAL_CASES_PATH
            ),
            "baseline_results_sha256": sha256_file(
                BASELINE_RESULTS_PATH
            ),
            "baseline_freeze_sha256": sha256_file(
                BASELINE_FREEZE_PATH
            ),
            "robustness_results_sha256": sha256_file(
                ROBUSTNESS_RESULTS_PATH
            ),
            "robustness_freeze_sha256": sha256_file(
                ROBUSTNESS_FREEZE_PATH
            ),
        },
    }


def build_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict:
    by_cell = defaultdict(list)
    paired = {}

    for row in rows:
        wording = str(
            row[
                "instruction_wording"
            ]
        )
        model = str(
            row["model"]
        )
        condition = str(
            row["condition"]
        )
        case_id = str(
            row["case_id"]
        )
        update_id = str(
            row["update_id"]
        )

        by_cell[
            (
                model,
                wording,
                condition,
            )
        ].append(
            row
        )

        key = (
            model,
            wording,
            case_id,
            update_id,
            condition,
        )
        if key in paired:
            raise RuntimeError(
                "Duplicate analysis key: {}".format(
                    key
                )
            )
        paired[
            key
        ] = row

    return {
        "by_cell": by_cell,
        "paired": paired,
    }


# ============================================================================
# Cell rates
# ============================================================================

def build_cell_summary(
    data: Mapping[str, Any],
    index: Mapping[str, Any],
) -> List[dict]:
    rng = random.Random(
        BOOTSTRAP_SEED
    )
    result = []

    for model in data[
        "models"
    ]:
        for wording in WORDINGS:
            for condition in CONDITIONS:
                rows = index[
                    "by_cell"
                ][
                    (
                        model,
                        wording,
                        condition,
                    )
                ]
                if len(rows) != 40:
                    raise RuntimeError(
                        "{} / {} / {} expected 40 rows, found {}".format(
                            model,
                            wording,
                            condition,
                            len(rows),
                        )
                    )

                by_case = defaultdict(list)
                for row in rows:
                    by_case[
                        str(
                            row[
                                "case_id"
                            ]
                        )
                    ].append(
                        bool_e2e(
                            row
                        )
                    )

                rate, low, high = (
                    cluster_bootstrap_mean_ci(
                        by_case,
                        rng,
                    )
                )

                locator_values = [
                    1.0
                    if row.get(
                        "locator_success"
                    ) is True
                    else 0.0
                    for row in rows
                    if isinstance(
                        row.get(
                            "locator_success"
                        ),
                        bool,
                    )
                ]

                result.append(
                    {
                        "model": model,
                        "instruction_wording": (
                            wording
                        ),
                        "condition": condition,
                        "condition_label": (
                            CONDITION_LABELS[
                                condition
                            ]
                        ),
                        "n_rows": len(
                            rows
                        ),
                        "n_case_clusters": len(
                            by_case
                        ),
                        "e2e_success_count": int(
                            sum(
                                bool_e2e(
                                    row
                                )
                                for row in rows
                            )
                        ),
                        "e2e_rate": rate,
                        "case_cluster_ci_low": (
                            low
                        ),
                        "case_cluster_ci_high": (
                            high
                        ),
                        "locator_rate": mean(
                            locator_values
                        ),
                        "locator_n": len(
                            locator_values
                        ),
                    }
                )

    return result


# ============================================================================
# Paired wording deltas
# ============================================================================

def build_delta_summary(
    data: Mapping[str, Any],
    index: Mapping[str, Any],
) -> List[dict]:
    rng = random.Random(
        BOOTSTRAP_SEED + 1
    )
    paired = index[
        "paired"
    ]
    result = []

    case_updates = []
    for case in data[
        "cases"
    ]:
        case_id = str(
            case["case_id"]
        )
        for update in case[
            "updates"
        ]:
            case_updates.append(
                (
                    case_id,
                    str(
                        update[
                            "update_id"
                        ]
                    ),
                )
            )

    for model in data[
        "models"
    ]:
        for wording in WORDINGS[1:]:
            for condition in CONDITIONS:
                by_case = defaultdict(list)
                improved = 0
                worsened = 0
                tied = 0

                for case_id, update_id in case_updates:
                    baseline_row = paired[
                        (
                            model,
                            "baseline",
                            case_id,
                            update_id,
                            condition,
                        )
                    ]
                    variant_row = paired[
                        (
                            model,
                            wording,
                            case_id,
                            update_id,
                            condition,
                        )
                    ]

                    delta = (
                        bool_e2e(
                            variant_row
                        )
                        - bool_e2e(
                            baseline_row
                        )
                    )
                    by_case[
                        case_id
                    ].append(
                        delta
                    )

                    if delta > 0:
                        improved += 1
                    elif delta < 0:
                        worsened += 1
                    else:
                        tied += 1

                effect, low, high = (
                    cluster_bootstrap_mean_ci(
                        by_case,
                        rng,
                    )
                )

                result.append(
                    {
                        "model": model,
                        "instruction_wording": (
                            wording
                        ),
                        "condition": condition,
                        "condition_label": (
                            CONDITION_LABELS[
                                condition
                            ]
                        ),
                        "delta_definition": (
                            "wording_minus_baseline"
                        ),
                        "e2e_rate_delta": (
                            effect
                        ),
                        "case_cluster_ci_low": (
                            low
                        ),
                        "case_cluster_ci_high": (
                            high
                        ),
                        "paired_target_improved": (
                            improved
                        ),
                        "paired_target_worsened": (
                            worsened
                        ),
                        "paired_target_tied": (
                            tied
                        ),
                        "n_paired_targets": (
                            improved
                            + worsened
                            + tied
                        ),
                        "n_case_clusters": len(
                            by_case
                        ),
                    }
                )

    return result


def build_range_summary(
    data: Mapping[str, Any],
    cell_rows: Sequence[Mapping[str, Any]],
) -> List[dict]:
    lookup = {
        (
            str(row["model"]),
            str(
                row[
                    "instruction_wording"
                ]
            ),
            str(row["condition"]),
        ): float(
            row[
                "e2e_rate"
            ]
        )
        for row in cell_rows
    }

    result = []

    for model in data[
        "models"
    ]:
        for condition in CONDITIONS:
            values = {
                wording: lookup[
                    (
                        model,
                        wording,
                        condition,
                    )
                ]
                for wording in WORDINGS
            }
            min_wording = min(
                values,
                key=values.get,
            )
            max_wording = max(
                values,
                key=values.get,
            )
            result.append(
                {
                    "model": model,
                    "condition": condition,
                    "condition_label": (
                        CONDITION_LABELS[
                            condition
                        ]
                    ),
                    "baseline_rate": values[
                        "baseline"
                    ],
                    "paraphrase_1_rate": values[
                        "instruction_paraphrase_1"
                    ],
                    "paraphrase_2_rate": values[
                        "instruction_paraphrase_2"
                    ],
                    "min_rate": values[
                        min_wording
                    ],
                    "min_wording": min_wording,
                    "max_rate": values[
                        max_wording
                    ],
                    "max_wording": max_wording,
                    "max_minus_min_range": (
                        values[
                            max_wording
                        ]
                        - values[
                            min_wording
                        ]
                    ),
                }
            )

    return result


# ============================================================================
# Interaction and D-vs-E
# ============================================================================

def build_interaction_summary(
    data: Mapping[str, Any],
    index: Mapping[str, Any],
) -> List[dict]:
    rng = random.Random(
        BOOTSTRAP_SEED + 2
    )
    paired = index[
        "paired"
    ]
    result = []

    for model in data[
        "models"
    ]:
        for wording in WORDINGS:
            by_case = defaultdict(list)

            for case in data[
                "cases"
            ]:
                case_id = str(
                    case["case_id"]
                )

                for update in case[
                    "updates"
                ]:
                    update_id = str(
                        update[
                            "update_id"
                        ]
                    )

                    values = {}
                    for condition in CONDITIONS[:4]:
                        row = paired[
                            (
                                model,
                                wording,
                                case_id,
                                update_id,
                                condition,
                            )
                        ]
                        values[
                            condition
                        ] = bool_e2e(
                            row
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

            effect, low, high = (
                cluster_bootstrap_mean_ci(
                    by_case,
                    rng,
                )
            )

            result.append(
                {
                    "model": model,
                    "instruction_wording": wording,
                    "formula": "(D-B)-(C-A)",
                    "interaction": effect,
                    "case_cluster_ci_low": low,
                    "case_cluster_ci_high": high,
                    "n_case_clusters": len(
                        by_case
                    ),
                }
            )

    return result


def build_d_vs_e_summary(
    data: Mapping[str, Any],
    index: Mapping[str, Any],
) -> List[dict]:
    rng = random.Random(
        BOOTSTRAP_SEED + 3
    )
    paired = index[
        "paired"
    ]
    result = []

    for model in data[
        "models"
    ]:
        for wording in WORDINGS:
            by_case = defaultdict(list)
            d_only = 0
            e_only = 0
            tie = 0

            for case in data[
                "cases"
            ]:
                case_id = str(
                    case["case_id"]
                )

                for update in case[
                    "updates"
                ]:
                    update_id = str(
                        update[
                            "update_id"
                        ]
                    )
                    d = bool_e2e(
                        paired[
                            (
                                model,
                                wording,
                                case_id,
                                update_id,
                                "D_atomic_select",
                            )
                        ]
                    )
                    e = bool_e2e(
                        paired[
                            (
                                model,
                                wording,
                                case_id,
                                update_id,
                                "E_compound_exact_patch",
                            )
                        ]
                    )

                    delta = d - e
                    by_case[
                        case_id
                    ].append(
                        delta
                    )

                    if delta > 0:
                        d_only += 1
                    elif delta < 0:
                        e_only += 1
                    else:
                        tie += 1

            effect, low, high = (
                cluster_bootstrap_mean_ci(
                    by_case,
                    rng,
                )
            )

            result.append(
                {
                    "model": model,
                    "instruction_wording": wording,
                    "difference_definition": (
                        "D_atomic_select_minus_E_compound_exact_patch"
                    ),
                    "e2e_rate_difference": (
                        effect
                    ),
                    "case_cluster_ci_low": (
                        low
                    ),
                    "case_cluster_ci_high": (
                        high
                    ),
                    "paired_target_d_only_success": (
                        d_only
                    ),
                    "paired_target_e_only_success": (
                        e_only
                    ),
                    "paired_target_tie": tie,
                    "n_case_clusters": len(
                        by_case
                    ),
                }
            )

    return result


# ============================================================================
# Locator and failures
# ============================================================================

def build_locator_summary(
    data: Mapping[str, Any],
    index: Mapping[str, Any],
) -> List[dict]:
    result = []

    for model in data[
        "models"
    ]:
        for wording in WORDINGS:
            for condition in (
                "B_compound_select",
                "D_atomic_select",
                "E_compound_exact_patch",
            ):
                rows = index[
                    "by_cell"
                ][
                    (
                        model,
                        wording,
                        condition,
                    )
                ]
                values = [
                    1.0
                    if row.get(
                        "locator_success"
                    ) is True
                    else 0.0
                    for row in rows
                    if isinstance(
                        row.get(
                            "locator_success"
                        ),
                        bool,
                    )
                ]
                result.append(
                    {
                        "model": model,
                        "instruction_wording": wording,
                        "condition": condition,
                        "locator_success_count": int(
                            sum(values)
                        ),
                        "locator_n": len(
                            values
                        ),
                        "locator_rate": mean(
                            values
                        ),
                    }
                )

    return result


def build_failure_summary(
    data: Mapping[str, Any],
    index: Mapping[str, Any],
) -> List[dict]:
    reasons = sorted(
        {
            str(reason)
            for row in data[
                "rows"
            ]
            for reason in row.get(
                "failure_reasons",
                [],
            )
        }
    )

    result = []

    for model in data[
        "models"
    ]:
        for wording in WORDINGS:
            for condition in CONDITIONS:
                rows = index[
                    "by_cell"
                ][
                    (
                        model,
                        wording,
                        condition,
                    )
                ]

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

                for reason in reasons:
                    result.append(
                        {
                            "model": model,
                            "instruction_wording": wording,
                            "condition": condition,
                            "failure_reason": reason,
                            "rows_with_reason": int(
                                counter.get(
                                    reason,
                                    0,
                                )
                            ),
                            "n_rows": len(
                                rows
                            ),
                            "multi_label_overlap": True,
                        }
                    )

    return result


# ============================================================================
# Report
# ============================================================================

def render_report(
    data: Mapping[str, Any],
    cell_rows: Sequence[Mapping[str, Any]],
    delta_rows: Sequence[Mapping[str, Any]],
    range_rows: Sequence[Mapping[str, Any]],
    interaction_rows: Sequence[Mapping[str, Any]],
    d_vs_e_rows: Sequence[Mapping[str, Any]],
) -> str:
    cell = {
        (
            str(row["model"]),
            str(
                row[
                    "instruction_wording"
                ]
            ),
            str(row["condition"]),
        ): row
        for row in cell_rows
    }

    delta = {
        (
            str(row["model"]),
            str(
                row[
                    "instruction_wording"
                ]
            ),
            str(row["condition"]),
        ): row
        for row in delta_rows
    }

    lines = []
    lines.append(
        "# Instruction-Only Robustness — Paired Offline Sensitivity Analysis"
    )
    lines.append("")
    lines.append(
        "This analysis reuses the same 20 formal cases. It is a sensitivity "
        "analysis, not an independent validation."
    )
    lines.append("")
    lines.append(
        "Only system-instruction wording differs across the baseline and two "
        "pre-specified paraphrases. Memory, evidence, condition definitions, "
        "user-prompt adapter, schemas, evaluator, models, and generation "
        "settings remain frozen."
    )
    lines.append("")

    lines.append(
        "## E2E success across instruction wordings"
    )
    lines.append("")

    for model in data[
        "models"
    ]:
        lines.append(
            "### {}".format(
                model
            )
        )
        lines.append("")
        lines.append(
            "| Condition | Baseline | Paraphrase 1 | Paraphrase 2 |"
        )
        lines.append(
            "|---|---:|---:|---:|"
        )

        for condition in CONDITIONS:
            values = []
            for wording in WORDINGS:
                row = cell[
                    (
                        model,
                        wording,
                        condition,
                    )
                ]
                values.append(
                    "{}/40 ({})".format(
                        int(
                            row[
                                "e2e_success_count"
                            ]
                        ),
                        fmt_rate(
                            row[
                                "e2e_rate"
                            ]
                        ),
                    )
                )

            lines.append(
                "| {} | {} | {} | {} |".format(
                    condition,
                    values[0],
                    values[1],
                    values[2],
                )
            )
        lines.append("")

    lines.append(
        "## Paired change relative to the baseline wording"
    )
    lines.append("")
    lines.append(
        "Differences are paraphrase minus baseline on the exact same "
        "case/update/condition/model. Confidence intervals resample whole cases."
    )
    lines.append("")
    lines.append(
        "| Model | Wording | Condition | E2E delta | 95% case-cluster CI | "
        "Improved / Worsened / Tied targets |"
    )
    lines.append(
        "|---|---|---|---:|---:|---:|"
    )

    for row in delta_rows:
        lines.append(
            "| {} | {} | {} | {} | [{}, {}] | {} / {} / {} |".format(
                row["model"],
                row[
                    "instruction_wording"
                ],
                row["condition"],
                fmt_effect(
                    row[
                        "e2e_rate_delta"
                    ]
                ),
                fmt_effect(
                    row[
                        "case_cluster_ci_low"
                    ]
                ),
                fmt_effect(
                    row[
                        "case_cluster_ci_high"
                    ]
                ),
                int(
                    row[
                        "paired_target_improved"
                    ]
                ),
                int(
                    row[
                        "paired_target_worsened"
                    ]
                ),
                int(
                    row[
                        "paired_target_tied"
                    ]
                ),
            )
        )

    lines.append("")
    lines.append(
        "## Wording range"
    )
    lines.append("")
    lines.append(
        "The range is max minus min E2E rate across the three wordings. It is "
        "descriptive sensitivity, not an independent uncertainty estimate."
    )
    lines.append("")
    lines.append(
        "| Model | Condition | Baseline | P1 | P2 | Max-min range |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|"
    )

    for row in range_rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                row["model"],
                row["condition"],
                fmt_rate(
                    row[
                        "baseline_rate"
                    ]
                ),
                fmt_rate(
                    row[
                        "paraphrase_1_rate"
                    ]
                ),
                fmt_rate(
                    row[
                        "paraphrase_2_rate"
                    ]
                ),
                fmt_effect(
                    row[
                        "max_minus_min_range"
                    ]
                ),
            )
        )

    lines.append("")
    lines.append(
        "## Representation x edit-scope interaction by wording"
    )
    lines.append("")
    lines.append(
        "Interaction = `(D-B)-(C-A)`."
    )
    lines.append("")
    lines.append(
        "| Model | Wording | Interaction | 95% case-cluster CI |"
    )
    lines.append(
        "|---|---|---:|---:|"
    )
    for row in interaction_rows:
        lines.append(
            "| {} | {} | {} | [{}, {}] |".format(
                row["model"],
                row[
                    "instruction_wording"
                ],
                fmt_effect(
                    row[
                        "interaction"
                    ]
                ),
                fmt_effect(
                    row[
                        "case_cluster_ci_low"
                    ]
                ),
                fmt_effect(
                    row[
                        "case_cluster_ci_high"
                    ]
                ),
            )
        )

    lines.append("")
    lines.append(
        "## Atomic Select versus Compound Exact Patch"
    )
    lines.append("")
    lines.append(
        "Difference = D Atomic Select minus E Compound Exact Patch."
    )
    lines.append("")
    lines.append(
        "| Model | Wording | D-E difference | 95% case-cluster CI | "
        "D-only / E-only / Tie |"
    )
    lines.append(
        "|---|---|---:|---:|---:|"
    )
    for row in d_vs_e_rows:
        lines.append(
            "| {} | {} | {} | [{}, {}] | {} / {} / {} |".format(
                row["model"],
                row[
                    "instruction_wording"
                ],
                fmt_effect(
                    row[
                        "e2e_rate_difference"
                    ]
                ),
                fmt_effect(
                    row[
                        "case_cluster_ci_low"
                    ]
                ),
                fmt_effect(
                    row[
                        "case_cluster_ci_high"
                    ]
                ),
                int(
                    row[
                        "paired_target_d_only_success"
                    ]
                ),
                int(
                    row[
                        "paired_target_e_only_success"
                    ]
                ),
                int(
                    row[
                        "paired_target_tie"
                    ]
                ),
            )
        )

    lines.append("")
    lines.append(
        "## Interpretation guardrails"
    )
    lines.append("")
    lines.append(
        "- This uses the same 20 cases, so it measures instruction-wording "
        "sensitivity rather than independent replication."
    )
    lines.append(
        "- Do not declare a configuration robust merely because its rank stays "
        "the same; compare paired effect magnitudes and clustered intervals."
    )
    lines.append(
        "- Large wording shifts should first be treated as instability of the "
        "design conclusion, not as a positive contribution by themselves."
    )
    lines.append(
        "- Exact Patch remains a core comparator; similar observed rates do "
        "not prove equivalence."
    )
    lines.append(
        "- Interaction estimates must be reported regardless of significance "
        "if the paper centers representation x edit scope."
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
        "REPRESENTATION x EDIT-SCOPE — INSTRUCTION ROBUSTNESS OFFLINE ANALYSIS"
    )
    print("=" * 78)
    print("Mode: ZERO MODEL CALLS")
    print(
        "Bootstrap: {} whole-case replicates".format(
            BOOTSTRAP_REPS
        )
    )
    print()

    print(
        "[1/6] Auditing frozen baseline + completed robustness run..."
    )
    data = audit_and_load()
    print(
        "  PASS | 600 baseline + 1200 robustness rows | same 20 cases"
    )

    print(
        "[2/6] Computing wording x model x condition rates..."
    )
    index = build_index(
        data[
            "rows"
        ]
    )
    cell_rows = build_cell_summary(
        data,
        index,
    )
    print(
        "  PASS | {} cells".format(
            len(cell_rows)
        )
    )

    print(
        "[3/6] Computing paired paraphrase-minus-baseline effects..."
    )
    delta_rows = build_delta_summary(
        data,
        index,
    )
    range_rows = build_range_summary(
        data,
        cell_rows,
    )
    print(
        "  PASS | {} paired deltas | {} wording ranges".format(
            len(delta_rows),
            len(range_rows),
        )
    )

    print(
        "[4/6] Computing interaction and D-vs-E sensitivity..."
    )
    interaction_rows = build_interaction_summary(
        data,
        index,
    )
    d_vs_e_rows = build_d_vs_e_summary(
        data,
        index,
    )
    print(
        "  PASS | {} interactions | {} D-vs-E comparisons".format(
            len(interaction_rows),
            len(d_vs_e_rows),
        )
    )

    print(
        "[5/6] Computing locator + multi-label failure summaries..."
    )
    locator_rows = build_locator_summary(
        data,
        index,
    )
    failure_rows = build_failure_summary(
        data,
        index,
    )
    print(
        "  PASS | failure labels remain overlapping"
    )

    print(
        "[6/6] Writing offline sensitivity outputs..."
    )
    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        CELL_SUMMARY_PATH,
        cell_rows,
        (
            "model",
            "instruction_wording",
            "condition",
            "condition_label",
            "n_rows",
            "n_case_clusters",
            "e2e_success_count",
            "e2e_rate",
            "case_cluster_ci_low",
            "case_cluster_ci_high",
            "locator_rate",
            "locator_n",
        ),
    )

    write_csv(
        DELTA_SUMMARY_PATH,
        delta_rows,
        (
            "model",
            "instruction_wording",
            "condition",
            "condition_label",
            "delta_definition",
            "e2e_rate_delta",
            "case_cluster_ci_low",
            "case_cluster_ci_high",
            "paired_target_improved",
            "paired_target_worsened",
            "paired_target_tied",
            "n_paired_targets",
            "n_case_clusters",
        ),
    )

    write_csv(
        RANGE_SUMMARY_PATH,
        range_rows,
        (
            "model",
            "condition",
            "condition_label",
            "baseline_rate",
            "paraphrase_1_rate",
            "paraphrase_2_rate",
            "min_rate",
            "min_wording",
            "max_rate",
            "max_wording",
            "max_minus_min_range",
        ),
    )

    write_csv(
        INTERACTION_SUMMARY_PATH,
        interaction_rows,
        (
            "model",
            "instruction_wording",
            "formula",
            "interaction",
            "case_cluster_ci_low",
            "case_cluster_ci_high",
            "n_case_clusters",
        ),
    )

    write_csv(
        D_VS_E_SUMMARY_PATH,
        d_vs_e_rows,
        (
            "model",
            "instruction_wording",
            "difference_definition",
            "e2e_rate_difference",
            "case_cluster_ci_low",
            "case_cluster_ci_high",
            "paired_target_d_only_success",
            "paired_target_e_only_success",
            "paired_target_tie",
            "n_case_clusters",
        ),
    )

    write_csv(
        LOCATOR_SUMMARY_PATH,
        locator_rows,
        (
            "model",
            "instruction_wording",
            "condition",
            "locator_success_count",
            "locator_n",
            "locator_rate",
        ),
    )

    write_csv(
        FAILURE_SUMMARY_PATH,
        failure_rows,
        (
            "model",
            "instruction_wording",
            "condition",
            "failure_reason",
            "rows_with_reason",
            "n_rows",
            "multi_label_overlap",
        ),
    )

    analysis_json = {
        "status": (
            "INSTRUCTION_ROBUSTNESS_OFFLINE_ANALYSIS_COMPLETE"
        ),
        "model_calls": 0,
        "independent_validation": False,
        "same_20_cases": True,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "cluster_unit": "case",
        "json_valid_rows_across_1800": data[
            "json_valid_count"
        ],
        "source_hashes": data[
            "source_hashes"
        ],
        "wording_condition_summary": (
            cell_rows
        ),
        "wording_delta_vs_baseline": (
            delta_rows
        ),
        "wording_range_summary": (
            range_rows
        ),
        "interaction_by_wording": (
            interaction_rows
        ),
        "atomic_select_vs_exact_patch": (
            d_vs_e_rows
        ),
        "locator_summary": (
            locator_rows
        ),
        "failure_reason_summary": (
            failure_rows
        ),
    }
    write_json(
        ANALYSIS_JSON_PATH,
        analysis_json,
    )

    report = render_report(
        data,
        cell_rows,
        delta_rows,
        range_rows,
        interaction_rows,
        d_vs_e_rows,
    )
    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(
            report
        )

    manifest = {
        "status": (
            "INSTRUCTION_ROBUSTNESS_OFFLINE_ANALYSIS_COMPLETE"
        ),
        "model_calls": 0,
        "analysis_script_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        "source_hashes": data[
            "source_hashes"
        ],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "outputs": {
            "wording_condition_summary": str(
                CELL_SUMMARY_PATH
            ),
            "wording_delta_vs_baseline": str(
                DELTA_SUMMARY_PATH
            ),
            "wording_range_summary": str(
                RANGE_SUMMARY_PATH
            ),
            "interaction_by_wording": str(
                INTERACTION_SUMMARY_PATH
            ),
            "atomic_select_vs_exact_patch": str(
                D_VS_E_SUMMARY_PATH
            ),
            "locator_summary": str(
                LOCATOR_SUMMARY_PATH
            ),
            "failure_reason_summary": str(
                FAILURE_SUMMARY_PATH
            ),
            "analysis_json": str(
                ANALYSIS_JSON_PATH
            ),
            "report": str(
                REPORT_PATH
            ),
        },
    }
    write_json(
        MANIFEST_PATH,
        manifest,
    )

    # Compact console summary.
    cell_lookup = {
        (
            str(row["model"]),
            str(
                row[
                    "instruction_wording"
                ]
            ),
            str(row["condition"]),
        ): row
        for row in cell_rows
    }

    print()
    print("=" * 78)
    print(
        "INSTRUCTION ROBUSTNESS OFFLINE ANALYSIS COMPLETE"
    )
    print("=" * 78)

    for model in data[
        "models"
    ]:
        print(model)
        for condition in CONDITIONS:
            values = []
            for wording in WORDINGS:
                row = cell_lookup[
                    (
                        model,
                        wording,
                        condition,
                    )
                ]
                values.append(
                    "{}={}/40 ({})".format(
                        wording,
                        int(
                            row[
                                "e2e_success_count"
                            ]
                        ),
                        fmt_rate(
                            row[
                                "e2e_rate"
                            ]
                        ),
                    )
                )
            print(
                "  {} | {}".format(
                    condition,
                    " | ".join(
                        values
                    ),
                )
            )

    print()
    print("Wrote:")
    for path in (
        CELL_SUMMARY_PATH,
        DELTA_SUMMARY_PATH,
        RANGE_SUMMARY_PATH,
        INTERACTION_SUMMARY_PATH,
        D_VS_E_SUMMARY_PATH,
        LOCATOR_SUMMARY_PATH,
        FAILURE_SUMMARY_PATH,
        ANALYSIS_JSON_PATH,
        REPORT_PATH,
        MANIFEST_PATH,
    ):
        print(
            "  {}".format(
                path
            )
        )

    print()
    print(
        "NO MODEL CALLS WERE MADE. Baseline and robustness raw outputs were "
        "not modified."
    )


if __name__ == "__main__":
    from atomic_memory.reanalysis import run_analysis
    run_analysis(main, globals(), paths.WORDING, Path(__file__).with_name("run_wording.py"))
