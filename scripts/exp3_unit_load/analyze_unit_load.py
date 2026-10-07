#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze paired unit-load effects with whole-case bootstrap intervals.

Use --check for read-only verification of saved analyses."""

import csv
import hashlib
import json
import os
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "atomic_memory").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atomic_memory import paths

from atomic_memory.unit_load import (
    CONDITION_L1_PATCH,
    CONDITION_L1_SELECT,
    CONDITION_L2_PATCH,
    CONDITION_L2_SELECT,
    CONDITION_L3_PATCH,
    CONDITION_L3_SELECT,
    UNIT_LOAD_CONDITIONS,
)

OUTPUT_DIR = paths.UNIT_LOAD
RESULTS_PATH = OUTPUT_DIR / "unit_load_results.jsonl"
FREEZE_MANIFEST_PATH = OUTPUT_DIR / "unit_load_freeze_manifest.json"
ANALYSIS_DIR = OUTPUT_DIR / "unit_load_analysis"

CONDITION_SUMMARY_PATH = ANALYSIS_DIR / "condition_summary.csv"
CONTRAST_SUMMARY_PATH = ANALYSIS_DIR / "contrast_summary.csv"
DIAGNOSTIC_SUMMARY_PATH = ANALYSIS_DIR / "diagnostic_summary.csv"
FAILURE_SUMMARY_PATH = ANALYSIS_DIR / "failure_reason_summary.csv"
COST_SUMMARY_PATH = ANALYSIS_DIR / "cost_summary.csv"
ANALYSIS_JSON_PATH = ANALYSIS_DIR / "unit_load_analysis.json"
REPORT_PATH = ANALYSIS_DIR / "report.md"
MANIFEST_PATH = ANALYSIS_DIR / "analysis_manifest.json"

EXPECTED_TOTAL_CALLS = 720
EXPECTED_CASES = 20
EXPECTED_TARGETS_PER_CASE = 2
BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20261006

MODELS = (
    "llama3.1:8b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "gemma3:12b-it-q4_K_M",
)

CONDITION_LABELS = {
    CONDITION_L1_SELECT: "Load 1 / Select",
    CONDITION_L1_PATCH: "Load 1 / Patch",
    CONDITION_L2_SELECT: "Load 2 / Select",
    CONDITION_L2_PATCH: "Load 2 / Patch",
    CONDITION_L3_SELECT: "Load 3 / Select",
    CONDITION_L3_PATCH: "Load 3 / Patch",
}


# ============================================================================
# Helpers
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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
                raise ValueError("Row {} is not an object".format(line_number))
            rows.append(value)
    return rows


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values)) / float(len(values))


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower]) * (1.0 - weight) + float(sorted_values[upper]) * weight


def bootstrap_ci(
    case_values: Mapping[str, float],
    reps: int,
    seed: int,
) -> Tuple[float, float]:
    case_ids = sorted(case_values)
    rng = random.Random(seed)
    draws = []
    n = len(case_ids)
    for _ in range(reps):
        sampled = [case_ids[rng.randrange(n)] for _ in range(n)]
        draws.append(mean([float(case_values[case_id]) for case_id in sampled]))
    draws.sort()
    return percentile(draws, 0.025), percentile(draws, 0.975)


def paired_bootstrap_ci(
    case_values: Mapping[str, float],
    reps: int,
    seed: int,
) -> Tuple[float, float]:
    return bootstrap_ci(case_values, reps, seed)


def safe_bool(value: Any) -> Any:
    if value is None:
        return None
    return bool(value)


# ============================================================================
# Validation and indexing
# ============================================================================

def validate_inputs(rows: Sequence[Mapping[str, Any]], freeze: Mapping[str, Any]) -> dict:
    immutable = freeze.get("immutable_protocol")
    if not isinstance(immutable, Mapping):
        raise RuntimeError("Freeze manifest lacks immutable_protocol")

    if immutable.get("expected_total_calls") != EXPECTED_TOTAL_CALLS:
        raise RuntimeError("Freeze manifest expected_total_calls mismatch")
    if immutable.get("bootstrap_reps") != BOOTSTRAP_REPS:
        raise RuntimeError("Frozen bootstrap reps differ from analyzer")
    if immutable.get("bootstrap_seed") != BOOTSTRAP_SEED:
        raise RuntimeError("Frozen bootstrap seed differs from analyzer")
    if tuple(immutable.get("models", [])) != MODELS:
        raise RuntimeError("Frozen model list differs from analyzer")
    if tuple(immutable.get("conditions", [])) != tuple(UNIT_LOAD_CONDITIONS):
        raise RuntimeError("Frozen condition list differs from analyzer")

    if len(rows) != EXPECTED_TOTAL_CALLS:
        raise RuntimeError(
            "Expected {} completed rows, found {}".format(
                EXPECTED_TOTAL_CALLS,
                len(rows),
            )
        )

    keys = set()
    case_ids = set()
    per_cell = defaultdict(int)
    per_case_target_ids = defaultdict(set)

    for row in rows:
        key = (
            str(row["case_id"]),
            str(row["update_id"]),
            str(row["condition"]),
            str(row["model"]),
        )
        if key in keys:
            raise RuntimeError("Duplicate result key: {}".format(key))
        keys.add(key)
        case_ids.add(str(row["case_id"]))
        per_cell[(str(row["model"]), str(row["condition"]))] += 1
        per_case_target_ids[str(row["case_id"])].add(str(row["update_id"]))

    if len(case_ids) != EXPECTED_CASES:
        raise RuntimeError("Expected 20 unique cases, found {}".format(len(case_ids)))

    for case_id, update_ids in per_case_target_ids.items():
        if len(update_ids) != EXPECTED_TARGETS_PER_CASE:
            raise RuntimeError("Case {} does not have two targets".format(case_id))

    for model in MODELS:
        for condition in UNIT_LOAD_CONDITIONS:
            if per_cell[(model, condition)] != 40:
                raise RuntimeError(
                    "Cell {} {} has {} rows, expected 40".format(
                        model,
                        condition,
                        per_cell[(model, condition)],
                    )
                )

    return {
        "row_count": len(rows),
        "unique_key_count": len(keys),
        "case_count": len(case_ids),
        "rows_per_model_condition": 40,
    }


def index_rows(rows: Sequence[Mapping[str, Any]]) -> dict:
    index = {}
    for row in rows:
        key = (
            str(row["model"]),
            str(row["case_id"]),
            str(row["update_id"]),
            str(row["condition"]),
        )
        index[key] = row
    return index


def case_means(index: Mapping[Tuple[str, str, str, str], Mapping[str, Any]], model: str, condition: str) -> Dict[str, float]:
    values = {}
    case_ids = sorted({key[1] for key in index if key[0] == model})
    for case_id in case_ids:
        target_rows = [
            row
            for key, row in index.items()
            if key[0] == model and key[1] == case_id and key[3] == condition
        ]
        if len(target_rows) != 2:
            raise RuntimeError("Expected two rows for {} {} {}".format(model, case_id, condition))
        values[case_id] = mean(
            [
                1.0 if bool(row["evaluation"]["end_to_end_success"]) else 0.0
                for row in target_rows
            ]
        )
    return values


def contrast_case_values(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> Dict[str, float]:
    # Subtract matched case means before resampling to retain pairing.
    if set(left) != set(right):
        raise RuntimeError("Paired contrast case sets differ")
    return {case_id: float(left[case_id]) - float(right[case_id]) for case_id in left}


# ============================================================================
# Summaries
# ============================================================================

def condition_summaries(rows: Sequence[Mapping[str, Any]], index: Mapping[Tuple[str, str, str, str], Mapping[str, Any]]) -> List[dict]:
    result = []
    for model_index, model in enumerate(MODELS):
        for condition_index, condition in enumerate(UNIT_LOAD_CONDITIONS):
            cell = [row for row in rows if row["model"] == model and row["condition"] == condition]
            success_count = sum(1 for row in cell if bool(row["evaluation"]["end_to_end_success"]))
            per_case = case_means(index, model, condition)
            ci_low, ci_high = bootstrap_ci(
                per_case,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED + model_index * 100 + condition_index,
            )
            spec_load = int(cell[0]["unit_load"])
            interface = str(cell[0]["interface"])
            result.append(
                {
                    "model": model,
                    "condition": condition,
                    "label": CONDITION_LABELS[condition],
                    "unit_load": spec_load,
                    "interface": interface,
                    "n_cases": EXPECTED_CASES,
                    "n_targets": len(cell),
                    "success_count": success_count,
                    "success_rate": success_count / float(len(cell)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return result


def target_level_pair_counts(
    index: Mapping[Tuple[str, str, str, str], Mapping[str, Any]],
    model: str,
    left_condition: str,
    right_condition: str,
) -> Tuple[int, int, int]:
    left_only = 0
    right_only = 0
    ties = 0
    case_ids = sorted({key[1] for key in index if key[0] == model})
    for case_id in case_ids:
        update_ids = sorted({key[2] for key in index if key[0] == model and key[1] == case_id})
        for update_id in update_ids:
            left = bool(index[(model, case_id, update_id, left_condition)]["evaluation"]["end_to_end_success"])
            right = bool(index[(model, case_id, update_id, right_condition)]["evaluation"]["end_to_end_success"])
            if left and not right:
                left_only += 1
            elif right and not left:
                right_only += 1
            else:
                ties += 1
    return left_only, right_only, ties


def contrast_summaries(index: Mapping[Tuple[str, str, str, str], Mapping[str, Any]]) -> List[dict]:
    result = []

    for model_index, model in enumerate(MODELS):
        cm = {
            condition: case_means(index, model, condition)
            for condition in UNIT_LOAD_CONDITIONS
        }

        # Primary: G_Patch = Patch L1 - Patch L3
        g_patch_cases = contrast_case_values(cm[CONDITION_L1_PATCH], cm[CONDITION_L3_PATCH])
        gp_low, gp_high = paired_bootstrap_ci(
            g_patch_cases,
            BOOTSTRAP_REPS,
            BOOTSTRAP_SEED + 1000 + model_index,
        )
        left_only, right_only, ties = target_level_pair_counts(
            index,
            model,
            CONDITION_L1_PATCH,
            CONDITION_L3_PATCH,
        )
        result.append(
            {
                "model": model,
                "role": "primary",
                "contrast": "G_Patch",
                "definition": "Patch L1 - Patch L3",
                "estimate": mean(list(g_patch_cases.values())),
                "ci_low": gp_low,
                "ci_high": gp_high,
                "left_only_targets": left_only,
                "right_only_targets": right_only,
                "tie_targets": ties,
            }
        )

        # Supporting: G_Select = Select L1 - Select L3
        g_select_cases = contrast_case_values(cm[CONDITION_L1_SELECT], cm[CONDITION_L3_SELECT])
        gs_low, gs_high = paired_bootstrap_ci(
            g_select_cases,
            BOOTSTRAP_REPS,
            BOOTSTRAP_SEED + 1100 + model_index,
        )
        left_only, right_only, ties = target_level_pair_counts(
            index,
            model,
            CONDITION_L1_SELECT,
            CONDITION_L3_SELECT,
        )
        result.append(
            {
                "model": model,
                "role": "supporting",
                "contrast": "G_Select",
                "definition": "Select L1 - Select L3",
                "estimate": mean(list(g_select_cases.values())),
                "ci_low": gs_low,
                "ci_high": gs_high,
                "left_only_targets": left_only,
                "right_only_targets": right_only,
                "tie_targets": ties,
            }
        )

        # Key secondary interaction S.
        s_cases = {
            case_id: float(g_select_cases[case_id]) - float(g_patch_cases[case_id])
            for case_id in g_patch_cases
        }
        s_low, s_high = paired_bootstrap_ci(
            s_cases,
            BOOTSTRAP_REPS,
            BOOTSTRAP_SEED + 1200 + model_index,
        )
        positive = sum(1 for value in s_cases.values() if value > 0)
        negative = sum(1 for value in s_cases.values() if value < 0)
        zero = sum(1 for value in s_cases.values() if value == 0)
        result.append(
            {
                "model": model,
                "role": "key_secondary",
                "contrast": "S",
                "definition": "(Select L1-L3) - (Patch L1-L3)",
                "estimate": mean(list(s_cases.values())),
                "ci_low": s_low,
                "ci_high": s_high,
                "left_only_targets": "",
                "right_only_targets": "",
                "tie_targets": "",
                "positive_case_interactions": positive,
                "negative_case_interactions": negative,
                "zero_case_interactions": zero,
            }
        )

        # Pre-specified supporting paired contrasts to show the six absolute cells.
        supporting_pairs = (
            ("Patch_L1_minus_L2", CONDITION_L1_PATCH, CONDITION_L2_PATCH),
            ("Patch_L2_minus_L3", CONDITION_L2_PATCH, CONDITION_L3_PATCH),
            ("Select_L1_minus_L2", CONDITION_L1_SELECT, CONDITION_L2_SELECT),
            ("Select_L2_minus_L3", CONDITION_L2_SELECT, CONDITION_L3_SELECT),
            ("Patch_minus_Select_L1", CONDITION_L1_PATCH, CONDITION_L1_SELECT),
            ("Patch_minus_Select_L2", CONDITION_L2_PATCH, CONDITION_L2_SELECT),
            ("Patch_minus_Select_L3", CONDITION_L3_PATCH, CONDITION_L3_SELECT),
        )
        for pair_index, (name, left_condition, right_condition) in enumerate(supporting_pairs):
            values = contrast_case_values(cm[left_condition], cm[right_condition])
            ci_low, ci_high = paired_bootstrap_ci(
                values,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED + 2000 + model_index * 20 + pair_index,
            )
            left_only, right_only, ties = target_level_pair_counts(
                index,
                model,
                left_condition,
                right_condition,
            )
            result.append(
                {
                    "model": model,
                    "role": "supporting",
                    "contrast": name,
                    "definition": "{} - {}".format(left_condition, right_condition),
                    "estimate": mean(list(values.values())),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "left_only_targets": left_only,
                    "right_only_targets": right_only,
                    "tie_targets": ties,
                }
            )

    return result


def diagnostic_summaries(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    fields = (
        ("locator_success", "row"),
        ("legal_action", "execution"),
        ("execution_success", "execution"),
        ("contract_valid", "evaluation"),
        ("structure_valid", "evaluation"),
        ("target_new_current", "evaluation"),
        ("target_old_not_current", "evaluation"),
        ("protected_facts_preserved", "evaluation"),
        ("no_unsupported_changes", "evaluation"),
        ("end_to_end_success", "evaluation"),
    )
    result = []

    for model in MODELS:
        for condition in UNIT_LOAD_CONDITIONS:
            cell = [row for row in rows if row["model"] == model and row["condition"] == condition]
            for field, source in fields:
                values = []
                for row in cell:
                    if source == "row":
                        value = row.get(field)
                    else:
                        value = row.get(source, {}).get(field)
                    values.append(safe_bool(value))
                success = sum(1 for value in values if value is True)
                failure = sum(1 for value in values if value is False)
                unscorable = sum(1 for value in values if value is None)
                result.append(
                    {
                        "model": model,
                        "condition": condition,
                        "metric": field,
                        "success_count": success,
                        "failure_count": failure,
                        "unscorable_count": unscorable,
                        "total_count": len(values),
                        "rate_over_all": success / float(len(values)),
                    }
                )

            correctly_localized = [row for row in cell if row.get("locator_success") is True]
            conditional_success = sum(
                1
                for row in correctly_localized
                if bool(row.get("execution", {}).get("execution_success"))
            )
            result.append(
                {
                    "model": model,
                    "condition": condition,
                    "metric": "execution_success_given_correct_localization",
                    "success_count": conditional_success,
                    "failure_count": len(correctly_localized) - conditional_success,
                    "unscorable_count": 0,
                    "total_count": len(correctly_localized),
                    "rate_over_all": (
                        conditional_success / float(len(correctly_localized))
                        if correctly_localized
                        else ""
                    ),
                }
            )

    return result


def failure_summaries(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    result = []
    for model in MODELS:
        for condition in UNIT_LOAD_CONDITIONS:
            cell = [row for row in rows if row["model"] == model and row["condition"] == condition]
            counts = defaultdict(int)
            for row in cell:
                for reason in row.get("failure_reasons", []):
                    counts[str(reason)] += 1
            for reason in sorted(counts):
                result.append(
                    {
                        "model": model,
                        "condition": condition,
                        "failure_reason": reason,
                        "row_count": counts[reason],
                        "note": "labels overlap; counts must not be summed as independent failures",
                    }
                )
    return result


def cost_summaries(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    result = []
    first_planned_by_model = {
        model: min(int(row["planned_call_index"]) for row in rows if row["model"] == model)
        for model in MODELS
    }

    for model in MODELS:
        for condition in UNIT_LOAD_CONDITIONS:
            cell = [row for row in rows if row["model"] == model and row["condition"] == condition]
            input_tokens = [float(row.get("usage", {}).get("input_tokens", 0)) for row in cell]
            output_tokens = [float(row.get("usage", {}).get("output_tokens", 0)) for row in cell]
            total_tokens = [a + b for a, b in zip(input_tokens, output_tokens)]
            latencies = [float(row.get("latency_seconds", 0.0)) for row in cell]
            warm_rows = [
                row
                for row in cell
                if int(row["planned_call_index"]) != first_planned_by_model[model]
            ]
            warm_latencies = [float(row.get("latency_seconds", 0.0)) for row in warm_rows]
            result.append(
                {
                    "model": model,
                    "condition": condition,
                    "unit_load": int(cell[0]["unit_load"]),
                    "interface": str(cell[0]["interface"]),
                    "mean_input_tokens": mean(input_tokens),
                    "mean_output_tokens": mean(output_tokens),
                    "mean_total_tokens": mean(total_tokens),
                    "mean_latency_seconds": mean(latencies),
                    "mean_warm_latency_seconds": mean(warm_latencies),
                    "n_calls": len(cell),
                    "warm_latency_n": len(warm_rows),
                }
            )
    return result


# ============================================================================
# Report
# ============================================================================

def pct(value: float) -> str:
    return "{:.1f}%".format(100.0 * float(value))


def pp(value: float) -> str:
    return "{:+.1f} pp".format(100.0 * float(value))


def render_report(
    condition_rows: Sequence[Mapping[str, Any]],
    contrast_rows: Sequence[Mapping[str, Any]],
    cost_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = []
    lines.append("# Unit-load extension analysis")
    lines.append("")
    lines.append("This is the frozen offline analysis of the 720-call extension. Models are analyzed separately. Confidence intervals resample whole cases and preserve both targets and all six conditions.")
    lines.append("")

    lines.append("## End-to-end success")
    lines.append("")
    lines.append("| Model | L1 Select | L1 Patch | L2 Select | L2 Patch | L3 Select | L3 Patch |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for model in MODELS:
        cells = {row["condition"]: row for row in condition_rows if row["model"] == model}
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                model,
                pct(cells[CONDITION_L1_SELECT]["success_rate"]),
                pct(cells[CONDITION_L1_PATCH]["success_rate"]),
                pct(cells[CONDITION_L2_SELECT]["success_rate"]),
                pct(cells[CONDITION_L2_PATCH]["success_rate"]),
                pct(cells[CONDITION_L3_SELECT]["success_rate"]),
                pct(cells[CONDITION_L3_PATCH]["success_rate"]),
            )
        )
    lines.append("")

    lines.append("## Frozen primary and key secondary contrasts")
    lines.append("")
    lines.append("| Model | Contrast | Estimate | 95% case-cluster bootstrap interval |")
    lines.append("|---|---|---:|---:|")
    for model in MODELS:
        for role in ("primary", "key_secondary"):
            row = next(item for item in contrast_rows if item["model"] == model and item["role"] == role)
            lines.append(
                "| {} | {} | {} | [{}, {}] |".format(
                    model,
                    row["contrast"],
                    pp(row["estimate"]),
                    pp(row["ci_low"]),
                    pp(row["ci_high"]),
                )
            )
    lines.append("")

    lines.append("Interpretation guardrails:")
    lines.append("")
    lines.append("- `G_Patch > 0` means finer grouping had higher observed Patch success in this sample; it does not by itself establish a deployment recommendation.")
    lines.append("- `S > 0` only means the load-1 versus load-3 grouping contrast is more positive under Select than Patch. It does not establish Patch as the better absolute configuration.")
    lines.append("- Similar observed success rates do not establish equivalence or noninferiority.")
    lines.append("- Grouping load changes unit size and unit count together, so these are configuration effects rather than a pure causal effect of abstract granularity.")
    lines.append("")

    lines.append("## Token overhead")
    lines.append("")
    lines.append("| Model | Condition | Mean input | Mean output | Mean total | Warm latency (s) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for model in MODELS:
        for condition in UNIT_LOAD_CONDITIONS:
            row = next(item for item in cost_rows if item["model"] == model and item["condition"] == condition)
            lines.append(
                "| {} | {} | {:.1f} | {:.1f} | {:.1f} | {:.3f} |".format(
                    model,
                    CONDITION_LABELS[condition],
                    row["mean_input_tokens"],
                    row["mean_output_tokens"],
                    row["mean_total_tokens"],
                    row["mean_warm_latency_seconds"],
                )
            )
    lines.append("")
    lines.append("Token counts are inference-overhead measurements within a model, not storage, retrieval, monetary, or utility estimates. Latency is descriptive for the recorded backend.")
    lines.append("")
    lines.append("## Analysis status")
    lines.append("")
    lines.append("The primary and key secondary estimands, bootstrap unit, number of repetitions, and analysis seed were frozen before the 720 formal calls. Supporting contrasts are reported to preserve the six-cell absolute context and must not be used as a post-hoc significance screen.")
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 78)
    print("UNIT-LOAD EXTENSION — FROZEN OFFLINE ANALYSIS")
    print("=" * 78)
    print("ZERO model calls")
    print()

    if not RESULTS_PATH.exists():
        raise FileNotFoundError("Missing unit-load results: {}".format(RESULTS_PATH))
    if not FREEZE_MANIFEST_PATH.exists():
        raise FileNotFoundError("Missing unit-load freeze manifest: {}".format(FREEZE_MANIFEST_PATH))

    rows = read_jsonl(RESULTS_PATH)
    freeze = read_json(FREEZE_MANIFEST_PATH)
    validation = validate_inputs(rows, freeze)
    index = index_rows(rows)

    condition_rows = condition_summaries(rows, index)
    contrast_rows = contrast_summaries(index)
    diagnostic_rows = diagnostic_summaries(rows)
    failure_rows = failure_summaries(rows)
    cost_rows = cost_summaries(rows)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    write_csv_atomic(
        CONDITION_SUMMARY_PATH,
        condition_rows,
        (
            "model", "condition", "label", "unit_load", "interface",
            "n_cases", "n_targets", "success_count", "success_rate",
            "ci_low", "ci_high",
        ),
    )
    write_csv_atomic(
        CONTRAST_SUMMARY_PATH,
        contrast_rows,
        (
            "model", "role", "contrast", "definition", "estimate", "ci_low", "ci_high",
            "left_only_targets", "right_only_targets", "tie_targets",
            "positive_case_interactions", "negative_case_interactions", "zero_case_interactions",
        ),
    )
    write_csv_atomic(
        DIAGNOSTIC_SUMMARY_PATH,
        diagnostic_rows,
        (
            "model", "condition", "metric", "success_count", "failure_count",
            "unscorable_count", "total_count", "rate_over_all",
        ),
    )
    write_csv_atomic(
        FAILURE_SUMMARY_PATH,
        failure_rows,
        ("model", "condition", "failure_reason", "row_count", "note"),
    )
    write_csv_atomic(
        COST_SUMMARY_PATH,
        cost_rows,
        (
            "model", "condition", "unit_load", "interface", "mean_input_tokens",
            "mean_output_tokens", "mean_total_tokens", "mean_latency_seconds",
            "mean_warm_latency_seconds", "n_calls", "warm_latency_n",
        ),
    )

    analysis = {
        "status": "UNIT_LOAD_ANALYSIS_COMPLETE",
        "created_at_utc": utc_now_iso(),
        "validation": validation,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "condition_summary": condition_rows,
        "contrasts": contrast_rows,
        "diagnostics": diagnostic_rows,
        "failure_reasons": failure_rows,
        "costs": cost_rows,
        "guardrails": [
            "Primary G_Patch is Patch L1 minus Patch L3.",
            "Key secondary S is (Select L1-L3) minus (Patch L1-L3).",
            "S does not identify Patch as a better substitute without the six absolute cell results.",
            "Similar observed success rates are not equivalence or noninferiority evidence.",
            "Failure labels overlap and must not be summed as independent failed calls.",
            "Models are analyzed separately with no primary pooled estimate.",
        ],
    }
    write_json_atomic(ANALYSIS_JSON_PATH, analysis)

    report = render_report(condition_rows, contrast_rows, cost_rows)
    write_text_atomic(REPORT_PATH, report)

    manifest = {
        "status": "UNIT_LOAD_ANALYSIS_COMPLETE",
        "created_at_utc": utc_now_iso(),
        "zero_model_calls": True,
        "results_sha256": sha256_file(RESULTS_PATH),
        "freeze_manifest_sha256": sha256_file(FREEZE_MANIFEST_PATH),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "outputs": {
            "condition_summary": str(CONDITION_SUMMARY_PATH),
            "contrast_summary": str(CONTRAST_SUMMARY_PATH),
            "diagnostic_summary": str(DIAGNOSTIC_SUMMARY_PATH),
            "failure_reason_summary": str(FAILURE_SUMMARY_PATH),
            "cost_summary": str(COST_SUMMARY_PATH),
            "analysis_json": str(ANALYSIS_JSON_PATH),
            "report": str(REPORT_PATH),
        },
    }
    write_json_atomic(MANIFEST_PATH, manifest)

    print("PASS | 720 rows | 20 case clusters | frozen estimands")
    print("Primary: G_Patch = Patch L1 - Patch L3")
    print("Key secondary: S = (Select L1-L3) - (Patch L1-L3)")
    print("Report: {}".format(REPORT_PATH))
    print("Analysis JSON: {}".format(ANALYSIS_JSON_PATH))


if __name__ == "__main__":
    from atomic_memory.reanalysis import run_analysis
    run_analysis(main, globals(), paths.UNIT_LOAD, Path(__file__).with_name("run_unit_load.py"))
