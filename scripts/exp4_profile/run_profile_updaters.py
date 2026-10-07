#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run or resume the qualified profile-study updater plan and score execution."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "atomic_memory").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atomic_memory import paths

from atomic_memory.backend import (  # noqa: E402
    GenerationConfig,
    OllamaBackend,
)
from atomic_memory.core import (  # noqa: E402
    ACTION_EXACT_PATCH,
    ACTION_REWRITE_UNIT,
    execute_and_evaluate,
)

PROTOCOL_VERSION = "profile-extraction-layer2-formal-v1"

OUTPUT_DIR = paths.PROFILE

CASES_PATH = OUTPUT_DIR / "formal_cases.jsonl"
TASKS_PATH = OUTPUT_DIR / "extraction_tasks.jsonl"
UPDATER_PLAN_PATH = OUTPUT_DIR / "updater_execution_plan.jsonl"
READER_CONTROL_PLAN_PATH = OUTPUT_DIR / "reader_control_plan.jsonl"
PREFLIGHT_PATH = OUTPUT_DIR / "formal_preflight.json"
FREEZE_PATH = OUTPUT_DIR / "freeze_manifest.json"

READER_CONTROL_RESULTS_PATH = (
    OUTPUT_DIR / "reader_control_results.jsonl"
)
READER_CONTROL_SUMMARY_PATH = (
    OUTPUT_DIR / "reader_control_run_summary.json"
)

UPDATER_RESULTS_PATH = OUTPUT_DIR / "updater_results.jsonl"
UPDATER_SUMMARY_PATH = OUTPUT_DIR / "updater_run_summary.json"

# These must not exist before the updater stage is complete.
FUTURE_STAGE_PATHS = (
    OUTPUT_DIR / "actual_reader_results.jsonl",
    OUTPUT_DIR / "actual_reader_run_summary.json",
    OUTPUT_DIR / "formal_analysis.json",
)

CORE_PATH = (
    PROJECT_ROOT
    / "atomic_memory"
    / "core.py"
)
BACKEND_PATH = PROJECT_ROOT / "atomic_memory" / "backend.py"
BUILDER_PATH = PROJECT_ROOT / "scripts/exp4_profile/build_profile.py"

EXPECTED_CALLS = 1200
EXPECTED_CASES = 50
EXPECTED_TASKS = 100
EXPECTED_MODELS = (
    "llama3.1:8b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "gemma3:12b-it-q4_K_M",
)
EXPECTED_CONDITIONS = (
    "L1_select",
    "L1_patch",
    "L3_select",
    "L3_patch",
)

CONDITION_SPECS = {
    "L1_select": {
        "grouping": "g1",
        "unit_load": 1,
        "interface": "select",
        "expected_action_type": ACTION_REWRITE_UNIT,
    },
    "L1_patch": {
        "grouping": "g1",
        "unit_load": 1,
        "interface": "patch",
        "expected_action_type": ACTION_EXACT_PATCH,
    },
    "L3_select": {
        "grouping": "g3",
        "unit_load": 3,
        "interface": "select",
        "expected_action_type": ACTION_REWRITE_UNIT,
    },
    "L3_patch": {
        "grouping": "g3",
        "unit_load": 3,
        "interface": "patch",
        "expected_action_type": ACTION_EXACT_PATCH,
    },
}

OLLAMA_BASE_URL = paths.OLLAMA_URL
OLLAMA_TIMEOUT = 600.0


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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(
                    "{}:{} must contain a JSON object.".format(
                        path,
                        line_number,
                    )
                )
            rows.append(value)
    return rows


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temp), str(path))


def append_jsonl_fsync(
    path: Path,
    row: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def public_memory(
    memory: Sequence[Mapping[str, Any]],
) -> List[dict]:
    return [
        {
            "memory_id": str(unit["memory_id"]),
            "text": str(unit["text"]),
        }
        for unit in memory
    ]


def snapshot_fingerprint(snapshot: Mapping[str, Any]) -> dict:
    return {
        "provider": snapshot.get("provider"),
        "ollama_version": snapshot.get("ollama_version"),
        "model": snapshot.get("model"),
        "digest": snapshot.get(
            "digest",
            snapshot.get("model_digest"),
        ),
        "model_digest": snapshot.get(
            "model_digest",
            snapshot.get("digest"),
        ),
        "parameter_size": snapshot.get("parameter_size"),
        "quantization_level": snapshot.get("quantization_level"),
    }


def result_key(
    row: Mapping[str, Any],
) -> Tuple[str, str, str, str]:
    return (
        str(row["case_id"]),
        str(row["update_id"]),
        str(row["model"]),
        str(row["condition"]),
    )


def parse_action_response(text: str) -> Dict[str, Any]:
    """Strict direct-JSON parser: no repair and no fence stripping."""
    value = json.loads(text.strip())
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


# ============================================================================
# Diagnostics
# ============================================================================

def hidden_target_unit_ids(
    case: Mapping[str, Any],
    update: Mapping[str, Any],
    grouping: str,
) -> List[str]:
    target_fact_id = str(update["target_fact_id"])
    units = case["layer2_representations"][grouping]
    result = [
        str(unit["memory_id"])
        for unit in units
        if target_fact_id
        in [str(value) for value in unit.get("fact_ids", [])]
    ]
    if len(result) != 1:
        raise RuntimeError(
            "Expected exactly one hidden target unit for {} / {} / {}; "
            "got {}.".format(
                case["case_id"],
                update["update_id"],
                grouping,
                result,
            )
        )
    return result


def locator_success(
    action: Mapping[str, Any],
    target_unit_ids: Sequence[str],
) -> bool:
    selected_id = action.get("memory_id")
    return (
        isinstance(selected_id, str)
        and selected_id in set(str(x) for x in target_unit_ids)
    )


def diagnostic_failure_reasons(
    response_json_valid: bool,
    locator: bool,
    result: Mapping[str, Any],
) -> List[str]:
    """Overlapping diagnostic labels; do not sum them as failed calls."""
    reasons: List[str] = []

    if not response_json_valid:
        reasons.append("response_json_invalid")

    if locator is False:
        reasons.append("locator_failure")

    for reason in result["evaluation"].get(
        "failure_reasons",
        [],
    ):
        reason_text = str(reason)
        if reason_text not in reasons:
            reasons.append(reason_text)

    return reasons


def primary_failure_stage(
    failure_reasons: Sequence[str],
    result: Mapping[str, Any],
) -> str:
    if result["evaluation"]["end_to_end_success"]:
        return "success"
    if failure_reasons:
        return str(failure_reasons[0])
    return "other_failure"


# ============================================================================
# Formal gate / frozen-plan verification
# ============================================================================

def verify_reader_control_gate(
    preflight_path: Path,
    freeze_path: Path,
) -> dict:
    if not READER_CONTROL_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Missing fresh formal reader-control results: {}".format(
                READER_CONTROL_RESULTS_PATH
            )
        )
    if not READER_CONTROL_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Missing fresh formal reader-control summary: {}".format(
                READER_CONTROL_SUMMARY_PATH
            )
        )

    summary = read_json(READER_CONTROL_SUMMARY_PATH)

    if summary.get("status") != "FORMAL_READER_CONTROL_GATE_PASS":
        raise RuntimeError(
            "Fresh formal reader-control gate is not PASS: {!r}".format(
                summary.get("status")
            )
        )

    if not bool(summary.get("updater_calls_unlocked")):
        raise RuntimeError(
            "Reader-control summary does not unlock updater calls."
        )

    if str(summary.get("protocol_version")) != PROTOCOL_VERSION:
        raise RuntimeError(
            "Reader-control protocol version mismatch."
        )

    expected_results_hash = str(summary.get("results_sha256", ""))
    actual_results_hash = sha256_file(READER_CONTROL_RESULTS_PATH)
    if expected_results_hash != actual_results_hash:
        raise RuntimeError(
            "Reader-control results hash mismatch."
        )

    expected_preflight_hash = str(
        summary.get("formal_preflight_sha256", "")
    )
    if expected_preflight_hash != sha256_file(preflight_path):
        raise RuntimeError(
            "Reader-control summary is not bound to current formal preflight."
        )

    expected_freeze_hash = str(
        summary.get("freeze_manifest_sha256", "")
    )
    if expected_freeze_hash != sha256_file(freeze_path):
        raise RuntimeError(
            "Reader-control summary is not bound to current freeze manifest."
        )

    rows = read_jsonl(READER_CONTROL_RESULTS_PATH)
    if len(rows) != 500:
        raise RuntimeError(
            "Expected 500 reader-control rows, found {}.".format(
                len(rows)
            )
        )

    return summary


def verify_frozen_design() -> Tuple[
    dict,
    dict,
    List[dict],
    List[dict],
    Dict[str, dict],
]:
    required = (
        CASES_PATH,
        TASKS_PATH,
        UPDATER_PLAN_PATH,
        READER_CONTROL_PLAN_PATH,
        PREFLIGHT_PATH,
        FREEZE_PATH,
        CORE_PATH,
        BACKEND_PATH,
        BUILDER_PATH,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                "Missing required formal artifact/code file: {}".format(
                    path
                )
            )

    future_started = [
        path
        for path in FUTURE_STAGE_PATHS
        if path.exists()
    ]
    if future_started:
        raise RuntimeError(
            "A future formal stage already has result artifacts. "
            "Do not run/resume updaters after downstream reading/analysis "
            "has begun:\n{}".format(
                "\n".join(
                    "  {}".format(path)
                    for path in future_started
                )
            )
        )

    preflight = read_json(PREFLIGHT_PATH)
    freeze = read_json(FREEZE_PATH)
    cases = read_jsonl(CASES_PATH)
    tasks = read_jsonl(TASKS_PATH)
    plan = read_jsonl(UPDATER_PLAN_PATH)

    if preflight.get("status") != "PASS":
        raise RuntimeError("Formal preflight is not PASS.")

    if not bool(preflight.get("fresh_prospective_formal_batch")):
        raise RuntimeError(
            "Preflight is not marked as a fresh prospective batch."
        )

    if freeze.get("status") != "FORMAL_PROTOCOL_FROZEN":
        raise RuntimeError("Formal protocol is not frozen.")

    if freeze.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(
            "Unexpected formal protocol version: {!r}".format(
                freeze.get("protocol_version")
            )
        )

    if not bool(freeze.get("fresh_prospective_formal_batch")):
        raise RuntimeError(
            "Freeze manifest is not marked as fresh prospective."
        )

    if len(cases) != EXPECTED_CASES:
        raise RuntimeError(
            "Expected {} formal cases, found {}.".format(
                EXPECTED_CASES,
                len(cases),
            )
        )

    if len(tasks) != EXPECTED_TASKS:
        raise RuntimeError(
            "Expected {} formal tasks, found {}.".format(
                EXPECTED_TASKS,
                len(tasks),
            )
        )

    if len(plan) != EXPECTED_CALLS:
        raise RuntimeError(
            "Expected {} updater rows, found {}.".format(
                EXPECTED_CALLS,
                len(plan),
            )
        )

    if tuple(freeze.get("updater_models", [])) != EXPECTED_MODELS:
        raise RuntimeError(
            "Frozen updater model list differs from expected models."
        )

    if tuple(freeze.get("conditions", [])) != EXPECTED_CONDITIONS:
        raise RuntimeError(
            "Frozen condition list differs from expected conditions."
        )

    if int(freeze.get("call_budget", {}).get("updater_calls", -1)) != (
        EXPECTED_CALLS
    ):
        raise RuntimeError("Frozen updater call budget changed.")

    frozen_hashes = freeze.get("frozen_file_sha256", {})
    for filename, path in (
        ("formal_cases.jsonl", CASES_PATH),
        ("extraction_tasks.jsonl", TASKS_PATH),
        ("updater_execution_plan.jsonl", UPDATER_PLAN_PATH),
        ("reader_control_plan.jsonl", READER_CONTROL_PLAN_PATH),
    ):
        expected = str(frozen_hashes.get(filename, ""))
        actual = sha256_file(path)
        if expected != actual:
            raise RuntimeError(
                "Frozen hash mismatch for {}: expected {}, got {}."
                .format(filename, expected, actual)
            )

    code_hashes = freeze.get("code_sha256", {})
    expected_core_hash = str(
        code_hashes.get("representation_edit_scope_core.py", "")
    )
    if expected_core_hash != sha256_file(CORE_PATH):
        raise RuntimeError(
            "Current core.py differs from the "
            "frozen formal protocol."
        )

    expected_builder_hash = str(
        code_hashes.get("build_profile_extraction_formal.py", "")
    )
    if expected_builder_hash != sha256_file(BUILDER_PATH):
        raise RuntimeError(
            "Current build_profile.py differs from the "
            "frozen formal protocol."
        )

    system_prompt = str(freeze["updater_system_prompt"])
    if sha256_text(system_prompt) != str(
        freeze["updater_system_prompt_sha256"]
    ):
        raise RuntimeError("Frozen updater system-prompt hash mismatch.")

    expected_generation = dict(freeze["updater_generation"])

    case_by_id: Dict[str, dict] = {}
    for case in cases:
        case_id = str(case["case_id"])
        if case_id in case_by_id:
            raise RuntimeError(
                "Duplicate case_id: {}".format(case_id)
            )
        case_by_id[case_id] = case

    task_ids = [str(task["task_id"]) for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("Duplicate formal task_id detected.")

    # The plan itself is the frozen execution order.
    planned_indexes = [
        int(row["planned_call_index"])
        for row in plan
    ]
    if planned_indexes != list(range(EXPECTED_CALLS)):
        raise RuntimeError(
            "Updater plan planned_call_index is not exactly 0..1199."
        )

    identity_keys = [result_key(row) for row in plan]
    if len(identity_keys) != len(set(identity_keys)):
        raise RuntimeError(
            "Updater plan contains duplicate result identities."
        )

    condition_position_counts: Dict[str, Dict[str, Counter]] = {}
    for model in EXPECTED_MODELS:
        condition_position_counts[model] = {}
        for condition in EXPECTED_CONDITIONS:
            condition_position_counts[model][condition] = Counter()

    for row in plan:
        case_id = str(row["case_id"])
        update_id = str(row["update_id"])
        model = str(row["model"])
        condition = str(row["condition"])

        if case_id not in case_by_id:
            raise RuntimeError(
                "Plan references unknown case_id={}".format(case_id)
            )
        case = case_by_id[case_id]

        if str(row["case_sha256"]) != str(case["case_sha256"]):
            raise RuntimeError(
                "case_sha256 mismatch in plan for {}".format(case_id)
            )

        update = next(
            (
                candidate
                for candidate in case["updates"]
                if str(candidate["update_id"]) == update_id
            ),
            None,
        )
        if update is None:
            raise RuntimeError(
                "Plan references unknown update_id={} in {}".format(
                    update_id,
                    case_id,
                )
            )

        if model not in EXPECTED_MODELS:
            raise RuntimeError(
                "Unexpected updater model in plan: {}".format(model)
            )
        if condition not in CONDITION_SPECS:
            raise RuntimeError(
                "Unexpected condition in plan: {}".format(condition)
            )

        spec = CONDITION_SPECS[condition]
        for field in (
            "grouping",
            "unit_load",
            "interface",
            "expected_action_type",
        ):
            if row.get(field) != spec[field]:
                raise RuntimeError(
                    "Condition spec mismatch for {} field {}.".format(
                        condition,
                        field,
                    )
                )

        grouping = str(row["grouping"])
        full_memory = case["layer2_representations"][grouping]
        public = public_memory(full_memory)

        if canonical_json(public) != canonical_json(row["memory"]):
            raise RuntimeError(
                "Frozen plan memory differs from case representation for "
                "{} / {} / {}.".format(
                    case_id,
                    update_id,
                    condition,
                )
            )

        if sha256_text(canonical_json(public)) != str(
            row["memory_sha256"]
        ):
            raise RuntimeError(
                "Frozen memory hash mismatch for plan row {}.".format(
                    row["planned_call_index"]
                )
            )

        if str(row["evidence_text"]) != str(update["evidence_text"]):
            raise RuntimeError(
                "Evidence mismatch for {} / {}.".format(
                    case_id,
                    update_id,
                )
            )

        generation_config = dict(row["generation_config"])
        expected_row_generation = {
            "model": model,
            "temperature": float(
                expected_generation["temperature"]
            ),
            "seed": int(expected_generation["seed"]),
            "max_output_tokens": int(
                expected_generation["max_output_tokens"]
            ),
            "context_window": int(
                expected_generation["context_window"]
            ),
        }
        if generation_config != expected_row_generation:
            raise RuntimeError(
                "Generation config mismatch for plan row {}.".format(
                    row["planned_call_index"]
                )
            )

        if sha256_text(str(row["user_prompt"])) != str(
            row["user_prompt_sha256"]
        ):
            raise RuntimeError(
                "User-prompt hash mismatch for plan row {}.".format(
                    row["planned_call_index"]
                )
            )

        if sha256_text(
            canonical_json(row["response_schema"])
        ) != str(row["response_schema_sha256"]):
            raise RuntimeError(
                "Response-schema hash mismatch for plan row {}.".format(
                    row["planned_call_index"]
                )
            )

        payload = {
            "system_prompt": system_prompt,
            "user_prompt": str(row["user_prompt"]),
            "generation_config": generation_config,
            "response_schema": row["response_schema"],
        }
        if sha256_text(canonical_json(payload)) != str(
            row["updater_input_sha256"]
        ):
            raise RuntimeError(
                "Updater-input hash mismatch for plan row {}.".format(
                    row["planned_call_index"]
                )
            )

        hidden_target_unit_ids(
            case,
            update,
            grouping,
        )

        position = int(row["condition_execution_position"])
        if position not in (1, 2, 3, 4):
            raise RuntimeError(
                "Invalid condition execution position."
            )
        condition_position_counts[model][condition][position] += 1

    for model in EXPECTED_MODELS:
        for condition in EXPECTED_CONDITIONS:
            counts = condition_position_counts[model][condition]
            if dict(counts) != {1: 25, 2: 25, 3: 25, 4: 25}:
                raise RuntimeError(
                    "Execution-position balance mismatch for {} / {}: {}"
                    .format(model, condition, dict(counts))
                )

    reader_gate_summary = verify_reader_control_gate(
        PREFLIGHT_PATH,
        FREEZE_PATH,
    )

    return (
        freeze,
        preflight,
        cases,
        plan,
        case_by_id,
    )


# ============================================================================
# Resume verification / summaries
# ============================================================================

def verify_existing_results(
    rows: Sequence[Mapping[str, Any]],
    plan_by_key: Mapping[
        Tuple[str, str, str, str],
        Mapping[str, Any],
    ],
    model_snapshots: Mapping[str, Mapping[str, Any]],
) -> Dict[Tuple[str, str, str, str], dict]:
    completed: Dict[Tuple[str, str, str, str], dict] = {}

    for row in rows:
        key = result_key(row)
        if key in completed:
            raise RuntimeError(
                "Duplicate saved updater result identity: {}".format(key)
            )
        if key not in plan_by_key:
            raise RuntimeError(
                "Saved updater result not present in frozen plan: {}".format(
                    key
                )
            )

        planned = plan_by_key[key]
        for field in (
            "planned_call_index",
            "task_id",
            "case_id",
            "case_sha256",
            "update_id",
            "model",
            "condition",
            "grouping",
            "unit_load",
            "interface",
            "expected_action_type",
            "condition_execution_position",
            "updater_input_sha256",
        ):
            if row.get(field) != planned.get(field):
                raise RuntimeError(
                    "Saved updater result differs from frozen plan for "
                    "{} field {}.".format(key, field)
                )

        expected_snapshot = snapshot_fingerprint(
            model_snapshots[str(row["model"])]
        )
        if row.get("model_snapshot") != expected_snapshot:
            raise RuntimeError(
                "Current model/runtime snapshot differs from saved formal "
                "result for {}.".format(key)
            )

        completed[key] = dict(row)

    return completed


def count_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict:
    by_model_condition: Dict[str, Dict[str, dict]] = {}
    for model in EXPECTED_MODELS:
        by_model_condition[model] = {}
        for condition in EXPECTED_CONDITIONS:
            subset = [
                row
                for row in rows
                if str(row["model"]) == model
                and str(row["condition"]) == condition
            ]
            by_model_condition[model][condition] = {
                "n": len(subset),
                "strict_success": sum(
                    1
                    for row in subset
                    if bool(row["evaluation"]["end_to_end_success"])
                ),
                "json_valid": sum(
                    1
                    for row in subset
                    if bool(row["response_json_valid"])
                ),
                "locator_success": sum(
                    1
                    for row in subset
                    if bool(row["locator_success"])
                ),
                "execution_success": sum(
                    1
                    for row in subset
                    if bool(
                        row["execution"]["execution_success"]
                    )
                ),
            }

    reason_counts = Counter()
    for row in rows:
        for reason in row.get("failure_reasons", []):
            reason_counts[str(reason)] += 1

    return {
        "completed_calls": len(rows),
        "strict_success_total": sum(
            1
            for row in rows
            if bool(row["evaluation"]["end_to_end_success"])
        ),
        "strict_failure_total": sum(
            1
            for row in rows
            if not bool(row["evaluation"]["end_to_end_success"])
        ),
        "by_model_condition": by_model_condition,
        "overlapping_failure_reason_row_counts": dict(
            sorted(reason_counts.items())
        ),
        "failure_reason_warning": (
            "Failure reasons overlap; do not sum reason counts as independent "
            "failed calls."
        ),
    }


def build_run_summary(
    *,
    status: str,
    rows: Sequence[Mapping[str, Any]],
    freeze: Mapping[str, Any],
    model_snapshots: Mapping[str, Mapping[str, Any]],
    started_at_unix: float,
    error: Optional[str] = None,
) -> dict:
    summary = count_summary(rows)
    summary.update(
        {
            "status": status,
            "protocol_version": PROTOCOL_VERSION,
            "expected_calls": EXPECTED_CALLS,
            "remaining_calls": EXPECTED_CALLS - len(rows),
            "reader_control_gate_status": (
                "FORMAL_READER_CONTROL_GATE_PASS"
            ),
            "model_snapshots": {
                model: snapshot_fingerprint(snapshot)
                for model, snapshot in model_snapshots.items()
            },
            "results_path": str(UPDATER_RESULTS_PATH),
            "results_sha256": (
                sha256_file(UPDATER_RESULTS_PATH)
                if UPDATER_RESULTS_PATH.exists()
                else None
            ),
            "formal_preflight_sha256": sha256_file(PREFLIGHT_PATH),
            "freeze_manifest_sha256": sha256_file(FREEZE_PATH),
            "updater_plan_sha256": sha256_file(UPDATER_PLAN_PATH),
            "core_sha256": sha256_file(CORE_PATH),
            "backend_sha256": sha256_file(BACKEND_PATH),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "elapsed_seconds_this_process": (
                time.perf_counter() - started_at_unix
            ),
            "downstream_reader_started": False,
        }
    )
    if error is not None:
        summary["error"] = error
    return summary


def print_final_summary(summary: Mapping[str, Any]) -> None:
    print()
    print("=" * 78)
    print("PROFILE EXTRACTION FORMAL UPDATER STAGE COMPLETE")
    print("=" * 78)
    print(
        "Completed updater calls: {}/{}".format(
            summary["completed_calls"],
            summary["expected_calls"],
        )
    )
    print(
        "Strict successes:       {}".format(
            summary["strict_success_total"]
        )
    )
    print(
        "Strict failures:        {}".format(
            summary["strict_failure_total"]
        )
    )
    print()
    print("Strict success by model / condition:")
    for model in EXPECTED_MODELS:
        print("  {}".format(model))
        for condition in EXPECTED_CONDITIONS:
            cell = summary["by_model_condition"][model][condition]
            print(
                "    {:10s} {:3d}/{:3d} | locator {:3d}/{:3d} | "
                "exec {:3d}/{:3d}".format(
                    condition,
                    int(cell["strict_success"]),
                    int(cell["n"]),
                    int(cell["locator_success"]),
                    int(cell["n"]),
                    int(cell["execution_success"]),
                    int(cell["n"]),
                )
            )
    print()
    print(
        "Results: {}".format(UPDATER_RESULTS_PATH)
    )
    print(
        "Summary: {}".format(UPDATER_SUMMARY_PATH)
    )
    print()
    print(
        "NEXT: run the frozen downstream Profile Extraction reader on the "
        "1,200 executed memories."
    )
    print(
        "Do not modify cases, updater outputs, reader prompt/model, or gates."
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    paths.require_output_stage(paths.PROFILE, Path(__file__).with_name("build_profile.py"))
    process_start = time.perf_counter()

    print("=" * 78)
    print("PROFILE EXTRACTION LAYER-2 — FORMAL UPDATER RUN")
    print("=" * 78)

    (
        freeze,
        preflight,
        cases,
        plan,
        case_by_id,
    ) = verify_frozen_design()

    system_prompt = str(freeze["updater_system_prompt"])

    print("Frozen protocol:       PASS")
    print("Fresh reader controls: PASS")
    print("Updater calls:         1200")
    print("Models:                3")
    print("Conditions:            g1/g3 x Select/Patch")
    print("Execution order:       frozen updater_execution_plan.jsonl")
    print("Strict evaluator:      frozen pre-existing Layer-1 evaluator")
    print("Downstream reader:     NOT run in this stage")
    print()

    backend = OllamaBackend(
        base_url=OLLAMA_BASE_URL,
        timeout_seconds=OLLAMA_TIMEOUT,
    )

    model_snapshots: Dict[str, dict] = {}
    for model in EXPECTED_MODELS:
        snapshot = backend.describe(model)
        fingerprint = snapshot_fingerprint(snapshot)

        if fingerprint.get("quantization_level") != "Q4_K_M":
            raise RuntimeError(
                "Unexpected quantization for {}: {!r}".format(
                    model,
                    fingerprint.get("quantization_level"),
                )
            )

        model_snapshots[model] = dict(snapshot)

        print(
            "Model snapshot: {} | {} | {} | {}".format(
                model,
                fingerprint.get("parameter_size"),
                fingerprint.get("quantization_level"),
                fingerprint.get(
                    "model_digest",
                    fingerprint.get("digest"),
                ),
            )
        )

    print()

    plan_by_key = {
        result_key(row): row
        for row in plan
    }

    existing_rows = read_jsonl(UPDATER_RESULTS_PATH)
    completed = verify_existing_results(
        existing_rows,
        plan_by_key,
        model_snapshots,
    )

    print(
        "Completed:             {}/{}".format(
            len(completed),
            EXPECTED_CALLS,
        )
    )
    print(
        "Remaining:             {}".format(
            EXPECTED_CALLS - len(completed)
        )
    )
    print()

    if len(completed) == EXPECTED_CALLS:
        summary = build_run_summary(
            status="FORMAL_UPDATER_STAGE_COMPLETE",
            rows=existing_rows,
            freeze=freeze,
            model_snapshots=model_snapshots,
            started_at_unix=process_start,
        )
        write_json_atomic(UPDATER_SUMMARY_PATH, summary)
        print_final_summary(summary)
        return

    running_summary = build_run_summary(
        status="FORMAL_UPDATER_STAGE_RUNNING",
        rows=existing_rows,
        freeze=freeze,
        model_snapshots=model_snapshots,
        started_at_unix=process_start,
    )
    write_json_atomic(UPDATER_SUMMARY_PATH, running_summary)

    rows = list(existing_rows)

    try:
        for plan_row in plan:
            key = result_key(plan_row)
            if key in completed:
                continue

            case = case_by_id[str(plan_row["case_id"])]
            update = next(
                candidate
                for candidate in case["updates"]
                if str(candidate["update_id"])
                == str(plan_row["update_id"])
            )

            model = str(plan_row["model"])
            condition = str(plan_row["condition"])
            grouping = str(plan_row["grouping"])
            expected_action_type = str(
                plan_row["expected_action_type"]
            )

            # IMPORTANT: evaluator uses the latent full memory with fact_ids;
            # the model still sees only the frozen public memory in user_prompt.
            initial_memory = case["layer2_representations"][grouping]

            if canonical_json(
                public_memory(initial_memory)
            ) != canonical_json(plan_row["memory"]):
                raise RuntimeError(
                    "Runtime full/public memory binding drift for {}.".format(
                        key
                    )
                )

            hidden_targets = hidden_target_unit_ids(
                case,
                update,
                grouping,
            )

            generation = dict(plan_row["generation_config"])
            config = GenerationConfig(
                model=model,
                temperature=float(generation["temperature"]),
                seed=int(generation["seed"]),
                max_output_tokens=int(
                    generation["max_output_tokens"]
                ),
                context_window=int(
                    generation["context_window"]
                ),
            )

            user_prompt = str(plan_row["user_prompt"])
            response_schema = dict(plan_row["response_schema"])

            payload = {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "generation_config": generation,
                "response_schema": response_schema,
            }
            runtime_input_hash = sha256_text(
                canonical_json(payload)
            )
            if runtime_input_hash != str(
                plan_row["updater_input_sha256"]
            ):
                raise RuntimeError(
                    "Runtime updater-input hash mismatch for {}.".format(
                        key
                    )
                )

            current_number = len(completed) + 1
            print(
                "[{}/{}] {} | {} | {} | {} | pos={}".format(
                    current_number,
                    EXPECTED_CALLS,
                    model,
                    plan_row["case_id"],
                    plan_row["update_id"],
                    condition,
                    plan_row["condition_execution_position"],
                )
            )

            request_id = (
                "profile_extraction_formal_updater__{:04d}__{}".format(
                    int(plan_row["planned_call_index"]),
                    runtime_input_hash[:12],
                )
            )

            call_start = time.perf_counter()
            response = backend.generate(
                request_id=request_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                config=config,
                response_schema=response_schema,
            )
            latency_seconds = (
                time.perf_counter() - call_start
            )

            response_json_valid = True
            response_parse_error: Optional[str] = None
            try:
                action = parse_action_response(response.text)
            except Exception as exc:
                response_json_valid = False
                response_parse_error = "{}: {}".format(
                    type(exc).__name__,
                    str(exc),
                )
                action = {}

            strict_result = execute_and_evaluate(
                initial_memory=initial_memory,
                action=action,
                expected_action_type=expected_action_type,
                update=update,
            )

            locator = locator_success(
                action,
                hidden_targets,
            )
            reasons = diagnostic_failure_reasons(
                response_json_valid=response_json_valid,
                locator=locator,
                result=strict_result,
            )
            failure_stage = primary_failure_stage(
                reasons,
                strict_result,
            )

            result_row = {
                "protocol_version": PROTOCOL_VERSION,
                "planned_call_index": int(
                    plan_row["planned_call_index"]
                ),
                "task_id": str(plan_row["task_id"]),
                "target_instance_index": int(
                    plan_row["target_instance_index"]
                ),
                "model_index": int(plan_row["model_index"]),
                "condition_execution_position": int(
                    plan_row["condition_execution_position"]
                ),
                "request_id": request_id,
                "case_id": str(plan_row["case_id"]),
                "case_sha256": str(plan_row["case_sha256"]),
                "update_id": str(plan_row["update_id"]),
                "target_fact_id": str(update["target_fact_id"]),
                "model": model,
                "condition": condition,
                "grouping": grouping,
                "unit_load": int(plan_row["unit_load"]),
                "interface": str(plan_row["interface"]),
                "expected_action_type": expected_action_type,
                "generation_config": generation,
                "evidence_text": str(plan_row["evidence_text"]),
                "initial_memory": public_memory(initial_memory),
                "gold_target_unit_ids": list(hidden_targets),
                "updater_system_prompt_sha256": str(
                    plan_row["updater_system_prompt_sha256"]
                ),
                "user_prompt_sha256": str(
                    plan_row["user_prompt_sha256"]
                ),
                "response_schema_sha256": str(
                    plan_row["response_schema_sha256"]
                ),
                "updater_input_sha256": str(
                    plan_row["updater_input_sha256"]
                ),
                "response_schema": response_schema,
                "raw_response": response.text,
                "response_json_valid": bool(response_json_valid),
                "response_parse_error": response_parse_error,
                "parsed_action": (
                    action
                    if response_json_valid
                    else None
                ),
                "locator_success": bool(locator),
                "execution": strict_result["execution"],
                # Convenience public view for the next frozen reader stage.
                "executed_public_memory": public_memory(
                    strict_result["execution"]["final_memory"]
                ),
                "evaluation": strict_result["evaluation"],
                "strict_state_success": bool(
                    strict_result["evaluation"][
                        "end_to_end_success"
                    ]
                ),
                "failure_stage": failure_stage,
                "failure_reasons": reasons,
                "usage": dict(response.usage),
                "latency_seconds": latency_seconds,
                "response_id": response.response_id,
                "response_metadata": dict(response.metadata),
                "model_snapshot": snapshot_fingerprint(
                    model_snapshots[model]
                ),
            }

            append_jsonl_fsync(
                UPDATER_RESULTS_PATH,
                result_row,
            )
            rows.append(result_row)
            completed[key] = result_row

            print(
                "    json={} locator={} legal={} exec={} contract={} "
                "structure={} strict={} reasons={} in_tok={} out_tok={} "
                "latency={:.2f}s".format(
                    response_json_valid,
                    locator,
                    strict_result["execution"]["legal_action"],
                    strict_result["execution"]["execution_success"],
                    strict_result["evaluation"]["contract_valid"],
                    strict_result["evaluation"]["structure_valid"],
                    strict_result["evaluation"]["end_to_end_success"],
                    reasons,
                    response.usage.get("input_tokens", 0),
                    response.usage.get("output_tokens", 0),
                    latency_seconds,
                )
            )

            # Keep a crash-resilient progress summary.
            progress_summary = build_run_summary(
                status="FORMAL_UPDATER_STAGE_RUNNING",
                rows=rows,
                freeze=freeze,
                model_snapshots=model_snapshots,
                started_at_unix=process_start,
            )
            write_json_atomic(
                UPDATER_SUMMARY_PATH,
                progress_summary,
            )

    except Exception as exc:
        failed_summary = build_run_summary(
            status="FORMAL_UPDATER_STAGE_FAILED",
            rows=rows,
            freeze=freeze,
            model_snapshots=model_snapshots,
            started_at_unix=process_start,
            error="{}: {}".format(
                type(exc).__name__,
                str(exc),
            ),
        )
        write_json_atomic(
            UPDATER_SUMMARY_PATH,
            failed_summary,
        )
        raise

    final_rows = read_jsonl(UPDATER_RESULTS_PATH)
    final_completed = verify_existing_results(
        final_rows,
        plan_by_key,
        model_snapshots,
    )

    if len(final_completed) != EXPECTED_CALLS:
        raise RuntimeError(
            "Updater stage ended incomplete: {}/{}.".format(
                len(final_completed),
                EXPECTED_CALLS,
            )
        )

    # The results file must preserve exact frozen plan order.
    saved_order = [
        int(row["planned_call_index"])
        for row in final_rows
    ]
    if saved_order != list(range(EXPECTED_CALLS)):
        raise RuntimeError(
            "Saved updater results are not in exact frozen planned order."
        )

    final_summary = build_run_summary(
        status="FORMAL_UPDATER_STAGE_COMPLETE",
        rows=final_rows,
        freeze=freeze,
        model_snapshots=model_snapshots,
        started_at_unix=process_start,
    )
    write_json_atomic(
        UPDATER_SUMMARY_PATH,
        final_summary,
    )
    print_final_summary(final_summary)


if __name__ == "__main__":
    main()
