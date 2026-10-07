#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the qualified profile reader on frozen executed memory states.

See the root README for stage dependencies and logical-row caching."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
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
from atomic_memory.reader import (  # noqa: E402
    UNKNOWN_VALUE,
    all_unknown,
    attribute_match_map,
    build_reader_user_prompt,
    canonical_json,
    classify_profile,
    exact_profile_match,
    parse_reader_response,
    public_memory,
    response_schema_for_attributes,
)

PROTOCOL_VERSION = "profile-extraction-layer2-formal-v1"

OUTPUT_DIR = paths.PROFILE

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

ACTUAL_READER_PLAN_PATH = OUTPUT_DIR / "actual_reader_plan.jsonl"
ACTUAL_READER_PLAN_MANIFEST_PATH = (
    OUTPUT_DIR / "actual_reader_plan_manifest.json"
)
ACTUAL_READER_RESULTS_PATH = (
    OUTPUT_DIR / "actual_reader_results.jsonl"
)
ACTUAL_READER_SUMMARY_PATH = (
    OUTPUT_DIR / "actual_reader_run_summary.json"
)

FINAL_ANALYSIS_PATH = OUTPUT_DIR / "formal_analysis.json"

EXPECTED_ROWS = 1200
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

OLLAMA_BASE_URL = paths.OLLAMA_URL
OLLAMA_TIMEOUT = 600.0


# ============================================================================
# Generic helpers
# ============================================================================

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


def write_jsonl_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
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


# ============================================================================
# Upstream / frozen-protocol verification
# ============================================================================

def verify_reader_interface_against_frozen_controls(
    *,
    freeze: Mapping[str, Any],
) -> None:
    """Zero-call proof that this runner mirrors the frozen reader interface."""
    controls = read_jsonl(READER_CONTROL_PLAN_PATH)
    if len(controls) != 500:
        raise RuntimeError(
            "Expected 500 frozen reader-control plan rows."
        )

    reader = freeze["reader"]
    generation_config = {
        "model": str(reader["model"]),
        "temperature": float(reader["temperature"]),
        "seed": int(reader["seed"]),
        "max_output_tokens": int(
            reader["max_output_tokens"]
        ),
        "context_window": int(
            reader["context_window"]
        ),
    }
    system_prompt = str(freeze["reader_system_prompt"])

    for row in controls:
        attributes = [
            str(attribute)
            for attribute in row["required_attributes"]
        ]
        labels = {
            str(key): str(value)
            for key, value in row["memory_labels"].items()
        }
        memory = row["memory"]

        rebuilt_prompt = build_reader_user_prompt(
            memory=memory,
            subject=str(row["subject"]),
            attributes=attributes,
            labels=labels,
        )
        if rebuilt_prompt != str(row["user_prompt"]):
            raise RuntimeError(
                "Actual-reader prompt helper differs from frozen control "
                "prompt interface at {}.".format(row["control_id"])
            )

        rebuilt_schema = response_schema_for_attributes(
            attributes
        )
        if canonical_json(rebuilt_schema) != canonical_json(
            row["response_schema"]
        ):
            raise RuntimeError(
                "Actual-reader schema helper differs from frozen control "
                "schema at {}.".format(row["control_id"])
            )

        payload = {
            "system_prompt": system_prompt,
            "user_prompt": rebuilt_prompt,
            "generation_config": generation_config,
            "response_schema": rebuilt_schema,
        }
        rebuilt_hash = sha256_text(canonical_json(payload))
        if rebuilt_hash != str(row["reader_input_sha256"]):
            raise RuntimeError(
                "Actual-reader input hashing/interface differs from frozen "
                "control row {}.".format(row["control_id"])
            )


def verify_upstream() -> Tuple[
    dict,
    dict,
    List[dict],
    List[dict],
    List[dict],
    Dict[str, dict],
]:
    required = (
        TASKS_PATH,
        UPDATER_PLAN_PATH,
        READER_CONTROL_PLAN_PATH,
        PREFLIGHT_PATH,
        FREEZE_PATH,
        READER_CONTROL_RESULTS_PATH,
        READER_CONTROL_SUMMARY_PATH,
        UPDATER_RESULTS_PATH,
        UPDATER_SUMMARY_PATH,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                "Missing required upstream artifact: {}".format(path)
            )

    if FINAL_ANALYSIS_PATH.exists():
        raise RuntimeError(
            "formal_analysis.json already exists. Do not alter/re-run the "
            "reader stage after final analysis has begun."
        )

    preflight = read_json(PREFLIGHT_PATH)
    freeze = read_json(FREEZE_PATH)
    control_summary = read_json(READER_CONTROL_SUMMARY_PATH)
    updater_summary = read_json(UPDATER_SUMMARY_PATH)

    tasks = read_jsonl(TASKS_PATH)
    updater_plan = read_jsonl(UPDATER_PLAN_PATH)
    updater_rows = read_jsonl(UPDATER_RESULTS_PATH)

    if preflight.get("status") != "PASS":
        raise RuntimeError("Formal preflight is not PASS.")

    if freeze.get("status") != "FORMAL_PROTOCOL_FROZEN":
        raise RuntimeError("Formal protocol is not frozen.")

    if freeze.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(
            "Unexpected frozen protocol version."
        )

    if control_summary.get("status") != (
        "FORMAL_READER_CONTROL_GATE_PASS"
    ):
        raise RuntimeError(
            "Fresh formal reader-control gate is not PASS."
        )

    if updater_summary.get("status") != (
        "FORMAL_UPDATER_STAGE_COMPLETE"
    ):
        raise RuntimeError(
            "Formal updater stage is not complete: {!r}".format(
                updater_summary.get("status")
            )
        )

    if int(updater_summary.get("completed_calls", -1)) != EXPECTED_ROWS:
        raise RuntimeError(
            "Updater summary does not contain 1,200 completed calls."
        )

    if len(tasks) != EXPECTED_TASKS:
        raise RuntimeError(
            "Expected 100 extraction tasks, found {}.".format(
                len(tasks)
            )
        )
    if len(updater_plan) != EXPECTED_ROWS:
        raise RuntimeError(
            "Expected 1,200 updater plan rows, found {}.".format(
                len(updater_plan)
            )
        )
    if len(updater_rows) != EXPECTED_ROWS:
        raise RuntimeError(
            "Expected 1,200 updater result rows, found {}.".format(
                len(updater_rows)
            )
        )

    if str(updater_summary.get("results_sha256", "")) != sha256_file(
        UPDATER_RESULTS_PATH
    ):
        raise RuntimeError(
            "Updater results hash differs from completed updater summary."
        )

    if str(updater_summary.get("freeze_manifest_sha256", "")) != (
        sha256_file(FREEZE_PATH)
    ):
        raise RuntimeError(
            "Updater summary is not bound to current freeze manifest."
        )

    frozen_hashes = freeze.get("frozen_file_sha256", {})
    for filename, path in (
        ("extraction_tasks.jsonl", TASKS_PATH),
        ("updater_execution_plan.jsonl", UPDATER_PLAN_PATH),
        ("reader_control_plan.jsonl", READER_CONTROL_PLAN_PATH),
    ):
        expected = str(frozen_hashes.get(filename, ""))
        actual = sha256_file(path)
        if expected != actual:
            raise RuntimeError(
                "Frozen hash mismatch for {}.".format(filename)
            )

    if tuple(freeze.get("updater_models", [])) != EXPECTED_MODELS:
        raise RuntimeError("Frozen updater models changed.")
    if tuple(freeze.get("conditions", [])) != EXPECTED_CONDITIONS:
        raise RuntimeError("Frozen conditions changed.")

    reader = freeze["reader"]
    if str(reader["model"]) != "llama3.1:8b-instruct-q4_K_M":
        raise RuntimeError(
            "Frozen primary reader model changed unexpectedly."
        )

    if sha256_text(str(freeze["reader_system_prompt"])) != str(
        freeze["reader_system_prompt_sha256"]
    ):
        raise RuntimeError(
            "Frozen reader system-prompt hash mismatch."
        )

    # Prove the runner's helper exactly reproduces the previously passed
    # frozen control interface before deriving any actual-reader prompts.
    verify_reader_interface_against_frozen_controls(
        freeze=freeze
    )

    task_by_id: Dict[str, dict] = {}
    for task in tasks:
        task_id = str(task["task_id"])
        if task_id in task_by_id:
            raise RuntimeError(
                "Duplicate task_id: {}".format(task_id)
            )
        task_by_id[task_id] = task

    # Updater rows must still align exactly with the frozen plan.
    for expected_index, (plan_row, result_row) in enumerate(
        zip(updater_plan, updater_rows)
    ):
        if int(plan_row["planned_call_index"]) != expected_index:
            raise RuntimeError(
                "Updater plan order is not 0..1199."
            )
        if int(result_row["planned_call_index"]) != expected_index:
            raise RuntimeError(
                "Updater results no longer preserve frozen plan order."
            )

        for field in (
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
            if result_row.get(field) != plan_row.get(field):
                raise RuntimeError(
                    "Updater result/plan mismatch at row {} field {}."
                    .format(expected_index, field)
                )

        if str(result_row["task_id"]) not in task_by_id:
            raise RuntimeError(
                "Updater row references unknown task_id."
            )

        if bool(result_row["strict_state_success"]) != bool(
            result_row["evaluation"]["end_to_end_success"]
        ):
            raise RuntimeError(
                "strict_state_success/evaluator mismatch at row {}."
                .format(expected_index)
            )

        execution = result_row.get("execution")
        if not isinstance(execution, Mapping):
            raise RuntimeError(
                "Missing execution object at updater row {}.".format(
                    expected_index
                )
            )
        final_memory = execution.get("final_memory")
        if not isinstance(final_memory, list):
            raise RuntimeError(
                "Missing final_memory at updater row {}.".format(
                    expected_index
                )
            )

        if canonical_json(
            public_memory(final_memory)
        ) != canonical_json(
            result_row["executed_public_memory"]
        ):
            raise RuntimeError(
                "executed_public_memory differs from execution.final_memory "
                "at updater row {}.".format(expected_index)
            )

    return (
        freeze,
        preflight,
        tasks,
        updater_plan,
        updater_rows,
        task_by_id,
    )


# ============================================================================
# Deterministically derive + freeze the actual-reader plan
# ============================================================================

def reader_generation_config(
    freeze: Mapping[str, Any],
) -> dict:
    reader = freeze["reader"]
    return {
        "model": str(reader["model"]),
        "temperature": float(reader["temperature"]),
        "seed": int(reader["seed"]),
        "max_output_tokens": int(
            reader["max_output_tokens"]
        ),
        "context_window": int(
            reader["context_window"]
        ),
    }


def build_actual_reader_plan(
    *,
    freeze: Mapping[str, Any],
    updater_rows: Sequence[Mapping[str, Any]],
    task_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[dict], dict]:
    # Retain a logical row for every updater state, including repeated reader inputs.
    system_prompt = str(freeze["reader_system_prompt"])
    generation_config = reader_generation_config(freeze)

    plan: List[dict] = []

    for updater_row in updater_rows:
        task = task_by_id[str(updater_row["task_id"])]

        attributes = [
            str(attribute)
            for attribute in task["required_attributes"]
        ]
        labels = {
            str(key): str(value)
            for key, value in task["memory_labels"].items()
        }
        memory = public_memory(
            updater_row["executed_public_memory"]
        )
        schema = response_schema_for_attributes(attributes)
        user_prompt = build_reader_user_prompt(
            memory=memory,
            subject=str(task["subject"]),
            attributes=attributes,
            labels=labels,
        )

        input_payload = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "generation_config": generation_config,
            "response_schema": schema,
        }

        oracle_values = {
            str(key): str(value)
            for key, value in task[
                "oracle_expected_values"
            ].items()
        }
        stale_values = {
            str(key): str(value)
            for key, value in task[
                "stale_expected_values"
            ].items()
        }

        plan.append(
            {
                "logical_reader_index": int(
                    updater_row["planned_call_index"]
                ),
                "source_updater_planned_call_index": int(
                    updater_row["planned_call_index"]
                ),
                "source_updater_request_id": str(
                    updater_row["request_id"]
                ),
                "task_id": str(updater_row["task_id"]),
                "case_id": str(updater_row["case_id"]),
                "case_sha256": str(
                    updater_row["case_sha256"]
                ),
                "update_id": str(updater_row["update_id"]),
                "updater_model": str(updater_row["model"]),
                "condition": str(updater_row["condition"]),
                "grouping": str(updater_row["grouping"]),
                "unit_load": int(updater_row["unit_load"]),
                "interface": str(updater_row["interface"]),
                "subject": str(task["subject"]),
                "required_attributes": list(attributes),
                "memory_labels": dict(labels),
                "reader_memory": memory,
                "reader_memory_sha256": sha256_text(
                    canonical_json(memory)
                ),
                "oracle_expected_values": oracle_values,
                "stale_expected_values": stale_values,
                "strict_state_success": bool(
                    updater_row["strict_state_success"]
                ),
                "strict_failure_stage": str(
                    updater_row["failure_stage"]
                ),
                "strict_failure_reasons": [
                    str(value)
                    for value in updater_row[
                        "failure_reasons"
                    ]
                ],
                "updater_execution_success": bool(
                    updater_row["execution"][
                        "execution_success"
                    ]
                ),
                "updater_contract_valid": bool(
                    updater_row["evaluation"][
                        "contract_valid"
                    ]
                ),
                "updater_structure_valid": bool(
                    updater_row["evaluation"][
                        "structure_valid"
                    ]
                ),
                "updater_target_new_current": bool(
                    updater_row["evaluation"][
                        "target_new_current"
                    ]
                ),
                "updater_protected_facts_preserved": bool(
                    updater_row["evaluation"][
                        "protected_facts_preserved"
                    ]
                ),
                "user_prompt": user_prompt,
                "response_schema": schema,
                "generation_config": dict(
                    generation_config
                ),
                "reader_system_prompt_sha256": sha256_text(
                    system_prompt
                ),
                "user_prompt_sha256": sha256_text(
                    user_prompt
                ),
                "response_schema_sha256": sha256_text(
                    canonical_json(schema)
                ),
                "reader_input_sha256": sha256_text(
                    canonical_json(input_payload)
                ),
            }
        )

    if len(plan) != EXPECTED_ROWS:
        raise RuntimeError(
            "Expected 1,200 actual-reader plan rows."
        )

    logical_indexes = [
        int(row["logical_reader_index"])
        for row in plan
    ]
    if logical_indexes != list(range(EXPECTED_ROWS)):
        raise RuntimeError(
            "Actual-reader plan order is not exactly 0..1199."
        )

    # Exact-input caching is a response-level optimization only.
    #
    # IMPORTANT: the same reader input may legitimately appear in multiple
    # logical updater trials with DIFFERENT oracle targets. Example: two
    # different intended updates for the same subject can both fail/no-op and
    # therefore leave the exact same executed memory. The downstream reader
    # should return the same deterministic extraction for the identical input,
    # while each logical trial is scored separately against its own frozen
    # oracle target.
    #
    # Therefore identical input requires identical prompt/schema, but NOT an
    # identical scoring target.
    by_input: Dict[str, List[dict]] = {}
    for row in plan:
        by_input.setdefault(
            str(row["reader_input_sha256"]),
            [],
        ).append(row)

    multi_oracle_input_count = 0
    multi_oracle_logical_rows = 0

    for input_hash, rows in by_input.items():
        first = rows[0]
        for row in rows[1:]:
            if row["user_prompt"] != first["user_prompt"]:
                raise RuntimeError(
                    "Same reader-input hash maps to different prompts."
                )
            if canonical_json(
                row["response_schema"]
            ) != canonical_json(
                first["response_schema"]
            ):
                raise RuntimeError(
                    "Same reader-input hash maps to different schemas."
                )

        oracle_targets = {
            canonical_json(row["oracle_expected_values"])
            for row in rows
        }
        if len(oracle_targets) > 1:
            multi_oracle_input_count += 1
            multi_oracle_logical_rows += len(rows)

    report = {
        "logical_rows": len(plan),
        "unique_reader_inputs": len(by_input),
        "cache_reusable_rows": (
            len(plan) - len(by_input)
        ),
        "identical_input_groups_with_multiple_oracle_targets": (
            multi_oracle_input_count
        ),
        "logical_rows_in_multiple_oracle_input_groups": (
            multi_oracle_logical_rows
        ),
        "strict_success_rows": sum(
            1
            for row in plan
            if bool(row["strict_state_success"])
        ),
        "strict_failure_rows": sum(
            1
            for row in plan
            if not bool(row["strict_state_success"])
        ),
        "candidate_profiles_used": False,
        "manual_annotation_required": False,
    }
    return plan, report


def build_plan_manifest(
    *,
    freeze: Mapping[str, Any],
    plan_path: Path,
    plan_report: Mapping[str, Any],
) -> dict:
    return {
        "status": "ACTUAL_READER_PLAN_FROZEN",
        "protocol_version": PROTOCOL_VERSION,
        "model_calls_before_plan_freeze": 0,
        "derived_only_from_frozen_updater_outputs": True,
        "reader_interface": (
            "direct_three_field_profile_extraction"
        ),
        "reader": dict(freeze["reader"]),
        "reader_system_prompt_sha256": str(
            freeze["reader_system_prompt_sha256"]
        ),
        "logical_rows": int(plan_report["logical_rows"]),
        "unique_reader_inputs": int(
            plan_report["unique_reader_inputs"]
        ),
        "cache_reusable_rows": int(
            plan_report["cache_reusable_rows"]
        ),
        "strict_success_rows": int(
            plan_report["strict_success_rows"]
        ),
        "strict_failure_rows": int(
            plan_report["strict_failure_rows"]
        ),
        "identical_input_groups_with_multiple_oracle_targets": int(
            plan_report[
                "identical_input_groups_with_multiple_oracle_targets"
            ]
        ),
        "logical_rows_in_multiple_oracle_input_groups": int(
            plan_report[
                "logical_rows_in_multiple_oracle_input_groups"
            ]
        ),
        "cache_semantics": (
            "Cache only the deterministic reader response for an exact "
            "reader input. Score every logical row separately against that "
            "row's own frozen oracle target."
        ),
        "source_updater_results_sha256": sha256_file(
            UPDATER_RESULTS_PATH
        ),
        "source_updater_summary_sha256": sha256_file(
            UPDATER_SUMMARY_PATH
        ),
        "source_tasks_sha256": sha256_file(TASKS_PATH),
        "source_freeze_manifest_sha256": sha256_file(
            FREEZE_PATH
        ),
        "source_reader_control_summary_sha256": sha256_file(
            READER_CONTROL_SUMMARY_PATH
        ),
        "actual_reader_plan_sha256": sha256_file(plan_path),
        "anti_rescue_rules": [
            "Do not modify reader model, prompt, schema, or expected values.",
            "Do not drop strict failures or difficult memory states.",
            "Do not repair executed memory before reader inference.",
            "Do not use updater evidence or gold update metadata in reader prompts.",
            "Do not manually relabel reader outputs.",
        ],
    }


def create_or_verify_actual_reader_plan(
    *,
    freeze: Mapping[str, Any],
    updater_rows: Sequence[Mapping[str, Any]],
    task_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[dict], dict]:
    # Freeze the derived plan before generating or reusing reader responses.
    expected_plan, report = build_actual_reader_plan(
        freeze=freeze,
        updater_rows=updater_rows,
        task_by_id=task_by_id,
    )

    results_started = ACTUAL_READER_RESULTS_PATH.exists()

    if ACTUAL_READER_PLAN_PATH.exists():
        existing = read_jsonl(ACTUAL_READER_PLAN_PATH)
        if canonical_json(existing) != canonical_json(
            expected_plan
        ):
            raise RuntimeError(
                "Existing actual_reader_plan.jsonl differs from the "
                "deterministic frozen derivation."
            )
        plan_status = "VERIFIED"
    else:
        if results_started:
            raise RuntimeError(
                "Reader results exist but actual_reader_plan.jsonl is missing."
            )
        write_jsonl_atomic(
            ACTUAL_READER_PLAN_PATH,
            expected_plan,
        )
        plan_status = "CREATED"

    expected_manifest = build_plan_manifest(
        freeze=freeze,
        plan_path=ACTUAL_READER_PLAN_PATH,
        plan_report=report,
    )

    if ACTUAL_READER_PLAN_MANIFEST_PATH.exists():
        existing_manifest = read_json(
            ACTUAL_READER_PLAN_MANIFEST_PATH
        )
        if canonical_json(existing_manifest) != canonical_json(
            expected_manifest
        ):
            raise RuntimeError(
                "Existing actual-reader plan manifest differs from expected."
            )
        manifest_status = "VERIFIED"
    else:
        if results_started:
            raise RuntimeError(
                "Reader results exist but actual-reader manifest is missing."
            )
        write_json_atomic(
            ACTUAL_READER_PLAN_MANIFEST_PATH,
            expected_manifest,
        )
        manifest_status = "CREATED"

    manifest = read_json(ACTUAL_READER_PLAN_MANIFEST_PATH)
    if manifest.get("status") != "ACTUAL_READER_PLAN_FROZEN":
        raise RuntimeError(
            "Actual-reader plan manifest is not frozen."
        )
    if str(
        manifest["actual_reader_plan_sha256"]
    ) != sha256_file(ACTUAL_READER_PLAN_PATH):
        raise RuntimeError(
            "Actual-reader plan hash mismatch."
        )
    if str(
        manifest["source_updater_results_sha256"]
    ) != sha256_file(UPDATER_RESULTS_PATH):
        raise RuntimeError(
            "Updater outputs changed after actual-reader plan freeze."
        )

    print(
        "Actual-reader plan:     {} | manifest {}".format(
            plan_status,
            manifest_status,
        )
    )

    return read_jsonl(ACTUAL_READER_PLAN_PATH), manifest


# ============================================================================
# Resume verification
# ============================================================================

def verify_existing_results(
    *,
    existing_rows: Sequence[Mapping[str, Any]],
    plan_by_index: Mapping[int, Mapping[str, Any]],
    reader_model: str,
    generation_config: Mapping[str, Any],
    expected_snapshot: Mapping[str, Any],
) -> Dict[int, dict]:
    completed: Dict[int, dict] = {}

    for row in existing_rows:
        index = int(row["logical_reader_index"])
        if index in completed:
            raise RuntimeError(
                "Duplicate actual-reader result index: {}".format(
                    index
                )
            )
        if index not in plan_by_index:
            raise RuntimeError(
                "Unexpected actual-reader result index: {}".format(
                    index
                )
            )

        planned = plan_by_index[index]
        for field in (
            "task_id",
            "case_id",
            "update_id",
            "updater_model",
            "condition",
            "grouping",
            "unit_load",
            "interface",
            "reader_input_sha256",
        ):
            if row.get(field) != planned.get(field):
                raise RuntimeError(
                    "Saved actual-reader result differs from plan for "
                    "index {} field {}.".format(index, field)
                )

        if str(row["reader_model"]) != reader_model:
            raise RuntimeError(
                "Saved reader model mismatch at index {}.".format(
                    index
                )
            )

        if row["generation_config"] != dict(
            generation_config
        ):
            raise RuntimeError(
                "Saved reader generation config mismatch at index {}."
                .format(index)
            )

        if row["reader_model_snapshot"] != dict(
            expected_snapshot
        ):
            raise RuntimeError(
                "Saved reader/runtime snapshot mismatch at index {}."
                .format(index)
            )

        completed[index] = dict(row)

    return completed


# ============================================================================
# Descriptive stage summary (NO bootstrap inference here)
# ============================================================================

def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict:
    by_model_condition: Dict[str, Dict[str, dict]] = {}

    overall_joint = Counter()

    for model in EXPECTED_MODELS:
        by_model_condition[model] = {}

        for condition in EXPECTED_CONDITIONS:
            subset = [
                row
                for row in rows
                if str(row["updater_model"]) == model
                and str(row["condition"]) == condition
            ]

            joint = Counter()
            for row in subset:
                strict_ok = bool(row["strict_state_success"])
                task_ok = bool(row["functional_task_pass"])

                if strict_ok and task_ok:
                    label = "strict_pass_task_pass"
                elif strict_ok and not task_ok:
                    label = "strict_pass_task_fail"
                elif not strict_ok and task_ok:
                    label = "strict_fail_task_pass"
                else:
                    label = "strict_fail_task_fail"

                joint[label] += 1
                overall_joint[label] += 1

            by_model_condition[model][condition] = {
                "n": len(subset),
                "task_success": sum(
                    1
                    for row in subset
                    if bool(row["functional_task_pass"])
                ),
                "strict_success": sum(
                    1
                    for row in subset
                    if bool(row["strict_state_success"])
                ),
                "valid_json": sum(
                    1
                    for row in subset
                    if bool(row["response_json_valid"])
                ),
                "profile_class_counts": dict(
                    sorted(
                        Counter(
                            str(row["profile_class"])
                            for row in subset
                        ).items()
                    )
                ),
                "joint_strict_task": {
                    key: int(joint.get(key, 0))
                    for key in (
                        "strict_pass_task_pass",
                        "strict_pass_task_fail",
                        "strict_fail_task_pass",
                        "strict_fail_task_fail",
                    )
                },
            }

    return {
        "logical_rows": len(rows),
        "task_success_total": sum(
            1
            for row in rows
            if bool(row["functional_task_pass"])
        ),
        "task_failure_total": sum(
            1
            for row in rows
            if not bool(row["functional_task_pass"])
        ),
        "strict_success_total": sum(
            1
            for row in rows
            if bool(row["strict_state_success"])
        ),
        "strict_failure_total": sum(
            1
            for row in rows
            if not bool(row["strict_state_success"])
        ),
        "valid_json_total": sum(
            1
            for row in rows
            if bool(row["response_json_valid"])
        ),
        "unique_reader_inputs": len(
            {
                str(row["reader_input_sha256"])
                for row in rows
            }
        ),
        "actual_reader_calls": sum(
            1
            for row in rows
            if not bool(row["cache_reused"])
        ),
        "cache_reused_rows": sum(
            1
            for row in rows
            if bool(row["cache_reused"])
        ),
        "overall_joint_strict_task": {
            key: int(overall_joint.get(key, 0))
            for key in (
                "strict_pass_task_pass",
                "strict_pass_task_fail",
                "strict_fail_task_pass",
                "strict_fail_task_fail",
            )
        },
        "by_model_condition": by_model_condition,
        "interpretation_note": (
            "These are descriptive observed joint outcomes only. "
            "Strict-fail/task-fail is not a causal claim that strict failure "
            "caused downstream failure."
        ),
    }


def build_run_summary(
    *,
    status: str,
    rows: Sequence[Mapping[str, Any]],
    reader_snapshot: Mapping[str, Any],
    plan_manifest: Mapping[str, Any],
    process_start: float,
    error: Optional[str] = None,
) -> dict:
    summary = summarize_rows(rows)
    summary.update(
        {
            "status": status,
            "protocol_version": PROTOCOL_VERSION,
            "expected_logical_rows": EXPECTED_ROWS,
            "remaining_rows": EXPECTED_ROWS - len(rows),
            "reader_model_snapshot": dict(reader_snapshot),
            "actual_reader_plan_sha256": sha256_file(
                ACTUAL_READER_PLAN_PATH
            ),
            "actual_reader_plan_manifest_sha256": sha256_file(
                ACTUAL_READER_PLAN_MANIFEST_PATH
            ),
            "source_updater_results_sha256": str(
                plan_manifest[
                    "source_updater_results_sha256"
                ]
            ),
            "results_path": str(
                ACTUAL_READER_RESULTS_PATH
            ),
            "results_sha256": (
                sha256_file(ACTUAL_READER_RESULTS_PATH)
                if ACTUAL_READER_RESULTS_PATH.exists()
                else None
            ),
            "elapsed_seconds_this_process": (
                time.perf_counter() - process_start
            ),
            "bootstrap_or_ci_computed": False,
            "final_analysis_started": False,
        }
    )
    if error is not None:
        summary["error"] = error
    return summary


def print_final_summary(summary: Mapping[str, Any]) -> None:
    print()
    print("=" * 78)
    print("FORMAL DOWNSTREAM PROFILE EXTRACTION COMPLETE")
    print("=" * 78)
    print(
        "Logical rows:           {}/1200".format(
            summary["logical_rows"]
        )
    )
    print(
        "Task successes:         {}".format(
            summary["task_success_total"]
        )
    )
    print(
        "Task failures:          {}".format(
            summary["task_failure_total"]
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
    print(
        "Valid JSON:             {}/1200".format(
            summary["valid_json_total"]
        )
    )
    print(
        "Unique reader inputs:   {}".format(
            summary["unique_reader_inputs"]
        )
    )
    print(
        "Actual reader calls:    {}".format(
            summary["actual_reader_calls"]
        )
    )
    print(
        "Cache reused rows:      {}".format(
            summary["cache_reused_rows"]
        )
    )
    print()
    print("Overall Strict x Task:")
    joint = summary["overall_joint_strict_task"]
    print(
        "  Strict pass / Task pass: {}".format(
            joint["strict_pass_task_pass"]
        )
    )
    print(
        "  Strict pass / Task fail: {}".format(
            joint["strict_pass_task_fail"]
        )
    )
    print(
        "  Strict fail / Task pass: {}".format(
            joint["strict_fail_task_pass"]
        )
    )
    print(
        "  Strict fail / Task fail: {}".format(
            joint["strict_fail_task_fail"]
        )
    )
    print()
    print("Task success by updater model / condition:")
    for model in EXPECTED_MODELS:
        print("  {}".format(model))
        for condition in EXPECTED_CONDITIONS:
            cell = summary["by_model_condition"][model][condition]
            print(
                "    {:10s} task {:3d}/{:3d} | strict {:3d}/{:3d}"
                .format(
                    condition,
                    int(cell["task_success"]),
                    int(cell["n"]),
                    int(cell["strict_success"]),
                    int(cell["n"]),
                )
            )
    print()
    print(
        "Results: {}".format(ACTUAL_READER_RESULTS_PATH)
    )
    print(
        "Summary: {}".format(ACTUAL_READER_SUMMARY_PATH)
    )
    print()
    print(
        "NEXT: run the frozen descriptive + whole-case bootstrap analysis."
    )
    print(
        "Do not change reader/updater outputs or manually relabel failures."
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    paths.require_output_stage(paths.PROFILE, Path(__file__).with_name("build_profile.py"))
    process_start = time.perf_counter()

    print("=" * 78)
    print("PROFILE EXTRACTION LAYER-2 — FORMAL DOWNSTREAM READER")
    print("=" * 78)

    (
        freeze,
        preflight,
        tasks,
        updater_plan,
        updater_rows,
        task_by_id,
    ) = verify_upstream()

    print("Formal protocol:        PASS / FROZEN")
    print("Fresh reader controls:  PASS")
    print("Updater stage:          COMPLETE 1200/1200")
    print("Reader interface check: PASS against 500 frozen controls")
    print()

    # Build and freeze the derived actual-reader plan BEFORE the first call.
    actual_plan, plan_manifest = (
        create_or_verify_actual_reader_plan(
            freeze=freeze,
            updater_rows=updater_rows,
            task_by_id=task_by_id,
        )
    )

    print(
        "Logical reader rows:    {}".format(
            plan_manifest["logical_rows"]
        )
    )
    print(
        "Unique reader inputs:   {}".format(
            plan_manifest["unique_reader_inputs"]
        )
    )
    print(
        "Potential cache reuse:  {}".format(
            plan_manifest["cache_reusable_rows"]
        )
    )
    print(
        "Strict failures kept:   {}".format(
            plan_manifest["strict_failure_rows"]
        )
    )
    print(
        "Multi-oracle input groups: {}".format(
            plan_manifest[
                "identical_input_groups_with_multiple_oracle_targets"
            ]
        )
    )
    print()

    reader = freeze["reader"]
    reader_model = str(reader["model"])
    generation_config = reader_generation_config(freeze)
    config = GenerationConfig(
        model=reader_model,
        temperature=float(
            generation_config["temperature"]
        ),
        seed=int(generation_config["seed"]),
        max_output_tokens=int(
            generation_config["max_output_tokens"]
        ),
        context_window=int(
            generation_config["context_window"]
        ),
    )
    system_prompt = str(freeze["reader_system_prompt"])

    backend = OllamaBackend(
        base_url=OLLAMA_BASE_URL,
        timeout_seconds=OLLAMA_TIMEOUT,
    )
    live_snapshot_raw = backend.describe(reader_model)
    live_snapshot = snapshot_fingerprint(
        live_snapshot_raw
    )

    control_summary = read_json(
        READER_CONTROL_SUMMARY_PATH
    )
    control_snapshot = control_summary.get(
        "reader_model_snapshot"
    )
    if live_snapshot != control_snapshot:
        raise RuntimeError(
            "Current fixed reader/runtime snapshot differs from the "
            "fresh formal control stage."
        )

    print(
        "Reader snapshot:        {} | {} | {}".format(
            live_snapshot.get("parameter_size"),
            live_snapshot.get("quantization_level"),
            live_snapshot.get(
                "model_digest",
                live_snapshot.get("digest"),
            ),
        )
    )
    print()

    plan_by_index = {
        int(row["logical_reader_index"]): row
        for row in actual_plan
    }
    if len(plan_by_index) != EXPECTED_ROWS:
        raise RuntimeError(
            "Duplicate logical_reader_index in actual reader plan."
        )

    existing_rows = read_jsonl(
        ACTUAL_READER_RESULTS_PATH
    )
    completed = verify_existing_results(
        existing_rows=existing_rows,
        plan_by_index=plan_by_index,
        reader_model=reader_model,
        generation_config=generation_config,
        expected_snapshot=live_snapshot,
    )

    # Persist deterministic exact-input cache across resume.
    cache: Dict[str, dict] = {}
    for row in existing_rows:
        input_hash = str(row["reader_input_sha256"])
        if input_hash not in cache:
            cache[input_hash] = {
                "reader_raw_response": row[
                    "reader_raw_response"
                ],
                "response_json_valid": row[
                    "response_json_valid"
                ],
                "response_parse_error": row[
                    "response_parse_error"
                ],
                "parsed_response": row[
                    "parsed_response"
                ],
                "reader_usage": row["reader_usage"],
                "reader_latency_seconds": row[
                    "reader_latency_seconds"
                ],
                "reader_response_id": row[
                    "reader_response_id"
                ],
                "reader_response_metadata": row[
                    "reader_response_metadata"
                ],
                "reader_model_snapshot": row[
                    "reader_model_snapshot"
                ],
            }

    print(
        "Completed:              {}/1200".format(
            len(completed)
        )
    )
    print(
        "Remaining:              {}".format(
            EXPECTED_ROWS - len(completed)
        )
    )
    print()

    if len(completed) == EXPECTED_ROWS:
        ordered = [
            completed[index]
            for index in range(EXPECTED_ROWS)
        ]
        summary = build_run_summary(
            status="FORMAL_ACTUAL_READER_STAGE_COMPLETE",
            rows=ordered,
            reader_snapshot=live_snapshot,
            plan_manifest=plan_manifest,
            process_start=process_start,
        )
        write_json_atomic(
            ACTUAL_READER_SUMMARY_PATH,
            summary,
        )
        print_final_summary(summary)
        return

    rows = list(existing_rows)

    running_summary = build_run_summary(
        status="FORMAL_ACTUAL_READER_STAGE_RUNNING",
        rows=rows,
        reader_snapshot=live_snapshot,
        plan_manifest=plan_manifest,
        process_start=process_start,
    )
    write_json_atomic(
        ACTUAL_READER_SUMMARY_PATH,
        running_summary,
    )

    try:
        for plan_row in actual_plan:
            index = int(plan_row["logical_reader_index"])
            if index in completed:
                continue

            input_hash = str(
                plan_row["reader_input_sha256"]
            )
            cache_reused = input_hash in cache

            if cache_reused:
                call_record = dict(cache[input_hash])
                call_kind = "CACHE"
            else:
                call_kind = "CALL"
                request_id = (
                    "formal_actual_reader__{:04d}__{}".format(
                        index,
                        input_hash[:12],
                    )
                )

                start = time.perf_counter()
                response = backend.generate(
                    request_id=request_id,
                    system_prompt=system_prompt,
                    user_prompt=str(plan_row["user_prompt"]),
                    config=config,
                    response_schema=dict(
                        plan_row["response_schema"]
                    ),
                )
                latency = time.perf_counter() - start

                response_json_valid = True
                response_parse_error: Optional[str] = None
                parsed_response: Optional[dict]

                try:
                    parsed_response = parse_reader_response(
                        response.text,
                        plan_row["response_schema"],
                    )
                except Exception as exc:
                    response_json_valid = False
                    response_parse_error = "{}: {}".format(
                        type(exc).__name__,
                        str(exc),
                    )
                    parsed_response = None

                call_record = {
                    "reader_raw_response": response.text,
                    "response_json_valid": bool(
                        response_json_valid
                    ),
                    "response_parse_error": (
                        response_parse_error
                    ),
                    "parsed_response": parsed_response,
                    "reader_usage": dict(response.usage),
                    "reader_latency_seconds": latency,
                    "reader_response_id": (
                        response.response_id
                    ),
                    "reader_response_metadata": dict(
                        response.metadata
                    ),
                    "reader_model_snapshot": dict(
                        live_snapshot
                    ),
                }
                cache[input_hash] = dict(call_record)

            parsed_raw = call_record.get(
                "parsed_response"
            )
            parsed = (
                dict(parsed_raw)
                if isinstance(parsed_raw, Mapping)
                else None
            )

            oracle = {
                str(key): str(value)
                for key, value in plan_row[
                    "oracle_expected_values"
                ].items()
            }
            stale = {
                str(key): str(value)
                for key, value in plan_row[
                    "stale_expected_values"
                ].items()
            }
            attributes = [
                str(attribute)
                for attribute in plan_row[
                    "required_attributes"
                ]
            ]

            task_pass = exact_profile_match(
                parsed,
                oracle,
            )
            profile_class = classify_profile(
                parsed=parsed,
                oracle=oracle,
                stale=stale,
                attributes=attributes,
            )

            strict_ok = bool(
                plan_row["strict_state_success"]
            )

            if strict_ok and task_pass:
                joint_label = "strict_pass_task_pass"
            elif strict_ok and not task_pass:
                joint_label = "strict_pass_task_fail"
            elif not strict_ok and task_pass:
                joint_label = "strict_fail_task_pass"
            else:
                joint_label = "strict_fail_task_fail"

            result_row = {
                "protocol_version": PROTOCOL_VERSION,
                "logical_reader_index": index,
                "source_updater_planned_call_index": int(
                    plan_row[
                        "source_updater_planned_call_index"
                    ]
                ),
                "source_updater_request_id": str(
                    plan_row["source_updater_request_id"]
                ),
                "task_id": str(plan_row["task_id"]),
                "case_id": str(plan_row["case_id"]),
                "case_sha256": str(
                    plan_row["case_sha256"]
                ),
                "update_id": str(plan_row["update_id"]),
                "updater_model": str(
                    plan_row["updater_model"]
                ),
                "condition": str(plan_row["condition"]),
                "grouping": str(plan_row["grouping"]),
                "unit_load": int(plan_row["unit_load"]),
                "interface": str(plan_row["interface"]),
                "subject": str(plan_row["subject"]),
                "required_attributes": list(attributes),
                "reader_memory_sha256": str(
                    plan_row["reader_memory_sha256"]
                ),
                "oracle_expected_values": oracle,
                "stale_expected_values": stale,
                "strict_state_success": strict_ok,
                "strict_failure_stage": str(
                    plan_row["strict_failure_stage"]
                ),
                "strict_failure_reasons": list(
                    plan_row["strict_failure_reasons"]
                ),
                "updater_execution_success": bool(
                    plan_row["updater_execution_success"]
                ),
                "updater_contract_valid": bool(
                    plan_row["updater_contract_valid"]
                ),
                "updater_structure_valid": bool(
                    plan_row["updater_structure_valid"]
                ),
                "updater_target_new_current": bool(
                    plan_row[
                        "updater_target_new_current"
                    ]
                ),
                "updater_protected_facts_preserved": bool(
                    plan_row[
                        "updater_protected_facts_preserved"
                    ]
                ),
                "parsed_response": parsed,
                "attribute_match_oracle": (
                    attribute_match_map(
                        parsed,
                        oracle,
                    )
                ),
                "functional_task_pass": bool(task_pass),
                "profile_class": profile_class,
                "joint_strict_task": joint_label,
                "reader_input_sha256": input_hash,
                "cache_reused": bool(cache_reused),
                "reader_model": reader_model,
                "generation_config": dict(
                    generation_config
                ),
                **{
                    key: value
                    for key, value in call_record.items()
                    if key != "parsed_response"
                },
            }

            append_jsonl_fsync(
                ACTUAL_READER_RESULTS_PATH,
                result_row,
            )
            rows.append(result_row)
            completed[index] = result_row

            print(
                "[{}/1200] {} | {} | {} | {} | {} | strict={} "
                "task={} class={}".format(
                    len(completed),
                    plan_row["updater_model"],
                    plan_row["case_id"],
                    plan_row["update_id"],
                    plan_row["condition"],
                    call_kind,
                    strict_ok,
                    task_pass,
                    profile_class,
                )
            )

            progress_summary = build_run_summary(
                status=(
                    "FORMAL_ACTUAL_READER_STAGE_RUNNING"
                ),
                rows=rows,
                reader_snapshot=live_snapshot,
                plan_manifest=plan_manifest,
                process_start=process_start,
            )
            write_json_atomic(
                ACTUAL_READER_SUMMARY_PATH,
                progress_summary,
            )

    except Exception as exc:
        failed_summary = build_run_summary(
            status="FORMAL_ACTUAL_READER_STAGE_FAILED",
            rows=rows,
            reader_snapshot=live_snapshot,
            plan_manifest=plan_manifest,
            process_start=process_start,
            error="{}: {}".format(
                type(exc).__name__,
                str(exc),
            ),
        )
        write_json_atomic(
            ACTUAL_READER_SUMMARY_PATH,
            failed_summary,
        )
        raise

    final_rows = read_jsonl(
        ACTUAL_READER_RESULTS_PATH
    )
    final_completed = verify_existing_results(
        existing_rows=final_rows,
        plan_by_index=plan_by_index,
        reader_model=reader_model,
        generation_config=generation_config,
        expected_snapshot=live_snapshot,
    )

    if len(final_completed) != EXPECTED_ROWS:
        raise RuntimeError(
            "Actual-reader stage ended incomplete: {}/1200."
            .format(len(final_completed))
        )

    saved_indexes = [
        int(row["logical_reader_index"])
        for row in final_rows
    ]
    if saved_indexes != list(range(EXPECTED_ROWS)):
        raise RuntimeError(
            "Saved actual-reader results do not preserve frozen plan order."
        )

    final_summary = build_run_summary(
        status="FORMAL_ACTUAL_READER_STAGE_COMPLETE",
        rows=final_rows,
        reader_snapshot=live_snapshot,
        plan_manifest=plan_manifest,
        process_start=process_start,
    )
    write_json_atomic(
        ACTUAL_READER_SUMMARY_PATH,
        final_summary,
    )
    print_final_summary(final_summary)


if __name__ == "__main__":
    main()
