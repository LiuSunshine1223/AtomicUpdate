#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze direct-extraction calibration using the candidate-selection cases.

This stage makes no model calls."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "atomic_memory").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atomic_memory import paths

from atomic_memory.core import (  # noqa: E402
    ATTRIBUTE_SPECS,
)

# ============================================================================
# Source calibration batch: immutable prior Profile Selection design/results
# ============================================================================

SOURCE_DIR = paths.SELECTION

SOURCE_TASKS_PATH = SOURCE_DIR / "profile_tasks.jsonl"
SOURCE_CONTROL_PLAN_PATH = SOURCE_DIR / "reader_control_plan.jsonl"
SOURCE_PREFLIGHT_PATH = SOURCE_DIR / "formal_preflight.json"
SOURCE_FREEZE_PATH = SOURCE_DIR / "freeze_manifest.json"
SOURCE_CONTROL_SUMMARY_PATH = SOURCE_DIR / "reader_control_run_summary.json"
SOURCE_ERROR_AUDIT_PATH = SOURCE_DIR / "reader_control_error_audit.json"

SOURCE_CORE_PATH = (
    PROJECT_ROOT
    / "atomic_memory"
    / "core.py"
)

EXPECTED_SOURCE_PROTOCOL = "profile-selection-layer2-formal-v1"
EXPECTED_SOURCE_TASKS = 100
EXPECTED_SOURCE_CONTROLS = 500

# ============================================================================
# New calibration design
# ============================================================================

PROTOCOL_VERSION = "profile-extraction-reader-calibration-v1"

OUTPUT_DIR = paths.CALIBRATION

TASKS_PATH = OUTPUT_DIR / "extraction_tasks.jsonl"
CONTROL_PLAN_PATH = OUTPUT_DIR / "reader_control_plan.jsonl"
PREFLIGHT_PATH = OUTPUT_DIR / "calibration_preflight.json"
FREEZE_PATH = OUTPUT_DIR / "freeze_manifest.json"

# Reserved result artifacts. Once any of these exist, this builder will refuse
# to rebuild the calibration design.
RESERVED_RESULT_PATHS = (
    OUTPUT_DIR / "reader_control_results.jsonl",
    OUTPUT_DIR / "reader_control_run_summary.json",
    OUTPUT_DIR / "reader_control_error_audit.json",
)

READER_MODEL = "llama3.1:8b-instruct-q4_K_M"
READER_TEMPERATURE = 0.0
READER_SEED = 42
READER_MAX_OUTPUT_TOKENS = 64
READER_CONTEXT_WINDOW = 8192

UNKNOWN_VALUE = "UNKNOWN"

READER_SYSTEM_PROMPT = """You are a downstream memory reader.

Your job is to extract the named person's CURRENT profile directly from the
provided memory state.

Rules:
1. Use the memory state exactly as provided. Do not repair or rewrite it.
2. For every required attribute, return the current value explicitly supported
   by the memory.
3. The request supplies both an output key and the corresponding memory label.
   Treat them as the same attribute.
4. If a required current value cannot be determined from the provided memory,
   return the exact string "UNKNOWN" for that attribute.
5. Do not use stereotypes, outside knowledge, update evidence, candidate
   profiles, or hidden facts.
6. Return exactly one JSON object containing exactly the required attribute
   keys.
7. Do not output explanations or any other text.
"""

# Frozen before any new extraction-reader model call.
#
# Exact-profile gates are deliberately retained at 95/100 per grouping.
# No-memory is now an abstention control rather than an 8-way chance control.
READER_GATES = {
    "oracle_exact_profile_min_per_grouping": 95,
    "stale_exact_profile_min_per_grouping": 95,
    "no_memory_all_unknown_min": 95,
    "valid_json_min_rows": 500,
    "targets_per_grouping_control": 100,
    "no_memory_target_count": 100,
}

EXPECTED_TASKS = 100
EXPECTED_CONTROL_ROWS = 500

if READER_GATES["targets_per_grouping_control"] != EXPECTED_TASKS:
    raise RuntimeError("Reader gate target count changed.")
if READER_GATES["no_memory_target_count"] != EXPECTED_TASKS:
    raise RuntimeError("No-memory target count changed.")


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
    temp.replace(path)


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
    temp.replace(path)


def write_or_verify_json(path: Path, value: Any) -> str:
    if not path.exists():
        write_json_atomic(path, value)
        return "CREATED"

    existing = read_json(path)
    if canonical_json(existing) != canonical_json(value):
        raise RuntimeError(
            "Existing file differs from frozen calibration design: {}".format(
                path
            )
        )
    return "VERIFIED"


def write_or_verify_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    if not path.exists():
        write_jsonl_atomic(path, rows)
        return "CREATED"

    existing = read_jsonl(path)
    if canonical_json(existing) != canonical_json(list(rows)):
        raise RuntimeError(
            "Existing file differs from frozen calibration design: {}".format(
                path
            )
        )
    return "VERIFIED"


def reader_generation_config() -> dict:
    return {
        "model": READER_MODEL,
        "temperature": READER_TEMPERATURE,
        "seed": READER_SEED,
        "max_output_tokens": READER_MAX_OUTPUT_TOKENS,
        "context_window": READER_CONTEXT_WINDOW,
    }


# ============================================================================
# Source verification
# ============================================================================

def verify_source_batch() -> Tuple[
    dict,
    dict,
    dict,
    dict,
    List[dict],
    List[dict],
]:
    required = (
        SOURCE_TASKS_PATH,
        SOURCE_CONTROL_PLAN_PATH,
        SOURCE_PREFLIGHT_PATH,
        SOURCE_FREEZE_PATH,
        SOURCE_CONTROL_SUMMARY_PATH,
        SOURCE_ERROR_AUDIT_PATH,
        SOURCE_CORE_PATH,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                "Missing required source artifact: {}".format(path)
            )

    source_preflight = read_json(SOURCE_PREFLIGHT_PATH)
    source_freeze = read_json(SOURCE_FREEZE_PATH)
    source_summary = read_json(SOURCE_CONTROL_SUMMARY_PATH)
    source_audit = read_json(SOURCE_ERROR_AUDIT_PATH)
    source_tasks = read_jsonl(SOURCE_TASKS_PATH)
    source_controls = read_jsonl(SOURCE_CONTROL_PLAN_PATH)

    if source_preflight.get("status") != "PASS":
        raise RuntimeError("Source formal preflight is not PASS.")

    if source_freeze.get("status") != "FORMAL_PROTOCOL_FROZEN":
        raise RuntimeError("Source formal protocol is not frozen.")

    if source_freeze.get("protocol_version") != EXPECTED_SOURCE_PROTOCOL:
        raise RuntimeError(
            "Unexpected source protocol version: {!r}".format(
                source_freeze.get("protocol_version")
            )
        )

    if source_summary.get("status") != "READER_CONTROL_GATE_FAIL":
        raise RuntimeError(
            "Expected the prior Profile Selection reader gate to be FAIL; "
            "got {!r}.".format(source_summary.get("status"))
        )

    if source_audit.get("status") != (
        "OFFLINE_READER_CONTROL_ERROR_AUDIT_COMPLETE"
    ):
        raise RuntimeError("Source reader error audit is incomplete.")

    if int(source_audit.get("model_calls", -1)) != 0:
        raise RuntimeError("Source error audit unexpectedly made model calls.")

    if len(source_tasks) != EXPECTED_SOURCE_TASKS:
        raise RuntimeError(
            "Expected {} source tasks, found {}.".format(
                EXPECTED_SOURCE_TASKS,
                len(source_tasks),
            )
        )

    if len(source_controls) != EXPECTED_SOURCE_CONTROLS:
        raise RuntimeError(
            "Expected {} source controls, found {}.".format(
                EXPECTED_SOURCE_CONTROLS,
                len(source_controls),
            )
        )

    frozen_file_hashes = source_freeze.get("frozen_file_sha256", {})
    for filename, path in (
        ("profile_tasks.jsonl", SOURCE_TASKS_PATH),
        ("reader_control_plan.jsonl", SOURCE_CONTROL_PLAN_PATH),
    ):
        expected = str(frozen_file_hashes.get(filename, ""))
        actual = sha256_file(path)
        if expected != actual:
            raise RuntimeError(
                "Source frozen hash mismatch for {}: expected {}, got {}."
                .format(filename, expected, actual)
            )

    expected_core_hash = str(
        source_freeze.get("code_sha256", {}).get(
            "representation_edit_scope_core.py",
            "",
        )
    )
    actual_core_hash = sha256_file(SOURCE_CORE_PATH)
    if expected_core_hash and expected_core_hash != actual_core_hash:
        raise RuntimeError(
            "Current core.py differs from the "
            "source frozen core."
        )

    return (
        source_preflight,
        source_freeze,
        source_summary,
        source_audit,
        source_tasks,
        source_controls,
    )


# ============================================================================
# Extraction task / schema / prompt construction
# ============================================================================

def ordered_values(
    values: Mapping[str, Any],
    attributes: Sequence[str],
) -> dict:
    return {
        str(attribute): str(values[str(attribute)])
        for attribute in attributes
    }


def build_extraction_tasks(
    source_tasks: Sequence[Mapping[str, Any]],
) -> Tuple[List[dict], dict]:
    # Reuse the observed cases while changing the reader interface.
    tasks: List[dict] = []

    for source_task in source_tasks:
        attributes = [
            str(attribute)
            for attribute in source_task["attributes"]
        ]
        if len(attributes) != 3 or len(set(attributes)) != 3:
            raise RuntimeError(
                "{} does not have exactly three unique attributes.".format(
                    source_task["task_id"]
                )
            )

        target_attribute = str(source_task["target_attribute"])
        if target_attribute not in attributes:
            raise RuntimeError(
                "{} target attribute is not in the subject schema.".format(
                    source_task["task_id"]
                )
            )

        oracle_values = ordered_values(
            source_task["correct_values"],
            attributes,
        )
        stale_values = dict(oracle_values)
        stale_values[target_attribute] = str(
            source_task["target_old_value"]
        )

        if oracle_values[target_attribute] != str(
            source_task["target_new_value"]
        ):
            raise RuntimeError(
                "{} oracle target != target_new_value.".format(
                    source_task["task_id"]
                )
            )

        if stale_values[target_attribute] != str(
            source_task["target_old_value"]
        ):
            raise RuntimeError(
                "{} stale target != target_old_value.".format(
                    source_task["task_id"]
                )
            )

        differing = [
            attribute
            for attribute in attributes
            if oracle_values[attribute] != stale_values[attribute]
        ]
        if differing != [target_attribute]:
            raise RuntimeError(
                "{} oracle/stale profiles must differ only on target; got {}."
                .format(source_task["task_id"], differing)
            )

        labels = {
            attribute: str(ATTRIBUTE_SPECS[attribute]["label"])
            for attribute in attributes
        }

        unknown_values = {
            attribute: UNKNOWN_VALUE
            for attribute in attributes
        }

        tasks.append(
            {
                "extraction_task_id": "extract__{}".format(
                    source_task["task_id"]
                ),
                "source_task_id": str(source_task["task_id"]),
                "case_id": str(source_task["case_id"]),
                "update_id": str(source_task["update_id"]),
                "subject": str(source_task["subject"]),
                "required_attributes": list(attributes),
                "memory_labels": labels,
                "target_attribute": target_attribute,
                "protected_attributes": [
                    str(value)
                    for value in source_task["protected_attributes"]
                ],
                "target_old_value": str(
                    source_task["target_old_value"]
                ),
                "target_new_value": str(
                    source_task["target_new_value"]
                ),
                "oracle_expected_values": oracle_values,
                "stale_expected_values": stale_values,
                "no_memory_expected_values": unknown_values,
                "source_case_sha256": str(
                    source_task["case_sha256"]
                ),
                "manual_annotation_required": False,
            }
        )

    if len(tasks) != EXPECTED_TASKS:
        raise RuntimeError(
            "Expected {} extraction tasks, got {}.".format(
                EXPECTED_TASKS,
                len(tasks),
            )
        )

    schema_counts = Counter(
        tuple(task["required_attributes"])
        for task in tasks
    )
    target_counts = Counter(
        str(task["target_attribute"])
        for task in tasks
    )

    report = {
        "task_count": len(tasks),
        "schema_counts": {
            "|".join(key): int(value)
            for key, value in sorted(schema_counts.items())
        },
        "target_attribute_counts": dict(
            sorted(target_counts.items())
        ),
        "all_have_three_unique_attributes": True,
        "oracle_stale_differ_only_on_target": True,
        "manual_annotation_required": False,
    }
    return tasks, report


def response_schema_for_attributes(
    attributes: Sequence[str],
) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [str(attribute) for attribute in attributes],
        "properties": {
            str(attribute): {"type": "string"}
            for attribute in attributes
        },
    }


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


def build_reader_user_prompt(
    *,
    memory: Any,
    subject: str,
    attributes: Sequence[str],
    labels: Mapping[str, str],
) -> str:
    if memory is None:
        memory_text = "(no memory state provided)"
    else:
        memory_text = json.dumps(
            public_memory(memory),
            ensure_ascii=False,
            indent=2,
        )

    attribute_spec = [
        {
            "key": str(attribute),
            "memory_label": str(labels[str(attribute)]),
        }
        for attribute in attributes
    ]

    attribute_text = json.dumps(
        attribute_spec,
        ensure_ascii=False,
        indent=2,
    )

    example_shape = {
        str(attribute): UNKNOWN_VALUE
        for attribute in attributes
    }

    return (
        "MEMORY STATE:\n"
        + memory_text
        + "\n\nTARGET PERSON:\n"
        + str(subject)
        + "\n\nREQUIRED ATTRIBUTES:\n"
        + attribute_text
        + "\n\nExtract the target person's CURRENT value for every required "
          "attribute.\n"
          "Use the exact value text supported by the memory.\n"
          "If a value cannot be determined from the memory, use "
          "\"UNKNOWN\" for that attribute.\n"
          "Return exactly one JSON object with this key shape:\n"
        + json.dumps(
            example_shape,
            ensure_ascii=False,
        )
    )


# ============================================================================
# Reader control construction
# ============================================================================

def build_source_control_index(
    source_controls: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str, str], dict]:
    index: Dict[Tuple[str, str, str], dict] = {}

    for row in source_controls:
        task_id = str(row["task_id"])
        arm = str(row["arm"])
        grouping = str(row["grouping"])

        key = (task_id, arm, grouping)
        if key in index:
            raise RuntimeError(
                "Duplicate source reader control key: {}".format(key)
            )
        index[key] = dict(row)

    expected_counts = Counter(
        (str(row["arm"]), str(row["grouping"]))
        for row in source_controls
    )
    expected = {
        ("oracle", "g1"): 100,
        ("oracle", "g3"): 100,
        ("stale", "g1"): 100,
        ("stale", "g3"): 100,
        ("no_memory", "None"): 100,
    }
    if dict(expected_counts) != expected:
        raise RuntimeError(
            "Unexpected source control counts: {}".format(
                dict(expected_counts)
            )
        )

    return index


def build_reader_control_plan(
    tasks: Sequence[Mapping[str, Any]],
    source_controls: Sequence[Mapping[str, Any]],
) -> Tuple[List[dict], dict]:
    source_index = build_source_control_index(source_controls)
    rows: List[dict] = []

    generation_config = reader_generation_config()

    for task in tasks:
        source_task_id = str(task["source_task_id"])
        attributes = [
            str(attribute)
            for attribute in task["required_attributes"]
        ]
        labels = dict(task["memory_labels"])
        response_schema = response_schema_for_attributes(attributes)

        for arm, grouping in (
            ("oracle", "g1"),
            ("oracle", "g3"),
            ("stale", "g1"),
            ("stale", "g3"),
        ):
            source_row = source_index[
                (source_task_id, arm, grouping)
            ]
            memory = source_row["memory"]
            expected_values = (
                dict(task["oracle_expected_values"])
                if arm == "oracle"
                else dict(task["stale_expected_values"])
            )

            user_prompt = build_reader_user_prompt(
                memory=memory,
                subject=str(task["subject"]),
                attributes=attributes,
                labels=labels,
            )
            reader_input_payload = {
                "system_prompt": READER_SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "generation_config": generation_config,
                "response_schema": response_schema,
            }

            rows.append(
                {
                    "logical_control_index": len(rows),
                    "control_id": "{}__{}__{}".format(
                        task["extraction_task_id"],
                        arm,
                        grouping,
                    ),
                    "extraction_task_id": str(
                        task["extraction_task_id"]
                    ),
                    "source_task_id": source_task_id,
                    "case_id": str(task["case_id"]),
                    "update_id": str(task["update_id"]),
                    "arm": arm,
                    "grouping": grouping,
                    "subject": str(task["subject"]),
                    "required_attributes": list(attributes),
                    "memory_labels": dict(labels),
                    "memory": memory,
                    "expected_values": expected_values,
                    # Store exact frozen strings/objects so the runner does not
                    # have to reconstruct them from serialized dict ordering.
                    "user_prompt": user_prompt,
                    "response_schema": response_schema,
                    "reader_system_prompt_sha256": sha256_text(
                        READER_SYSTEM_PROMPT
                    ),
                    "user_prompt_sha256": sha256_text(
                        user_prompt
                    ),
                    "response_schema_sha256": sha256_text(
                        canonical_json(response_schema)
                    ),
                    "reader_input_sha256": sha256_text(
                        canonical_json(reader_input_payload)
                    ),
                }
            )

        # One no-memory logical row per task, shared across groupings.
        response_schema = response_schema_for_attributes(attributes)
        user_prompt = build_reader_user_prompt(
            memory=None,
            subject=str(task["subject"]),
            attributes=attributes,
            labels=labels,
        )
        reader_input_payload = {
            "system_prompt": READER_SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "generation_config": generation_config,
            "response_schema": response_schema,
        }

        rows.append(
            {
                "logical_control_index": len(rows),
                "control_id": "{}__no_memory".format(
                    task["extraction_task_id"]
                ),
                "extraction_task_id": str(
                    task["extraction_task_id"]
                ),
                "source_task_id": source_task_id,
                "case_id": str(task["case_id"]),
                "update_id": str(task["update_id"]),
                "arm": "no_memory",
                "grouping": None,
                "subject": str(task["subject"]),
                "required_attributes": list(attributes),
                "memory_labels": dict(labels),
                "memory": None,
                "expected_values": dict(
                    task["no_memory_expected_values"]
                ),
                "user_prompt": user_prompt,
                "response_schema": response_schema,
                "reader_system_prompt_sha256": sha256_text(
                    READER_SYSTEM_PROMPT
                ),
                "user_prompt_sha256": sha256_text(user_prompt),
                "response_schema_sha256": sha256_text(
                    canonical_json(response_schema)
                ),
                "reader_input_sha256": sha256_text(
                    canonical_json(reader_input_payload)
                ),
            }
        )

    if len(rows) != EXPECTED_CONTROL_ROWS:
        raise RuntimeError(
            "Expected {} calibration controls, got {}.".format(
                EXPECTED_CONTROL_ROWS,
                len(rows),
            )
        )

    counts = Counter(
        (str(row["arm"]), str(row["grouping"]))
        for row in rows
    )
    expected_counts = {
        ("oracle", "g1"): 100,
        ("oracle", "g3"): 100,
        ("stale", "g1"): 100,
        ("stale", "g3"): 100,
        ("no_memory", "None"): 100,
    }
    if dict(counts) != expected_counts:
        raise RuntimeError(
            "Unexpected calibration control counts: {}".format(
                dict(counts)
            )
        )

    unique_inputs = len(
        {
            str(row["reader_input_sha256"])
            for row in rows
        }
    )

    report = {
        "row_count": len(rows),
        "arm_grouping_counts": {
            "{}__{}".format(arm, grouping): int(count)
            for (arm, grouping), count in sorted(counts.items())
        },
        "logical_rows": len(rows),
        "unique_reader_inputs_before_runtime_cache": unique_inputs,
        "exact_input_caching_allowed": True,
        "candidate_profiles_removed": True,
        "direct_three_field_extraction": True,
    }
    return rows, report


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    paths.require_output_stage(paths.CALIBRATION, Path(__file__))
    print("=" * 78)
    print("PROFILE EXTRACTION READER CALIBRATION — BUILD / FREEZE")
    print("=" * 78)
    print("Mode:                 ZERO MODEL CALLS")
    print("Reader:               {}".format(READER_MODEL))
    print("Reader interface:     direct 3-field extraction")
    print("Calibration tasks:    {}".format(EXPECTED_TASKS))
    print("Logical controls:     {}".format(EXPECTED_CONTROL_ROWS))
    print("Calibration batch:    prior observed 50-case reader batch")
    print("Formal updater calls: BLOCKED")
    print()

    started_results = [
        path
        for path in RESERVED_RESULT_PATHS
        if path.exists()
    ]
    if started_results:
        raise RuntimeError(
            "Calibration result artifacts already exist. Do not rebuild the "
            "frozen calibration design:\n{}".format(
                "\n".join(
                    "  {}".format(path)
                    for path in started_results
                )
            )
        )

    print("[1/4] Verifying prior frozen Profile Selection calibration...")
    (
        source_preflight,
        source_freeze,
        source_summary,
        source_audit,
        source_tasks,
        source_controls,
    ) = verify_source_batch()
    print(
        "  PASS | source protocol frozen | prior reader gate = FAIL | "
        "error audit verified"
    )

    print("[2/4] Building 100 direct Profile Extraction tasks...")
    tasks, task_report = build_extraction_tasks(source_tasks)
    print(
        "  PASS | 100 tasks | 3 required attributes each | "
        "oracle/stale differ only on target"
    )

    print("[3/4] Building 500-row extraction-reader control plan...")
    controls, control_report = build_reader_control_plan(
        tasks,
        source_controls,
    )
    print(
        "  PASS | 200 oracle + 200 stale + 100 no-memory | "
        "no candidate profiles"
    )

    file_status: Dict[str, str] = {}
    file_status["extraction_tasks.jsonl"] = write_or_verify_jsonl(
        TASKS_PATH,
        tasks,
    )
    file_status["reader_control_plan.jsonl"] = write_or_verify_jsonl(
        CONTROL_PLAN_PATH,
        controls,
    )

    print("[4/4] Freezing calibration protocol and provenance...")

    frozen_file_hashes = {
        "extraction_tasks.jsonl": sha256_file(TASKS_PATH),
        "reader_control_plan.jsonl": sha256_file(
            CONTROL_PLAN_PATH
        ),
    }

    source_hashes = {
        "profile_tasks.jsonl": sha256_file(
            SOURCE_TASKS_PATH
        ),
        "reader_control_plan.jsonl": sha256_file(
            SOURCE_CONTROL_PLAN_PATH
        ),
        "formal_preflight.json": sha256_file(
            SOURCE_PREFLIGHT_PATH
        ),
        "freeze_manifest.json": sha256_file(
            SOURCE_FREEZE_PATH
        ),
        "reader_control_run_summary.json": sha256_file(
            SOURCE_CONTROL_SUMMARY_PATH
        ),
        "reader_control_error_audit.json": sha256_file(
            SOURCE_ERROR_AUDIT_PATH
        ),
        "representation_edit_scope_core.py": sha256_file(
            SOURCE_CORE_PATH
        ),
    }

    builder_path = Path(__file__).resolve()

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "CALIBRATION_PROTOCOL_FROZEN",
        "zero_model_call_build": True,
        "calibration_only": True,
        "formal_updater_calls_allowed_from_this_batch": False,
        "fresh_formal_batch_required_after_calibration_pass": True,
        "source_protocol_version": str(
            source_freeze["protocol_version"]
        ),
        "source_reader_gate_status": str(
            source_summary["status"]
        ),
        "source_error_audit_status": str(
            source_audit["status"]
        ),
        "source_case_reuse_reason": (
            "These 50 cases are reused only to calibrate a revised reader "
            "interface after the prior 8-way Profile Selection reader failed "
            "its frozen suitability gate. They are permanently excluded from "
            "the future prospective formal updater batch."
        ),
        "interface_change": {
            "old": (
                "memory + target + 8 anonymous candidate profiles -> "
                "profile_id"
            ),
            "new": (
                "memory + target + 3 required attributes -> direct "
                "attribute-value JSON"
            ),
            "reader_model_changed": False,
            "candidate_profiles_removed": True,
            "position_choice_removed": True,
            "unknown_policy": UNKNOWN_VALUE,
        },
        "reader": {
            "model": READER_MODEL,
            "temperature": READER_TEMPERATURE,
            "seed": READER_SEED,
            "max_output_tokens": READER_MAX_OUTPUT_TOKENS,
            "context_window": READER_CONTEXT_WINDOW,
            "exact_input_caching": True,
        },
        "reader_system_prompt": READER_SYSTEM_PROMPT,
        "reader_system_prompt_sha256": sha256_text(
            READER_SYSTEM_PROMPT
        ),
        "reader_gates": dict(READER_GATES),
        "calibration_task_count": EXPECTED_TASKS,
        "reader_control_logical_rows": EXPECTED_CONTROL_ROWS,
        "task_report": task_report,
        "reader_control_report": control_report,
        "source_file_sha256": source_hashes,
        "frozen_file_sha256": frozen_file_hashes,
        "code_sha256": {
            "build_profile_extraction_reader_calibration.py": (
                sha256_file(builder_path)
            ),
        },
        "anti_rescue_rules": [
            "Do not lower gates after observing calibration outputs.",
            "Do not delete difficult calibration tasks.",
            "Do not alter attribute vocabularies or expected values.",
            "Do not switch reader model within this calibration protocol.",
            "Do not start updater calls from this reused calibration batch.",
        ],
        "next_step_if_build_passes": (
            "Run the frozen 500-row Profile Extraction reader calibration."
        ),
        "next_step_if_reader_calibration_passes": (
            "Generate a fresh, non-overlapping 50-case prospective formal "
            "batch, freeze it, and run its reader controls before any updater "
            "calls."
        ),
    }

    preflight = {
        "status": "PASS",
        "model_calls": 0,
        "calibration_only": True,
        "formal_updater_calls_blocked": True,
        "tasks": len(tasks),
        "reader_control_rows": len(controls),
        "reader_model": READER_MODEL,
        "reader_interface": "direct_three_field_extraction",
        "candidate_profiles_removed": True,
        "source_reader_gate_status": str(
            source_summary["status"]
        ),
        "file_status": file_status,
        "frozen_file_sha256": frozen_file_hashes,
        "reader_gates": dict(READER_GATES),
        "next_step": (
            "Run the frozen 500-row Profile Extraction reader calibration. "
            "Do not run any updater calls."
        ),
    }

    preflight_status = write_or_verify_json(
        PREFLIGHT_PATH,
        preflight,
    )
    freeze_status = write_or_verify_json(
        FREEZE_PATH,
        protocol,
    )

    print(
        "  PASS | source provenance hashed | extraction prompts/schemas "
        "frozen | gates frozen"
    )
    print()
    print("=" * 78)
    print("PROFILE EXTRACTION CALIBRATION PREFLIGHT PASS")
    print("=" * 78)
    print("Tasks:                    {}/100".format(len(tasks)))
    print("Reader controls:          {}/500".format(len(controls)))
    print(
        "Unique frozen inputs:     {}".format(
            control_report[
                "unique_reader_inputs_before_runtime_cache"
            ]
        )
    )
    print("Reader model changed:     NO")
    print("Candidate profiles used:  NO")
    print("Model calls made:         0")
    print("Formal updater calls:     BLOCKED")
    print("Calibration protocol:     FROZEN")
    print()
    print("Files:")
    print(
        "  {} [{}]".format(
            TASKS_PATH,
            file_status["extraction_tasks.jsonl"],
        )
    )
    print(
        "  {} [{}]".format(
            CONTROL_PLAN_PATH,
            file_status["reader_control_plan.jsonl"],
        )
    )
    print(
        "  {} [{}]".format(
            PREFLIGHT_PATH,
            preflight_status,
        )
    )
    print(
        "  {} [{}]".format(
            FREEZE_PATH,
            freeze_status,
        )
    )
    print()
    print(
        "NEXT: run the frozen 500-row Profile Extraction reader calibration."
    )
    print(
        "Do not run any updater calls from the prior 50-case calibration batch."
    )


if __name__ == "__main__":
    main()
