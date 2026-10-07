#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze fresh profile-study cases, updater plans, and reader controls.

Uses saved calibration evidence without model inference."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "atomic_memory").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atomic_memory import paths

from atomic_memory.core import (  # noqa: E402
    ACTION_EXACT_PATCH,
    ACTION_REWRITE_UNIT,
    ATTRIBUTE_SPECS,
    ATTRIBUTES_PER_SUBJECT,
    generate_cases,
    sha256_json,
)

# ============================================================================
# Frozen Layer-2 formal design
# ============================================================================

PROTOCOL_VERSION = "profile-extraction-layer2-formal-v1"

# New prospective formal batch. Chosen after reader calibration and before any
# formal updater output.
FORMAL_MASTER_SEED = 20261011
FORMAL_NUM_CASES = 50
FORMAL_TARGETS_PER_CASE = 2

# Historical batches excluded by semantic initial-fact signature.
BASELINE_MASTER_SEED = 20261001
BASELINE_NUM_CASES = 20

UNIT_LOAD_MASTER_SEED = 20261005
UNIT_LOAD_NUM_CASES = 20

FEASIBILITY_MASTER_SEED = 20261007
FEASIBILITY_NUM_CASES = 200

# This is the prior 50-case reader-calibration batch. It MUST NOT be reused for
# the prospective formal updater study.
CALIBRATION_MASTER_SEED = 20261009
CALIBRATION_NUM_CASES = 50

MODELS = (
    "llama3.1:8b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "gemma3:12b-it-q4_K_M",
)

CONDITION_L1_SELECT = "L1_select"
CONDITION_L1_PATCH = "L1_patch"
CONDITION_L3_SELECT = "L3_select"
CONDITION_L3_PATCH = "L3_patch"
CONDITIONS = (
    CONDITION_L1_SELECT,
    CONDITION_L1_PATCH,
    CONDITION_L3_SELECT,
    CONDITION_L3_PATCH,
)

READER_MODEL = "llama3.1:8b-instruct-q4_K_M"

UPDATER_TEMPERATURE = 0.0
UPDATER_SEED = 42
UPDATER_MAX_OUTPUT_TOKENS = 1024
UPDATER_CONTEXT_WINDOW = 8192

READER_TEMPERATURE = 0.0
READER_SEED = 42
READER_MAX_OUTPUT_TOKENS = 64
READER_CONTEXT_WINDOW = 8192

UNKNOWN_VALUE = "UNKNOWN"

EXPECTED_TARGET_INSTANCES = FORMAL_NUM_CASES * FORMAL_TARGETS_PER_CASE
EXPECTED_UPDATER_CALLS = (
    EXPECTED_TARGET_INSTANCES * len(CONDITIONS) * len(MODELS)
)
EXPECTED_READER_CONTROL_ROWS = EXPECTED_TARGET_INSTANCES * 5
EXPECTED_ACTUAL_READER_ROWS = EXPECTED_UPDATER_CALLS
EXPECTED_MAX_MANDATORY_CALLS = (
    EXPECTED_UPDATER_CALLS
    + EXPECTED_READER_CONTROL_ROWS
    + EXPECTED_ACTUAL_READER_ROWS
)

if EXPECTED_TARGET_INSTANCES != 100:
    raise RuntimeError("Target-count invariant changed.")
if EXPECTED_UPDATER_CALLS != 1200:
    raise RuntimeError("Updater call-count invariant changed.")
if EXPECTED_READER_CONTROL_ROWS != 500:
    raise RuntimeError("Reader-control invariant changed.")
if EXPECTED_MAX_MANDATORY_CALLS != 2900:
    raise RuntimeError("Total call-budget invariant changed.")

OUTPUT_DIR = paths.PROFILE
CASES_PATH = OUTPUT_DIR / "formal_cases.jsonl"
TASKS_PATH = OUTPUT_DIR / "extraction_tasks.jsonl"
UPDATER_PLAN_PATH = OUTPUT_DIR / "updater_execution_plan.jsonl"
READER_CONTROL_PLAN_PATH = OUTPUT_DIR / "reader_control_plan.jsonl"
PREFLIGHT_PATH = OUTPUT_DIR / "formal_preflight.json"
FREEZE_PATH = OUTPUT_DIR / "freeze_manifest.json"

# Prior reader calibration provenance. These files are read-only.
CALIBRATION_DIR = paths.CALIBRATION
CALIBRATION_PREFLIGHT_PATH = (
    CALIBRATION_DIR / "calibration_preflight.json"
)
CALIBRATION_FREEZE_PATH = CALIBRATION_DIR / "freeze_manifest.json"
CALIBRATION_RESULTS_PATH = (
    CALIBRATION_DIR / "reader_control_results.jsonl"
)
CALIBRATION_SUMMARY_PATH = (
    CALIBRATION_DIR / "reader_control_run_summary.json"
)
EXPECTED_CALIBRATION_PROTOCOL = (
    "profile-extraction-reader-calibration-v1"
)

CORE_PATH = (
    PROJECT_ROOT
    / "atomic_memory"
    / "core.py"
)

# Future result names are reserved so the builder cannot rewrite the design
# after formal execution begins.
RESERVED_RESULT_PATHS = (
    OUTPUT_DIR / "reader_control_results.jsonl",
    OUTPUT_DIR / "reader_control_run_summary.json",
    OUTPUT_DIR / "updater_results.jsonl",
    OUTPUT_DIR / "updater_run_summary.json",
    OUTPUT_DIR / "actual_reader_results.jsonl",
    OUTPUT_DIR / "actual_reader_run_summary.json",
    OUTPUT_DIR / "formal_analysis.json",
)

UPDATER_SYSTEM_PROMPT = """You are a memory editing controller.

You will receive a memory state and one piece of new evidence.
Return exactly one JSON object matching the required action schema.

Rules:
1. Infer which existing fact the new evidence updates.
2. Preserve every unrelated fact.
3. Preserve all memory IDs.
4. Each memory ID defines a fixed unit. Facts must remain assigned to the same
   memory_id as in the input. Do not move, swap, regroup, merge, or split facts
   across units. Update the target fact inside its original unit.
5. Do not invent unsupported facts.
6. Memory text must remain in the controlled sentence grammar shown in the
   input examples:
   <Name>'s current <attribute> is <value>.
   <Name>'s previous <attribute> was <value>.
7. Do not output explanations, markdown, or any text outside the JSON object.
8. Do not ask for the correct memory ID. Select it yourself when the action
   requires one.
"""

# Exactly the reader interface that passed the frozen calibration.
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

# Frozen before any formal reader output.
READER_GATES = {
    "oracle_exact_profile_min_per_grouping": 95,
    "stale_exact_profile_min_per_grouping": 95,
    "no_memory_all_unknown_min": 95,
    "valid_json_min_rows": 500,
    "targets_per_grouping_control": 100,
    "no_memory_target_count": 100,
}


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


def stable_int(*parts: object) -> int:
    payload = "||".join(str(x) for x in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


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
                raise ValueError(
                    "{}:{} must contain a JSON object".format(
                        path,
                        line_number,
                    )
                )
            rows.append(value)
    return rows


def write_or_verify_json(
    path: Path,
    value: Mapping[str, Any],
) -> str:
    if path.exists():
        existing = read_json(path)
        if canonical_json(existing) != canonical_json(value):
            raise RuntimeError(
                "Existing frozen file differs from deterministic rebuild: "
                "{}".format(path)
            )
        return "VERIFIED_EXISTING"

    write_json_atomic(path, value)
    return "CREATED"


def write_or_verify_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    if path.exists():
        existing = read_jsonl(path)
        if canonical_json(existing) != canonical_json(list(rows)):
            raise RuntimeError(
                "Existing frozen file differs from deterministic rebuild: "
                "{}".format(path)
            )
        return "VERIFIED_EXISTING"

    write_jsonl_atomic(path, rows)
    return "CREATED"


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


# ============================================================================
# Formal case construction
# ============================================================================

def initial_fact_signature(case: Mapping[str, Any]) -> str:
    rows = sorted(
        (
            str(fact["subject"]),
            str(fact["attribute"]),
            str(fact["value"]),
        )
        for fact in case["facts"]
    )
    return sha256_text(canonical_json(rows))


def build_grouped_memory(
    case: Mapping[str, Any],
    facts_per_unit: int,
) -> List[dict]:
    if facts_per_unit not in (1, 3):
        raise ValueError("Layer-2 uses only g=1 and g=3.")

    fact_by_id = {
        str(fact["fact_id"]): dict(fact)
        for fact in case["facts"]
    }
    ordered_facts = [
        fact_by_id[str(fact_id)]
        for fact_id in case["shared_fact_order"]
    ]

    if len(ordered_facts) != 6:
        raise ValueError("Expected six ordered facts.")

    units = []
    for start in range(0, 6, facts_per_unit):
        unit_facts = ordered_facts[start : start + facts_per_unit]
        units.append(
            {
                "memory_id": "m{:02d}".format(len(units)),
                "text": " ".join(
                    str(fact["current_text"])
                    for fact in unit_facts
                ),
                "fact_ids": [
                    str(fact["fact_id"])
                    for fact in unit_facts
                ],
            }
        )
    return units


def build_layer2_cases() -> Tuple[List[dict], dict]:
    base_cases = generate_cases(
        num_cases=FORMAL_NUM_CASES,
        master_seed=FORMAL_MASTER_SEED,
        targets_per_case=FORMAL_TARGETS_PER_CASE,
    )

    historical_batches = {
        "baseline_20261001": generate_cases(
            BASELINE_NUM_CASES,
            BASELINE_MASTER_SEED,
            FORMAL_TARGETS_PER_CASE,
        ),
        "unit_load_20261005": generate_cases(
            UNIT_LOAD_NUM_CASES,
            UNIT_LOAD_MASTER_SEED,
            FORMAL_TARGETS_PER_CASE,
        ),
        "feasibility_20261007": generate_cases(
            FEASIBILITY_NUM_CASES,
            FEASIBILITY_MASTER_SEED,
            FORMAL_TARGETS_PER_CASE,
        ),
        "reader_calibration_20261009": generate_cases(
            CALIBRATION_NUM_CASES,
            CALIBRATION_MASTER_SEED,
            FORMAL_TARGETS_PER_CASE,
        ),
    }

    historical_signatures = {
        batch_name: {
            initial_fact_signature(case)
            for case in cases
        }
        for batch_name, cases in historical_batches.items()
    }

    layer2_cases: List[dict] = []
    overlap_by_batch = Counter()

    for case_index, base_case in enumerate(base_cases):
        case = copy.deepcopy(base_case)
        case["source_generator_case_id"] = str(case["case_id"])
        case["case_id"] = "layer2_formal_case_{:04d}".format(
            case_index
        )

        g1 = build_grouped_memory(case, 1)
        g3 = build_grouped_memory(case, 3)
        case["layer2_representations"] = {
            "g1": g1,
            "g3": g3,
        }
        case["layer2_design_invariants"] = {
            "fact_count": 6,
            "g1_unit_count": 6,
            "g1_facts_per_unit": 1,
            "g3_unit_count": 2,
            "g3_facts_per_unit": 3,
            "same_shared_fact_order": True,
            "only_endpoint_groupings": True,
            "monotonicity_claim_allowed": False,
            "fresh_prospective_formal_batch": True,
        }
        case["initial_fact_signature"] = initial_fact_signature(case)

        for batch_name, signatures in historical_signatures.items():
            if case["initial_fact_signature"] in signatures:
                overlap_by_batch[batch_name] += 1

        case_without_hash = dict(case)
        case_without_hash.pop("case_sha256", None)
        case["case_sha256"] = sha256_json(case_without_hash)
        layer2_cases.append(case)

    signatures = [
        str(case["initial_fact_signature"])
        for case in layer2_cases
    ]
    duplicate_new_case_count = len(signatures) - len(set(signatures))

    overlap_report = {
        "new_case_count": len(layer2_cases),
        "duplicate_new_initial_fact_sets": duplicate_new_case_count,
        "historical_overlap_counts": {
            name: int(overlap_by_batch.get(name, 0))
            for name in historical_batches
        },
        "historical_batches_checked": {
            "baseline": {
                "master_seed": BASELINE_MASTER_SEED,
                "num_cases": BASELINE_NUM_CASES,
            },
            "unit_load": {
                "master_seed": UNIT_LOAD_MASTER_SEED,
                "num_cases": UNIT_LOAD_NUM_CASES,
            },
            "zero_call_feasibility": {
                "master_seed": FEASIBILITY_MASTER_SEED,
                "num_cases": FEASIBILITY_NUM_CASES,
            },
            "reader_calibration": {
                "master_seed": CALIBRATION_MASTER_SEED,
                "num_cases": CALIBRATION_NUM_CASES,
            },
        },
    }

    if duplicate_new_case_count != 0:
        raise RuntimeError(
            "New formal batch contains duplicate initial full fact sets."
        )
    if any(overlap_by_batch.values()):
        raise RuntimeError(
            "New formal batch overlaps a historical/calibration batch: {}"
            .format(dict(overlap_by_batch))
        )

    return layer2_cases, overlap_report


# ============================================================================
# Direct Profile Extraction task construction
# ============================================================================

def subject_schema(
    attributes: Sequence[str],
) -> Tuple[str, str, str]:
    attr_set = set(str(x) for x in attributes)
    for schema in ATTRIBUTES_PER_SUBJECT:
        if attr_set == set(schema):
            return tuple(schema)
    raise ValueError(
        "Unexpected subject attribute set: {}".format(
            sorted(attr_set)
        )
    )


def ordered_values(
    values: Mapping[str, Any],
    attributes: Sequence[str],
) -> dict:
    return {
        str(attribute): str(values[str(attribute)])
        for attribute in attributes
    }


def replace_target_sentence(
    memory: Sequence[Mapping[str, Any]],
    update: Mapping[str, Any],
) -> List[dict]:
    old_fact = update["old_fact"]
    new_fact = update["new_fact"]
    old_sentence = str(old_fact["current_text"])
    new_sentence = str(new_fact["current_text"])

    result = public_memory(memory)
    count = 0
    for unit in result:
        occurrences = str(unit["text"]).count(old_sentence)
        count += occurrences
        if occurrences:
            unit["text"] = str(unit["text"]).replace(
                old_sentence,
                new_sentence,
                1,
            )

    if count != 1:
        raise RuntimeError(
            "Oracle replacement expected one old target sentence, got {}."
            .format(count)
        )
    return result


def build_extraction_task(
    case: Mapping[str, Any],
    update: Mapping[str, Any],
    task_index: int,
) -> dict:
    task_id = "{}__{}".format(
        case["case_id"],
        update["update_id"],
    )

    subject = str(update["new_fact"]["subject"])
    target_attribute = str(update["new_fact"]["attribute"])
    target_old_value = str(update["old_fact"]["value"])
    target_new_value = str(update["new_fact"]["value"])

    expected_subject_facts = [
        dict(fact)
        for fact in update["expected_current_facts"]
        if str(fact["subject"]) == subject
    ]
    if len(expected_subject_facts) != 3:
        raise RuntimeError(
            "{} does not have exactly three target-subject facts.".format(
                task_id
            )
        )

    raw_correct_values = {
        str(fact["attribute"]): str(fact["value"])
        for fact in expected_subject_facts
    }
    attributes = subject_schema(list(raw_correct_values.keys()))
    correct_values = ordered_values(
        raw_correct_values,
        attributes,
    )

    if target_attribute not in attributes:
        raise RuntimeError(
            "{} target attribute is not in subject schema.".format(
                task_id
            )
        )

    protected_attributes = [
        attribute
        for attribute in attributes
        if attribute != target_attribute
    ]
    if len(protected_attributes) != 2:
        raise RuntimeError(
            "{} must have exactly two protected profile attributes."
            .format(task_id)
        )

    oracle_expected_values = dict(correct_values)
    stale_expected_values = dict(correct_values)
    stale_expected_values[target_attribute] = target_old_value

    if oracle_expected_values[target_attribute] != target_new_value:
        raise RuntimeError(
            "{} oracle target does not equal target_new_value.".format(
                task_id
            )
        )

    differing = [
        attribute
        for attribute in attributes
        if oracle_expected_values[attribute]
        != stale_expected_values[attribute]
    ]
    if differing != [target_attribute]:
        raise RuntimeError(
            "{} oracle/stale profiles must differ only on target; got {}."
            .format(task_id, differing)
        )

    memory_labels = {
        attribute: str(ATTRIBUTE_SPECS[attribute]["label"])
        for attribute in attributes
    }
    no_memory_expected_values = {
        attribute: UNKNOWN_VALUE
        for attribute in attributes
    }

    checks = {
        "three_unique_attributes": (
            len(attributes) == 3
            and len(set(attributes)) == 3
        ),
        "target_in_schema": target_attribute in attributes,
        "two_protected_attributes": len(protected_attributes) == 2,
        "oracle_target_is_new": (
            oracle_expected_values[target_attribute]
            == target_new_value
        ),
        "stale_target_is_old": (
            stale_expected_values[target_attribute]
            == target_old_value
        ),
        "oracle_stale_differ_only_on_target": (
            differing == [target_attribute]
        ),
        "all_memory_labels_defined": all(
            attribute in ATTRIBUTE_SPECS
            and str(ATTRIBUTE_SPECS[attribute]["label"])
            for attribute in attributes
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "{} failed extraction-task checks: {}".format(
                task_id,
                {
                    key: value
                    for key, value in checks.items()
                    if not value
                },
            )
        )

    return {
        "task_id": task_id,
        "task_index": int(task_index),
        "case_id": str(case["case_id"]),
        "case_sha256": str(case["case_sha256"]),
        "update_id": str(update["update_id"]),
        "subject": subject,
        "required_attributes": list(attributes),
        "memory_labels": memory_labels,
        "target_attribute": target_attribute,
        "target_old_value": target_old_value,
        "target_new_value": target_new_value,
        "protected_attributes": protected_attributes,
        "oracle_expected_values": oracle_expected_values,
        "stale_expected_values": stale_expected_values,
        "no_memory_expected_values": no_memory_expected_values,
        "checks": checks,
        "manual_annotation_required": False,
    }


def build_extraction_tasks(
    cases: Sequence[Mapping[str, Any]],
) -> Tuple[List[dict], dict]:
    tasks: List[dict] = []
    task_index = 0

    for case in cases:
        for update in case["updates"]:
            tasks.append(
                build_extraction_task(
                    case,
                    update,
                    task_index,
                )
            )
            task_index += 1

    if len(tasks) != EXPECTED_TARGET_INSTANCES:
        raise RuntimeError(
            "Expected {} extraction tasks, got {}.".format(
                EXPECTED_TARGET_INSTANCES,
                len(tasks),
            )
        )

    schema_counts = Counter(
        tuple(task["required_attributes"])
        for task in tasks
    )
    target_attribute_counts = Counter(
        str(task["target_attribute"])
        for task in tasks
    )

    task_report = {
        "task_count": len(tasks),
        "all_structural_checks_pass": all(
            all(bool(value) for value in task["checks"].values())
            for task in tasks
        ),
        "schema_counts": {
            "|".join(schema): int(count)
            for schema, count in sorted(schema_counts.items())
        },
        "target_attribute_counts": dict(
            sorted(target_attribute_counts.items())
        ),
        "candidate_profiles_used": False,
        "direct_three_field_extraction": True,
        "manual_annotation_required": False,
    }
    return tasks, task_report


# ============================================================================
# Updater prompts, schemas, and execution plan
# ============================================================================

def condition_spec(condition: str) -> dict:
    mapping = {
        CONDITION_L1_SELECT: {
            "grouping": "g1",
            "unit_load": 1,
            "interface": "select",
            "expected_action_type": ACTION_REWRITE_UNIT,
        },
        CONDITION_L1_PATCH: {
            "grouping": "g1",
            "unit_load": 1,
            "interface": "patch",
            "expected_action_type": ACTION_EXACT_PATCH,
        },
        CONDITION_L3_SELECT: {
            "grouping": "g3",
            "unit_load": 3,
            "interface": "select",
            "expected_action_type": ACTION_REWRITE_UNIT,
        },
        CONDITION_L3_PATCH: {
            "grouping": "g3",
            "unit_load": 3,
            "interface": "patch",
            "expected_action_type": ACTION_EXACT_PATCH,
        },
    }
    if condition not in mapping:
        raise ValueError("Unknown condition: {}".format(condition))
    result = dict(mapping[condition])
    result["condition"] = condition
    return result


def action_instruction(expected_action_type: str) -> str:
    if expected_action_type == ACTION_REWRITE_UNIT:
        return (
            "Select exactly one existing memory unit and return the complete "
            "replacement text for that unit. Do not return or rewrite any "
            "other unit. The output action must be rewrite_unit."
        )
    if expected_action_type == ACTION_EXACT_PATCH:
        return (
            "Select one existing memory unit and provide one exact local "
            "string replacement. old_span must occur exactly once in the "
            "selected unit before editing. The output action must be "
            "exact_patch."
        )
    raise ValueError(
        "Unsupported Layer-2 action type: {}".format(
            expected_action_type
        )
    )


def action_schema(
    expected_action_type: str,
    memory: Sequence[Mapping[str, Any]],
) -> dict:
    memory_ids = [
        str(unit["memory_id"])
        for unit in memory
    ]

    if expected_action_type == ACTION_REWRITE_UNIT:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "memory_id", "text"],
            "properties": {
                "action": {
                    "type": "string",
                    "const": ACTION_REWRITE_UNIT,
                },
                "memory_id": {
                    "type": "string",
                    "enum": memory_ids,
                },
                "text": {
                    "type": "string",
                },
            },
        }

    if expected_action_type == ACTION_EXACT_PATCH:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "action",
                "memory_id",
                "old_span",
                "new_span",
            ],
            "properties": {
                "action": {
                    "type": "string",
                    "const": ACTION_EXACT_PATCH,
                },
                "memory_id": {
                    "type": "string",
                    "enum": memory_ids,
                },
                "old_span": {
                    "type": "string",
                    "minLength": 1,
                },
                "new_span": {
                    "type": "string",
                },
            },
        }

    raise ValueError("Unsupported action type.")


def build_updater_user_prompt(
    memory: Sequence[Mapping[str, Any]],
    evidence_text: str,
    expected_action_type: str,
) -> str:
    memory_json = json.dumps(
        public_memory(memory),
        ensure_ascii=False,
        indent=2,
    )
    return (
        "MEMORY STATE:\n"
        + memory_json
        + "\n\nNEW EVIDENCE:\n"
        + str(evidence_text)
        + "\n\nREQUIRED EDIT SCOPE:\n"
        + action_instruction(expected_action_type)
        + "\n\nReturn only the JSON action."
    )


def rotated_conditions(
    target_instance_index: int,
    model_index: int,
) -> Tuple[str, ...]:
    # 100 target instances / 4 positions = exactly 25 occurrences per
    # condition per position for every model.
    offset = (
        int(target_instance_index)
        + int(model_index)
    ) % len(CONDITIONS)
    values = list(CONDITIONS)
    return tuple(values[offset:] + values[:offset])


def build_updater_execution_plan(
    cases: Sequence[Mapping[str, Any]],
) -> Tuple[List[dict], dict]:
    plan = []
    target_instance_index = 0

    for case in cases:
        for update in case["updates"]:
            for model_index, model in enumerate(MODELS):
                condition_order = rotated_conditions(
                    target_instance_index,
                    model_index,
                )
                for position, condition in enumerate(
                    condition_order,
                    start=1,
                ):
                    spec = condition_spec(condition)
                    memory = public_memory(
                        case["layer2_representations"][
                            spec["grouping"]
                        ]
                    )
                    schema = action_schema(
                        spec["expected_action_type"],
                        memory,
                    )
                    user_prompt = build_updater_user_prompt(
                        memory,
                        update["evidence_text"],
                        spec["expected_action_type"],
                    )

                    generation_config = {
                        "model": str(model),
                        "temperature": UPDATER_TEMPERATURE,
                        "seed": UPDATER_SEED,
                        "max_output_tokens": UPDATER_MAX_OUTPUT_TOKENS,
                        "context_window": UPDATER_CONTEXT_WINDOW,
                    }
                    updater_input_payload = {
                        "system_prompt": UPDATER_SYSTEM_PROMPT,
                        "user_prompt": user_prompt,
                        "generation_config": generation_config,
                        "response_schema": schema,
                    }

                    plan.append(
                        {
                            "planned_call_index": len(plan),
                            "case_id": str(case["case_id"]),
                            "case_sha256": str(case["case_sha256"]),
                            "update_id": str(update["update_id"]),
                            "task_id": "{}__{}".format(
                                case["case_id"],
                                update["update_id"],
                            ),
                            "target_instance_index": (
                                target_instance_index
                            ),
                            "model_index": model_index,
                            "model": model,
                            "condition": condition,
                            "grouping": spec["grouping"],
                            "unit_load": spec["unit_load"],
                            "interface": spec["interface"],
                            "expected_action_type": spec[
                                "expected_action_type"
                            ],
                            "condition_execution_position": position,
                            "memory": memory,
                            "memory_sha256": sha256_text(
                                canonical_json(memory)
                            ),
                            "evidence_text": str(
                                update["evidence_text"]
                            ),
                            "user_prompt": user_prompt,
                            "response_schema": schema,
                            "generation_config": generation_config,
                            "updater_system_prompt_sha256": sha256_text(
                                UPDATER_SYSTEM_PROMPT
                            ),
                            "user_prompt_sha256": sha256_text(
                                user_prompt
                            ),
                            "response_schema_sha256": sha256_text(
                                canonical_json(schema)
                            ),
                            "updater_input_sha256": sha256_text(
                                canonical_json(
                                    updater_input_payload
                                )
                            ),
                        }
                    )

            target_instance_index += 1

    if len(plan) != EXPECTED_UPDATER_CALLS:
        raise RuntimeError(
            "Expected {} updater calls, got {}.".format(
                EXPECTED_UPDATER_CALLS,
                len(plan),
            )
        )

    identity_keys = {
        (
            row["case_id"],
            row["update_id"],
            row["model"],
            row["condition"],
        )
        for row in plan
    }
    if len(identity_keys) != EXPECTED_UPDATER_CALLS:
        raise RuntimeError(
            "Updater execution plan has duplicate result identities."
        )

    balance = {}
    for model in MODELS:
        balance[model] = {}
        for condition in CONDITIONS:
            position_counts = {
                str(position): sum(
                    1
                    for row in plan
                    if row["model"] == model
                    and row["condition"] == condition
                    and row["condition_execution_position"] == position
                )
                for position in range(1, 5)
            }
            if set(position_counts.values()) != {25}:
                raise RuntimeError(
                    "Execution-order imbalance for {} / {}: {}".format(
                        model,
                        condition,
                        position_counts,
                    )
                )
            balance[model][condition] = position_counts

    report = {
        "row_count": len(plan),
        "unique_result_identity_count": len(identity_keys),
        "condition_position_balance": balance,
        "each_condition_per_position_per_model": 25,
    }
    return plan, report


# ============================================================================
# Fixed direct-extraction reader control plan
# ============================================================================

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


def reader_generation_config() -> dict:
    return {
        "model": READER_MODEL,
        "temperature": READER_TEMPERATURE,
        "seed": READER_SEED,
        "max_output_tokens": READER_MAX_OUTPUT_TOKENS,
        "context_window": READER_CONTEXT_WINDOW,
    }


def build_reader_user_prompt(
    *,
    memory: Optional[Sequence[Mapping[str, Any]]],
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


def build_reader_control_plan(
    cases: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
) -> Tuple[List[dict], dict]:
    case_by_id = {
        str(case["case_id"]): case
        for case in cases
    }
    task_by_id = {
        str(task["task_id"]): task
        for task in tasks
    }

    rows: List[dict] = []
    generation_config = reader_generation_config()

    for task_index in range(EXPECTED_TARGET_INSTANCES):
        task = tasks[task_index]
        case = case_by_id[str(task["case_id"])]
        update = next(
            update
            for update in case["updates"]
            if str(update["update_id"]) == str(task["update_id"])
        )

        attributes = [
            str(attribute)
            for attribute in task["required_attributes"]
        ]
        labels = {
            str(key): str(value)
            for key, value in task["memory_labels"].items()
        }
        response_schema = response_schema_for_attributes(attributes)

        for arm, grouping in (
            ("oracle", "g1"),
            ("oracle", "g3"),
            ("stale", "g1"),
            ("stale", "g3"),
        ):
            stale_memory = public_memory(
                case["layer2_representations"][grouping]
            )
            memory = (
                replace_target_sentence(stale_memory, update)
                if arm == "oracle"
                else stale_memory
            )
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
                        task["task_id"],
                        arm,
                        grouping,
                    ),
                    "task_id": str(task["task_id"]),
                    "case_id": str(task["case_id"]),
                    "update_id": str(task["update_id"]),
                    "arm": arm,
                    "grouping": grouping,
                    "subject": str(task["subject"]),
                    "required_attributes": list(attributes),
                    "memory_labels": dict(labels),
                    "memory": memory,
                    "expected_values": expected_values,
                    "user_prompt": user_prompt,
                    "response_schema": response_schema,
                    "generation_config": dict(
                        generation_config
                    ),
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

        # One no-memory row per target; shared across groupings.
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
                    task["task_id"]
                ),
                "task_id": str(task["task_id"]),
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
                "generation_config": dict(generation_config),
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

    if len(rows) != EXPECTED_READER_CONTROL_ROWS:
        raise RuntimeError(
            "Expected {} reader controls, got {}.".format(
                EXPECTED_READER_CONTROL_ROWS,
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
            "Unexpected reader-control counts: {}".format(
                dict(counts)
            )
        )

    # Verify exact-input caching cannot alter scoring targets.
    by_input: Dict[str, List[dict]] = {}
    for row in rows:
        by_input.setdefault(
            str(row["reader_input_sha256"]),
            [],
        ).append(row)

    for input_hash, group in by_input.items():
        first = group[0]
        for row in group[1:]:
            if row["user_prompt"] != first["user_prompt"]:
                raise RuntimeError(
                    "Same reader-input hash maps to different prompts."
                )
            if canonical_json(row["response_schema"]) != canonical_json(
                first["response_schema"]
            ):
                raise RuntimeError(
                    "Same reader-input hash maps to different schemas."
                )
            if canonical_json(row["expected_values"]) != canonical_json(
                first["expected_values"]
            ):
                raise RuntimeError(
                    "Same reader-input hash maps to different expected values."
                )

    report = {
        "row_count": len(rows),
        "arm_grouping_counts": {
            "{}__{}".format(arm, grouping): int(count)
            for (arm, grouping), count in sorted(counts.items())
        },
        "unique_reader_inputs_before_runtime_cache": len(by_input),
        "exact_input_caching_allowed": True,
        "candidate_profiles_used": False,
        "direct_three_field_extraction": True,
        "no_memory_expected_policy": UNKNOWN_VALUE,
    }
    return rows, report


def verify_reader_calibration() -> dict:
    # Calibration qualifies the interface; formal cases are still generated fresh.
    required = (
        CALIBRATION_PREFLIGHT_PATH,
        CALIBRATION_FREEZE_PATH,
        CALIBRATION_RESULTS_PATH,
        CALIBRATION_SUMMARY_PATH,
        CORE_PATH,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                "Missing required reader-calibration provenance: {}"
                .format(path)
            )

    calibration_preflight = read_json(
        CALIBRATION_PREFLIGHT_PATH
    )
    calibration_freeze = read_json(
        CALIBRATION_FREEZE_PATH
    )
    calibration_summary = read_json(
        CALIBRATION_SUMMARY_PATH
    )

    if calibration_preflight.get("status") != "PASS":
        raise RuntimeError(
            "Reader calibration preflight is not PASS."
        )

    if calibration_freeze.get("status") != (
        "CALIBRATION_PROTOCOL_FROZEN"
    ):
        raise RuntimeError(
            "Reader calibration protocol is not frozen."
        )

    if calibration_freeze.get("protocol_version") != (
        EXPECTED_CALIBRATION_PROTOCOL
    ):
        raise RuntimeError(
            "Unexpected reader calibration protocol: {!r}".format(
                calibration_freeze.get("protocol_version")
            )
        )

    if calibration_summary.get("status") != (
        "READER_CALIBRATION_GATE_PASS"
    ):
        raise RuntimeError(
            "Reader calibration did not pass its frozen gate."
        )

    expected_reader = {
        "model": READER_MODEL,
        "temperature": READER_TEMPERATURE,
        "seed": READER_SEED,
        "max_output_tokens": READER_MAX_OUTPUT_TOKENS,
        "context_window": READER_CONTEXT_WINDOW,
        "exact_input_caching": True,
    }
    if calibration_freeze.get("reader") != expected_reader:
        raise RuntimeError(
            "Formal reader config differs from calibrated reader."
        )

    if str(calibration_freeze.get("reader_system_prompt")) != (
        READER_SYSTEM_PROMPT
    ):
        raise RuntimeError(
            "Formal reader system prompt differs from calibrated prompt."
        )

    if calibration_freeze.get("reader_gates") != READER_GATES:
        raise RuntimeError(
            "Formal reader gates differ from calibrated gates."
        )

    if str(calibration_summary.get("reader_model")) != READER_MODEL:
        raise RuntimeError(
            "Calibration summary reader model mismatch."
        )

    if str(
        calibration_summary.get("freeze_manifest_sha256", "")
    ) != sha256_file(CALIBRATION_FREEZE_PATH):
        raise RuntimeError(
            "Calibration summary freeze-manifest hash mismatch."
        )

    if str(
        calibration_summary.get("calibration_preflight_sha256", "")
    ) != sha256_file(CALIBRATION_PREFLIGHT_PATH):
        raise RuntimeError(
            "Calibration summary preflight hash mismatch."
        )

    if str(
        calibration_summary.get("results_sha256", "")
    ) != sha256_file(CALIBRATION_RESULTS_PATH):
        raise RuntimeError(
            "Calibration summary results hash mismatch."
        )

    calibrated_core_hash = str(
        calibration_freeze.get(
            "source_file_sha256",
            {},
        ).get(
            "representation_edit_scope_core.py",
            "",
        )
    )
    if calibrated_core_hash and calibrated_core_hash != sha256_file(
        CORE_PATH
    ):
        raise RuntimeError(
            "Current core differs from the core used for reader calibration."
        )

    return {
        "status": str(calibration_summary["status"]),
        "protocol_version": str(
            calibration_freeze["protocol_version"]
        ),
        "reader_model": READER_MODEL,
        "calibration_preflight_sha256": sha256_file(
            CALIBRATION_PREFLIGHT_PATH
        ),
        "calibration_freeze_sha256": sha256_file(
            CALIBRATION_FREEZE_PATH
        ),
        "calibration_results_sha256": sha256_file(
            CALIBRATION_RESULTS_PATH
        ),
        "calibration_summary_sha256": sha256_file(
            CALIBRATION_SUMMARY_PATH
        ),
        "observed": dict(
            calibration_summary.get("observed", {})
        ),
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    paths.require_output_stage(paths.PROFILE, Path(__file__))
    print("=" * 78)
    print("PROFILE EXTRACTION LAYER-2 FORMAL BUILD / PREFLIGHT")
    print("=" * 78)
    print("Mode:              ZERO MODEL CALLS")
    print("Formal cases:      {}".format(FORMAL_NUM_CASES))
    print("Targets/case:      {}".format(FORMAL_TARGETS_PER_CASE))
    print("Target instances:  {}".format(EXPECTED_TARGET_INSTANCES))
    print("Updater calls:     {}".format(EXPECTED_UPDATER_CALLS))
    print(
        "Reader controls:   {}".format(
            EXPECTED_READER_CONTROL_ROWS
        )
    )
    print(
        "Actual readers:    {}".format(
            EXPECTED_ACTUAL_READER_ROWS
        )
    )
    print(
        "Max total calls:   {}".format(
            EXPECTED_MAX_MANDATORY_CALLS
        )
    )
    print("Master seed:       {}".format(FORMAL_MASTER_SEED))
    print("Reader interface:  direct 3-field extraction")
    print()

    started_results = [
        path
        for path in RESERVED_RESULT_PATHS
        if path.exists()
    ]
    if started_results:
        raise RuntimeError(
            "Formal result artifacts already exist. The design is frozen and "
            "must not be rebuilt:\n{}".format(
                "\n".join(
                    "  {}".format(path)
                    for path in started_results
                )
            )
        )

    if not CORE_PATH.exists():
        raise FileNotFoundError(
            "Missing current core: {}".format(CORE_PATH)
        )

    print("[1/6] Verifying passed Profile Extraction reader calibration...")
    calibration_provenance = verify_reader_calibration()
    print(
        "  PASS | calibrated reader unchanged | calibration gate = PASS"
    )

    print("[2/6] Building 50 fresh cases and checking all prior overlap...")
    cases, overlap_report = build_layer2_cases()
    print(
        "  PASS | 50 unique new fact sets | historical/calibration "
        "overlaps = 0"
    )

    print("[3/6] Building 100 direct Profile Extraction tasks...")
    tasks, task_report = build_extraction_tasks(cases)
    print(
        "  PASS | 100 tasks | 3 required attributes each | "
        "oracle/stale differ only on target"
    )

    print("[4/6] Building balanced 1,200-call updater plan...")
    updater_plan, updater_plan_report = (
        build_updater_execution_plan(cases)
    )
    print(
        "  PASS | 1,200 rows | each condition appears 25x "
        "in each execution position per model"
    )

    print("[5/6] Building 500-row formal reader-control plan...")
    reader_controls, reader_control_report = (
        build_reader_control_plan(cases, tasks)
    )
    print(
        "  PASS | 200 oracle + 200 stale + 100 no-memory | "
        "direct extraction only"
    )

    file_status: Dict[str, str] = {}
    file_status["formal_cases.jsonl"] = write_or_verify_jsonl(
        CASES_PATH,
        cases,
    )
    file_status["extraction_tasks.jsonl"] = (
        write_or_verify_jsonl(
            TASKS_PATH,
            tasks,
        )
    )
    file_status["updater_execution_plan.jsonl"] = (
        write_or_verify_jsonl(
            UPDATER_PLAN_PATH,
            updater_plan,
        )
    )
    file_status["reader_control_plan.jsonl"] = (
        write_or_verify_jsonl(
            READER_CONTROL_PLAN_PATH,
            reader_controls,
        )
    )

    print("[6/6] Freezing hashes, estimands, gates, and provenance...")

    frozen_file_hashes = {
        "formal_cases.jsonl": sha256_file(CASES_PATH),
        "extraction_tasks.jsonl": sha256_file(TASKS_PATH),
        "updater_execution_plan.jsonl": sha256_file(
            UPDATER_PLAN_PATH
        ),
        "reader_control_plan.jsonl": sha256_file(
            READER_CONTROL_PLAN_PATH
        ),
    }

    builder_path = Path(__file__).resolve()

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "FORMAL_PROTOCOL_FROZEN",
        "zero_model_call_build": True,
        "fresh_prospective_formal_batch": True,
        "formal_master_seed": FORMAL_MASTER_SEED,
        "formal_num_cases": FORMAL_NUM_CASES,
        "targets_per_case": FORMAL_TARGETS_PER_CASE,
        "target_instances": EXPECTED_TARGET_INSTANCES,
        "cluster_unit": "case",
        "updater_models": list(MODELS),
        "conditions": list(CONDITIONS),
        "grouping_scope": ["g1", "g3"],
        "monotonicity_claim_allowed": False,
        "updater_generation": {
            "temperature": UPDATER_TEMPERATURE,
            "seed": UPDATER_SEED,
            "max_output_tokens": UPDATER_MAX_OUTPUT_TOKENS,
            "context_window": UPDATER_CONTEXT_WINDOW,
        },
        "reader_interface": {
            "name": "direct_three_field_profile_extraction",
            "candidate_profiles_used": False,
            "unknown_policy": UNKNOWN_VALUE,
            "functional_task_pass_definition": (
                "Reader-extracted complete target-subject profile exactly "
                "matches the frozen oracle post-update profile."
            ),
            "local_functional_scope": (
                "Only the target subject's three profile attributes are "
                "scored; corruption outside that subject may remain "
                "functionally irrelevant to this task."
            ),
        },
        "reader": {
            "model": READER_MODEL,
            "temperature": READER_TEMPERATURE,
            "seed": READER_SEED,
            "max_output_tokens": READER_MAX_OUTPUT_TOKENS,
            "context_window": READER_CONTEXT_WINDOW,
            "exact_input_caching": True,
        },
        "reader_gates": dict(READER_GATES),
        "reader_calibration_provenance": (
            calibration_provenance
        ),
        "call_budget": {
            "updater_calls": EXPECTED_UPDATER_CALLS,
            "reader_control_logical_rows": (
                EXPECTED_READER_CONTROL_ROWS
            ),
            "actual_reader_logical_rows": (
                EXPECTED_ACTUAL_READER_ROWS
            ),
            "max_mandatory_model_calls_without_cache": (
                EXPECTED_MAX_MANDATORY_CALLS
            ),
            "absolute_contingency_ceiling": (
                EXPECTED_MAX_MANDATORY_CALLS
            ),
            "note": (
                "Exact reader-input caching may reduce physical reader "
                "calls, but all logical rows remain in analysis."
            ),
        },
        "updater_system_prompt": UPDATER_SYSTEM_PROMPT,
        "updater_system_prompt_sha256": sha256_text(
            UPDATER_SYSTEM_PROMPT
        ),
        "reader_system_prompt": READER_SYSTEM_PROMPT,
        "reader_system_prompt_sha256": sha256_text(
            READER_SYSTEM_PROMPT
        ),
        "analysis_commitments": {
            "primary_functional_contrast": (
                "TaskAcc(g1,Patch)-TaskAcc(g3,Patch)"
            ),
            "matching_state_contrast": (
                "StrictAcc(g1,Patch)-StrictAcc(g3,Patch)"
            ),
            "supporting_select_contrast": (
                "TaskAcc(g1,Select)-TaskAcc(g3,Select)"
            ),
            "task_accuracy_definition": (
                "Exact match of all three extracted target-subject current "
                "attributes to frozen oracle post-update values."
            ),
            "strict_state_success_definition": (
                "Use the pre-existing strict Layer-1 evaluator without "
                "rescoring or relaxing its contract."
            ),
            "joint_strict_task_table": True,
            "models_analyzed_separately": True,
            "whole_case_bootstrap": True,
            "bootstrap_replicates": 20000,
            "failure_class_analysis": "descriptive",
            "natural_rate_failures_only": True,
            "reader_calibration_cases_pooled_into_formal": False,
            "old_profile_selection_results_pooled_into_formal": False,
            "no_monotonicity_claim_from_g1_vs_g3_endpoints": True,
        },
        "anti_rescue_rules": [
            "Do not lower frozen reader gates after observing formal controls.",
            "Do not delete or replace difficult formal cases.",
            "Do not modify reader prompt, reader model, or task schema within "
            "this formal protocol.",
            "Do not tune updater prompts or conditions to manufacture failure.",
            "Do not start the 1,200 updater calls unless the fresh formal "
            "reader-control stage passes.",
            "Do not reuse the prior 20261009 reader-calibration cases as "
            "formal updater cases.",
        ],
        "overlap_report": overlap_report,
        "task_report": task_report,
        "updater_plan_report": updater_plan_report,
        "reader_control_report": reader_control_report,
        "frozen_file_sha256": frozen_file_hashes,
        "code_sha256": {
            "representation_edit_scope_core.py": sha256_file(
                CORE_PATH
            ),
            "build_profile_extraction_formal.py": sha256_file(
                builder_path
            ),
        },
    }

    preflight = {
        "status": "PASS",
        "model_calls": 0,
        "manual_annotation_required": False,
        "fresh_prospective_formal_batch": True,
        "reader_calibration_pass_verified": True,
        "reader_interface": "direct_three_field_extraction",
        "candidate_profiles_used": False,
        "cases": len(cases),
        "tasks": len(tasks),
        "updater_plan_rows": len(updater_plan),
        "reader_control_rows": len(reader_controls),
        "historical_and_calibration_overlap_pass": all(
            value == 0
            for value in overlap_report[
                "historical_overlap_counts"
            ].values()
        ),
        "new_case_uniqueness_pass": (
            overlap_report[
                "duplicate_new_initial_fact_sets"
            ]
            == 0
        ),
        "task_structural_gate_pass": bool(
            task_report["all_structural_checks_pass"]
        ),
        "updater_execution_balance_pass": True,
        "reader_control_design_pass": True,
        "expected_updater_calls": EXPECTED_UPDATER_CALLS,
        "expected_reader_control_rows": (
            EXPECTED_READER_CONTROL_ROWS
        ),
        "expected_actual_reader_rows": (
            EXPECTED_ACTUAL_READER_ROWS
        ),
        "max_mandatory_call_budget": (
            EXPECTED_MAX_MANDATORY_CALLS
        ),
        "reader_gates": dict(READER_GATES),
        "file_status": file_status,
        "frozen_file_sha256": frozen_file_hashes,
        "next_step": (
            "Run the frozen 500-row fresh formal reader-control stage. "
            "Do not start the 1,200 updater calls until it passes."
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
        "  PASS | cases/tasks/plans hashed | reader calibration linked | "
        "formal protocol frozen"
    )
    print()
    print("=" * 78)
    print("PROFILE EXTRACTION LAYER-2 FORMAL PREFLIGHT PASS")
    print("=" * 78)
    print("Cases:                    {}/50".format(len(cases)))
    print("Tasks:                    {}/100".format(len(tasks)))
    print(
        "Updater plan:             {}/1200".format(
            len(updater_plan)
        )
    )
    print(
        "Reader controls:          {}/500".format(
            len(reader_controls)
        )
    )
    print(
        "Unique reader inputs:     {}".format(
            reader_control_report[
                "unique_reader_inputs_before_runtime_cache"
            ]
        )
    )
    print(
        "Historical/calib overlap: {}".format(
            overlap_report["historical_overlap_counts"]
        )
    )
    print("Reader calibration:       PASS VERIFIED")
    print("Candidate profiles used:  NO")
    print("Model calls made:         0")
    print("Formal protocol frozen:   YES")
    print("Updater calls allowed:    NO — reader controls must pass first")
    print()
    print("Files:")
    print(
        "  {} [{}]".format(
            CASES_PATH,
            file_status["formal_cases.jsonl"],
        )
    )
    print(
        "  {} [{}]".format(
            TASKS_PATH,
            file_status["extraction_tasks.jsonl"],
        )
    )
    print(
        "  {} [{}]".format(
            UPDATER_PLAN_PATH,
            file_status["updater_execution_plan.jsonl"],
        )
    )
    print(
        "  {} [{}]".format(
            READER_CONTROL_PLAN_PATH,
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
        "NEXT: run the fresh formal 500-row reader-control stage."
    )
    print(
        "Do not start the 1,200 updater calls until those controls PASS."
    )


if __name__ == "__main__":
    main()
