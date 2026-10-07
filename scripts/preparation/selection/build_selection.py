#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze the candidate-selection design and execution plans without inference."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "atomic_memory").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atomic_memory import paths

from atomic_memory.core import (  # noqa: E402
    ACTION_EXACT_PATCH,
    ACTION_REWRITE_UNIT,
    ATTRIBUTE_SPECS,
    ATTRIBUTES_PER_SUBJECT,
    CURRENT_TEMPLATE,
    generate_cases,
    sha256_json,
)

# ============================================================================
# Frozen Layer-2 design
# ============================================================================

PROTOCOL_VERSION = "profile-selection-layer2-formal-v1"

FORMAL_MASTER_SEED = 20261009
FORMAL_NUM_CASES = 50
FORMAL_TARGETS_PER_CASE = 2

# Historical batches used only for ZERO-call overlap checks.
BASELINE_MASTER_SEED = 20261001
BASELINE_NUM_CASES = 20
UNIT_LOAD_MASTER_SEED = 20261005
UNIT_LOAD_NUM_CASES = 20

# The zero-call feasibility audit used these generated cases. They had no model
# outputs, but we also avoid reusing their initial full fact sets.
FEASIBILITY_MASTER_SEED = 20261007
FEASIBILITY_NUM_CASES = 200

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

OUTPUT_DIR = paths.SELECTION
CASES_PATH = OUTPUT_DIR / "formal_cases.jsonl"
TASKS_PATH = OUTPUT_DIR / "profile_tasks.jsonl"
UPDATER_PLAN_PATH = OUTPUT_DIR / "updater_execution_plan.jsonl"
READER_CONTROL_PLAN_PATH = OUTPUT_DIR / "reader_control_plan.jsonl"
PREFLIGHT_PATH = OUTPUT_DIR / "formal_preflight.json"
FREEZE_PATH = OUTPUT_DIR / "freeze_manifest.json"

# Future result names are reserved here so the builder can refuse any attempt
# to rebuild the design after formal execution has started.
RESERVED_RESULT_PATHS = (
    OUTPUT_DIR / "reader_control_results.jsonl",
    OUTPUT_DIR / "updater_results.jsonl",
    OUTPUT_DIR / "actual_reader_results.jsonl",
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

READER_SYSTEM_PROMPT = """You are a downstream memory reader.

Your job is to select the candidate profile that matches the named person's
CURRENT information in the provided memory state.

Rules:
1. Use the memory state exactly as provided. Do not repair or rewrite it.
2. Match all listed profile attributes for the named person.
3. Do not use stereotypes or outside knowledge.
4. If no memory state is provided, do not assume hidden facts.
5. Return exactly one JSON object with one key: profile_id.
6. Do not output explanations or any other text.
"""

READER_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["profile_id"],
    "properties": {
        "profile_id": {
            "type": "string",
            "enum": ["p{}".format(i) for i in range(8)],
        }
    },
}

# Formal reader gates frozen before any prospective model output.
READER_GATES = {
    "oracle_min_correct_per_grouping": 95,
    "stale_expected_profile_min_per_grouping": 95,
    "no_memory_max_gold_correct": 20,
    "targets_per_grouping_control": 100,
    "no_memory_target_count": 100,
    "chance_accuracy": 0.125,
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

    # Historical semantic signatures.
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
    }
    historical_signatures = {
        batch_name: {
            initial_fact_signature(case)
            for case in cases
        }
        for batch_name, cases in historical_batches.items()
    }

    layer2_cases = []
    overlap_by_batch = Counter()

    for case_index, base_case in enumerate(base_cases):
        case = copy.deepcopy(base_case)
        case["source_generator_case_id"] = str(case["case_id"])
        case["case_id"] = "layer2_case_{:04d}".format(case_index)

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
        }
        case["initial_fact_signature"] = initial_fact_signature(case)

        for batch_name, signatures in historical_signatures.items():
            if case["initial_fact_signature"] in signatures:
                overlap_by_batch[batch_name] += 1

        # Recompute after Layer-2 additions / case-id change.
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
        },
    }

    if duplicate_new_case_count != 0:
        raise RuntimeError(
            "New formal batch contains duplicate initial full fact sets."
        )
    if any(overlap_by_batch.values()):
        raise RuntimeError(
            "New formal batch overlaps a historical generated batch: {}".format(
                dict(overlap_by_batch)
            )
        )

    return layer2_cases, overlap_report


# ============================================================================
# Profile Selection task construction
# ============================================================================

def subject_schema(attributes: Sequence[str]) -> Tuple[str, str, str]:
    attr_set = set(str(x) for x in attributes)
    for schema in ATTRIBUTES_PER_SUBJECT:
        if attr_set == set(schema):
            return tuple(schema)
    raise ValueError(
        "Unexpected subject attribute set: {}".format(
            sorted(attr_set)
        )
    )


def pick_protected_alternative(
    attribute: str,
    correct_value: str,
    *,
    task_id: str,
) -> str:
    values = [
        str(value)
        for value in ATTRIBUTE_SPECS[attribute]["values"]
        if str(value) != str(correct_value)
    ]
    if not values:
        raise ValueError("No alternative value available.")
    index = stable_int(
        "formal_protected_alt",
        task_id,
        attribute,
        correct_value,
    ) % len(values)
    return values[index]


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


def build_profile_task(
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

    correct_values = {
        str(fact["attribute"]): str(fact["value"])
        for fact in expected_subject_facts
    }
    attrs = subject_schema(list(correct_values.keys()))
    protected_attributes = [
        attr
        for attr in attrs
        if attr != target_attribute
    ]
    if len(protected_attributes) != 2:
        raise RuntimeError(
            "{} does not have exactly two protected subject facts.".format(
                task_id
            )
        )

    alternatives = {}
    for attr in attrs:
        if attr == target_attribute:
            alternatives[attr] = target_old_value
        else:
            alternatives[attr] = pick_protected_alternative(
                attr,
                correct_values[attr],
                task_id=task_id,
            )

    raw_candidates = []
    for bits in (
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ):
        values = {
            attr: (
                alternatives[attr]
                if bit
                else correct_values[attr]
            )
            for attr, bit in zip(attrs, bits)
        }
        wrong_attributes = [
            attr
            for attr, bit in zip(attrs, bits)
            if bit
        ]
        if not wrong_attributes:
            role = "gold"
        elif wrong_attributes == [target_attribute]:
            role = "stale_target"
        elif len(wrong_attributes) == 1:
            role = "protected_alt:{}".format(wrong_attributes[0])
        else:
            role = "multi_alt:" + ",".join(wrong_attributes)

        raw_candidates.append(
            {
                "attributes": values,
                "wrong_attributes": wrong_attributes,
                "semantic_role": role,
            }
        )

    gold_raw = [
        row for row in raw_candidates
        if row["semantic_role"] == "gold"
    ]
    if len(gold_raw) != 1:
        raise RuntimeError("Gold candidate construction failed.")

    desired_gold_position = int(task_index) % 8
    non_gold = [
        dict(row)
        for row in raw_candidates
        if row["semantic_role"] != "gold"
    ]
    rng = random.Random(
        stable_int("formal_candidate_order", task_id)
    )
    rng.shuffle(non_gold)

    ordered = []
    iterator = iter(non_gold)
    for position in range(8):
        if position == desired_gold_position:
            ordered.append(dict(gold_raw[0]))
        else:
            ordered.append(next(iterator))

    candidates = []
    for position, row in enumerate(ordered):
        item = dict(row)
        item["profile_id"] = "p{}".format(position)
        candidates.append(item)

    gold_profile_id = next(
        row["profile_id"]
        for row in candidates
        if row["semantic_role"] == "gold"
    )
    stale_profile_id = next(
        row["profile_id"]
        for row in candidates
        if row["semantic_role"] == "stale_target"
    )

    necessity = {}
    for removed_attr in attrs:
        constraints = {
            attr: correct_values[attr]
            for attr in attrs
            if attr != removed_attr
        }
        matches = [
            candidate["profile_id"]
            for candidate in candidates
            if all(
                candidate["attributes"][attr] == value
                for attr, value in constraints.items()
            )
        ]
        necessity[removed_attr] = {
            "match_count_without_constraint": len(matches),
            "matching_profile_ids": matches,
            "necessary": len(matches) == 2,
        }

    value_balance = {}
    for attr in attrs:
        counts = Counter(
            str(candidate["attributes"][attr])
            for candidate in candidates
        )
        value_balance[attr] = {
            "counts": dict(counts),
            "balanced_4_4": sorted(counts.values()) == [4, 4],
        }

    candidate_tuples = {
        tuple(
            candidate["attributes"][attr]
            for attr in attrs
        )
        for candidate in candidates
    }

    checks = {
        "candidate_count_8": len(candidates) == 8,
        "candidate_tuples_unique": len(candidate_tuples) == 8,
        "gold_unique": sum(
            1 for row in candidates
            if row["semantic_role"] == "gold"
        ) == 1,
        "stale_unique": sum(
            1 for row in candidates
            if row["semantic_role"] == "stale_target"
        ) == 1,
        "gold_stale_different": gold_profile_id != stale_profile_id,
        "target_necessary": necessity[target_attribute]["necessary"],
        "protected_1_necessary": necessity[
            protected_attributes[0]
        ]["necessary"],
        "protected_2_necessary": necessity[
            protected_attributes[1]
        ]["necessary"],
        "all_value_pairs_balanced_4_4": all(
            row["balanced_4_4"]
            for row in value_balance.values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "{} failed task checks: {}".format(
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
        "attributes": list(attrs),
        "target_attribute": target_attribute,
        "target_old_value": target_old_value,
        "target_new_value": target_new_value,
        "protected_attributes": protected_attributes,
        "correct_values": correct_values,
        "alternative_values": alternatives,
        "candidates": candidates,
        "gold_profile_id": gold_profile_id,
        "stale_profile_id": stale_profile_id,
        "necessity": necessity,
        "value_balance": value_balance,
        "checks": checks,
    }


def build_profile_tasks(
    cases: Sequence[Mapping[str, Any]],
) -> Tuple[List[dict], dict]:
    tasks = []
    task_index = 0
    for case in cases:
        for update in case["updates"]:
            tasks.append(
                build_profile_task(
                    case,
                    update,
                    task_index,
                )
            )
            task_index += 1

    if len(tasks) != EXPECTED_TARGET_INSTANCES:
        raise RuntimeError(
            "Expected {} tasks, got {}.".format(
                EXPECTED_TARGET_INSTANCES,
                len(tasks),
            )
        )

    gold_positions = Counter(
        str(task["gold_profile_id"])
        for task in tasks
    )
    counts = list(gold_positions.values())
    position_range = max(counts) - min(counts)

    if position_range > 1:
        raise RuntimeError(
            "Gold candidate positions are not balanced: {}".format(
                dict(gold_positions)
            )
        )

    task_report = {
        "task_count": len(tasks),
        "all_structural_checks_pass": all(
            all(bool(v) for v in task["checks"].values())
            for task in tasks
        ),
        "gold_position_counts": dict(
            sorted(gold_positions.items())
        ),
        "gold_position_balance_range": int(position_range),
        "chance_accuracy": 0.125,
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
                            "memory_sha256": sha256_text(
                                canonical_json(memory)
                            ),
                            "evidence_text": str(
                                update["evidence_text"]
                            ),
                            "updater_system_prompt_sha256": sha256_text(
                                UPDATER_SYSTEM_PROMPT
                            ),
                            "user_prompt_sha256": sha256_text(
                                user_prompt
                            ),
                            "response_schema_sha256": sha256_text(
                                canonical_json(schema)
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
# Reader control plan
# ============================================================================

def candidate_public_view(
    task: Mapping[str, Any],
) -> List[dict]:
    return [
        {
            "profile_id": str(candidate["profile_id"]),
            "attributes": dict(candidate["attributes"]),
        }
        for candidate in task["candidates"]
    ]


def build_reader_user_prompt(
    *,
    memory: Sequence[Mapping[str, Any]] | None,
    subject: str,
    candidates: Sequence[Mapping[str, Any]],
) -> str:
    if memory is None:
        memory_text = "(no memory state provided)"
    else:
        memory_text = json.dumps(
            public_memory(memory),
            ensure_ascii=False,
            indent=2,
        )

    candidates_text = json.dumps(
        list(candidates),
        ensure_ascii=False,
        indent=2,
    )

    return (
        "MEMORY STATE:\n"
        + memory_text
        + "\n\nTARGET PERSON:\n"
        + str(subject)
        + "\n\nCANDIDATE PROFILES:\n"
        + candidates_text
        + "\n\nChoose the one candidate profile that matches the target person's "
          "CURRENT information in the memory state.\n"
          "Return only: {\"profile_id\":\"pX\"}"
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

    rows = []
    for task_index in range(EXPECTED_TARGET_INSTANCES):
        task = tasks[task_index]
        case = case_by_id[str(task["case_id"])]
        update = next(
            update
            for update in case["updates"]
            if str(update["update_id"]) == str(task["update_id"])
        )
        candidates = candidate_public_view(task)

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
            user_prompt = build_reader_user_prompt(
                memory=memory,
                subject=str(task["subject"]),
                candidates=candidates,
            )
            reader_input_payload = {
                "system_prompt": READER_SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "generation_config": {
                    "model": READER_MODEL,
                    "temperature": READER_TEMPERATURE,
                    "seed": READER_SEED,
                    "max_output_tokens": (
                        READER_MAX_OUTPUT_TOKENS
                    ),
                    "context_window": READER_CONTEXT_WINDOW,
                },
                "response_schema": READER_RESPONSE_SCHEMA,
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
                    "gold_profile_id": str(
                        task["gold_profile_id"]
                    ),
                    "stale_profile_id": str(
                        task["stale_profile_id"]
                    ),
                    "memory": memory,
                    "reader_system_prompt_sha256": sha256_text(
                        READER_SYSTEM_PROMPT
                    ),
                    "user_prompt_sha256": sha256_text(
                        user_prompt
                    ),
                    "response_schema_sha256": sha256_text(
                        canonical_json(READER_RESPONSE_SCHEMA)
                    ),
                    "reader_input_sha256": sha256_text(
                        canonical_json(reader_input_payload)
                    ),
                }
            )

        # One no-memory row per target task; it is shared across groupings.
        user_prompt = build_reader_user_prompt(
            memory=None,
            subject=str(task["subject"]),
            candidates=candidates,
        )
        reader_input_payload = {
            "system_prompt": READER_SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "generation_config": {
                "model": READER_MODEL,
                "temperature": READER_TEMPERATURE,
                "seed": READER_SEED,
                "max_output_tokens": READER_MAX_OUTPUT_TOKENS,
                "context_window": READER_CONTEXT_WINDOW,
            },
            "response_schema": READER_RESPONSE_SCHEMA,
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
                "gold_profile_id": str(
                    task["gold_profile_id"]
                ),
                "stale_profile_id": str(
                    task["stale_profile_id"]
                ),
                "memory": None,
                "reader_system_prompt_sha256": sha256_text(
                    READER_SYSTEM_PROMPT
                ),
                "user_prompt_sha256": sha256_text(
                    user_prompt
                ),
                "response_schema_sha256": sha256_text(
                    canonical_json(READER_RESPONSE_SCHEMA)
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
        (
            str(row["arm"]),
            str(row["grouping"]),
        )
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
            "Reader control counts differ from frozen design: {}".format(
                dict(counts)
            )
        )

    unique_inputs = {
        str(row["reader_input_sha256"])
        for row in rows
    }

    report = {
        "logical_row_count": len(rows),
        "unique_reader_input_count_before_execution": len(
            unique_inputs
        ),
        "arm_grouping_counts": {
            "{}__{}".format(key[0], key[1]): int(value)
            for key, value in sorted(counts.items())
        },
        "gates": dict(READER_GATES),
    }
    return rows, report


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    paths.require_output_stage(paths.SELECTION, Path(__file__))
    print("=" * 78)
    print("PROFILE SELECTION LAYER-2 FORMAL BUILD / PREFLIGHT")
    print("=" * 78)
    print("Mode:              ZERO MODEL CALLS")
    print("Formal cases:      {}".format(FORMAL_NUM_CASES))
    print("Targets/case:      {}".format(FORMAL_TARGETS_PER_CASE))
    print("Target instances:  {}".format(EXPECTED_TARGET_INSTANCES))
    print("Updater calls:     {}".format(EXPECTED_UPDATER_CALLS))
    print("Reader controls:   {}".format(
        EXPECTED_READER_CONTROL_ROWS
    ))
    print("Max total calls:   {}".format(
        EXPECTED_MAX_MANDATORY_CALLS
    ))
    print("Master seed:       {}".format(FORMAL_MASTER_SEED))
    print()

    # Never rebuild the design after formal result production has begun.
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

    core_path = (
        PROJECT_ROOT
        / "atomic_memory"
        / "core.py"
    )
    if not core_path.exists():
        raise FileNotFoundError(
            "Missing current core: {}".format(core_path)
        )

    print("[1/5] Building 50 fresh cases and checking historical overlap...")
    cases, overlap_report = build_layer2_cases()
    print(
        "  PASS | 50 unique new fact sets | historical overlaps = 0"
    )

    print("[2/5] Building and auditing 100 Profile Selection tasks...")
    tasks, task_report = build_profile_tasks(cases)
    print(
        "  PASS | 100 tasks | all necessity checks pass | "
        "gold-position range={}".format(
            task_report["gold_position_balance_range"]
        )
    )

    print("[3/5] Building balanced 1,200-call updater plan...")
    updater_plan, updater_plan_report = (
        build_updater_execution_plan(cases)
    )
    print(
        "  PASS | 1,200 rows | each condition appears 25x "
        "in each execution position per model"
    )

    print("[4/5] Building 500-row reader-control plan...")
    reader_controls, reader_control_report = (
        build_reader_control_plan(cases, tasks)
    )
    print(
        "  PASS | 200 oracle + 200 stale + 100 no-memory logical rows"
    )

    # Freeze deterministic design files before the manifest references hashes.
    file_status = {}
    file_status["formal_cases.jsonl"] = write_or_verify_jsonl(
        CASES_PATH,
        cases,
    )
    file_status["profile_tasks.jsonl"] = write_or_verify_jsonl(
        TASKS_PATH,
        tasks,
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

    print("[5/5] Freezing hashes and call budget...")

    frozen_file_hashes = {
        "formal_cases.jsonl": sha256_file(CASES_PATH),
        "profile_tasks.jsonl": sha256_file(TASKS_PATH),
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
        "reader": {
            "model": READER_MODEL,
            "temperature": READER_TEMPERATURE,
            "seed": READER_SEED,
            "max_output_tokens": READER_MAX_OUTPUT_TOKENS,
            "context_window": READER_CONTEXT_WINDOW,
            "exact_input_caching": True,
        },
        "reader_gates": dict(READER_GATES),
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
            "optional_mismatched_memory_calls_if_triggered": 100,
            "absolute_contingency_ceiling": 3000,
        },
        "updater_system_prompt": UPDATER_SYSTEM_PROMPT,
        "updater_system_prompt_sha256": sha256_text(
            UPDATER_SYSTEM_PROMPT
        ),
        "reader_system_prompt": READER_SYSTEM_PROMPT,
        "reader_system_prompt_sha256": sha256_text(
            READER_SYSTEM_PROMPT
        ),
        "reader_response_schema": READER_RESPONSE_SCHEMA,
        "reader_response_schema_sha256": sha256_text(
            canonical_json(READER_RESPONSE_SCHEMA)
        ),
        "analysis_commitments": {
            "primary_task_contrast": (
                "TaskAcc(g1,Patch)-TaskAcc(g3,Patch)"
            ),
            "matching_state_contrast": (
                "StrictAcc(g1,Patch)-StrictAcc(g3,Patch)"
            ),
            "supporting_select_contrast": (
                "TaskAcc(g1,Select)-TaskAcc(g3,Select)"
            ),
            "models_analyzed_separately": True,
            "whole_case_bootstrap": True,
            "bootstrap_replicates": 20000,
            "failure_class_analysis": "descriptive",
            "pilot_rows_pooled_into_formal": False,
            "natural_rate_failures_only": True,
        },
        "overlap_report": overlap_report,
        "task_report": task_report,
        "updater_plan_report": updater_plan_report,
        "reader_control_report": reader_control_report,
        "frozen_file_sha256": frozen_file_hashes,
        "code_sha256": {
            "representation_edit_scope_core.py": sha256_file(
                core_path
            ),
            "build_profile_selection_formal.py": sha256_file(
                builder_path
            ),
        },
    }

    preflight = {
        "status": "PASS",
        "model_calls": 0,
        "manual_annotation_required": False,
        "cases": len(cases),
        "tasks": len(tasks),
        "updater_plan_rows": len(updater_plan),
        "reader_control_rows": len(reader_controls),
        "historical_overlap_pass": all(
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
        "gold_position_balance_pass": (
            task_report["gold_position_balance_range"] <= 1
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
        "file_status": file_status,
        "frozen_file_sha256": frozen_file_hashes,
        "next_step": (
            "Run the frozen 500-row reader-control stage before any "
            "prospective updater calls."
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
        "  PASS | cases/tasks/plans hashed | protocol frozen"
    )
    print()
    print("=" * 78)
    print("LAYER-2 FORMAL PREFLIGHT PASS")
    print("=" * 78)
    print("Cases:                  {}/50".format(len(cases)))
    print("Tasks:                  {}/100".format(len(tasks)))
    print("Updater plan:           {}/1200".format(
        len(updater_plan)
    ))
    print("Reader controls:        {}/500".format(
        len(reader_controls)
    ))
    print(
        "Historical overlaps:    {}".format(
            overlap_report["historical_overlap_counts"]
        )
    )
    print(
        "Gold positions:         {}".format(
            task_report["gold_position_counts"]
        )
    )
    print("Model calls made:       0")
    print("Formal protocol frozen: YES")
    print()
    print("Files:")
    print("  {} [{}]".format(
        CASES_PATH,
        file_status["formal_cases.jsonl"],
    ))
    print("  {} [{}]".format(
        TASKS_PATH,
        file_status["profile_tasks.jsonl"],
    ))
    print("  {} [{}]".format(
        UPDATER_PLAN_PATH,
        file_status["updater_execution_plan.jsonl"],
    ))
    print("  {} [{}]".format(
        READER_CONTROL_PLAN_PATH,
        file_status["reader_control_plan.jsonl"],
    ))
    print("  {} [{}]".format(
        PREFLIGHT_PATH,
        preflight_status,
    ))
    print("  {} [{}]".format(
        FREEZE_PATH,
        freeze_status,
    ))
    print()
    print(
        "NEXT: run the frozen reader-control stage. "
        "Do not start the 1,200 updater calls until those controls pass."
    )


if __name__ == "__main__":
    main()
