#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construct unit-load cases and local-edit tasks with the shared core evaluator.

This module makes no model calls."""

import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from atomic_memory.core import (
    ACTION_EXACT_PATCH,
    ACTION_REWRITE_UNIT,
    ATTRIBUTES_PER_SUBJECT,
    ATTRIBUTE_SPECS,
    EVIDENCE_TEMPLATE,
    Fact,
    SUBJECT_POOL,
    clone_memory,
    execute_and_evaluate,
    sha256_json,
)


UNIT_LOADS = (1, 2, 3)
INTERFACE_SELECT = "select"
INTERFACE_PATCH = "patch"

CONDITION_L1_SELECT = "L1_select"
CONDITION_L1_PATCH = "L1_patch"
CONDITION_L2_SELECT = "L2_select"
CONDITION_L2_PATCH = "L2_patch"
CONDITION_L3_SELECT = "L3_select"
CONDITION_L3_PATCH = "L3_patch"

UNIT_LOAD_CONDITIONS = (
    CONDITION_L1_SELECT,
    CONDITION_L1_PATCH,
    CONDITION_L2_SELECT,
    CONDITION_L2_PATCH,
    CONDITION_L3_SELECT,
    CONDITION_L3_PATCH,
)


_CONDITION_SPECS = {
    CONDITION_L1_SELECT: {
        "unit_load": 1,
        "interface": INTERFACE_SELECT,
        "expected_action_type": ACTION_REWRITE_UNIT,
    },
    CONDITION_L1_PATCH: {
        "unit_load": 1,
        "interface": INTERFACE_PATCH,
        "expected_action_type": ACTION_EXACT_PATCH,
    },
    CONDITION_L2_SELECT: {
        "unit_load": 2,
        "interface": INTERFACE_SELECT,
        "expected_action_type": ACTION_REWRITE_UNIT,
    },
    CONDITION_L2_PATCH: {
        "unit_load": 2,
        "interface": INTERFACE_PATCH,
        "expected_action_type": ACTION_EXACT_PATCH,
    },
    CONDITION_L3_SELECT: {
        "unit_load": 3,
        "interface": INTERFACE_SELECT,
        "expected_action_type": ACTION_REWRITE_UNIT,
    },
    CONDITION_L3_PATCH: {
        "unit_load": 3,
        "interface": INTERFACE_PATCH,
        "expected_action_type": ACTION_EXACT_PATCH,
    },
}


def unit_load_representation_key(unit_load: int) -> str:
    if int(unit_load) not in UNIT_LOADS:
        raise ValueError("unit_load must be one of {}".format(UNIT_LOADS))
    return "load_{}".format(int(unit_load))


def unit_load_condition_spec(condition: str) -> dict:
    if condition not in _CONDITION_SPECS:
        raise ValueError("Unknown unit-load condition: {}".format(condition))
    result = dict(_CONDITION_SPECS[condition])
    result["condition"] = condition
    result["representation"] = unit_load_representation_key(
        result["unit_load"]
    )
    return result


def _build_latent_facts(rng: random.Random) -> List[Fact]:
    """Match the baseline generator's semantic fact distribution."""
    subject_a, subject_b = rng.sample(list(SUBJECT_POOL), 2)
    facts = []

    subject_specs = (
        (subject_a, ATTRIBUTES_PER_SUBJECT[0]),
        (subject_b, ATTRIBUTES_PER_SUBJECT[1]),
    )

    for subject_index, (subject, attrs) in enumerate(subject_specs):
        for attr_index, attr in enumerate(attrs):
            values = list(ATTRIBUTE_SPECS[attr]["values"])
            value = rng.choice(values)
            facts.append(
                Fact(
                    "f{}{}".format(subject_index, attr_index),
                    subject,
                    attr,
                    value,
                )
            )

    return facts


def _shared_fact_order(
    facts: Sequence[Fact],
    rng: random.Random,
) -> List[Fact]:
    indices = list(range(len(facts)))
    rng.shuffle(indices)
    return [facts[index] for index in indices]


def _build_grouped_memory(
    ordered_facts: Sequence[Fact],
    unit_load: int,
) -> List[dict]:
    """Group the same ordered six facts into units of size 1, 2, or 3.

    IDs deliberately use the same format (m00, m01, ...) for every grouping
    load.  Only the number and membership of units change.
    """
    unit_load = int(unit_load)
    if unit_load not in UNIT_LOADS:
        raise ValueError("Unsupported unit_load: {}".format(unit_load))
    if len(ordered_facts) != 6:
        raise ValueError("Unit-load study requires exactly six facts")
    if len(ordered_facts) % unit_load != 0:
        raise ValueError("unit_load must divide the six-fact state")

    units = []
    unit_index = 0
    for start in range(0, len(ordered_facts), unit_load):
        unit_facts = list(ordered_facts[start : start + unit_load])
        units.append(
            {
                "memory_id": "m{:02d}".format(unit_index),
                "text": " ".join(
                    fact.current_text()
                    for fact in unit_facts
                ),
                "fact_ids": [fact.fact_id for fact in unit_facts],
            }
        )
        unit_index += 1

    return units


def _sample_new_value(rng: random.Random, old_fact: Fact) -> str:
    values = [
        value
        for value in ATTRIBUTE_SPECS[old_fact.attribute]["values"]
        if value != old_fact.value
    ]
    return rng.choice(values)


def _updated_fact(old_fact: Fact, new_value: str) -> Fact:
    return Fact(
        old_fact.fact_id,
        old_fact.subject,
        old_fact.attribute,
        new_value,
    )


def generate_unit_load_case(
    case_index: int,
    master_seed: int,
    targets_per_case: int = 2,
) -> dict:
    """Generate all three representations before sampling future targets."""
    if targets_per_case < 1 or targets_per_case > 6:
        raise ValueError("targets_per_case must be between 1 and 6")

    case_seed = (
        int(master_seed) * 1000003
        + int(case_index) * 9176
        + 41
    ) & 0x7FFFFFFF

    latent_rng = random.Random(case_seed)
    facts = _build_latent_facts(latent_rng)

    fact_order_seed = case_seed ^ 0x5A5A5A5A
    ordered_facts = _shared_fact_order(
        facts,
        random.Random(fact_order_seed),
    )

    representations = {
        unit_load_representation_key(load): _build_grouped_memory(
            ordered_facts,
            load,
        )
        for load in UNIT_LOADS
    }

    # Freeze every representation BEFORE any future update target is sampled.
    representation_hash = sha256_json(representations)

    target_seed = case_seed ^ 0x13579BDF
    target_rng = random.Random(target_seed)
    target_indices = target_rng.sample(
        list(range(len(ordered_facts))),
        targets_per_case,
    )

    updates = []
    for order, target_index in enumerate(target_indices, start=1):
        old_fact = ordered_facts[target_index]
        new_fact = _updated_fact(
            old_fact,
            _sample_new_value(target_rng, old_fact),
        )

        expected_current_facts = []
        for fact in ordered_facts:
            expected_current_facts.append(
                (
                    new_fact
                    if fact.fact_id == old_fact.fact_id
                    else fact
                ).to_dict()
            )

        updates.append(
            {
                "update_id": "u{:02d}".format(order),
                "target_fact_id": old_fact.fact_id,
                "old_fact": old_fact.to_dict(),
                "new_fact": new_fact.to_dict(),
                "evidence_text": EVIDENCE_TEMPLATE.format(
                    subject=new_fact.subject,
                    label=new_fact.label,
                    value=new_fact.value,
                ),
                "expected_current_facts": expected_current_facts,
                "allowed_historical_facts": [
                    {
                        "subject": old_fact.subject,
                        "attribute": old_fact.attribute,
                        "label": old_fact.label,
                        "value": old_fact.value,
                        "historical_text": old_fact.historical_text(),
                    }
                ],
            }
        )

    case = {
        "case_id": "unit_load_case_{:04d}".format(case_index),
        "master_seed": int(master_seed),
        "case_seed": int(case_seed),
        "fact_order_seed": int(fact_order_seed),
        "target_seed": int(target_seed),
        "facts": [fact.to_dict() for fact in facts],
        "shared_fact_order": [fact.fact_id for fact in ordered_facts],
        "representations": representations,
        "representation_hash": representation_hash,
        "updates": updates,
        "design_invariants": {
            "fact_count": 6,
            "unit_loads": list(UNIT_LOADS),
            "unit_counts": {
                unit_load_representation_key(load): 6 // load
                for load in UNIT_LOADS
            },
            "representations_built_before_target": True,
            "targets_share_same_initial_memory": True,
            "shared_fact_order_across_representations": True,
            "fixed_unit_membership_required": True,
            "controlled_language_version": "controlled-memory-v0",
        },
    }
    case["case_sha256"] = sha256_json(case)
    return case


def generate_unit_load_cases(
    num_cases: int,
    master_seed: int,
    targets_per_case: int = 2,
) -> List[dict]:
    if num_cases < 1:
        raise ValueError("num_cases must be >= 1")
    return [
        generate_unit_load_case(
            case_index=index,
            master_seed=master_seed,
            targets_per_case=targets_per_case,
        )
        for index in range(num_cases)
    ]


def initial_memory_for_unit_load_condition(
    case: Mapping[str, Any],
    condition: str,
) -> List[dict]:
    spec = unit_load_condition_spec(condition)
    return clone_memory(
        case["representations"][spec["representation"]]
    )


def target_unit_id_for_load(
    case: Mapping[str, Any],
    update: Mapping[str, Any],
    unit_load: int,
) -> str:
    target_fact_id = str(update["target_fact_id"])
    representation = unit_load_representation_key(unit_load)
    matches = [
        str(unit["memory_id"])
        for unit in case["representations"][representation]
        if target_fact_id in [
            str(fact_id)
            for fact_id in unit.get("fact_ids", [])
        ]
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one target unit for load {}, found {}".format(
                unit_load,
                matches,
            )
        )
    return matches[0]


def target_unit_id_for_condition(
    case: Mapping[str, Any],
    update: Mapping[str, Any],
    condition: str,
) -> str:
    spec = unit_load_condition_spec(condition)
    return target_unit_id_for_load(
        case,
        update,
        spec["unit_load"],
    )


def unit_load_locator_success(
    case: Mapping[str, Any],
    update: Mapping[str, Any],
    condition: str,
    action: Mapping[str, Any],
) -> bool:
    selected_id = action.get("memory_id")
    if not isinstance(selected_id, str):
        return False
    return selected_id == target_unit_id_for_condition(
        case,
        update,
        condition,
    )


def execute_and_evaluate_unit_load(
    case: Mapping[str, Any],
    update: Mapping[str, Any],
    condition: str,
    action: Mapping[str, Any],
) -> dict:
    # Each condition starts from the original memory; updates never accumulate.
    spec = unit_load_condition_spec(condition)
    initial_memory = initial_memory_for_unit_load_condition(
        case,
        condition,
    )
    return execute_and_evaluate(
        initial_memory=initial_memory,
        action=action,
        expected_action_type=spec["expected_action_type"],
        update=update,
    )


def oracle_correct_action(
    case: Mapping[str, Any],
    update: Mapping[str, Any],
    condition: str,
) -> dict:
    """Deterministic test helper. Never expose this helper's output to a model."""
    spec = unit_load_condition_spec(condition)
    memory = initial_memory_for_unit_load_condition(
        case,
        condition,
    )
    target_id = target_unit_id_for_condition(
        case,
        update,
        condition,
    )
    target_unit = next(
        unit
        for unit in memory
        if str(unit["memory_id"]) == target_id
    )

    old_text = str(update["old_fact"]["current_text"])
    new_text = str(update["new_fact"]["current_text"])

    if spec["expected_action_type"] == ACTION_REWRITE_UNIT:
        if target_unit["text"].count(old_text) != 1:
            raise ValueError("Target sentence is not unique in target unit")
        return {
            "action": ACTION_REWRITE_UNIT,
            "memory_id": target_id,
            "text": str(target_unit["text"]).replace(
                old_text,
                new_text,
                1,
            ),
        }

    if spec["expected_action_type"] == ACTION_EXACT_PATCH:
        return {
            "action": ACTION_EXACT_PATCH,
            "memory_id": target_id,
            "old_span": old_text,
            "new_span": new_text,
        }

    raise ValueError("Unsupported unit-load action type")


def oracle_wrong_existing_id_action(
    case: Mapping[str, Any],
    update: Mapping[str, Any],
    condition: str,
) -> dict:
    """Test helper for the existing-but-wrong-ID execution semantics.

    The returned action is deliberately executable under the interface contract
    while selecting a non-target *existing* unit.  Gold target knowledge is used
    only to construct a deterministic test case, never by the executor.
    """
    spec = unit_load_condition_spec(condition)
    memory = initial_memory_for_unit_load_condition(
        case,
        condition,
    )
    target_id = target_unit_id_for_condition(
        case,
        update,
        condition,
    )
    wrong_unit = next(
        unit
        for unit in memory
        if str(unit["memory_id"]) != target_id
    )

    if spec["expected_action_type"] == ACTION_REWRITE_UNIT:
        return {
            "action": ACTION_REWRITE_UNIT,
            "memory_id": str(wrong_unit["memory_id"]),
            "text": str(wrong_unit["text"]),
        }

    if spec["expected_action_type"] == ACTION_EXACT_PATCH:
        # Use an exact span that exists once in the wrong unit and replace it
        # with itself.  This proves that selecting a wrong *existing* ID is not
        # gold-censored by the executor: the action executes, then evaluation
        # fails because the requested target remains stale.
        wrong_span = str(wrong_unit["text"])
        if not wrong_span:
            raise ValueError("Wrong unit unexpectedly has empty text")
        return {
            "action": ACTION_EXACT_PATCH,
            "memory_id": str(wrong_unit["memory_id"]),
            "old_span": wrong_span,
            "new_span": wrong_span,
        }

    raise ValueError("Unsupported unit-load action type")


def public_unit_load_memory(
    memory: Sequence[Mapping[str, Any]],
) -> List[dict]:
    """Remove latent fact_ids before any model-facing serialization."""
    return [
        {
            "memory_id": str(unit["memory_id"]),
            "text": str(unit["text"]),
        }
        for unit in memory
    ]


def unit_load_action_schema(
    condition: str,
    memory: Sequence[Mapping[str, Any]],
) -> dict:
    """Strict schema for one of the six local-edit conditions."""
    spec = unit_load_condition_spec(condition)
    memory_ids = [str(unit["memory_id"]) for unit in memory]

    if spec["expected_action_type"] == ACTION_REWRITE_UNIT:
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
                "text": {"type": "string"},
            },
        }

    if spec["expected_action_type"] == ACTION_EXACT_PATCH:
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
                "new_span": {"type": "string"},
            },
        }

    raise ValueError("Unsupported unit-load action type")


def unit_load_action_instruction(condition: str) -> str:
    spec = unit_load_condition_spec(condition)

    if spec["expected_action_type"] == ACTION_REWRITE_UNIT:
        return (
            "Select exactly one existing memory unit and return the complete "
            "replacement text for that unit. Do not return or rewrite any "
            "other unit. The output action must be rewrite_unit."
        )

    if spec["expected_action_type"] == ACTION_EXACT_PATCH:
        return (
            "Select one existing memory unit and provide one exact local "
            "string replacement. old_span must occur exactly once in the "
            "selected unit before editing. The output action must be "
            "exact_patch."
        )

    raise ValueError("Unsupported unit-load action type")


def build_unit_load_user_prompt(
    condition: str,
    memory: Sequence[Mapping[str, Any]],
    evidence_text: str,
) -> str:
    import json

    memory_json = json.dumps(
        public_unit_load_memory(memory),
        ensure_ascii=False,
        indent=2,
    )
    return (
        "MEMORY STATE:\n"
        + memory_json
        + "\n\nNEW EVIDENCE:\n"
        + str(evidence_text)
        + "\n\nREQUIRED EDIT SCOPE:\n"
        + unit_load_action_instruction(condition)
        + "\n\nReturn only the JSON action."
    )


def flattened_fact_order(
    case: Mapping[str, Any],
    unit_load: int,
) -> List[str]:
    representation = unit_load_representation_key(unit_load)
    result = []
    for unit in case["representations"][representation]:
        result.extend(str(fact_id) for fact_id in unit["fact_ids"])
    return result


def condition_matrix() -> List[Tuple[int, str, str]]:
    """Stable six-cell order: load 1/2/3, Select then Patch."""
    return [
        (
            int(unit_load_condition_spec(condition)["unit_load"]),
            str(unit_load_condition_spec(condition)["interface"]),
            condition,
        )
        for condition in UNIT_LOAD_CONDITIONS
    ]
