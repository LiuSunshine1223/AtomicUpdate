#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate, execute, and strictly evaluate controlled memory-update cases.

All operations are deterministic and make no model calls."""

import copy
import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

ATTRIBUTE_SPECS = {
    "city": {"label": "city", "values": ("Seattle", "Chicago", "Austin", "Denver", "Boston", "Portland", "Phoenix", "Atlanta")},
    "job": {"label": "job", "values": ("designer", "teacher", "engineer", "writer", "analyst", "nurse", "architect", "researcher")},
    "drink": {"label": "morning drink", "values": ("latte", "green tea", "black coffee", "smoothie", "espresso", "cocoa", "herbal tea", "orange juice")},
    "device": {"label": "primary phone", "values": ("Pixel phone", "iPhone", "Galaxy phone", "OnePlus phone", "Motorola phone", "Xiaomi phone", "Nothing phone", "Sony phone")},
    "car": {"label": "car", "values": ("Honda Civic", "Toyota Corolla", "Mazda 3", "Subaru Impreza", "Ford Focus", "Hyundai Elantra", "Kia Forte", "Volkswagen Jetta")},
    "hobby": {"label": "weekend hobby", "values": ("hiking", "painting", "cycling", "baking", "photography", "gardening", "swimming", "woodworking")},
}

SUBJECT_POOL = ("Emma", "Liam", "Maya", "Noah", "Sofia", "Ethan", "Ava", "Lucas", "Nora", "Owen")
ATTRIBUTES_PER_SUBJECT = (("city", "job", "drink"), ("city", "device", "hobby"))

CURRENT_TEMPLATE = "{subject}'s current {label} is {value}."
HISTORY_TEMPLATE = "{subject}'s previous {label} was {value}."
EVIDENCE_TEMPLATE = "New evidence: {subject}'s {label} is now {value}."

CURRENT_RE = re.compile(r"^(?P<subject>[A-Za-z][A-Za-z0-9_-]*)'s current (?P<label>[a-z][a-z ]*[a-z]) is (?P<value>.+)\.$")
HISTORY_RE = re.compile(r"^(?P<subject>[A-Za-z][A-Za-z0-9_-]*)'s previous (?P<label>[a-z][a-z ]*[a-z]) was (?P<value>.+)\.$")
ALLOWED_LABELS = {spec["label"]: attr for attr, spec in ATTRIBUTE_SPECS.items()}

REPRESENTATION_COMPOUND = "compound"
REPRESENTATION_ATOMIC = "atomic"
ACTION_REWRITE_ALL = "rewrite_all"
ACTION_REWRITE_UNIT = "rewrite_unit"
ACTION_EXACT_PATCH = "exact_patch"

CONDITION_A = "A_compound_global"
CONDITION_B = "B_compound_select"
CONDITION_C = "C_atomic_global"
CONDITION_D = "D_atomic_select"
CONDITION_E = "E_compound_exact_patch"
CONDITIONS = (CONDITION_A, CONDITION_B, CONDITION_C, CONDITION_D, CONDITION_E)


@dataclass(frozen=True)
class Fact:
    fact_id: str
    subject: str
    attribute: str
    value: str

    @property
    def label(self):
        return str(ATTRIBUTE_SPECS[self.attribute]["label"])

    def current_text(self):
        return CURRENT_TEMPLATE.format(subject=self.subject, label=self.label, value=self.value)

    def historical_text(self):
        return HISTORY_TEMPLATE.format(subject=self.subject, label=self.label, value=self.value)

    def to_dict(self):
        return {
            "fact_id": self.fact_id,
            "subject": self.subject,
            "attribute": self.attribute,
            "label": self.label,
            "value": self.value,
            "current_text": self.current_text(),
        }


@dataclass(frozen=True)
class ParsedFact:
    subject: str
    attribute: str
    value: str
    status: str
    source_sentence: str


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def clone_memory(memory: Sequence[Mapping[str, Any]]) -> List[dict]:
    return copy.deepcopy([dict(unit) for unit in memory])


def _build_latent_facts(rng: random.Random) -> List[Fact]:
    subject_a, subject_b = rng.sample(list(SUBJECT_POOL), 2)
    facts = []
    for subject_index, (subject, attrs) in enumerate(((subject_a, ATTRIBUTES_PER_SUBJECT[0]), (subject_b, ATTRIBUTES_PER_SUBJECT[1]))):
        for attr_index, attr in enumerate(attrs):
            value = rng.choice(list(ATTRIBUTE_SPECS[attr]["values"]))
            facts.append(Fact("f{}{}".format(subject_index, attr_index), subject, attr, value))
    return facts


def _shared_fact_order(
    facts: Sequence[Fact],
    rng: random.Random,
) -> List[Fact]:
    """Create one shared permutation used by both representations."""
    indices = list(range(len(facts)))
    rng.shuffle(indices)
    return [facts[index] for index in indices]


def _build_atomic_memory(
    ordered_facts: Sequence[Fact],
) -> List[dict]:
    return [
        {
            "memory_id": "a{:02d}".format(i),
            "text": fact.current_text(),
            "fact_ids": [fact.fact_id],
        }
        for i, fact in enumerate(ordered_facts)
    ]


def _build_compound_memory(
    ordered_facts: Sequence[Fact],
) -> List[dict]:
    units = []
    for unit_index in range(3):
        start = unit_index * 2
        unit_facts = list(
            ordered_facts[start : start + 2]
        )
        units.append(
            {
                "memory_id": "c{:02d}".format(unit_index),
                "text": " ".join(
                    fact.current_text()
                    for fact in unit_facts
                ),
                "fact_ids": [
                    fact.fact_id
                    for fact in unit_facts
                ],
            }
        )
    return units


def _updated_fact(old_fact: Fact, new_value: str) -> Fact:
    return Fact(old_fact.fact_id, old_fact.subject, old_fact.attribute, new_value)


def _sample_new_value(rng: random.Random, old_fact: Fact) -> str:
    values = [v for v in ATTRIBUTE_SPECS[old_fact.attribute]["values"] if v != old_fact.value]
    return rng.choice(values)


def generate_case(case_index: int, master_seed: int, targets_per_case: int = 2) -> dict:
    if targets_per_case < 1 or targets_per_case > 6:
        raise ValueError("targets_per_case must be between 1 and 6")

    case_seed = (int(master_seed) * 1000003 + int(case_index) * 9176 + 41) & 0x7FFFFFFF
    latent_rng = random.Random(case_seed)
    facts = _build_latent_facts(latent_rng)

    fact_order_seed = case_seed ^ 0x5A5A5A5A
    ordered_facts = _shared_fact_order(
        facts,
        random.Random(fact_order_seed),
    )
    compound_memory = _build_compound_memory(
        ordered_facts
    )
    atomic_memory = _build_atomic_memory(
        ordered_facts
    )

    # Freeze both representations BEFORE sampling any future update target.
    representation_payload = {
        "compound": compound_memory,
        "atomic": atomic_memory,
    }
    representation_hash = sha256_json(
        representation_payload
    )

    target_seed = case_seed ^ 0x13579BDF
    target_rng = random.Random(target_seed)
    target_indices = target_rng.sample(
        list(range(len(ordered_facts))),
        targets_per_case,
    )

    updates = []
    for order, target_index in enumerate(
        target_indices,
        start=1,
    ):
        old_fact = ordered_facts[target_index]
        new_fact = _updated_fact(
            old_fact,
            _sample_new_value(target_rng, old_fact),
        )
        expected = []
        for fact in ordered_facts:
            expected.append(
                (
                    new_fact
                    if fact.fact_id == old_fact.fact_id
                    else fact
                ).to_dict()
            )
        updates.append({
            "update_id": "u{:02d}".format(order),
            "target_fact_id": old_fact.fact_id,
            "old_fact": old_fact.to_dict(),
            "new_fact": new_fact.to_dict(),
            "evidence_text": EVIDENCE_TEMPLATE.format(subject=new_fact.subject, label=new_fact.label, value=new_fact.value),
            "expected_current_facts": expected,
            "allowed_historical_facts": [{
                "subject": old_fact.subject,
                "attribute": old_fact.attribute,
                "label": old_fact.label,
                "value": old_fact.value,
                "historical_text": old_fact.historical_text(),
            }],
        })

    case = {
        "case_id": "case_{:04d}".format(case_index),
        "master_seed": int(master_seed),
        "case_seed": int(case_seed),
        "fact_order_seed": int(fact_order_seed),
        "target_seed": int(target_seed),
        "facts": [f.to_dict() for f in facts],
        "shared_fact_order": [
            fact.fact_id
            for fact in ordered_facts
        ],
        "representations": {"compound": compound_memory, "atomic": atomic_memory},
        "representation_hash": representation_hash,
        "updates": updates,
        "design_invariants": {
            "fact_count": 6,
            "compound_unit_count": 3,
            "compound_facts_per_unit": 2,
            "atomic_unit_count": 6,
            "atomic_facts_per_unit": 1,
            "representations_built_before_target": True,
            "targets_share_same_initial_memory": True,
            "shared_fact_order_across_representations": True,
            "fixed_unit_membership_required": True,
            "controlled_language_version": "controlled-memory-v0",
        },
    }
    case["case_sha256"] = sha256_json(case)
    return case


def generate_cases(num_cases: int, master_seed: int, targets_per_case: int = 2) -> List[dict]:
    if num_cases < 1:
        raise ValueError("num_cases must be >= 1")
    return [generate_case(i, master_seed, targets_per_case) for i in range(num_cases)]


def _split_sentences(text: str) -> List[str]:
    stripped = str(text).strip()
    if not stripped:
        return []
    return [chunk.strip() for chunk in re.split(r"(?<=\.)\s+", stripped) if chunk.strip()]


def parse_fact_sentence(sentence: str) -> ParsedFact:
    match = CURRENT_RE.match(sentence)
    status = "current"
    if match is None:
        match = HISTORY_RE.match(sentence)
        status = "history"
    if match is None:
        raise ValueError("Sentence outside controlled grammar: {!r}".format(sentence))
    label = str(match.group("label")).strip()
    attribute = ALLOWED_LABELS.get(label)
    if attribute is None:
        raise ValueError("Unknown attribute label: {!r}".format(label))
    value = str(match.group("value")).strip()
    if not value:
        raise ValueError("Empty fact value")
    return ParsedFact(str(match.group("subject")).strip(), attribute, value, status, sentence)


def parse_memory(memory: Sequence[Mapping[str, Any]]) -> dict:
    parsed, errors = [], []
    seen_ids, duplicate_ids = set(), []
    for unit in memory:
        memory_id = str(unit.get("memory_id", ""))
        text = unit.get("text")
        if not memory_id:
            errors.append({"memory_id": memory_id, "error": "missing_memory_id"})
            continue
        if memory_id in seen_ids:
            duplicate_ids.append(memory_id)
        seen_ids.add(memory_id)
        if not isinstance(text, str):
            errors.append({"memory_id": memory_id, "error": "text_not_string"})
            continue
        for sentence in _split_sentences(text):
            try:
                fact = parse_fact_sentence(sentence)
            except Exception as exc:
                errors.append({"memory_id": memory_id, "sentence": sentence, "error": str(exc)})
                continue
            parsed.append({
                "memory_id": memory_id,
                "subject": fact.subject,
                "attribute": fact.attribute,
                "value": fact.value,
                "status": fact.status,
                "source_sentence": fact.source_sentence,
            })
    if duplicate_ids:
        errors.append({"error": "duplicate_memory_ids", "memory_ids": sorted(set(duplicate_ids))})
    keys = [(r["subject"], r["attribute"], r["value"], r["status"]) for r in parsed]
    per_unit_fact_counts = {}
    per_unit_current_fact_counts = {}

    for row in parsed:
        memory_id = str(row["memory_id"])
        per_unit_fact_counts[memory_id] = (
            per_unit_fact_counts.get(memory_id, 0) + 1
        )

        if row["status"] == "current":
            per_unit_current_fact_counts[memory_id] = (
                per_unit_current_fact_counts.get(
                    memory_id,
                    0,
                )
                + 1
            )

    return {
        "contract_valid": not errors,
        "parsed_facts": parsed,
        "errors": errors,
        "duplicate_fact_count": len(keys) - len(set(keys)),
        "per_unit_fact_counts": per_unit_fact_counts,
        "per_unit_current_fact_counts": (
            per_unit_current_fact_counts
        ),
    }


def _strict_keys(action: Mapping[str, Any], required: Set[str]) -> Tuple[bool, List[str]]:
    actual = set(str(k) for k in action.keys())
    if actual == required:
        return True, []
    problems = []
    missing, extra = sorted(required - actual), sorted(actual - required)
    if missing:
        problems.append("missing_keys={}".format(missing))
    if extra:
        problems.append("extra_keys={}".format(extra))
    return False, problems


def _memory_id_map(memory: Sequence[Mapping[str, Any]]) -> Dict[str, dict]:
    result = {}
    for unit in memory:
        memory_id = str(unit.get("memory_id", ""))
        if not memory_id:
            raise ValueError("Initial memory has missing memory_id")
        if memory_id in result:
            raise ValueError("Initial memory has duplicate ID: {}".format(memory_id))
        result[memory_id] = dict(unit)
    return result


def execute_action(initial_memory: Sequence[Mapping[str, Any]], action: Mapping[str, Any], expected_action_type: str) -> dict:
    untouched = clone_memory(initial_memory)
    if not isinstance(action, Mapping):
        return {"legal_action": False, "execution_success": False, "error": "action_not_mapping", "final_memory": untouched}
    if action.get("action") != expected_action_type:
        return {"legal_action": False, "execution_success": False, "error": "unexpected_action_type: {!r}".format(action.get("action")), "final_memory": untouched}
    try:
        by_id = _memory_id_map(initial_memory)
    except Exception as exc:
        return {"legal_action": False, "execution_success": False, "error": str(exc), "final_memory": untouched}

    if expected_action_type == ACTION_REWRITE_ALL:
        ok, issues = _strict_keys(action, {"action", "units"})
        if not ok:
            return {"legal_action": False, "execution_success": False, "error": "; ".join(issues), "final_memory": untouched}
        units = action.get("units")
        if not isinstance(units, list) or len(units) != len(by_id):
            return {"legal_action": False, "execution_success": False, "error": "rewrite_all_unit_count_or_type_invalid", "final_memory": untouched}
        new_by_id = {}
        for unit in units:
            if not isinstance(unit, Mapping) or set(unit.keys()) != {"memory_id", "text"}:
                return {"legal_action": False, "execution_success": False, "error": "rewrite_all_unit_schema_invalid", "final_memory": untouched}
            memory_id, text = str(unit.get("memory_id", "")), unit.get("text")
            if memory_id not in by_id or memory_id in new_by_id or not isinstance(text, str):
                return {"legal_action": False, "execution_success": False, "error": "rewrite_all_unit_invalid", "final_memory": untouched}
            new_by_id[memory_id] = {"memory_id": memory_id, "text": text, "fact_ids": list(by_id[memory_id].get("fact_ids", []))}
        if set(new_by_id) != set(by_id):
            return {"legal_action": False, "execution_success": False, "error": "rewrite_all_id_set_mismatch", "final_memory": untouched}
        final_memory = [new_by_id[str(unit["memory_id"])] for unit in initial_memory]
        return {"legal_action": True, "execution_success": True, "error": None, "final_memory": final_memory}

    if expected_action_type == ACTION_REWRITE_UNIT:
        ok, issues = _strict_keys(action, {"action", "memory_id", "text"})
        if not ok:
            return {"legal_action": False, "execution_success": False, "error": "; ".join(issues), "final_memory": untouched}
        memory_id, text = str(action.get("memory_id", "")), action.get("text")
        if memory_id not in by_id or not isinstance(text, str):
            return {"legal_action": False, "execution_success": False, "error": "rewrite_unit_invalid", "final_memory": untouched}
        final_memory = clone_memory(initial_memory)
        for unit in final_memory:
            if str(unit["memory_id"]) == memory_id:
                unit["text"] = text
                break
        return {"legal_action": True, "execution_success": True, "error": None, "final_memory": final_memory}

    if expected_action_type == ACTION_EXACT_PATCH:
        ok, issues = _strict_keys(action, {"action", "memory_id", "old_span", "new_span"})
        if not ok:
            return {"legal_action": False, "execution_success": False, "error": "; ".join(issues), "final_memory": untouched}
        memory_id = str(action.get("memory_id", ""))
        old_span, new_span = action.get("old_span"), action.get("new_span")
        if memory_id not in by_id or not isinstance(old_span, str) or not isinstance(new_span, str) or not old_span:
            return {"legal_action": False, "execution_success": False, "error": "exact_patch_schema_or_id_invalid", "final_memory": untouched}
        source_text = str(by_id[memory_id].get("text", ""))
        count = source_text.count(old_span)
        if count != 1:
            return {"legal_action": True, "execution_success": False, "error": "old_span_occurrence_count={}".format(count), "final_memory": untouched}
        final_memory = clone_memory(initial_memory)
        for unit in final_memory:
            if str(unit["memory_id"]) == memory_id:
                unit["text"] = str(unit["text"]).replace(old_span, new_span, 1)
                break
        return {"legal_action": True, "execution_success": True, "error": None, "final_memory": final_memory}

    return {"legal_action": False, "execution_success": False, "error": "unsupported_expected_action_type", "final_memory": untouched}


def _fact_tuple(fact: Mapping[str, Any], status: str) -> Tuple[str, str, str, str]:
    return (str(fact["subject"]), str(fact["attribute"]), str(fact["value"]), str(status))


def _expected_slot_to_fact_id(
    update: Mapping[str, Any],
) -> Dict[Tuple[str, str], str]:
    result = {}
    for fact in update["expected_current_facts"]:
        slot = (
            str(fact["subject"]),
            str(fact["attribute"]),
        )
        fact_id = str(fact["fact_id"])
        if slot in result and result[slot] != fact_id:
            raise ValueError(
                "Ambiguous semantic slot in expected_current_facts: {}".format(
                    slot
                )
            )
        result[slot] = fact_id
    return result


def _actual_unit_membership(
    final_memory: Sequence[Mapping[str, Any]],
    parsed_result: Mapping[str, Any],
    update: Mapping[str, Any],
) -> Dict[str, List[str]]:
    slot_to_fact_id = _expected_slot_to_fact_id(
        update
    )
    result = {
        str(unit.get("memory_id", "")): []
        for unit in final_memory
    }

    for row in parsed_result["parsed_facts"]:
        if row["status"] != "current":
            continue

        memory_id = str(row["memory_id"])
        slot = (
            str(row["subject"]),
            str(row["attribute"]),
        )
        fact_id = slot_to_fact_id.get(slot)

        if fact_id is None:
            fact_id = (
                "__unsupported_slot__:{}|{}|{}".format(
                    row["subject"],
                    row["attribute"],
                    row["value"],
                )
            )

        result.setdefault(memory_id, []).append(
            fact_id
        )

    return {
        memory_id: sorted(fact_ids)
        for memory_id, fact_ids in result.items()
    }


def _failure_reasons(
    *,
    legal_action: bool,
    execution_success: bool,
    contract_valid: bool,
    structure_valid: Optional[bool],
    target_new_current: Optional[bool],
    target_old_not_current: Optional[bool],
    protected_facts_preserved: Optional[bool],
    no_unsupported_changes: Optional[bool],
) -> List[str]:
    reasons = []

    if not legal_action:
        reasons.append("illegal_action")
    if not execution_success:
        reasons.append("execution_failure")
    if not contract_valid:
        reasons.append("memory_contract_failure")
    if structure_valid is False:
        reasons.append("structure_failure")
    if target_new_current is False:
        reasons.append("target_new_missing")
    if target_old_not_current is False:
        reasons.append("stale_old_retained")
    if protected_facts_preserved is False:
        reasons.append("protected_fact_loss")
    if no_unsupported_changes is False:
        reasons.append("unsupported_change")

    return reasons


def evaluate_final_memory(
    final_memory: Sequence[Mapping[str, Any]],
    update: Mapping[str, Any],
    legal_action: bool,
    execution_success: bool,
    expected_unit_fact_counts: Optional[Mapping[str, int]] = None,
    expected_unit_membership: Optional[
        Mapping[str, Sequence[str]]
    ] = None,
) -> dict:
    parsed_result = parse_memory(final_memory)
    if not parsed_result["contract_valid"]:
        failure_reasons = _failure_reasons(
            legal_action=bool(legal_action),
            execution_success=bool(
                execution_success
            ),
            contract_valid=False,
            structure_valid=None,
            target_new_current=None,
            target_old_not_current=None,
            protected_facts_preserved=None,
            no_unsupported_changes=None,
        )
        return {
            "legal_action": bool(legal_action),
            "execution_success": bool(
                execution_success
            ),
            "contract_valid": False,
            "structure_valid": False,
            "membership_valid": None,
            "target_new_current": None,
            "target_old_not_current": None,
            "protected_facts_preserved": None,
            "no_unsupported_changes": None,
            "end_to_end_success": False,
            "failure_reasons": failure_reasons,
            "unsupported_facts": [],
            "missing_protected_facts": [],
            "parse_errors": parsed_result["errors"],
            "structure_errors": [
                {
                    "error": (
                        "structure_not_scorable_due_to_"
                        "contract_failure"
                    ),
                }
            ],
            "duplicate_fact_count": parsed_result[
                "duplicate_fact_count"
            ],
            "per_unit_fact_counts": parsed_result[
                "per_unit_fact_counts"
            ],
            "per_unit_membership": {},
        }

    parsed_keys = {
        (
            row["subject"],
            row["attribute"],
            row["value"],
            row["status"],
        )
        for row in parsed_result["parsed_facts"]
    }

    old_fact = update["old_fact"]
    new_fact = update["new_fact"]
    new_current = _fact_tuple(
        new_fact,
        "current",
    )
    old_current = _fact_tuple(
        old_fact,
        "current",
    )
    old_history = _fact_tuple(
        old_fact,
        "history",
    )

    expected_current = {
        _fact_tuple(fact, "current")
        for fact in update["expected_current_facts"]
    }
    target_id = str(update["target_fact_id"])
    protected = {
        _fact_tuple(fact, "current")
        for fact in update["expected_current_facts"]
        if str(fact["fact_id"]) != target_id
    }

    # Stale old target content is scored separately. It is not an
    # unsupported addition merely because the requested update failed.
    allowed_for_unsupported = set(expected_current)
    allowed_for_unsupported.add(old_history)
    allowed_for_unsupported.add(old_current)

    target_new_current = (
        new_current in parsed_keys
    )
    target_old_not_current = (
        old_current not in parsed_keys
    )
    missing_protected = sorted(
        protected - parsed_keys
    )
    unsupported = sorted(
        parsed_keys - allowed_for_unsupported
    )
    protected_ok = not missing_protected

    duplicate_fact_count = int(
        parsed_result["duplicate_fact_count"]
    )
    unsupported_ok = (
        not unsupported
        and duplicate_fact_count == 0
    )

    actual_unit_fact_counts = {
        str(unit.get("memory_id", "")): int(
            parsed_result[
                "per_unit_current_fact_counts"
            ].get(
                str(unit.get("memory_id", "")),
                0,
            )
        )
        for unit in final_memory
    }

    actual_unit_membership = (
        _actual_unit_membership(
            final_memory=final_memory,
            parsed_result=parsed_result,
            update=update,
        )
    )

    structure_errors = []

    if expected_unit_fact_counts is None:
        count_valid = True
    else:
        normalized_expected_counts = {
            str(memory_id): int(count)
            for memory_id, count
            in expected_unit_fact_counts.items()
        }
        count_valid = (
            actual_unit_fact_counts
            == normalized_expected_counts
        )
        if not count_valid:
            structure_errors.append(
                {
                    "error": "unit_fact_count_mismatch",
                    "expected": (
                        normalized_expected_counts
                    ),
                    "actual": actual_unit_fact_counts,
                }
            )

    if expected_unit_membership is None:
        membership_valid = True
    else:
        normalized_expected_membership = {
            str(memory_id): sorted(
                str(fact_id)
                for fact_id in fact_ids
            )
            for memory_id, fact_ids
            in expected_unit_membership.items()
        }
        membership_valid = (
            actual_unit_membership
            == normalized_expected_membership
        )
        if not membership_valid:
            structure_errors.append(
                {
                    "error": "unit_membership_mismatch",
                    "expected": (
                        normalized_expected_membership
                    ),
                    "actual": actual_unit_membership,
                }
            )

    structure_valid = bool(
        count_valid and membership_valid
    )

    e2e = all(
        (
            bool(legal_action),
            bool(execution_success),
            bool(structure_valid),
            target_new_current,
            target_old_not_current,
            protected_ok,
            unsupported_ok,
        )
    )

    failure_reasons = _failure_reasons(
        legal_action=bool(legal_action),
        execution_success=bool(
            execution_success
        ),
        contract_valid=True,
        structure_valid=bool(structure_valid),
        target_new_current=bool(
            target_new_current
        ),
        target_old_not_current=bool(
            target_old_not_current
        ),
        protected_facts_preserved=bool(
            protected_ok
        ),
        no_unsupported_changes=bool(
            unsupported_ok
        ),
    )

    def unpack(keys):
        return [
            {
                "subject": key[0],
                "attribute": key[1],
                "value": key[2],
                "status": key[3],
            }
            for key in keys
        ]

    return {
        "legal_action": bool(legal_action),
        "execution_success": bool(
            execution_success
        ),
        "contract_valid": True,
        "structure_valid": bool(
            structure_valid
        ),
        "membership_valid": bool(
            membership_valid
        ),
        "target_new_current": bool(
            target_new_current
        ),
        "target_old_not_current": bool(
            target_old_not_current
        ),
        "protected_facts_preserved": bool(
            protected_ok
        ),
        "no_unsupported_changes": bool(
            unsupported_ok
        ),
        "end_to_end_success": bool(e2e),
        "failure_reasons": failure_reasons,
        "unsupported_facts": unpack(
            unsupported
        ),
        "missing_protected_facts": unpack(
            missing_protected
        ),
        "parse_errors": [],
        "structure_errors": structure_errors,
        "duplicate_fact_count": (
            duplicate_fact_count
        ),
        "per_unit_fact_counts": (
            actual_unit_fact_counts
        ),
        "per_unit_membership": (
            actual_unit_membership
        ),
    }


def execute_and_evaluate(
    initial_memory: Sequence[Mapping[str, Any]],
    action: Mapping[str, Any],
    expected_action_type: str,
    update: Mapping[str, Any],
) -> dict:
    initial_parse = parse_memory(initial_memory)

    if not initial_parse["contract_valid"]:
        raise ValueError(
            "Initial memory is outside the controlled contract"
        )

    expected_unit_fact_counts = {
        str(unit.get("memory_id", "")): int(
            initial_parse[
                "per_unit_current_fact_counts"
            ].get(
                str(unit.get("memory_id", "")),
                0,
            )
        )
        for unit in initial_memory
    }

    expected_unit_membership = {
        str(unit.get("memory_id", "")): sorted(
            str(fact_id)
            for fact_id in unit.get("fact_ids", [])
        )
        for unit in initial_memory
    }

    execution = execute_action(
        initial_memory,
        action,
        expected_action_type,
    )
    evaluation = evaluate_final_memory(
        execution["final_memory"],
        update,
        execution["legal_action"],
        execution["execution_success"],
        expected_unit_fact_counts=(
            expected_unit_fact_counts
        ),
        expected_unit_membership=(
            expected_unit_membership
        ),
    )
    return {
        "execution": execution,
        "evaluation": evaluation,
    }


def condition_spec(condition: str) -> dict:
    mapping = {
        CONDITION_A: {"representation": REPRESENTATION_COMPOUND, "expected_action_type": ACTION_REWRITE_ALL},
        CONDITION_B: {"representation": REPRESENTATION_COMPOUND, "expected_action_type": ACTION_REWRITE_UNIT},
        CONDITION_C: {"representation": REPRESENTATION_ATOMIC, "expected_action_type": ACTION_REWRITE_ALL},
        CONDITION_D: {"representation": REPRESENTATION_ATOMIC, "expected_action_type": ACTION_REWRITE_UNIT},
        CONDITION_E: {"representation": REPRESENTATION_COMPOUND, "expected_action_type": ACTION_EXACT_PATCH},
    }
    if condition not in mapping:
        raise ValueError("Unknown condition: {}".format(condition))
    result = dict(mapping[condition])
    result["condition"] = condition
    return result


def initial_memory_for_condition(case: Mapping[str, Any], condition: str) -> List[dict]:
    spec = condition_spec(condition)
    return clone_memory(case["representations"][spec["representation"]])


def target_unit_ids(case: Mapping[str, Any], update: Mapping[str, Any]) -> dict:
    """Diagnostic helper only. Never reveal these IDs to the model."""
    target_id = str(update["target_fact_id"])
    result = {}
    for representation in (REPRESENTATION_COMPOUND, REPRESENTATION_ATOMIC):
        result[representation] = [
            str(unit["memory_id"])
            for unit in case["representations"][representation]
            if target_id in [str(x) for x in unit.get("fact_ids", [])]
        ]
    return result
