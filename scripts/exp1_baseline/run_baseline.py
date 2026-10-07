#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run or resume the frozen baseline after its integration gate.

See the root README for stage dependencies."""

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "atomic_memory").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atomic_memory import paths

from atomic_memory.backend import GenerationConfig, OllamaBackend
from atomic_memory.core import (
    ACTION_EXACT_PATCH,
    ACTION_REWRITE_ALL,
    ACTION_REWRITE_UNIT,
    CONDITIONS,
    condition_spec,
    execute_and_evaluate,
    generate_cases,
    initial_memory_for_condition,
    sha256_json,
    target_unit_ids,
)


# ============================================================================
# Frozen formal design
# ============================================================================

FORMAL_MASTER_SEED = 20261001
FORMAL_NUM_CASES = 20
FORMAL_TARGETS_PER_CASE = 2

OLLAMA_BASE_URL = paths.OLLAMA_URL
OLLAMA_TIMEOUT = 600.0
RUNTIME_POLICY = "historical_model_identity_match; current_runtime_frozen_per_batch"

MODELS = (
    "llama3.1:8b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "gemma3:12b-it-q4_K_M",
)

TEMPERATURE = 0.0
SEED = 42
MAX_OUTPUT_TOKENS = 1024
CONTEXT_WINDOW = 8192

EXPECTED_TOTAL_CALLS = (
    FORMAL_NUM_CASES
    * FORMAL_TARGETS_PER_CASE
    * len(CONDITIONS)
    * len(MODELS)
)

if EXPECTED_TOTAL_CALLS != 600:
    raise RuntimeError(
        "Formal call-count invariant changed: {}".format(
            EXPECTED_TOTAL_CALLS
        )
    )

DEV_OUTPUT_DIR = paths.DEV
INTEGRATION_RESULTS_PATH = (
    DEV_OUTPUT_DIR / "integration_smoke_results.jsonl"
)
INTEGRATION_MANIFEST_PATH = (
    DEV_OUTPUT_DIR / "integration_smoke_manifest.json"
)

FORMAL_OUTPUT_DIR = paths.BASELINE
FORMAL_CASES_PATH = FORMAL_OUTPUT_DIR / "formal_cases.jsonl"
PREFLIGHT_PATH = FORMAL_OUTPUT_DIR / "formal_preflight.json"
FREEZE_MANIFEST_PATH = (
    FORMAL_OUTPUT_DIR / "formal_freeze_manifest.json"
)
RESULTS_PATH = FORMAL_OUTPUT_DIR / "formal_results.jsonl"
RUN_SUMMARY_PATH = FORMAL_OUTPUT_DIR / "formal_run_summary.json"

SYSTEM_PROMPT = """You are a memory editing controller.

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

HIDDEN_PROMPT_KEYS = (
    "fact_ids",
    "target_fact_id",
    "expected_current_facts",
    "allowed_historical_facts",
    "case_sha256",
    "representation_hash",
)

PROTOCOL_VERSION = "representation-edit-scope-controlled-pilot-v1"


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
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


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
    return datetime.now(
        timezone.utc
    ).isoformat()


def read_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(
                    "JSONL row {} must be an object: {}".format(
                        line_number,
                        path,
                    )
                )
            rows.append(value)
    return rows


def write_json_atomic(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_name(
        path.name + ".tmp"
    )
    with temporary.open(
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
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(
        str(temporary),
        str(path),
    )


def write_jsonl_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_name(
        path.name + ".tmp"
    )
    with temporary.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
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
    os.replace(
        str(temporary),
        str(path),
    )


def append_jsonl_fsync(
    path: Path,
    row: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as handle:
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


def parse_action_response(
    text: str,
) -> Dict[str, Any]:
    """Strict JSON-object parser. No repair and no fence removal."""
    candidate = text.strip()
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError(
            "response JSON must be an object"
        )
    return value


def public_memory(
    memory: Sequence[Mapping[str, Any]],
) -> List[dict]:
    """Expose only public memory_id/text, never latent fact_ids."""
    return [
        {
            "memory_id": str(
                unit["memory_id"]
            ),
            "text": str(
                unit["text"]
            ),
        }
        for unit in memory
    ]


def model_slug(model: str) -> str:
    return (
        model.replace(":", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def result_key(
    row: Mapping[str, Any],
) -> Tuple[str, str, str, str]:
    return (
        str(row["case_id"]),
        str(row["update_id"]),
        str(row["condition"]),
        str(row["model"]),
    )


# ============================================================================
# Exact action schemas and prompt adapter
# ============================================================================

def action_schema(
    expected_action_type: str,
    memory: Sequence[Mapping[str, Any]],
) -> dict:
    memory_ids = [
        str(unit["memory_id"])
        for unit in memory
    ]

    if expected_action_type == ACTION_REWRITE_ALL:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "action",
                "units",
            ],
            "properties": {
                "action": {
                    "type": "string",
                    "const": ACTION_REWRITE_ALL,
                },
                "units": {
                    "type": "array",
                    "minItems": len(
                        memory_ids
                    ),
                    "maxItems": len(
                        memory_ids
                    ),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "memory_id",
                            "text",
                        ],
                        "properties": {
                            "memory_id": {
                                "type": "string",
                                "enum": memory_ids,
                            },
                            "text": {
                                "type": "string",
                            },
                        },
                    },
                },
            },
        }

    if expected_action_type == ACTION_REWRITE_UNIT:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "action",
                "memory_id",
                "text",
            ],
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

    raise ValueError(
        "Unsupported action type: {}".format(
            expected_action_type
        )
    )


def action_instruction(
    expected_action_type: str,
) -> str:
    if expected_action_type == ACTION_REWRITE_ALL:
        return (
            "Rewrite the complete memory state. Return every existing "
            "memory_id exactly once. Keep unrelated text unchanged whenever "
            "possible. The output action must be rewrite_all."
        )

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
        "Unsupported action type: {}".format(
            expected_action_type
        )
    )


def build_user_prompt(
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
        + action_instruction(
            expected_action_type
        )
        + "\n\nReturn only the JSON action."
    )


# ============================================================================
# Diagnostics
# ============================================================================

def locator_success(
    case: Mapping[str, Any],
    update: Mapping[str, Any],
    condition: str,
    action: Mapping[str, Any],
) -> Any:
    spec = condition_spec(condition)
    expected_action_type = spec[
        "expected_action_type"
    ]

    if expected_action_type == ACTION_REWRITE_ALL:
        return None

    selected_id = action.get(
        "memory_id"
    )
    if not isinstance(
        selected_id,
        str,
    ):
        return False

    hidden_targets = target_unit_ids(
        case,
        update,
    )[spec["representation"]]
    return selected_id in hidden_targets


def diagnostic_failure_reasons(
    response_json_valid: bool,
    locator: Any,
    result: Mapping[str, Any],
) -> List[str]:
    """Overlapping diagnostic labels; never sum them as independent failures."""
    reasons = []

    if not response_json_valid:
        reasons.append(
            "response_json_invalid"
        )

    if locator is False:
        reasons.append(
            "locator_failure"
        )

    for reason in result[
        "evaluation"
    ].get(
        "failure_reasons",
        [],
    ):
        reason_text = str(
            reason
        )
        if reason_text not in reasons:
            reasons.append(
                reason_text
            )

    return reasons


def primary_failure_stage(
    failure_reasons: Sequence[str],
    result: Mapping[str, Any],
) -> str:
    if result["evaluation"][
        "end_to_end_success"
    ]:
        return "success"
    if failure_reasons:
        return str(
            failure_reasons[0]
        )
    return "other_failure"


# ============================================================================
# Integration-smoke binding
# ============================================================================

def verify_integration_gate() -> dict:
    if not INTEGRATION_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Missing integration smoke manifest: {}".format(
                INTEGRATION_MANIFEST_PATH
            )
        )
    if not INTEGRATION_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Missing integration smoke results: {}".format(
                INTEGRATION_RESULTS_PATH
            )
        )

    manifest = read_json(
        INTEGRATION_MANIFEST_PATH
    )
    rows = read_jsonl(
        INTEGRATION_RESULTS_PATH
    )

    required_manifest = {
        "status": (
            "INTEGRATION_SMOKE_COMPLETE"
        ),
        "completed_call_count": 15,
        "implementation_smoke": "PASS",
        "shared_order_pass": True,
        "fixed_unit_membership_required": True,
        "expected_call_count": 15,
    }

    for key, expected in required_manifest.items():
        actual = manifest.get(
            key
        )
        if actual != expected:
            raise RuntimeError(
                "Integration gate failed: {} expected {!r}, got {!r}".format(
                    key,
                    expected,
                    actual,
                )
            )

    if tuple(
        manifest.get("models", [])
    ) != tuple(MODELS):
        raise RuntimeError(
            "Integration model list differs from formal model list."
        )

    if tuple(
        manifest.get("conditions", [])
    ) != tuple(CONDITIONS):
        raise RuntimeError(
            "Integration condition list differs from formal conditions."
        )

    expected_generation = {
        "temperature": TEMPERATURE,
        "seed": SEED,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "context_window": CONTEXT_WINDOW,
    }
    if manifest.get(
        "generation_config"
    ) != expected_generation:
        raise RuntimeError(
            "Integration generation config differs from formal config."
        )

    if manifest.get(
        "system_prompt_sha256"
    ) != sha256_text(SYSTEM_PROMPT):
        raise RuntimeError(
            "Formal system prompt does not match the tested integration prompt."
        )

    core_path = (
        PROJECT_ROOT
        / "atomic_memory"
        / "core.py"
    )
    backend_path = (
        PROJECT_ROOT
        / "atomic_memory"
        / "backend.py"
    )

    if manifest.get(
        "core_sha256"
    ) != sha256_file(core_path):
        raise RuntimeError(
            "Current core file differs from the integration-tested core."
        )

    if manifest.get(
        "backend_sha256"
    ) != sha256_file(
        backend_path
    ):
        raise RuntimeError(
            "Current backend file differs from the integration-tested backend."
        )

    if len(rows) != 15:
        raise RuntimeError(
            "Expected 15 integration rows, found {}.".format(
                len(rows)
            )
        )

    expected_pairs = {
        (
            condition,
            model,
        )
        for condition in CONDITIONS
        for model in MODELS
    }
    actual_pairs = {
        (
            str(row["condition"]),
            str(row["model"]),
        )
        for row in rows
    }
    if actual_pairs != expected_pairs:
        raise RuntimeError(
            "Integration rows do not cover exactly 5 conditions x 3 models."
        )

    if len(
        {
            result_key(row)
            for row in rows
        }
    ) != 15:
        raise RuntimeError(
            "Duplicate integration result keys detected."
        )

    for row in rows:
        if not bool(
            row.get("response_json_valid")
        ):
            raise RuntimeError(
                "Integration smoke contains invalid JSON response: {}".format(
                    row.get("request_id")
                )
            )

        rebuilt_prompt = build_user_prompt(
            row["initial_memory"],
            row["evidence_text"],
            row["expected_action_type"],
        )
        if rebuilt_prompt != row.get(
            "user_prompt"
        ):
            raise RuntimeError(
                "Formal user-prompt adapter differs from integration smoke: {}".format(
                    row.get("request_id")
                )
            )

        rebuilt_schema = action_schema(
            row["expected_action_type"],
            row["initial_memory"],
        )
        if canonical_json(
            rebuilt_schema
        ) != canonical_json(
            row["response_schema"]
        ):
            raise RuntimeError(
                "Formal response schema differs from integration smoke: {}".format(
                    row.get("request_id")
                )
            )

        if row.get(
            "system_prompt_sha256"
        ) != sha256_text(
            SYSTEM_PROMPT
        ):
            raise RuntimeError(
                "Integration row system prompt hash mismatch: {}".format(
                    row.get("request_id")
                )
            )

    return {
        "manifest": manifest,
        "rows": rows,
        "manifest_sha256": sha256_file(
            INTEGRATION_MANIFEST_PATH
        ),
        "results_sha256": sha256_file(
            INTEGRATION_RESULTS_PATH
        ),
    }


# ============================================================================
# Formal-case build and deterministic validation
# ============================================================================

def case_without_sha(
    case: Mapping[str, Any],
) -> dict:
    result = dict(case)
    result.pop(
        "case_sha256",
        None,
    )
    return result


def validate_formal_cases(
    cases: Sequence[Mapping[str, Any]],
    integration_manifest: Mapping[str, Any],
) -> dict:
    errors = []

    if len(cases) != FORMAL_NUM_CASES:
        errors.append(
            "case_count"
        )

    case_ids = [
        str(case["case_id"])
        for case in cases
    ]
    if len(
        set(case_ids)
    ) != FORMAL_NUM_CASES:
        errors.append(
            "case_ids_not_unique"
        )

    integration_case_sha = str(
        integration_manifest.get(
            "case_sha256",
            "",
        )
    )

    # Exclude initial fact sets from the fixed development seed without
    # requiring legacy snapshots from an earlier state-layout implementation.
    legacy_fact_sets = {
        sha256_json(case["facts"])
        for case in generate_cases(num_cases=20, master_seed=20260926, targets_per_case=2)
    }

    prompt_hash_rows = []
    target_instance_count = 0

    for case in cases:
        stored_case_sha = str(
            case.get(
                "case_sha256",
                "",
            )
        )
        computed_case_sha = sha256_json(
            case_without_sha(case)
        )
        if stored_case_sha != computed_case_sha:
            errors.append(
                "{}:case_sha".format(
                    case["case_id"]
                )
            )

        if stored_case_sha == integration_case_sha:
            errors.append(
                "{}:reuses_integration_case".format(
                    case["case_id"]
                )
            )

        if sha256_json(case["facts"]) in legacy_fact_sets:
            errors.append("{}:reuses_legacy_dev_fact_set".format(case["case_id"]))

        invariants = case.get(
            "design_invariants",
            {},
        )
        required_invariants = {
            "fact_count": 6,
            "compound_unit_count": 3,
            "compound_facts_per_unit": 2,
            "atomic_unit_count": 6,
            "atomic_facts_per_unit": 1,
            "representations_built_before_target": True,
            "targets_share_same_initial_memory": True,
            "shared_fact_order_across_representations": True,
            "fixed_unit_membership_required": True,
        }
        for key, expected in required_invariants.items():
            if invariants.get(
                key
            ) != expected:
                errors.append(
                    "{}:invariant:{}".format(
                        case["case_id"],
                        key,
                    )
                )

        shared = [
            str(x)
            for x in case[
                "shared_fact_order"
            ]
        ]

        compound_order = []
        for unit in case[
            "representations"
        ]["compound"]:
            compound_order.extend(
                str(x)
                for x in unit[
                    "fact_ids"
                ]
            )

        atomic_order = [
            str(
                unit["fact_ids"][0]
            )
            for unit in case[
                "representations"
            ]["atomic"]
        ]

        if (
            compound_order != shared
            or atomic_order != shared
        ):
            errors.append(
                "{}:shared_order".format(
                    case["case_id"]
                )
            )

        updates = list(
            case["updates"]
        )
        if len(
            updates
        ) != FORMAL_TARGETS_PER_CASE:
            errors.append(
                "{}:update_count".format(
                    case["case_id"]
                )
            )
            continue

        target_fact_ids = [
            str(
                update[
                    "target_fact_id"
                ]
            )
            for update in updates
        ]
        if len(
            set(target_fact_ids)
        ) != FORMAL_TARGETS_PER_CASE:
            errors.append(
                "{}:duplicate_targets".format(
                    case["case_id"]
                )
            )

        for update in updates:
            target_instance_count += 1

            if (
                str(
                    update["old_fact"]["value"]
                )
                == str(
                    update["new_fact"]["value"]
                )
            ):
                errors.append(
                    "{}:{}:old_equals_new".format(
                        case["case_id"],
                        update["update_id"],
                    )
                )

            hidden_targets = target_unit_ids(
                case,
                update,
            )
            if (
                len(
                    hidden_targets["compound"]
                ) != 1
                or len(
                    hidden_targets["atomic"]
                ) != 1
            ):
                errors.append(
                    "{}:{}:target_unit_count".format(
                        case["case_id"],
                        update["update_id"],
                    )
                )

            for condition in CONDITIONS:
                spec = condition_spec(
                    condition
                )
                memory = initial_memory_for_condition(
                    case,
                    condition,
                )
                prompt = build_user_prompt(
                    memory,
                    update[
                        "evidence_text"
                    ],
                    spec[
                        "expected_action_type"
                    ],
                )
                lowered_prompt = prompt.lower()
                for hidden_key in HIDDEN_PROMPT_KEYS:
                    if (
                        hidden_key.lower()
                        in lowered_prompt
                    ):
                        errors.append(
                            "{}:{}:{}:hidden_prompt_key:{}".format(
                                case["case_id"],
                                update["update_id"],
                                condition,
                                hidden_key,
                            )
                        )

                schema = action_schema(
                    spec[
                        "expected_action_type"
                    ],
                    memory,
                )

                prompt_hash_rows.append(
                    {
                        "case_id": case[
                            "case_id"
                        ],
                        "update_id": update[
                            "update_id"
                        ],
                        "condition": condition,
                        "user_prompt_sha256": (
                            sha256_text(
                                prompt
                            )
                        ),
                        "response_schema_sha256": (
                            sha256_text(
                                canonical_json(
                                    schema
                                )
                            )
                        ),
                    }
                )

    if target_instance_count != (
        FORMAL_NUM_CASES
        * FORMAL_TARGETS_PER_CASE
    ):
        errors.append(
            "target_instance_count"
        )

    if errors:
        raise RuntimeError(
            "Formal case validation failed: {}".format(
                "; ".join(
                    errors[:20]
                )
            )
        )

    return {
        "case_count": len(cases),
        "target_instance_count": (
            target_instance_count
        ),
        "prompt_schema_binding_sha256": (
            sha256_json(
                prompt_hash_rows
            )
        ),
        "case_sha256_values": [
            str(
                case["case_sha256"]
            )
            for case in cases
        ],
    }


def build_or_verify_formal_cases(
    integration_manifest: Mapping[str, Any],
) -> Tuple[List[dict], dict, str]:
    FORMAL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated = generate_cases(
        num_cases=FORMAL_NUM_CASES,
        master_seed=FORMAL_MASTER_SEED,
        targets_per_case=FORMAL_TARGETS_PER_CASE,
    )

    if FORMAL_CASES_PATH.exists():
        cases = read_jsonl(
            FORMAL_CASES_PATH
        )
        if canonical_json(
            cases
        ) != canonical_json(
            generated
        ):
            raise RuntimeError(
                "Existing formal_cases.jsonl does not equal deterministic "
                "generation from the frozen formal seed."
            )
        case_status = "VERIFIED_EXISTING"
    else:
        if RESULTS_PATH.exists():
            raise RuntimeError(
                "Formal results exist but formal_cases.jsonl is missing."
            )
        if FREEZE_MANIFEST_PATH.exists():
            raise RuntimeError(
                "Freeze manifest exists but formal_cases.jsonl is missing."
            )

        write_jsonl_atomic(
            FORMAL_CASES_PATH,
            generated,
        )
        cases = generated
        case_status = "CREATED"

    validation = validate_formal_cases(
        cases,
        integration_manifest,
    )

    return (
        cases,
        validation,
        case_status,
    )


# ============================================================================
# Balanced formal execution order
# ============================================================================

def rotated_conditions(
    target_instance_index: int,
) -> Tuple[str, ...]:
    """5-condition cyclic rotation; 40 target instances => perfect balance."""
    offset = int(
        target_instance_index
    ) % len(
        CONDITIONS
    )
    values = list(
        CONDITIONS
    )
    return tuple(
        values[offset:]
        + values[:offset]
    )


def build_expected_plan(
    cases: Sequence[Mapping[str, Any]],
) -> List[dict]:
    plan = []
    target_instance_index = 0

    for case_index, case in enumerate(
        cases
    ):
        updates = list(
            case["updates"]
        )
        for update_index, update in enumerate(
            updates
        ):
            condition_order = rotated_conditions(
                target_instance_index
            )

            for condition_position, condition in enumerate(
                condition_order,
                start=1,
            ):
                for model_index, model in enumerate(
                    MODELS
                ):
                    # planned_call_index is model-major below; this temporary
                    # plan is used only for key validation and balance checks.
                    plan.append(
                        {
                            "case_index": case_index,
                            "case_id": case[
                                "case_id"
                            ],
                            "update_index": update_index,
                            "update_id": update[
                                "update_id"
                            ],
                            "target_instance_index": (
                                target_instance_index
                            ),
                            "condition": condition,
                            "condition_execution_position": (
                                condition_position
                            ),
                            "model_index": model_index,
                            "model": model,
                        }
                    )

            target_instance_index += 1

    if len(plan) != EXPECTED_TOTAL_CALLS:
        raise RuntimeError(
            "Expected plan length {}, found {}.".format(
                EXPECTED_TOTAL_CALLS,
                len(plan),
            )
        )

    keys = {
        (
            str(row["case_id"]),
            str(row["update_id"]),
            str(row["condition"]),
            str(row["model"]),
        )
        for row in plan
    }
    if len(keys) != EXPECTED_TOTAL_CALLS:
        raise RuntimeError(
            "Formal plan contains duplicate result keys."
        )

    # Verify each condition appears exactly 8 times in each execution position
    # per model: 40 target instances / 5 positions = 8.
    balance = {}
    for model in MODELS:
        balance[model] = {}
        for condition in CONDITIONS:
            position_counts = {}
            for position in range(
                1,
                len(CONDITIONS) + 1,
            ):
                count = sum(
                    1
                    for row in plan
                    if row["model"] == model
                    and row["condition"] == condition
                    and row[
                        "condition_execution_position"
                    ] == position
                )
                position_counts[
                    str(position)
                ] = count
                if count != 8:
                    raise RuntimeError(
                        "Condition-order balance failed: {} {} position {} "
                        "count {}".format(
                            model,
                            condition,
                            position,
                            count,
                        )
                    )
            balance[model][
                condition
            ] = position_counts

    return plan


def model_major_execution_rows(
    cases: Sequence[Mapping[str, Any]],
) -> List[dict]:
    """Efficient model-major execution with balanced A-E positions."""
    base = []
    target_instance_index = 0

    for case_index, case in enumerate(
        cases
    ):
        for update_index, update in enumerate(
            case["updates"]
        ):
            condition_order = rotated_conditions(
                target_instance_index
            )

            for condition_position, condition in enumerate(
                condition_order,
                start=1,
            ):
                base.append(
                    {
                        "case_index": case_index,
                        "case_id": case[
                            "case_id"
                        ],
                        "update_index": update_index,
                        "update_id": update[
                            "update_id"
                        ],
                        "target_instance_index": (
                            target_instance_index
                        ),
                        "condition": condition,
                        "condition_execution_position": (
                            condition_position
                        ),
                    }
                )

            target_instance_index += 1

    rows = []
    planned_call_index = 0

    for model_index, model in enumerate(
        MODELS
    ):
        for base_row in base:
            planned_call_index += 1
            row = dict(
                base_row
            )
            row.update(
                {
                    "model_index": model_index,
                    "model": model,
                    "planned_call_index": (
                        planned_call_index
                    ),
                }
            )
            rows.append(
                row
            )

    if len(rows) != EXPECTED_TOTAL_CALLS:
        raise RuntimeError(
            "Model-major execution plan size mismatch."
        )

    return rows


# ============================================================================
# Model preflight and freeze
# ============================================================================

def snapshot_fingerprint(
    snapshot: Mapping[str, Any],
) -> dict:
    """Fields that materially bind the local model/runtime snapshot."""
    return {
        "provider": snapshot.get(
            "provider"
        ),
        "ollama_version": snapshot.get(
            "ollama_version"
        ),
        "model": snapshot.get(
            "model"
        ),
        "digest": snapshot.get(
            "digest",
            snapshot.get(
                "model_digest"
            ),
        ),
        "model_digest": snapshot.get(
            "model_digest",
            snapshot.get(
                "digest"
            ),
        ),
        "parameter_size": snapshot.get(
            "parameter_size"
        ),
        "quantization_level": snapshot.get(
            "quantization_level"
        ),
    }


def model_identity_fingerprint(snapshot: Mapping[str, Any]) -> dict:
    """Match model artifacts independently of the historical Ollama version."""
    identity = snapshot_fingerprint(snapshot)
    identity.pop("ollama_version")
    return identity


def integration_runtime_reference(
    integration_manifest: Mapping[str, Any],
    model_snapshots: Mapping[str, Any],
) -> dict:
    historical = {
        model: integration_manifest["model_snapshots"][model].get("ollama_version")
        for model in MODELS
    }
    current = {
        model: model_snapshots[model].get("ollama_version")
        for model in MODELS
    }
    return {
        "policy": RUNTIME_POLICY,
        "historical_smoke_scope": "prompt/schema adapter and model artifact identity",
        "historical_integration_versions": historical,
        "current_versions": current,
        "version_changed_models": [
            model for model in MODELS if historical[model] != current[model]
        ],
    }


def preflight_models(
    backend: OllamaBackend,
    integration_manifest: Mapping[str, Any],
) -> dict:
    tested_snapshots = integration_manifest.get(
        "model_snapshots",
        {},
    )
    current = {}

    for model in MODELS:
        snapshot = backend.describe(
            model
        )
        current[model] = snapshot

        if snapshot.get(
            "quantization_level"
        ) != "Q4_K_M":
            raise RuntimeError(
                "Unexpected quantization for {}: {}".format(
                    model,
                    snapshot.get(
                        "quantization_level"
                    ),
                )
            )

        tested = tested_snapshots.get(
            model
        )
        if not isinstance(
            tested,
            Mapping,
        ):
            raise RuntimeError(
                "Integration manifest lacks model snapshot for {}".format(
                    model
                )
            )

        observed_fingerprint = model_identity_fingerprint(snapshot)
        tested_fingerprint = model_identity_fingerprint(tested)
        if observed_fingerprint != tested_fingerprint:
            differences = "\n".join(
                "  {}: expected={!r}, current={!r}".format(
                    field, expected, observed_fingerprint[field]
                )
                for field, expected in tested_fingerprint.items()
                if expected != observed_fingerprint[field]
            )
            raise RuntimeError(
                "Current model identity differs from integration "
                "smoke for {}:\n{}\n"
                "Ollama version changes are recorded separately; model names, "
                "weights and quantization must still match.".format(
                    model, differences
                )
            )

    versions = {snapshot.get("ollama_version") for snapshot in current.values()}
    if len(versions) != 1 or versions.intersection({None, "", "unknown"}):
        raise RuntimeError(
            "All models must report the same known Ollama version before freeze."
        )
    return current


def schema_hashes_for_freeze(
    cases: Sequence[Mapping[str, Any]],
) -> dict:
    first_case = cases[0]
    result = {}

    for condition in CONDITIONS:
        spec = condition_spec(
            condition
        )
        memory = initial_memory_for_condition(
            first_case,
            condition,
        )
        schema = action_schema(
            spec[
                "expected_action_type"
            ],
            memory,
        )
        result[condition] = sha256_text(
            canonical_json(
                schema
            )
        )

    return result


def build_immutable_freeze_payload(
    cases: Sequence[Mapping[str, Any]],
    case_validation: Mapping[str, Any],
    integration_gate: Mapping[str, Any],
    model_snapshots: Mapping[str, Any],
    execution_plan: Sequence[Mapping[str, Any]],
) -> dict:
    core_path = (
        PROJECT_ROOT
        / "atomic_memory"
        / "core.py"
    )
    backend_path = (
        PROJECT_ROOT
        / "atomic_memory"
        / "backend.py"
    )
    runner_path = Path(
        __file__
    ).resolve()

    immutable = {
        "protocol_version": PROTOCOL_VERSION,
        "formal_pilot_frozen": True,
        "study_scope": (
            "controlled synthetic representation x edit-scope formal pilot"
        ),
        "primary_endpoint": (
            "Legal Action AND Execution Success AND Structure Valid AND "
            "Target Correct AND Protected Facts Preserved AND "
            "No Unsupported Changes"
        ),
        "cluster_unit": "case",
        "formal_master_seed": (
            FORMAL_MASTER_SEED
        ),
        "formal_num_cases": (
            FORMAL_NUM_CASES
        ),
        "formal_targets_per_case": (
            FORMAL_TARGETS_PER_CASE
        ),
        "conditions": list(
            CONDITIONS
        ),
        "models": list(
            MODELS
        ),
        "expected_total_calls": (
            EXPECTED_TOTAL_CALLS
        ),
        "generation_config": {
            "temperature": TEMPERATURE,
            "seed": SEED,
            "max_output_tokens": (
                MAX_OUTPUT_TOKENS
            ),
            "context_window": (
                CONTEXT_WINDOW
            ),
        },
        "ollama_base_url": (
            OLLAMA_BASE_URL
        ),
        "system_prompt": (
            SYSTEM_PROMPT
        ),
        "system_prompt_sha256": (
            sha256_text(
                SYSTEM_PROMPT
            )
        ),
        "response_schema_sha256_by_condition": (
            schema_hashes_for_freeze(
                cases
            )
        ),
        "formal_cases_sha256": (
            sha256_json(
                list(cases)
            )
        ),
        "formal_cases_file_sha256": (
            sha256_file(
                FORMAL_CASES_PATH
            )
        ),
        "prompt_schema_binding_sha256": (
            case_validation[
                "prompt_schema_binding_sha256"
            ]
        ),
        "condition_order_policy": (
            "5-condition cyclic rotation by target_instance_index; "
            "40 target instances give each condition 8 appearances in each "
            "execution position per model"
        ),
        "execution_plan_sha256": (
            sha256_json(
                list(
                    execution_plan
                )
            )
        ),
        "core_sha256": (
            sha256_file(
                core_path
            )
        ),
        "backend_sha256": (
            sha256_file(
                backend_path
            )
        ),
        "runner_sha256": (
            sha256_file(
                runner_path
            )
        ),
        "integration_manifest_sha256": (
            integration_gate[
                "manifest_sha256"
            ]
        ),
        "integration_results_sha256": (
            integration_gate[
                "results_sha256"
            ]
        ),
        "integration_runtime_reference": integration_runtime_reference(
            integration_gate["manifest"], model_snapshots
        ),
        "model_snapshots": {
            model: snapshot_fingerprint(
                model_snapshots[
                    model
                ]
            )
            for model in MODELS
        },
        "failure_labels_overlap": True,
        "failure_label_reporting_rule": (
            "Do not sum overlapping failure-reason counts as independent "
            "failed calls."
        ),
        "cost_measurement_scope": (
            "input tokens, output tokens, and wall-clock generation latency; "
            "retrieval cost is not measured"
        ),
        "no_prompt_rescue": True,
        "outcome_dependent_protocol_changes_allowed": False,
    }

    immutable[
        "freeze_fingerprint_sha256"
    ] = sha256_json(
        immutable
    )

    return immutable


def freeze_or_verify(
    immutable: Mapping[str, Any],
) -> str:
    if FREEZE_MANIFEST_PATH.exists():
        existing = read_json(
            FREEZE_MANIFEST_PATH
        )
        existing_immutable = existing.get(
            "immutable_protocol"
        )
        if existing_immutable != dict(
            immutable
        ):
            raise RuntimeError(
                "Freeze manifest mismatch. Protocol/code/model/cases changed "
                "after freeze. Do not continue formal calls."
            )
        return "VERIFIED_EXISTING"

    if RESULTS_PATH.exists():
        raise RuntimeError(
            "Formal results exist but freeze manifest is missing."
        )

    manifest = {
        "status": "FORMAL_PROTOCOL_FROZEN",
        "created_at_utc": utc_now_iso(),
        "immutable_protocol": dict(
            immutable
        ),
    }
    write_json_atomic(
        FREEZE_MANIFEST_PATH,
        manifest,
    )
    return "CREATED"


# ============================================================================
# Existing-result validation and summary
# ============================================================================

def expected_key_set(
    cases: Sequence[Mapping[str, Any]],
) -> Set[Tuple[str, str, str, str]]:
    return {
        (
            str(case["case_id"]),
            str(update["update_id"]),
            str(condition),
            str(model),
        )
        for case in cases
        for update in case["updates"]
        for condition in CONDITIONS
        for model in MODELS
    }


def load_and_validate_existing_results(
    cases: Sequence[Mapping[str, Any]],
    immutable: Mapping[str, Any],
) -> Tuple[List[dict], Set[Tuple[str, str, str, str]]]:
    expected = expected_key_set(
        cases
    )

    if not RESULTS_PATH.exists():
        return [], set()

    rows = read_jsonl(
        RESULTS_PATH
    )
    seen = set()

    case_by_id = {
        str(case["case_id"]): case
        for case in cases
    }

    for row in rows:
        key = result_key(
            row
        )
        if key not in expected:
            raise RuntimeError(
                "Unexpected formal result key: {}".format(
                    key
                )
            )
        if key in seen:
            raise RuntimeError(
                "Duplicate formal result key: {}".format(
                    key
                )
            )
        seen.add(
            key
        )

        case = case_by_id[
            key[0]
        ]
        if row.get(
            "case_sha256"
        ) != case.get(
            "case_sha256"
        ):
            raise RuntimeError(
                "Stored case hash mismatch for {}".format(
                    row.get(
                        "request_id"
                    )
                )
            )

        if row.get(
            "system_prompt_sha256"
        ) != immutable[
            "system_prompt_sha256"
        ]:
            raise RuntimeError(
                "Stored prompt hash mismatch for {}".format(
                    row.get(
                        "request_id"
                    )
                )
            )

        expected_schema_hash = immutable[
            "response_schema_sha256_by_condition"
        ][key[2]]
        if row.get(
            "response_schema_sha256"
        ) != expected_schema_hash:
            raise RuntimeError(
                "Stored schema hash mismatch for {}".format(
                    row.get(
                        "request_id"
                    )
                )
            )

    return rows, seen


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict:
    e2e_success = 0
    reason_counts = {}

    for row in rows:
        if bool(
            row.get(
                "evaluation",
                {},
            ).get(
                "end_to_end_success"
            )
        ):
            e2e_success += 1

        for reason in row.get(
            "failure_reasons",
            [],
        ):
            reason_text = str(
                reason
            )
            reason_counts[
                reason_text
            ] = int(
                reason_counts.get(
                    reason_text,
                    0,
                )
            ) + 1

    return {
        "completed_calls": len(
            rows
        ),
        "expected_calls": (
            EXPECTED_TOTAL_CALLS
        ),
        "remaining_calls": (
            EXPECTED_TOTAL_CALLS
            - len(rows)
        ),
        "e2e_success_count": (
            e2e_success
        ),
        "e2e_failure_count": (
            len(rows)
            - e2e_success
        ),
        "failure_reason_row_counts": (
            reason_counts
        ),
        "failure_reason_count_warning": (
            "Reasons overlap; do not sum them as independent failed calls."
        ),
    }


def write_run_summary(
    rows: Sequence[Mapping[str, Any]],
    status: str,
    error: Optional[str] = None,
) -> None:
    summary = summarize_rows(
        rows
    )
    summary.update(
        {
            "status": status,
            "formal_pilot_frozen": True,
            "formal_results_path": str(
                RESULTS_PATH
            ),
            "freeze_manifest_path": str(
                FREEZE_MANIFEST_PATH
            ),
            "updated_at_utc": utc_now_iso(),
        }
    )
    if error is not None:
        summary["error"] = str(
            error
        )

    if RESULTS_PATH.exists():
        summary[
            "formal_results_file_sha256"
        ] = sha256_file(
            RESULTS_PATH
        )

    write_json_atomic(
        RUN_SUMMARY_PATH,
        summary,
    )


# ============================================================================
# Formal call execution
# ============================================================================

def run_formal_calls(
    backend: OllamaBackend,
    cases: Sequence[Mapping[str, Any]],
    immutable: Mapping[str, Any],
    existing_rows: List[dict],
    seen: Set[Tuple[str, str, str, str]],
) -> None:
    case_by_id = {
        str(case["case_id"]): case
        for case in cases
    }

    config_by_model = {
        model: GenerationConfig(
            model=model,
            temperature=TEMPERATURE,
            seed=SEED,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            context_window=CONTEXT_WINDOW,
        )
        for model in MODELS
    }

    execution_rows = model_major_execution_rows(
        cases
    )

    print()
    print("=" * 78)
    print(
        "FORMAL CONTROLLED PILOT — RUN / RESUME"
    )
    print("=" * 78)
    print(
        "Completed: {}/{}".format(
            len(seen),
            EXPECTED_TOTAL_CALLS,
        )
    )
    print(
        "Remaining: {}".format(
            EXPECTED_TOTAL_CALLS
            - len(seen)
        )
    )
    print(
        "Execution: model-major; A-E position is cyclically balanced."
    )
    print()

    if len(seen) == EXPECTED_TOTAL_CALLS:
        write_run_summary(
            existing_rows,
            "FORMAL_PILOT_COMPLETE",
        )
        print(
            "All 600 formal calls are already complete. No new calls."
        )
        return

    write_run_summary(
        existing_rows,
        "FORMAL_PILOT_RUNNING",
    )

    rows = list(
        existing_rows
    )

    try:
        for plan_row in execution_rows:
            key = (
                str(
                    plan_row[
                        "case_id"
                    ]
                ),
                str(
                    plan_row[
                        "update_id"
                    ]
                ),
                str(
                    plan_row[
                        "condition"
                    ]
                ),
                str(
                    plan_row[
                        "model"
                    ]
                ),
            )
            if key in seen:
                continue

            case = case_by_id[
                key[0]
            ]
            update = next(
                candidate
                for candidate in case[
                    "updates"
                ]
                if str(
                    candidate[
                        "update_id"
                    ]
                ) == key[1]
            )
            condition = key[2]
            model = key[3]

            spec = condition_spec(
                condition
            )
            initial_memory = initial_memory_for_condition(
                case,
                condition,
            )
            expected_action_type = spec[
                "expected_action_type"
            ]

            schema = action_schema(
                expected_action_type,
                initial_memory,
            )
            schema_hash = sha256_text(
                canonical_json(
                    schema
                )
            )
            expected_schema_hash = immutable[
                "response_schema_sha256_by_condition"
            ][condition]
            if schema_hash != expected_schema_hash:
                raise RuntimeError(
                    "Runtime schema hash drift before call: {}".format(
                        key
                    )
                )

            user_prompt = build_user_prompt(
                initial_memory,
                update[
                    "evidence_text"
                ],
                expected_action_type,
            )

            lowered_prompt = user_prompt.lower()
            for hidden_key in HIDDEN_PROMPT_KEYS:
                if hidden_key.lower() in lowered_prompt:
                    raise RuntimeError(
                        "Hidden metadata leaked into prompt before call: {} {}".format(
                            key,
                            hidden_key,
                        )
                    )

            request_id = (
                "formal__{}__{}__{}__{}".format(
                    model_slug(
                        model
                    ),
                    case[
                        "case_id"
                    ],
                    update[
                        "update_id"
                    ],
                    condition,
                )
            )

            current_number = (
                len(seen) + 1
            )
            print(
                "[{}/{}] {} | {} | {} | {}".format(
                    current_number,
                    EXPECTED_TOTAL_CALLS,
                    model,
                    case[
                        "case_id"
                    ],
                    update[
                        "update_id"
                    ],
                    condition,
                )
            )

            call_start = time.perf_counter()
            response = backend.generate(
                request_id=request_id,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                config=config_by_model[
                    model
                ],
                response_schema=schema,
            )
            latency_seconds = (
                time.perf_counter()
                - call_start
            )

            response_json_valid = True
            parse_error = None
            try:
                action = parse_action_response(
                    response.text
                )
            except Exception as exc:
                response_json_valid = False
                parse_error = "{}: {}".format(
                    type(exc).__name__,
                    str(exc),
                )
                action = {}

            result = execute_and_evaluate(
                initial_memory=initial_memory,
                action=action,
                expected_action_type=(
                    expected_action_type
                ),
                update=update,
            )

            locator = locator_success(
                case=case,
                update=update,
                condition=condition,
                action=action,
            )
            reasons = diagnostic_failure_reasons(
                response_json_valid=(
                    response_json_valid
                ),
                locator=locator,
                result=result,
            )
            stage = primary_failure_stage(
                reasons,
                result,
            )

            hidden_targets = target_unit_ids(
                case,
                update,
            )[spec["representation"]]

            row = {
                "protocol_version": (
                    PROTOCOL_VERSION
                ),
                "planned_call_index": (
                    plan_row[
                        "planned_call_index"
                    ]
                ),
                "model_block_index": (
                    plan_row[
                        "model_index"
                    ]
                ),
                "target_instance_index": (
                    plan_row[
                        "target_instance_index"
                    ]
                ),
                "condition_execution_position": (
                    plan_row[
                        "condition_execution_position"
                    ]
                ),
                "request_id": request_id,
                "case_id": case[
                    "case_id"
                ],
                "case_sha256": case[
                    "case_sha256"
                ],
                "representation_hash": case[
                    "representation_hash"
                ],
                "update_id": update[
                    "update_id"
                ],
                "target_fact_id": update[
                    "target_fact_id"
                ],
                "condition": condition,
                "representation": spec[
                    "representation"
                ],
                "expected_action_type": (
                    expected_action_type
                ),
                "model": model,
                "generation_config": (
                    config_by_model[
                        model
                    ].to_dict()
                ),
                "evidence_text": update[
                    "evidence_text"
                ],
                "initial_memory": public_memory(
                    initial_memory
                ),
                "gold_target_unit_ids": (
                    list(
                        hidden_targets
                    )
                ),
                "system_prompt_sha256": (
                    immutable[
                        "system_prompt_sha256"
                    ]
                ),
                "user_prompt": user_prompt,
                "user_prompt_sha256": (
                    sha256_text(
                        user_prompt
                    )
                ),
                "response_schema": schema,
                "response_schema_sha256": (
                    schema_hash
                ),
                "raw_response": response.text,
                "response_json_valid": (
                    response_json_valid
                ),
                "response_parse_error": (
                    parse_error
                ),
                "parsed_action": (
                    action
                    if response_json_valid
                    else None
                ),
                "locator_success": locator,
                "execution": result[
                    "execution"
                ],
                "evaluation": result[
                    "evaluation"
                ],
                "failure_stage": stage,
                "failure_reasons": reasons,
                "usage": dict(
                    response.usage
                ),
                "latency_seconds": (
                    latency_seconds
                ),
                "response_id": (
                    response.response_id
                ),
                "response_metadata": dict(
                    response.metadata
                ),
            }

            append_jsonl_fsync(
                RESULTS_PATH,
                row,
            )
            rows.append(
                row
            )
            seen.add(
                key
            )

            write_run_summary(
                rows,
                "FORMAL_PILOT_RUNNING",
            )

            print(
                "    json={} locator={} e2e={} reasons={} "
                "in_tok={} out_tok={} latency={:.2f}s".format(
                    response_json_valid,
                    locator,
                    result[
                        "evaluation"
                    ][
                        "end_to_end_success"
                    ],
                    reasons,
                    response.usage.get(
                        "input_tokens",
                        0,
                    ),
                    response.usage.get(
                        "output_tokens",
                        0,
                    ),
                    latency_seconds,
                )
            )

        if len(seen) != EXPECTED_TOTAL_CALLS:
            raise RuntimeError(
                "Formal loop ended with {}/{} calls.".format(
                    len(seen),
                    EXPECTED_TOTAL_CALLS,
                )
            )

        write_run_summary(
            rows,
            "FORMAL_PILOT_COMPLETE",
        )

    except Exception as exc:
        write_run_summary(
            rows,
            "FORMAL_PILOT_INTERRUPTED",
            error="{}: {}".format(
                type(exc).__name__,
                str(exc),
            ),
        )
        raise

    print()
    print("=" * 78)
    print(
        "FORMAL CONTROLLED PILOT COMPLETE"
    )
    print("=" * 78)
    print(
        "Completed calls: {}/{}".format(
            len(seen),
            EXPECTED_TOTAL_CALLS,
        )
    )
    print(
        "Raw results:     {}".format(
            RESULTS_PATH
        )
    )
    print(
        "Freeze manifest: {}".format(
            FREEZE_MANIFEST_PATH
        )
    )
    print(
        "Run summary:     {}".format(
            RUN_SUMMARY_PATH
        )
    )
    print()
    print(
        "Do not rerun, tune prompts, or change the frozen protocol based on "
        "these outcomes. Next step is frozen offline analysis."
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    paths.require_output_stage(paths.BASELINE, Path(__file__))
    print("=" * 78)
    print(
        "REPRESENTATION x EDIT-SCOPE — FORMAL CONTROLLED PILOT"
    )
    print("=" * 78)
    print(
        "Design: 20 cases x 2 targets x 5 conditions x 3 models = 600 calls"
    )
    print(
        "Cluster unit: case"
    )
    print(
        "Formal seed: {}".format(
            FORMAL_MASTER_SEED
        )
    )
    print()

    print(
        "[1/6] Verifying completed integration smoke..."
    )
    integration_gate = verify_integration_gate()
    integration_manifest = integration_gate[
        "manifest"
    ]
    print(
        "  PASS | exact prompt/schema adapter reproduced from 15 smoke rows"
    )

    integration_seed = int(
        integration_manifest.get(
            "master_seed",
            -1,
        )
    )
    if FORMAL_MASTER_SEED in (
        20260926,
        integration_seed,
    ):
        raise RuntimeError(
            "Formal master seed reuses a development seed."
        )


    print(
        "[2/6] Building/verifying fresh formal cases..."
    )
    (
        cases,
        case_validation,
        case_status,
    ) = build_or_verify_formal_cases(
        integration_manifest
    )
    print(
        "  {} | {} cases | {} target instances | shared order + membership PASS".format(
            case_status,
            case_validation[
                "case_count"
            ],
            case_validation[
                "target_instance_count"
            ],
        )
    )

    print(
        "[3/6] Building balanced 600-call execution plan..."
    )
    plan_for_validation = build_expected_plan(
        cases
    )
    execution_plan = model_major_execution_rows(
        cases
    )
    if len(
        plan_for_validation
    ) != len(
        execution_plan
    ):
        raise RuntimeError(
            "Execution-plan size mismatch."
        )
    print(
        "  PASS | every A-E condition occupies each within-target position "
        "8 times per model"
    )

    print(
        "[4/6] Checking model identities and recording the current Ollama runtime..."
    )
    backend = OllamaBackend(
        base_url=OLLAMA_BASE_URL,
        timeout_seconds=OLLAMA_TIMEOUT,
    )
    model_snapshots = preflight_models(
        backend,
        integration_manifest,
    )
    for model in MODELS:
        snapshot = model_snapshots[
            model
        ]
        print(
            "  OK  {}  [Ollama {} / {} / {} / {}]".format(
                model,
                snapshot.get("ollama_version"),
                snapshot.get(
                    "parameter_size"
                ),
                snapshot.get(
                    "quantization_level"
                ),
                snapshot.get(
                    "model_digest",
                    snapshot.get(
                        "digest"
                    ),
                ),
            )
        )

    print(
        "[5/6] Freezing/verifying protocol..."
    )
    immutable = build_immutable_freeze_payload(
        cases=cases,
        case_validation=(
            case_validation
        ),
        integration_gate=(
            integration_gate
        ),
        model_snapshots=(
            model_snapshots
        ),
        execution_plan=(
            execution_plan
        ),
    )
    freeze_status = freeze_or_verify(
        immutable
    )
    runtime_reference = immutable["integration_runtime_reference"]
    print(
        "  Historical integration Ollama: {} | current batch Ollama: {}".format(
            ", ".join(sorted(set(runtime_reference["historical_integration_versions"].values()))),
            ", ".join(sorted(set(runtime_reference["current_versions"].values()))),
        )
    )

    preflight_report = {
        "status": "PASS",
        "model_calls": 0,
        "checked_at_utc": utc_now_iso(),
        "integration_gate": "PASS",
        "integration_manifest_sha256": (
            integration_gate[
                "manifest_sha256"
            ]
        ),
        "integration_results_sha256": (
            integration_gate[
                "results_sha256"
            ]
        ),
        "formal_case_status": (
            case_status
        ),
        "formal_case_validation": (
            case_validation
        ),
        "expected_total_calls": (
            EXPECTED_TOTAL_CALLS
        ),
        "condition_order_balance": "PASS",
        "model_snapshot_gate": "PASS",
        "integration_runtime_reference": runtime_reference,
        "freeze_status": freeze_status,
        "freeze_fingerprint_sha256": (
            immutable[
                "freeze_fingerprint_sha256"
            ]
        ),
    }
    write_json_atomic(
        PREFLIGHT_PATH,
        preflight_report,
    )

    print(
        "  {} | fingerprint {}".format(
            freeze_status,
            immutable[
                "freeze_fingerprint_sha256"
            ][:16],
        )
    )
    print(
        "  Preflight report: {}".format(
            PREFLIGHT_PATH
        )
    )

    print(
        "[6/6] Loading/resuming formal results..."
    )
    existing_rows, seen = (
        load_and_validate_existing_results(
            cases,
            immutable,
        )
    )
    print(
        "  Existing: {}/{} | Remaining: {}".format(
            len(seen),
            EXPECTED_TOTAL_CALLS,
            EXPECTED_TOTAL_CALLS
            - len(seen),
        )
    )

    run_formal_calls(
        backend=backend,
        cases=cases,
        immutable=immutable,
        existing_rows=(
            existing_rows
        ),
        seen=seen,
    )


if __name__ == "__main__":
    main()
