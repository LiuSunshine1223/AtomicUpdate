#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the two frozen instruction paraphrases on the baseline cases.

The baseline responses are reused; only the additional paraphrases are run."""

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
    initial_memory_for_condition,
    target_unit_ids,
)


# ============================================================================
# Frozen baseline artifacts
# ============================================================================

FORMAL_DIR = paths.BASELINE
FORMAL_CASES_PATH = FORMAL_DIR / "formal_cases.jsonl"
FORMAL_FREEZE_PATH = FORMAL_DIR / "formal_freeze_manifest.json"
FORMAL_RESULTS_PATH = FORMAL_DIR / "formal_results.jsonl"
FORMAL_RUN_SUMMARY_PATH = FORMAL_DIR / "formal_run_summary.json"

ROBUSTNESS_DIR = paths.WORDING
ROBUSTNESS_FREEZE_PATH = (
    ROBUSTNESS_DIR / "robustness_freeze_manifest.json"
)
ROBUSTNESS_PREFLIGHT_PATH = (
    ROBUSTNESS_DIR / "robustness_preflight.json"
)
ROBUSTNESS_RESULTS_PATH = (
    ROBUSTNESS_DIR / "robustness_results.jsonl"
)
ROBUSTNESS_RUN_SUMMARY_PATH = (
    ROBUSTNESS_DIR / "robustness_run_summary.json"
)

OLLAMA_BASE_URL = paths.OLLAMA_URL
OLLAMA_TIMEOUT = 600.0

EXPECTED_CASES = 20
EXPECTED_TARGETS_PER_CASE = 2
EXPECTED_BASELINE_CALLS = 600

MODELS = (
    "llama3.1:8b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "gemma3:12b-it-q4_K_M",
)

TEMPERATURE = 0.0
SEED = 42
MAX_OUTPUT_TOKENS = 1024
CONTEXT_WINDOW = 8192

VARIANT_ORDER = (
    "instruction_paraphrase_1",
    "instruction_paraphrase_2",
)

EXPECTED_NEW_CALLS_PER_VARIANT = (
    EXPECTED_CASES
    * EXPECTED_TARGETS_PER_CASE
    * len(CONDITIONS)
    * len(MODELS)
)
EXPECTED_NEW_CALLS = (
    EXPECTED_NEW_CALLS_PER_VARIANT
    * len(VARIANT_ORDER)
)

if EXPECTED_NEW_CALLS_PER_VARIANT != 600:
    raise RuntimeError(
        "Unexpected per-variant call count: {}".format(
            EXPECTED_NEW_CALLS_PER_VARIANT
        )
    )

if EXPECTED_NEW_CALLS != 1200:
    raise RuntimeError(
        "Unexpected total robustness call count: {}".format(
            EXPECTED_NEW_CALLS
        )
    )

PROTOCOL_VERSION = (
    "representation-edit-scope-instruction-robustness-v1"
)


# ============================================================================
# Pre-specified instruction paraphrases
# ============================================================================

# The rule ORDER is intentionally identical to the baseline:
# 1 target identification
# 2 unrelated-fact preservation
# 3 memory-ID preservation
# 4 fixed unit membership
# 5 no unsupported facts
# 6 controlled grammar
# 7 JSON only
# 8 model selects required unit itself

INSTRUCTION_VARIANTS = {
    "instruction_paraphrase_1": """You control an evidence-based edit to an existing memory state.

You will be given the current memory and one new piece of evidence.
Respond with exactly one JSON object that conforms to the required action schema.

Rules:
1. Determine which existing fact is updated by the new evidence.
2. Leave every fact unrelated to that update unchanged.
3. Keep every memory ID unchanged.
4. Treat each memory ID as a fixed unit. Every fact must stay assigned to the
   same memory_id where it appears in the input. Do not move, swap, regroup,
   merge, or split facts across units. Apply the target update inside its
   original unit.
5. Do not add any fact that is unsupported by the memory or the evidence.
6. Keep memory text in the controlled sentence grammar used by the input:
   <Name>'s current <attribute> is <value>.
   <Name>'s previous <attribute> was <value>.
7. Output only the JSON object. Do not include explanations, markdown, or any
   other text.
8. If the required action needs a memory ID, infer and select that ID yourself.
""",
    "instruction_paraphrase_2": """Perform one update to a memory state using the supplied new evidence.

You receive the existing memory plus a single evidence statement.
Return exactly one JSON object that matches the required action schema.

Requirements:
1. Identify the existing fact that the evidence updates.
2. Preserve all other facts that are not part of this update.
3. Preserve all memory IDs.
4. Unit membership is fixed by memory_id. A fact that starts in one memory
   unit must remain in that same unit. Do not transfer, swap, regroup, merge,
   or split facts between units. Modify the target fact within its original
   unit.
5. Introduce no information that is unsupported by the memory or evidence.
6. Use the same controlled sentence grammar as the input memory:
   <Name>'s current <attribute> is <value>.
   <Name>'s previous <attribute> was <value>.
7. Return the JSON object alone, with no explanation, markdown, or surrounding
   text.
8. When the action requires choosing a memory ID, choose the appropriate ID
   yourself from the provided memory.
""",
}

RULE_INVENTORY = (
    "identify_updated_existing_fact",
    "preserve_unrelated_facts",
    "preserve_memory_ids",
    "fixed_unit_membership",
    "no_unsupported_facts",
    "controlled_sentence_grammar",
    "json_object_only",
    "self_select_memory_id_when_required",
)

HIDDEN_PROMPT_KEYS = (
    "fact_ids",
    "target_fact_id",
    "expected_current_facts",
    "allowed_historical_facts",
    "case_sha256",
    "representation_hash",
)


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


def sha256_json(value: Any) -> str:
    return sha256_text(
        canonical_json(value)
    )


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
                raise RuntimeError(
                    "{} line {} is not a JSON object".format(
                        path,
                        line_number,
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
) -> Tuple[str, str, str, str, str]:
    return (
        str(row["instruction_variant"]),
        str(row["case_id"]),
        str(row["update_id"]),
        str(row["condition"]),
        str(row["model"]),
    )


# ============================================================================
# Exact frozen user prompt + response schemas
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
    spec = condition_spec(
        condition
    )

    if spec[
        "expected_action_type"
    ] == ACTION_REWRITE_ALL:
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
    if result[
        "evaluation"
    ][
        "end_to_end_success"
    ]:
        return "success"

    if failure_reasons:
        return str(
            failure_reasons[0]
        )

    return "other_failure"


# ============================================================================
# Frozen-baseline audit
# ============================================================================

def snapshot_fingerprint(
    snapshot: Mapping[str, Any],
) -> dict:
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


def load_and_audit_baseline() -> dict:
    required = (
        FORMAL_CASES_PATH,
        FORMAL_FREEZE_PATH,
        FORMAL_RESULTS_PATH,
        FORMAL_RUN_SUMMARY_PATH,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                "Missing frozen baseline artifact: {}".format(
                    path
                )
            )

    cases = read_jsonl(
        FORMAL_CASES_PATH
    )
    freeze = read_json(
        FORMAL_FREEZE_PATH
    )
    baseline_rows = read_jsonl(
        FORMAL_RESULTS_PATH
    )
    run_summary = read_json(
        FORMAL_RUN_SUMMARY_PATH
    )

    if len(cases) != EXPECTED_CASES:
        raise RuntimeError(
            "Expected 20 frozen cases, found {}".format(
                len(cases)
            )
        )

    if len(baseline_rows) != EXPECTED_BASELINE_CALLS:
        raise RuntimeError(
            "Expected 600 frozen baseline rows, found {}".format(
                len(baseline_rows)
            )
        )

    if run_summary.get(
        "status"
    ) != "FORMAL_PILOT_COMPLETE":
        raise RuntimeError(
            "Frozen formal pilot is not marked complete."
        )

    if int(
        run_summary.get(
            "completed_calls",
            -1,
        )
    ) != EXPECTED_BASELINE_CALLS:
        raise RuntimeError(
            "Frozen formal completed_call count is not 600."
        )

    immutable = freeze.get(
        "immutable_protocol"
    )
    if not isinstance(
        immutable,
        dict,
    ):
        raise RuntimeError(
            "Frozen manifest lacks immutable_protocol."
        )

    if immutable.get(
        "formal_pilot_frozen"
    ) is not True:
        raise RuntimeError(
            "Baseline formal pilot is not frozen."
        )

    if tuple(
        immutable.get(
            "models",
            [],
        )
    ) != tuple(MODELS):
        raise RuntimeError(
            "Baseline model list differs from robustness design."
        )

    if tuple(
        immutable.get(
            "conditions",
            [],
        )
    ) != tuple(CONDITIONS):
        raise RuntimeError(
            "Baseline condition list differs from robustness design."
        )

    expected_generation = {
        "temperature": TEMPERATURE,
        "seed": SEED,
        "max_output_tokens": (
            MAX_OUTPUT_TOKENS
        ),
        "context_window": (
            CONTEXT_WINDOW
        ),
    }
    if immutable.get(
        "generation_config"
    ) != expected_generation:
        raise RuntimeError(
            "Baseline generation settings differ from robustness design."
        )

    if immutable.get(
        "formal_cases_file_sha256"
    ) != sha256_file(
        FORMAL_CASES_PATH
    ):
        raise RuntimeError(
            "Frozen formal_cases.jsonl hash mismatch."
        )

    expected_results_sha = run_summary.get(
        "formal_results_file_sha256"
    )
    if (
        expected_results_sha
        and expected_results_sha
        != sha256_file(
            FORMAL_RESULTS_PATH
        )
    ):
        raise RuntimeError(
            "Frozen formal_results.jsonl hash mismatch."
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

    if immutable.get(
        "core_sha256"
    ) != sha256_file(
        core_path
    ):
        raise RuntimeError(
            "Core changed after frozen formal pilot."
        )

    if immutable.get(
        "backend_sha256"
    ) != sha256_file(
        backend_path
    ):
        raise RuntimeError(
            "Backend changed after frozen formal pilot."
        )

    # Every reconstructed USER prompt and response schema must be identical to
    # the baseline formal run. Only the SYSTEM instruction changes later.
    case_by_id = {
        str(case["case_id"]): case
        for case in cases
    }

    seen_keys = set()

    for row in baseline_rows:
        key = (
            str(row["case_id"]),
            str(row["update_id"]),
            str(row["condition"]),
            str(row["model"]),
        )
        if key in seen_keys:
            raise RuntimeError(
                "Duplicate baseline result key: {}".format(
                    key
                )
            )
        seen_keys.add(
            key
        )

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
        spec = condition_spec(
            condition
        )
        memory = initial_memory_for_condition(
            case,
            condition,
        )

        rebuilt_user_prompt = build_user_prompt(
            memory,
            update[
                "evidence_text"
            ],
            spec[
                "expected_action_type"
            ],
        )
        if rebuilt_user_prompt != row.get(
            "user_prompt"
        ):
            raise RuntimeError(
                "User-prompt adapter differs from baseline: {}".format(
                    row.get(
                        "request_id"
                    )
                )
            )

        schema = action_schema(
            spec[
                "expected_action_type"
            ],
            memory,
        )
        schema_hash = sha256_text(
            canonical_json(
                schema
            )
        )
        if schema_hash != row.get(
            "response_schema_sha256"
        ):
            raise RuntimeError(
                "Response schema differs from baseline: {}".format(
                    row.get(
                        "request_id"
                    )
                )
            )

        frozen_schema_hash = immutable.get(
            "response_schema_sha256_by_condition",
            {},
        ).get(
            condition
        )
        if schema_hash != frozen_schema_hash:
            raise RuntimeError(
                "Response schema differs from frozen condition hash."
            )

        lowered_prompt = rebuilt_user_prompt.lower()
        for hidden_key in HIDDEN_PROMPT_KEYS:
            if hidden_key.lower() in lowered_prompt:
                raise RuntimeError(
                    "Hidden metadata found in user prompt: {}".format(
                        hidden_key
                    )
                )

    if len(seen_keys) != EXPECTED_BASELINE_CALLS:
        raise RuntimeError(
            "Baseline unique-key coverage is not 600."
        )

    return {
        "cases": cases,
        "case_by_id": case_by_id,
        "freeze": freeze,
        "immutable": immutable,
        "baseline_rows": baseline_rows,
        "source_hashes": {
            "formal_cases_sha256": sha256_file(
                FORMAL_CASES_PATH
            ),
            "formal_freeze_manifest_sha256": sha256_file(
                FORMAL_FREEZE_PATH
            ),
            "formal_results_sha256": sha256_file(
                FORMAL_RESULTS_PATH
            ),
            "formal_run_summary_sha256": sha256_file(
                FORMAL_RUN_SUMMARY_PATH
            ),
            "core_sha256": sha256_file(
                core_path
            ),
            "backend_sha256": sha256_file(
                backend_path
            ),
        },
    }


def validate_instruction_variants(
    baseline_system_prompt: str,
) -> dict:
    hashes = {}
    texts = []

    for variant in VARIANT_ORDER:
        prompt = INSTRUCTION_VARIANTS[
            variant
        ]
        if not isinstance(
            prompt,
            str,
        ) or not prompt.strip():
            raise RuntimeError(
                "Empty instruction variant: {}".format(
                    variant
                )
            )
        if prompt == baseline_system_prompt:
            raise RuntimeError(
                "{} is identical to baseline instruction.".format(
                    variant
                )
            )
        hashes[
            variant
        ] = sha256_text(
            prompt
        )
        texts.append(
            prompt
        )

    if len(
        set(texts)
    ) != len(texts):
        raise RuntimeError(
            "Instruction paraphrases are not unique."
        )

    return hashes


def preflight_models(
    backend: OllamaBackend,
    immutable: Mapping[str, Any],
) -> dict:
    frozen_snapshots = immutable.get(
        "model_snapshots",
        {},
    )
    current = {}

    for model in MODELS:
        snapshot = backend.describe(
            model
        )
        current[
            model
        ] = snapshot

        frozen = frozen_snapshots.get(
            model
        )
        if not isinstance(
            frozen,
            Mapping,
        ):
            raise RuntimeError(
                "Baseline freeze lacks model snapshot for {}".format(
                    model
                )
            )

        if snapshot_fingerprint(
            snapshot
        ) != dict(
            frozen
        ):
            raise RuntimeError(
                "Current model/runtime snapshot differs from frozen baseline "
                "for {}".format(
                    model
                )
            )

    return current


# ============================================================================
# Balanced execution plan
# ============================================================================

def rotated_conditions(
    target_instance_index: int,
) -> Tuple[str, ...]:
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


def variant_pair_order(
    model_index: int,
    target_instance_index: int,
    condition_position: int,
) -> Tuple[str, str]:
    """Exactly balance which paraphrase is first within each condition pair."""
    parity = (
        int(model_index)
        + int(target_instance_index)
        + int(condition_position)
    ) % 2

    if parity == 0:
        return (
            VARIANT_ORDER[0],
            VARIANT_ORDER[1],
        )

    return (
        VARIANT_ORDER[1],
        VARIANT_ORDER[0],
    )


def build_execution_plan(
    cases: Sequence[Mapping[str, Any]],
) -> List[dict]:
    plan = []
    planned_call_index = 0

    # model-major, same as formal pilot
    for model_index, model in enumerate(
        MODELS
    ):
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
                    pair_order = variant_pair_order(
                        model_index,
                        target_instance_index,
                        condition_position,
                    )

                    for variant_position, variant in enumerate(
                        pair_order,
                        start=1,
                    ):
                        planned_call_index += 1
                        plan.append(
                            {
                                "planned_call_index": (
                                    planned_call_index
                                ),
                                "model_index": model_index,
                                "model": model,
                                "case_index": case_index,
                                "case_id": str(
                                    case["case_id"]
                                ),
                                "update_index": update_index,
                                "update_id": str(
                                    update["update_id"]
                                ),
                                "target_instance_index": (
                                    target_instance_index
                                ),
                                "condition": condition,
                                "condition_execution_position": (
                                    condition_position
                                ),
                                "instruction_variant": (
                                    variant
                                ),
                                "variant_execution_position_within_condition": (
                                    variant_position
                                ),
                            }
                        )

                target_instance_index += 1

    if len(plan) != EXPECTED_NEW_CALLS:
        raise RuntimeError(
            "Expected 1200 planned calls, found {}".format(
                len(plan)
            )
        )

    keys = {
        (
            row["instruction_variant"],
            row["case_id"],
            row["update_id"],
            row["condition"],
            row["model"],
        )
        for row in plan
    }
    if len(keys) != EXPECTED_NEW_CALLS:
        raise RuntimeError(
            "Robustness plan contains duplicate keys."
        )

    # Each model/variant/condition must have exactly 40 rows.
    for model in MODELS:
        for variant in VARIANT_ORDER:
            for condition in CONDITIONS:
                count = sum(
                    1
                    for row in plan
                    if row[
                        "model"
                    ] == model
                    and row[
                        "instruction_variant"
                    ] == variant
                    and row[
                        "condition"
                    ] == condition
                )
                if count != 40:
                    raise RuntimeError(
                        "Plan cell {} / {} / {} has {} rows, not 40.".format(
                            model,
                            variant,
                            condition,
                            count,
                        )
                    )

    # Across each model, each variant should appear exactly 100 times in first
    # position and 100 times in second position within a condition pair.
    for model in MODELS:
        for variant in VARIANT_ORDER:
            first_count = sum(
                1
                for row in plan
                if row[
                    "model"
                ] == model
                and row[
                    "instruction_variant"
                ] == variant
                and row[
                    "variant_execution_position_within_condition"
                ] == 1
            )
            second_count = sum(
                1
                for row in plan
                if row[
                    "model"
                ] == model
                and row[
                    "instruction_variant"
                ] == variant
                and row[
                    "variant_execution_position_within_condition"
                ] == 2
            )
            if (
                first_count != 100
                or second_count != 100
            ):
                raise RuntimeError(
                    "Variant-order balance failed for {} / {}: {}/{}".format(
                        model,
                        variant,
                        first_count,
                        second_count,
                    )
                )

    return plan


# ============================================================================
# Freeze / resume
# ============================================================================

def build_freeze_payload(
    baseline: Mapping[str, Any],
    instruction_hashes: Mapping[str, str],
    model_snapshots: Mapping[str, Any],
    plan: Sequence[Mapping[str, Any]],
) -> dict:
    immutable = baseline[
        "immutable"
    ]

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "study_type": (
            "same-case instruction-wording sensitivity analysis"
        ),
        "independent_validation": False,
        "new_model_calls": EXPECTED_NEW_CALLS,
        "new_calls_per_instruction_variant": (
            EXPECTED_NEW_CALLS_PER_VARIANT
        ),
        "baseline_calls_reused_without_rerun": (
            EXPECTED_BASELINE_CALLS
        ),
        "models": list(
            MODELS
        ),
        "conditions": list(
            CONDITIONS
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
        "baseline_system_prompt_sha256": (
            immutable[
                "system_prompt_sha256"
            ]
        ),
        "instruction_variants": {
            variant: {
                "system_prompt": INSTRUCTION_VARIANTS[
                    variant
                ],
                "system_prompt_sha256": (
                    instruction_hashes[
                        variant
                    ]
                ),
                "rule_inventory": list(
                    RULE_INVENTORY
                ),
                "rule_order_matches_baseline": True,
            }
            for variant in VARIANT_ORDER
        },
        "only_system_instruction_wording_changes": True,
        "memory_unchanged": True,
        "evidence_unchanged": True,
        "user_prompt_adapter_unchanged": True,
        "response_schema_unchanged": True,
        "evaluator_unchanged": True,
        "sentence_expression_unchanged": True,
        "formal_cases_sha256": baseline[
            "source_hashes"
        ][
            "formal_cases_sha256"
        ],
        "formal_freeze_manifest_sha256": baseline[
            "source_hashes"
        ][
            "formal_freeze_manifest_sha256"
        ],
        "formal_results_sha256": baseline[
            "source_hashes"
        ][
            "formal_results_sha256"
        ],
        "formal_run_summary_sha256": baseline[
            "source_hashes"
        ][
            "formal_run_summary_sha256"
        ],
        "core_sha256": baseline[
            "source_hashes"
        ][
            "core_sha256"
        ],
        "backend_sha256": baseline[
            "source_hashes"
        ][
            "backend_sha256"
        ],
        "response_schema_sha256_by_condition": dict(
            immutable[
                "response_schema_sha256_by_condition"
            ]
        ),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "model_snapshots": {
            model: snapshot_fingerprint(
                model_snapshots[
                    model
                ]
            )
            for model in MODELS
        },
        "execution_plan_sha256": sha256_json(
            list(plan)
        ),
        "variant_order_policy": (
            "Within each case/update/condition pair, paraphrase order is "
            "alternated by model + target-instance + condition-position parity; "
            "each paraphrase is first 100/200 times per model."
        ),
        "analysis_guardrail": (
            "Compare paired effect sizes and case-cluster uncertainty against "
            "the frozen baseline. This uses the same 20 cases and is a "
            "sensitivity analysis, not independent replication."
        ),
        "no_outcome_dependent_prompt_editing": True,
    }

    payload[
        "freeze_fingerprint_sha256"
    ] = sha256_json(
        payload
    )

    return payload


def freeze_or_verify(
    payload: Mapping[str, Any],
) -> str:
    if ROBUSTNESS_FREEZE_PATH.exists():
        existing = read_json(
            ROBUSTNESS_FREEZE_PATH
        )
        if existing.get(
            "immutable_protocol"
        ) != dict(
            payload
        ):
            raise RuntimeError(
                "Robustness freeze mismatch. Do not continue with changed "
                "prompts/code/models after robustness calls have started."
            )
        return "VERIFIED_EXISTING"

    if ROBUSTNESS_RESULTS_PATH.exists():
        raise RuntimeError(
            "Robustness results exist but freeze manifest is missing."
        )

    write_json_atomic(
        ROBUSTNESS_FREEZE_PATH,
        {
            "status": (
                "INSTRUCTION_ROBUSTNESS_FROZEN"
            ),
            "created_at_utc": utc_now_iso(),
            "immutable_protocol": dict(
                payload
            ),
        },
    )
    return "CREATED"


def expected_key_set(
    plan: Sequence[Mapping[str, Any]],
) -> Set[Tuple[str, str, str, str, str]]:
    return {
        (
            str(row["instruction_variant"]),
            str(row["case_id"]),
            str(row["update_id"]),
            str(row["condition"]),
            str(row["model"]),
        )
        for row in plan
    }


def load_existing_results(
    expected_keys: Set[
        Tuple[str, str, str, str, str]
    ],
    freeze_payload: Mapping[str, Any],
) -> Tuple[List[dict], Set[Tuple[str, str, str, str, str]]]:
    if not ROBUSTNESS_RESULTS_PATH.exists():
        return [], set()

    rows = read_jsonl(
        ROBUSTNESS_RESULTS_PATH
    )
    seen = set()

    for row in rows:
        key = result_key(
            row
        )
        if key not in expected_keys:
            raise RuntimeError(
                "Unexpected robustness result key: {}".format(
                    key
                )
            )
        if key in seen:
            raise RuntimeError(
                "Duplicate robustness result key: {}".format(
                    key
                )
            )
        seen.add(
            key
        )

        variant = key[0]
        expected_prompt_hash = freeze_payload[
            "instruction_variants"
        ][variant][
            "system_prompt_sha256"
        ]
        if row.get(
            "system_prompt_sha256"
        ) != expected_prompt_hash:
            raise RuntimeError(
                "Stored robustness system prompt hash mismatch: {}".format(
                    row.get(
                        "request_id"
                    )
                )
            )

        expected_schema_hash = freeze_payload[
            "response_schema_sha256_by_condition"
        ][key[3]]
        if row.get(
            "response_schema_sha256"
        ) != expected_schema_hash:
            raise RuntimeError(
                "Stored robustness schema hash mismatch: {}".format(
                    row.get(
                        "request_id"
                    )
                )
            )

    return rows, seen


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict:
    by_variant = {}
    for variant in VARIANT_ORDER:
        subset = [
            row
            for row in rows
            if str(
                row.get(
                    "instruction_variant"
                )
            ) == variant
        ]
        successes = sum(
            1
            for row in subset
            if bool(
                row.get(
                    "evaluation",
                    {},
                ).get(
                    "end_to_end_success"
                )
            )
        )
        by_variant[
            variant
        ] = {
            "completed_calls": len(
                subset
            ),
            "e2e_successes": successes,
            "e2e_failures": (
                len(subset)
                - successes
            ),
        }

    return {
        "completed_calls": len(
            rows
        ),
        "expected_calls": (
            EXPECTED_NEW_CALLS
        ),
        "remaining_calls": (
            EXPECTED_NEW_CALLS
            - len(rows)
        ),
        "by_instruction_variant": (
            by_variant
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
            "independent_validation": False,
            "same_formal_cases_as_baseline": True,
            "only_system_instruction_wording_changed": True,
            "updated_at_utc": utc_now_iso(),
            "results_path": str(
                ROBUSTNESS_RESULTS_PATH
            ),
            "freeze_manifest_path": str(
                ROBUSTNESS_FREEZE_PATH
            ),
        }
    )

    if error is not None:
        summary[
            "error"
        ] = str(
            error
        )

    if ROBUSTNESS_RESULTS_PATH.exists():
        summary[
            "robustness_results_file_sha256"
        ] = sha256_file(
            ROBUSTNESS_RESULTS_PATH
        )

    write_json_atomic(
        ROBUSTNESS_RUN_SUMMARY_PATH,
        summary,
    )


# ============================================================================
# Calls
# ============================================================================

def run_calls(
    backend: OllamaBackend,
    baseline: Mapping[str, Any],
    freeze_payload: Mapping[str, Any],
    plan: Sequence[Mapping[str, Any]],
    existing_rows: List[dict],
    seen: Set[Tuple[str, str, str, str, str]],
) -> None:
    case_by_id = baseline[
        "case_by_id"
    ]

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

    rows = list(
        existing_rows
    )

    print()
    print("=" * 78)
    print(
        "INSTRUCTION-ONLY ROBUSTNESS — RUN / RESUME"
    )
    print("=" * 78)
    print(
        "Completed: {}/{}".format(
            len(seen),
            EXPECTED_NEW_CALLS,
        )
    )
    print(
        "Remaining: {}".format(
            EXPECTED_NEW_CALLS
            - len(seen)
        )
    )
    print(
        "Baseline 600 calls are reused; they are NOT rerun."
    )
    print()

    if len(seen) == EXPECTED_NEW_CALLS:
        write_run_summary(
            rows,
            "INSTRUCTION_ROBUSTNESS_COMPLETE",
        )
        print(
            "All 1200 robustness calls are already complete. No new calls."
        )
        return

    write_run_summary(
        rows,
        "INSTRUCTION_ROBUSTNESS_RUNNING",
    )

    try:
        for plan_row in plan:
            key = (
                str(
                    plan_row[
                        "instruction_variant"
                    ]
                ),
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

            variant = key[0]
            case = case_by_id[
                key[1]
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
                ) == key[2]
            )
            condition = key[3]
            model = key[4]

            spec = condition_spec(
                condition
            )
            expected_action_type = spec[
                "expected_action_type"
            ]
            initial_memory = initial_memory_for_condition(
                case,
                condition,
            )

            user_prompt = build_user_prompt(
                initial_memory,
                update[
                    "evidence_text"
                ],
                expected_action_type,
            )
            schema = action_schema(
                expected_action_type,
                initial_memory,
            )
            schema_hash = sha256_text(
                canonical_json(
                    schema
                )
            )

            if schema_hash != freeze_payload[
                "response_schema_sha256_by_condition"
            ][condition]:
                raise RuntimeError(
                    "Runtime response schema drift before call: {}".format(
                        key
                    )
                )

            lowered_user = user_prompt.lower()
            for hidden_key in HIDDEN_PROMPT_KEYS:
                if hidden_key.lower() in lowered_user:
                    raise RuntimeError(
                        "Hidden metadata leaked into user prompt: {}".format(
                            hidden_key
                        )
                    )

            system_prompt = INSTRUCTION_VARIANTS[
                variant
            ]
            system_prompt_hash = sha256_text(
                system_prompt
            )
            if system_prompt_hash != freeze_payload[
                "instruction_variants"
            ][variant][
                "system_prompt_sha256"
            ]:
                raise RuntimeError(
                    "Runtime instruction prompt drift: {}".format(
                        variant
                    )
                )

            request_id = (
                "instruction_robustness__{}__{}__{}__{}__{}".format(
                    model_slug(
                        model
                    ),
                    variant,
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
                "[{}/{}] {} | {} | {} | {} | {}".format(
                    current_number,
                    EXPECTED_NEW_CALLS,
                    model,
                    variant,
                    case[
                        "case_id"
                    ],
                    update[
                        "update_id"
                    ],
                    condition,
                )
            )

            start = time.perf_counter()
            response = backend.generate(
                request_id=request_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                config=config_by_model[
                    model
                ],
                response_schema=schema,
            )
            latency_seconds = (
                time.perf_counter()
                - start
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
            failure_stage = primary_failure_stage(
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
                "model_index": (
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
                "variant_execution_position_within_condition": (
                    plan_row[
                        "variant_execution_position_within_condition"
                    ]
                ),
                "instruction_variant": (
                    variant
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
                "gold_target_unit_ids": list(
                    hidden_targets
                ),
                "system_prompt_sha256": (
                    system_prompt_hash
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
                "failure_stage": (
                    failure_stage
                ),
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
                ROBUSTNESS_RESULTS_PATH,
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
                "INSTRUCTION_ROBUSTNESS_RUNNING",
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

        if len(seen) != EXPECTED_NEW_CALLS:
            raise RuntimeError(
                "Robustness loop ended with {}/{} calls.".format(
                    len(seen),
                    EXPECTED_NEW_CALLS,
                )
            )

        write_run_summary(
            rows,
            "INSTRUCTION_ROBUSTNESS_COMPLETE",
        )

    except Exception as exc:
        write_run_summary(
            rows,
            "INSTRUCTION_ROBUSTNESS_INTERRUPTED",
            error="{}: {}".format(
                type(exc).__name__,
                str(exc),
            ),
        )
        raise

    print()
    print("=" * 78)
    print(
        "INSTRUCTION-ONLY ROBUSTNESS COMPLETE"
    )
    print("=" * 78)
    print(
        "New calls complete: {}/{}".format(
            len(seen),
            EXPECTED_NEW_CALLS,
        )
    )
    print(
        "Baseline calls reused: {}".format(
            EXPECTED_BASELINE_CALLS
        )
    )
    print(
        "Results: {}".format(
            ROBUSTNESS_RESULTS_PATH
        )
    )
    print(
        "Freeze:  {}".format(
            ROBUSTNESS_FREEZE_PATH
        )
    )
    print(
        "Summary: {}".format(
            ROBUSTNESS_RUN_SUMMARY_PATH
        )
    )
    print()
    print(
        "Do not alter the two frozen paraphrases after seeing outcomes. "
        "Next step is paired offline sensitivity analysis against the baseline."
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    paths.require_output_stage(paths.WORDING, Path(__file__))
    print("=" * 78)
    print(
        "REPRESENTATION x EDIT-SCOPE — INSTRUCTION-ONLY ROBUSTNESS"
    )
    print("=" * 78)
    print(
        "Same 20 formal cases; NOT an independent validation set."
    )
    print(
        "Existing baseline calls: 600 (reused, not rerun)"
    )
    print(
        "New instruction variants: 2"
    )
    print(
        "New calls: 2 x 20 x 2 x 5 x 3 = 1200"
    )
    print()

    print(
        "[1/5] Auditing frozen baseline and exact user-prompt/schema binding..."
    )
    baseline = load_and_audit_baseline()
    baseline_system_prompt = baseline[
        "immutable"
    ][
        "system_prompt"
    ]
    print(
        "  PASS | 20 cases | 600/600 baseline rows | user prompt unchanged | "
        "schemas unchanged"
    )

    print(
        "[2/5] Freezing two pre-specified instruction paraphrases..."
    )
    instruction_hashes = validate_instruction_variants(
        baseline_system_prompt
    )
    for variant in VARIANT_ORDER:
        print(
            "  {} | SHA256 {}".format(
                variant,
                instruction_hashes[
                    variant
                ][:16],
            )
        )

    print(
        "[3/5] Building balanced 1200-call plan..."
    )
    plan = build_execution_plan(
        baseline[
            "cases"
        ]
    )
    print(
        "  PASS | 600 calls/variant | each paraphrase first 100/200 times "
        "per model"
    )

    print(
        "[4/5] Preflighting the same three frozen model snapshots..."
    )
    backend = OllamaBackend(
        base_url=OLLAMA_BASE_URL,
        timeout_seconds=OLLAMA_TIMEOUT,
    )
    model_snapshots = preflight_models(
        backend,
        baseline[
            "immutable"
        ],
    )
    for model in MODELS:
        snapshot = model_snapshots[
            model
        ]
        print(
            "  OK  {}  [{} / {} / {}]".format(
                model,
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

    freeze_payload = build_freeze_payload(
        baseline=baseline,
        instruction_hashes=(
            instruction_hashes
        ),
        model_snapshots=(
            model_snapshots
        ),
        plan=plan,
    )
    freeze_status = freeze_or_verify(
        freeze_payload
    )

    preflight = {
        "status": "PASS",
        "checked_at_utc": utc_now_iso(),
        "independent_validation": False,
        "same_20_formal_cases": True,
        "baseline_calls_reused": (
            EXPECTED_BASELINE_CALLS
        ),
        "new_calls_expected": (
            EXPECTED_NEW_CALLS
        ),
        "new_calls_per_variant": (
            EXPECTED_NEW_CALLS_PER_VARIANT
        ),
        "only_system_instruction_wording_changes": True,
        "sentence_expression_changes": False,
        "user_prompt_adapter_unchanged": True,
        "response_schema_unchanged": True,
        "evaluator_unchanged": True,
        "model_snapshot_gate": "PASS",
        "execution_order_balance": "PASS",
        "freeze_status": freeze_status,
        "freeze_fingerprint_sha256": (
            freeze_payload[
                "freeze_fingerprint_sha256"
            ]
        ),
    }
    write_json_atomic(
        ROBUSTNESS_PREFLIGHT_PATH,
        preflight,
    )

    print(
        "[5/5] Loading/resuming robustness calls..."
    )
    expected_keys = expected_key_set(
        plan
    )
    existing_rows, seen = load_existing_results(
        expected_keys,
        freeze_payload,
    )
    print(
        "  Freeze: {} | Existing: {}/{} | Remaining: {}".format(
            freeze_status,
            len(seen),
            EXPECTED_NEW_CALLS,
            EXPECTED_NEW_CALLS
            - len(seen),
        )
    )

    run_calls(
        backend=backend,
        baseline=baseline,
        freeze_payload=(
            freeze_payload
        ),
        plan=plan,
        existing_rows=(
            existing_rows
        ),
        seen=seen,
    )


if __name__ == "__main__":
    main()
