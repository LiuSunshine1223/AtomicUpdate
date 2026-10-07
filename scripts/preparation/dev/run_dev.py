#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the 15-call integration gate required by the baseline experiment.

This is an implementation check, excluded from paper effect estimates.
"""

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

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
# Development smoke configuration
# ============================================================================

OLLAMA_BASE_URL = paths.OLLAMA_URL
OLLAMA_TIMEOUT = 600.0

MODELS = (
    "llama3.1:8b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "gemma3:12b-it-q4_K_M",
)

TEMPERATURE = 0.0
SEED = 42
MAX_OUTPUT_TOKENS = 1024
CONTEXT_WINDOW = 8192


# Keep this intentionally tiny. This is implementation smoke testing only.
INTEGRATION_SMOKE_MASTER_SEED = 2026092501

# Stable development output names. No _v2/_fixed/_final suffixes.
OUTPUT_DIR = paths.DEV
INTEGRATION_RESULTS_PATH = (
    OUTPUT_DIR / "integration_smoke_results.jsonl"
)
INTEGRATION_MANIFEST_PATH = (
    OUTPUT_DIR / "integration_smoke_manifest.json"
)

# False by default so a previous development smoke run is never silently
# overwritten. Change to True only for an intentional development rerun.
RESET_DEV_SMOKE = False

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


# ============================================================================
# Small utilities
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
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(
                    "JSONL row {} must be an object".format(line_number)
                )
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
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


def parse_action_response(text: str) -> Dict[str, Any]:
    """Strict parser: direct JSON object only; no repair and no fence removal."""
    candidate = text.strip()
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def public_memory(memory: Sequence[Mapping[str, Any]]) -> List[dict]:
    """Hide latent fact_ids from the model."""
    return [
        {
            "memory_id": str(unit["memory_id"]),
            "text": str(unit["text"]),
        }
        for unit in memory
    ]


# ============================================================================
# Action schemas
# ============================================================================

def action_schema(
    expected_action_type: str,
    memory: Sequence[Mapping[str, Any]],
) -> dict:
    memory_ids = [str(unit["memory_id"]) for unit in memory]

    if expected_action_type == ACTION_REWRITE_ALL:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "units"],
            "properties": {
                "action": {
                    "type": "string",
                    "const": ACTION_REWRITE_ALL,
                },
                "units": {
                    "type": "array",
                    "minItems": len(memory_ids),
                    "maxItems": len(memory_ids),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["memory_id", "text"],
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

    raise ValueError(
        "Unsupported action type: {}".format(expected_action_type)
    )


# ============================================================================
# Prompts
# ============================================================================

def action_instruction(expected_action_type: str) -> str:
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
        "Unsupported action type: {}".format(expected_action_type)
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
        + action_instruction(expected_action_type)
        + "\n\nReturn only the JSON action."
    )


# ============================================================================
# Diagnostics and evaluation
# ============================================================================

def locator_success(
    case: Mapping[str, Any],
    update: Mapping[str, Any],
    condition: str,
    action: Mapping[str, Any],
) -> Any:
    spec = condition_spec(condition)
    expected_action_type = spec["expected_action_type"]

    if expected_action_type == ACTION_REWRITE_ALL:
        return None

    selected_id = action.get("memory_id")
    if not isinstance(selected_id, str):
        return False

    hidden_targets = target_unit_ids(case, update)[
        spec["representation"]
    ]
    return selected_id in hidden_targets


def diagnostic_failure_reasons(
    response_json_valid: bool,
    locator: Any,
    result: Mapping[str, Any],
) -> List[str]:
    """Return overlapping diagnostic labels.

    Labels are intentionally non-exclusive. Their counts must never be summed
    and reported as a number of independent failed calls.
    """
    reasons = []

    if not response_json_valid:
        reasons.append("response_json_invalid")

    if locator is False:
        reasons.append("locator_failure")

    for reason in result["evaluation"].get(
        "failure_reasons",
        [],
    ):
        if reason not in reasons:
            reasons.append(str(reason))

    return reasons


def primary_failure_stage(
    failure_reasons: Sequence[str],
    result: Mapping[str, Any],
) -> str:
    """Backward-compatible single display label only."""
    if result["evaluation"]["end_to_end_success"]:
        return "success"
    if failure_reasons:
        return str(failure_reasons[0])
    return "other_failure"


def _shared_order_check() -> dict:
    """Check the current generator only; ZERO model calls."""
    generated = generate_cases(
        num_cases=20,
        master_seed=20260926,
        targets_per_case=2,
    )

    failures = []
    for case in generated:
        shared = list(case["shared_fact_order"])
        compound = []
        for unit in case["representations"]["compound"]:
            compound.extend(
                str(fact_id)
                for fact_id in unit["fact_ids"]
            )
        atomic = [
            str(unit["fact_ids"][0])
            for unit in case["representations"]["atomic"]
        ]

        if compound != shared or atomic != shared:
            failures.append(
                {
                    "case_id": case["case_id"],
                    "shared": shared,
                    "compound": compound,
                    "atomic": atomic,
                }
            )

    return {
        "checked_cases": len(generated),
        "pass": not failures,
        "failures": failures,
    }




def run_fresh_integration_smoke() -> None:
    """Run one fresh shared-order case through all 5 conditions x 3 models.

    This is the final development integration smoke before protocol freeze.
    The case is generated directly; no saved development-case file is required.
    """
    if INTEGRATION_RESULTS_PATH.exists() or INTEGRATION_MANIFEST_PATH.exists():
        raise RuntimeError(
            "Integration smoke outputs already exist. Preserve them; do not "
            "silently overwrite or rerun."
        )

    cases = generate_cases(
        num_cases=1,
        master_seed=INTEGRATION_SMOKE_MASTER_SEED,
        targets_per_case=1,
    )
    case = cases[0]
    update = case["updates"][0]

    shared = list(case["shared_fact_order"])
    compound_order = []
    for unit in case["representations"]["compound"]:
        compound_order.extend(str(x) for x in unit["fact_ids"])
    atomic_order = [
        str(unit["fact_ids"][0])
        for unit in case["representations"]["atomic"]
    ]
    if compound_order != shared or atomic_order != shared:
        raise RuntimeError("Shared-order invariant failed before model calls.")

    if not bool(
        case.get("design_invariants", {}).get(
            "fixed_unit_membership_required",
            False,
        )
    ):
        raise RuntimeError(
            "Fixed-unit-membership invariant missing before model calls."
        )

    backend = OllamaBackend(
        base_url=OLLAMA_BASE_URL,
        timeout_seconds=OLLAMA_TIMEOUT,
    )

    print("=" * 78)
    print("REPRESENTATION x EDIT-SCOPE — FINAL DEVELOPMENT INTEGRATION SMOKE")
    print("=" * 78)
    print("Fresh generated case:   YES")
    print("Shared fact order:      PASS")
    print("Fixed membership rule: PASS")
    print("Models:                 {}".format(len(MODELS)))
    print("Conditions:             {}".format(len(CONDITIONS)))
    print("Expected calls:         15")
    print()

    model_snapshots = {}
    print("Preflighting Ollama and all models...")
    for model in MODELS:
        snapshot = backend.describe(model)
        model_snapshots[model] = snapshot
        print(
            "  OK  {}  [{} / {}]".format(
                model,
                snapshot.get("parameter_size"),
                snapshot.get("quantization_level"),
            )
        )
    print()

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

    manifest = {
        "status": "INTEGRATION_SMOKE_RUNNING",
        "formal_pilot_frozen": False,
        "purpose": "Final fresh development integration smoke before freeze.",
        "master_seed": INTEGRATION_SMOKE_MASTER_SEED,
        "case_id": case["case_id"],
        "case_sha256": case["case_sha256"],
        "shared_fact_order": shared,
        "shared_order_pass": True,
        "fixed_unit_membership_required": True,
        "models": list(MODELS),
        "conditions": list(CONDITIONS),
        "expected_call_count": 15,
        "completed_call_count": 0,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "generation_config": {
            "temperature": TEMPERATURE,
            "seed": SEED,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "context_window": CONTEXT_WINDOW,
        },
        "model_snapshots": model_snapshots,
        "core_sha256": sha256_file(
            PROJECT_ROOT / "atomic_memory" / "core.py"
        ),
        "backend_sha256": sha256_file(
            PROJECT_ROOT / "atomic_memory" / "backend.py"
        ),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(INTEGRATION_MANIFEST_PATH, manifest)

    completed = 0
    e2e_successes = 0
    failure_reason_counts = {}
    start_all = time.perf_counter()

    try:
        for condition in CONDITIONS:
            spec = condition_spec(condition)
            initial_memory = initial_memory_for_condition(case, condition)
            expected_action_type = spec["expected_action_type"]
            schema = action_schema(expected_action_type, initial_memory)
            user_prompt = build_user_prompt(
                initial_memory,
                update["evidence_text"],
                expected_action_type,
            )

            for model in MODELS:
                request_id = "integration__{}__{}__{}".format(
                    condition,
                    update["update_id"],
                    model.replace(":", "_"),
                )
                print(
                    "[{}/15] {} | {}".format(
                        completed + 1,
                        condition,
                        model,
                    )
                )

                call_start = time.perf_counter()
                response = backend.generate(
                    request_id=request_id,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    config=config_by_model[model],
                    response_schema=schema,
                )
                latency_seconds = time.perf_counter() - call_start

                response_json_valid = True
                parse_error = None
                try:
                    action = parse_action_response(response.text)
                except Exception as exc:
                    response_json_valid = False
                    parse_error = str(exc)
                    action = {}

                result = execute_and_evaluate(
                    initial_memory=initial_memory,
                    action=action,
                    expected_action_type=expected_action_type,
                    update=update,
                )
                locator = locator_success(
                    case=case,
                    update=update,
                    condition=condition,
                    action=action,
                )
                reasons = diagnostic_failure_reasons(
                    response_json_valid=response_json_valid,
                    locator=locator,
                    result=result,
                )
                stage = primary_failure_stage(reasons, result)

                for reason in reasons:
                    failure_reason_counts[reason] = (
                        int(failure_reason_counts.get(reason, 0)) + 1
                    )

                row = {
                    "request_id": request_id,
                    "case_id": case["case_id"],
                    "case_sha256": case["case_sha256"],
                    "update_id": update["update_id"],
                    "condition": condition,
                    "representation": spec["representation"],
                    "expected_action_type": expected_action_type,
                    "model": model,
                    "generation_config": config_by_model[model].to_dict(),
                    "evidence_text": update["evidence_text"],
                    "initial_memory": public_memory(initial_memory),
                    "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
                    "user_prompt": user_prompt,
                    "user_prompt_sha256": sha256_text(user_prompt),
                    "response_schema": schema,
                    "raw_response": response.text,
                    "response_json_valid": response_json_valid,
                    "response_parse_error": parse_error,
                    "parsed_action": action if response_json_valid else None,
                    "locator_success": locator,
                    "execution": result["execution"],
                    "evaluation": result["evaluation"],
                    "failure_stage": stage,
                    "failure_reasons": reasons,
                    "usage": dict(response.usage),
                    "latency_seconds": latency_seconds,
                    "response_id": response.response_id,
                    "response_metadata": dict(response.metadata),
                }
                append_jsonl(INTEGRATION_RESULTS_PATH, row)

                completed += 1
                if result["evaluation"]["end_to_end_success"]:
                    e2e_successes += 1

                manifest["completed_call_count"] = completed
                write_json(INTEGRATION_MANIFEST_PATH, manifest)

                print(
                    "    json={} locator={} e2e={} reasons={} latency={:.2f}s".format(
                        response_json_valid,
                        locator,
                        result["evaluation"]["end_to_end_success"],
                        reasons,
                        latency_seconds,
                    )
                )

        total_seconds = time.perf_counter() - start_all
        manifest.update(
            {
                "status": "INTEGRATION_SMOKE_COMPLETE",
                "completed_call_count": completed,
                "e2e_success_count": e2e_successes,
                "e2e_failure_count": completed - e2e_successes,
                "failure_reason_row_counts": failure_reason_counts,
                "failure_reason_count_warning": (
                    "Reasons overlap; do not sum as independent failed calls."
                ),
                "implementation_smoke": "PASS",
                "freeze_ready_if_no_manual_anomaly": True,
                "total_wall_seconds": total_seconds,
            }
        )
        write_json(INTEGRATION_MANIFEST_PATH, manifest)

    except Exception as exc:
        manifest.update(
            {
                "status": "INTEGRATION_SMOKE_FAILED",
                "completed_call_count": completed,
                "implementation_smoke": "FAIL",
                "error": "{}: {}".format(type(exc).__name__, str(exc)),
            }
        )
        write_json(INTEGRATION_MANIFEST_PATH, manifest)
        raise

    print()
    print("=" * 78)
    print("FINAL DEVELOPMENT INTEGRATION SMOKE COMPLETE")
    print("=" * 78)
    print("Completed calls:        {}".format(completed))
    print("E2E successes:          {}".format(e2e_successes))
    print("E2E failures:           {}".format(completed - e2e_successes))
    print("Implementation smoke:   PASS")
    print("Formal pilot frozen:    NO")
    print()
    print("Wrote:")
    print("  {}".format(INTEGRATION_RESULTS_PATH))
    print("  {}".format(INTEGRATION_MANIFEST_PATH))
    print()
    print(
        "If there is no manual implementation anomaly in these 15 rows, "
        "the next step is protocol freeze and the 600-call formal pilot."
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    paths.require_output_stage(paths.DEV, Path(__file__))
    run_fresh_integration_smoke()


if __name__ == "__main__":
    main()
