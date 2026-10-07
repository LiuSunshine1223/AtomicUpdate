#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run or resume the frozen Select/Patch unit-load study.

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
from atomic_memory.core import sha256_json
from atomic_memory.unit_load import (
    UNIT_LOADS,
    UNIT_LOAD_CONDITIONS,
    build_unit_load_user_prompt,
    execute_and_evaluate_unit_load,
    flattened_fact_order,
    generate_unit_load_cases,
    initial_memory_for_unit_load_condition,
    oracle_correct_action,
    oracle_wrong_existing_id_action,
    public_unit_load_memory,
    target_unit_id_for_condition,
    unit_load_action_schema,
    unit_load_condition_spec,
    unit_load_locator_success,
)


# ============================================================================
# Frozen study constants
# ============================================================================

UNIT_LOAD_MASTER_SEED = 20261005
UNIT_LOAD_NUM_CASES = 20
UNIT_LOAD_TARGETS_PER_CASE = 2

MODELS = (
    "llama3.1:8b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "gemma3:12b-it-q4_K_M",
)

OLLAMA_BASE_URL = paths.OLLAMA_URL
OLLAMA_TIMEOUT = 600.0
TEMPERATURE = 0.0
GENERATION_SEED = 42
MAX_OUTPUT_TOKENS = 1024
CONTEXT_WINDOW = 8192

EXPECTED_TOTAL_CALLS = (
    UNIT_LOAD_NUM_CASES
    * UNIT_LOAD_TARGETS_PER_CASE
    * len(UNIT_LOAD_CONDITIONS)
    * len(MODELS)
)
if EXPECTED_TOTAL_CALLS != 720:
    raise RuntimeError("Unit-load call-count invariant changed")

PROTOCOL_VERSION = "representation-edit-scope-unit-load-v1"
ANALYSIS_BOOTSTRAP_REPS = 20000
ANALYSIS_BOOTSTRAP_SEED = 20261006

BASELINE_OUTPUT_DIR = paths.BASELINE
BASELINE_FREEZE_PATH = BASELINE_OUTPUT_DIR / "formal_freeze_manifest.json"
BASELINE_CASES_PATH = BASELINE_OUTPUT_DIR / "formal_cases.jsonl"

OUTPUT_DIR = paths.UNIT_LOAD
CASES_PATH = OUTPUT_DIR / "unit_load_cases.jsonl"
PREFLIGHT_PATH = OUTPUT_DIR / "unit_load_preflight.json"
FREEZE_MANIFEST_PATH = OUTPUT_DIR / "unit_load_freeze_manifest.json"
RESULTS_PATH = OUTPUT_DIR / "unit_load_results.jsonl"
RUN_SUMMARY_PATH = OUTPUT_DIR / "unit_load_run_summary.json"

ANALYSIS_SCRIPT_PATH = PROJECT_ROOT / "scripts/exp3_unit_load/analyze_unit_load.py"
CORE_PATH = (
    PROJECT_ROOT / "atomic_memory" / "core.py"
)
UNIT_LOAD_CORE_PATH = (
    PROJECT_ROOT / "atomic_memory" / "unit_load.py"
)
BACKEND_PATH = PROJECT_ROOT / "atomic_memory" / "backend.py"

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
    "gold_target_unit",
)

PRIMARY_ESTIMAND = {
    "name": "G_Patch",
    "definition": "E2E(Patch, load=1) - E2E(Patch, load=3)",
    "aggregation": "average two targets within case, then average paired case differences",
}
KEY_SECONDARY_ESTIMAND = {
    "name": "S",
    "definition": (
        "[E2E(Select, load=1) - E2E(Select, load=3)] - "
        "[E2E(Patch, load=1) - E2E(Patch, load=3)]"
    ),
    "aggregation": "average two targets within case, then average paired case interactions",
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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
                    "JSONL row {} must be an object: {}".format(
                        line_number,
                        path,
                    )
                )
            rows.append(value)
    return rows


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
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
    os.replace(str(temporary), str(path))


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True)
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def append_jsonl_fsync(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def parse_action_response(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def model_slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_").replace(" ", "_")


def result_key(row: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(row["case_id"]),
        str(row["update_id"]),
        str(row["condition"]),
        str(row["model"]),
    )


def case_without_sha(case: Mapping[str, Any]) -> dict:
    result = dict(case)
    result.pop("case_sha256", None)
    return result


def semantic_case_signature(case: Mapping[str, Any]) -> str:
    """Signature independent of representation format for freshness checks."""
    payload = {
        "facts": case.get("facts", []),
        "shared_fact_order": case.get("shared_fact_order", []),
        "updates": [
            {
                "target_fact_id": update.get("target_fact_id"),
                "old_fact": update.get("old_fact"),
                "new_fact": update.get("new_fact"),
                "evidence_text": update.get("evidence_text"),
            }
            for update in case.get("updates", [])
        ],
    }
    return sha256_json(payload)


# ============================================================================
# Baseline compatibility and deterministic case validation
# ============================================================================

def snapshot_fingerprint(snapshot: Mapping[str, Any]) -> dict:
    digest = snapshot.get("model_digest", snapshot.get("digest", ""))
    return {
        "provider": snapshot.get("provider"),
        "ollama_version": snapshot.get("ollama_version"),
        "model": snapshot.get("model"),
        "model_digest": digest,
        "parameter_size": snapshot.get("parameter_size"),
        "quantization_level": snapshot.get("quantization_level"),
    }


def load_baseline_reference() -> dict:
    if not BASELINE_FREEZE_PATH.exists():
        raise FileNotFoundError(
            "Missing completed baseline freeze manifest: {}".format(
                BASELINE_FREEZE_PATH
            )
        )
    if not BASELINE_CASES_PATH.exists():
        raise FileNotFoundError(
            "Missing completed baseline cases: {}".format(BASELINE_CASES_PATH)
        )

    manifest = read_json(BASELINE_FREEZE_PATH)
    immutable = manifest.get("immutable_protocol")
    if not isinstance(immutable, Mapping):
        raise RuntimeError("Baseline freeze manifest lacks immutable_protocol")

    if immutable.get("expected_total_calls") != 600:
        raise RuntimeError("Baseline freeze manifest is not the completed 600-call design")

    baseline_prompt_hash = immutable.get("system_prompt_sha256")
    if baseline_prompt_hash != sha256_text(SYSTEM_PROMPT):
        raise RuntimeError(
            "Unit-load SYSTEM_PROMPT does not match the frozen baseline prompt"
        )

    baseline_models = tuple(immutable.get("models", []))
    if baseline_models != MODELS:
        raise RuntimeError("Unit-load model list differs from frozen baseline")

    baseline_cases = read_jsonl(BASELINE_CASES_PATH)
    if len(baseline_cases) != 20:
        raise RuntimeError("Expected 20 frozen baseline cases")

    return {
        "manifest": manifest,
        "immutable": dict(immutable),
        "cases": baseline_cases,
        "manifest_sha256": sha256_file(BASELINE_FREEZE_PATH),
        "cases_file_sha256": sha256_file(BASELINE_CASES_PATH),
    }


def validate_cases(
    cases: Sequence[Mapping[str, Any]],
    baseline_reference: Mapping[str, Any],
) -> dict:
    errors = []
    if len(cases) != UNIT_LOAD_NUM_CASES:
        errors.append("case_count")

    ids = [str(case.get("case_id")) for case in cases]
    if len(set(ids)) != UNIT_LOAD_NUM_CASES:
        errors.append("case_ids_not_unique")

    baseline_signatures = {
        semantic_case_signature(case)
        for case in baseline_reference["cases"]
    }

    oracle_success_checks = 0
    wrong_existing_id_checks = 0
    prompt_schema_rows = []
    target_instances = 0

    for case in cases:
        case_id = str(case.get("case_id"))
        stored_sha = str(case.get("case_sha256", ""))
        computed_sha = sha256_json(case_without_sha(case))
        if stored_sha != computed_sha:
            errors.append("{}:case_sha".format(case_id))

        if semantic_case_signature(case) in baseline_signatures:
            errors.append("{}:reuses_baseline_semantic_case".format(case_id))

        invariants = case.get("design_invariants", {})
        expected_invariants = {
            "fact_count": 6,
            "unit_loads": [1, 2, 3],
            "representations_built_before_target": True,
            "targets_share_same_initial_memory": True,
            "shared_fact_order_across_representations": True,
            "fixed_unit_membership_required": True,
        }
        for key, expected in expected_invariants.items():
            if invariants.get(key) != expected:
                errors.append("{}:invariant:{}".format(case_id, key))

        shared = [str(x) for x in case.get("shared_fact_order", [])]
        if len(shared) != 6 or len(set(shared)) != 6:
            errors.append("{}:shared_fact_order".format(case_id))

        for load in UNIT_LOADS:
            flat = flattened_fact_order(case, load)
            if flat != shared:
                errors.append("{}:load_{}:shared_order".format(case_id, load))

            representation = case["representations"]["load_{}".format(load)]
            if len(representation) != 6 // load:
                errors.append("{}:load_{}:unit_count".format(case_id, load))
            if not all(len(unit.get("fact_ids", [])) == load for unit in representation):
                errors.append("{}:load_{}:fact_count_per_unit".format(case_id, load))

        updates = list(case.get("updates", []))
        if len(updates) != UNIT_LOAD_TARGETS_PER_CASE:
            errors.append("{}:update_count".format(case_id))
            continue

        target_ids = [str(update.get("target_fact_id")) for update in updates]
        if len(set(target_ids)) != len(target_ids):
            errors.append("{}:targets_not_distinct".format(case_id))

        for update in updates:
            target_instances += 1
            evidence = str(update.get("evidence_text", ""))
            if not evidence:
                errors.append("{}:{}:empty_evidence".format(case_id, update.get("update_id")))

            for condition in UNIT_LOAD_CONDITIONS:
                spec = unit_load_condition_spec(condition)
                memory = initial_memory_for_unit_load_condition(case, condition)
                target_unit_id = target_unit_id_for_condition(case, update, condition)
                if sum(1 for unit in memory if str(unit["memory_id"]) == target_unit_id) != 1:
                    errors.append(
                        "{}:{}:{}:target_unit_count".format(
                            case_id,
                            update.get("update_id"),
                            condition,
                        )
                    )

                old_text = str(update["old_fact"]["current_text"])
                target_unit = next(
                    unit for unit in memory if str(unit["memory_id"]) == target_unit_id
                )
                if target_unit["text"].count(old_text) != 1:
                    errors.append(
                        "{}:{}:{}:old_span_not_unique".format(
                            case_id,
                            update.get("update_id"),
                            condition,
                        )
                    )

                schema = unit_load_action_schema(condition, memory)
                enum_ids = list(schema["properties"]["memory_id"]["enum"])
                actual_ids = [str(unit["memory_id"]) for unit in memory]
                if enum_ids != actual_ids:
                    errors.append(
                        "{}:{}:{}:schema_ids".format(
                            case_id,
                            update.get("update_id"),
                            condition,
                        )
                    )

                prompt = build_unit_load_user_prompt(condition, memory, evidence)
                lowered = prompt.lower()
                for hidden in HIDDEN_PROMPT_KEYS:
                    if hidden.lower() in lowered:
                        errors.append(
                            "{}:{}:{}:hidden_prompt_key:{}".format(
                                case_id,
                                update.get("update_id"),
                                condition,
                                hidden,
                            )
                        )
                if str(update["target_fact_id"]) in prompt:
                    errors.append(
                        "{}:{}:{}:target_fact_id_leak".format(
                            case_id,
                            update.get("update_id"),
                            condition,
                        )
                    )

                correct_action = oracle_correct_action(case, update, condition)
                correct_result = execute_and_evaluate_unit_load(
                    case,
                    update,
                    condition,
                    correct_action,
                )
                oracle_success_checks += 1
                if not bool(correct_result["evaluation"]["end_to_end_success"]):
                    errors.append(
                        "{}:{}:{}:oracle_correct_failed".format(
                            case_id,
                            update.get("update_id"),
                            condition,
                        )
                    )

                wrong_action = oracle_wrong_existing_id_action(
                    case,
                    update,
                    condition,
                )
                wrong_result = execute_and_evaluate_unit_load(
                    case,
                    update,
                    condition,
                    wrong_action,
                )
                wrong_existing_id_checks += 1
                if not bool(wrong_result["execution"]["legal_action"]):
                    errors.append(
                        "{}:{}:{}:wrong_existing_id_illegal".format(
                            case_id,
                            update.get("update_id"),
                            condition,
                        )
                    )
                if not bool(wrong_result["execution"]["execution_success"]):
                    errors.append(
                        "{}:{}:{}:wrong_existing_id_not_executed".format(
                            case_id,
                            update.get("update_id"),
                            condition,
                        )
                    )
                if unit_load_locator_success(case, update, condition, wrong_action):
                    errors.append(
                        "{}:{}:{}:wrong_existing_id_locator_true".format(
                            case_id,
                            update.get("update_id"),
                            condition,
                        )
                    )
                if bool(wrong_result["evaluation"]["end_to_end_success"]):
                    errors.append(
                        "{}:{}:{}:wrong_existing_id_e2e_true".format(
                            case_id,
                            update.get("update_id"),
                            condition,
                        )
                    )

                prompt_schema_rows.append(
                    {
                        "case_id": case_id,
                        "update_id": str(update["update_id"]),
                        "condition": condition,
                        "unit_load": int(spec["unit_load"]),
                        "interface": str(spec["interface"]),
                        "evidence_sha256": sha256_text(evidence),
                        "user_prompt_sha256": sha256_text(prompt),
                        "response_schema_sha256": sha256_text(canonical_json(schema)),
                    }
                )

    if target_instances != UNIT_LOAD_NUM_CASES * UNIT_LOAD_TARGETS_PER_CASE:
        errors.append("target_instance_count")

    if oracle_success_checks != UNIT_LOAD_NUM_CASES * UNIT_LOAD_TARGETS_PER_CASE * 6:
        errors.append("oracle_success_check_count")
    if wrong_existing_id_checks != UNIT_LOAD_NUM_CASES * UNIT_LOAD_TARGETS_PER_CASE * 6:
        errors.append("wrong_existing_id_check_count")

    if errors:
        raise RuntimeError(
            "Unit-load deterministic validation failed: {}".format(
                "; ".join(errors[:30])
            )
        )

    return {
        "status": "PASS",
        "case_count": len(cases),
        "target_instance_count": target_instances,
        "oracle_correct_action_checks": oracle_success_checks,
        "wrong_existing_id_execution_checks": wrong_existing_id_checks,
        "fresh_against_baseline_cases": True,
        "prompt_schema_binding_sha256": sha256_json(prompt_schema_rows),
        "case_sha256_values": [str(case["case_sha256"]) for case in cases],
    }


def build_or_verify_cases(
    baseline_reference: Mapping[str, Any],
) -> Tuple[List[dict], dict, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated = generate_unit_load_cases(
        num_cases=UNIT_LOAD_NUM_CASES,
        master_seed=UNIT_LOAD_MASTER_SEED,
        targets_per_case=UNIT_LOAD_TARGETS_PER_CASE,
    )

    if CASES_PATH.exists():
        cases = read_jsonl(CASES_PATH)
        if canonical_json(cases) != canonical_json(generated):
            raise RuntimeError(
                "Existing unit_load_cases.jsonl differs from deterministic generation"
            )
        status = "VERIFIED_EXISTING"
    else:
        if RESULTS_PATH.exists() or FREEZE_MANIFEST_PATH.exists():
            raise RuntimeError(
                "Results/freeze exist but unit_load_cases.jsonl is missing"
            )
        write_jsonl_atomic(CASES_PATH, generated)
        cases = generated
        status = "CREATED"

    validation = validate_cases(cases, baseline_reference)
    return cases, validation, status


# ============================================================================
# Execution plan
# ============================================================================

def rotated_conditions(target_instance_index: int) -> Tuple[str, ...]:
    offset = int(target_instance_index) % len(UNIT_LOAD_CONDITIONS)
    values = list(UNIT_LOAD_CONDITIONS)
    return tuple(values[offset:] + values[:offset])


def model_major_execution_rows(
    cases: Sequence[Mapping[str, Any]],
) -> Tuple[List[dict], dict]:
    base = []
    target_instance_index = 0

    for case_index, case in enumerate(cases):
        for update_index, update in enumerate(case["updates"]):
            order = rotated_conditions(target_instance_index)
            for position, condition in enumerate(order, start=1):
                base.append(
                    {
                        "case_index": case_index,
                        "case_id": str(case["case_id"]),
                        "update_index": update_index,
                        "update_id": str(update["update_id"]),
                        "target_instance_index": target_instance_index,
                        "condition": condition,
                        "condition_execution_position": position,
                    }
                )
            target_instance_index += 1

    rows = []
    planned_call_index = 0
    for model_index, model in enumerate(MODELS):
        for base_row in base:
            planned_call_index += 1
            row = dict(base_row)
            row.update(
                {
                    "model_index": model_index,
                    "model": model,
                    "planned_call_index": planned_call_index,
                }
            )
            rows.append(row)

    if len(rows) != EXPECTED_TOTAL_CALLS:
        raise RuntimeError("Execution plan is not 720 calls")

    balance = {}
    for model in MODELS:
        balance[model] = {}
        for condition in UNIT_LOAD_CONDITIONS:
            counts = {}
            for position in range(1, 7):
                count = sum(
                    1
                    for row in rows
                    if row["model"] == model
                    and row["condition"] == condition
                    and row["condition_execution_position"] == position
                )
                counts[str(position)] = count
                if count not in (6, 7):
                    raise RuntimeError(
                        "Condition position balance failed: {} {} pos {} count {}".format(
                            model,
                            condition,
                            position,
                            count,
                        )
                    )
            if sum(counts.values()) != 40:
                raise RuntimeError("Condition position counts do not sum to 40")
            balance[model][condition] = counts

    return rows, balance


# ============================================================================
# Model preflight and protocol freeze
# ============================================================================

def preflight_models(
    backend: OllamaBackend,
    baseline_reference: Mapping[str, Any],
) -> dict:
    baseline_snapshots = baseline_reference["immutable"].get(
        "model_snapshots",
        {},
    )
    current = {}

    for model in MODELS:
        snapshot = backend.describe(model)
        current[model] = snapshot
        baseline = baseline_snapshots.get(model)
        if not isinstance(baseline, Mapping):
            raise RuntimeError("Baseline freeze lacks model snapshot for {}".format(model))
        if snapshot_fingerprint(snapshot) != snapshot_fingerprint(baseline):
            raise RuntimeError(
                "Current model/runtime snapshot differs from frozen baseline for {}".format(
                    model
                )
            )

    return current


def schema_hashes_for_freeze(cases: Sequence[Mapping[str, Any]]) -> dict:
    first_case = cases[0]
    result = {}
    for condition in UNIT_LOAD_CONDITIONS:
        memory = initial_memory_for_unit_load_condition(first_case, condition)
        schema = unit_load_action_schema(condition, memory)
        result[condition] = sha256_text(canonical_json(schema))
    return result


def build_immutable_freeze_payload(
    cases: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    baseline_reference: Mapping[str, Any],
    model_snapshots: Mapping[str, Any],
    execution_plan: Sequence[Mapping[str, Any]],
    position_balance: Mapping[str, Any],
) -> dict:
    required_paths = [
        CORE_PATH,
        UNIT_LOAD_CORE_PATH,
        BACKEND_PATH,
        Path(__file__).resolve(),
        ANALYSIS_SCRIPT_PATH,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError("Required frozen source missing: {}".format(path))

    immutable = {
        "protocol_version": PROTOCOL_VERSION,
        "study_scope": "controlled synthetic unit-load x local-edit-interface extension",
        "research_question": (
            "When Patch is available, how does finer fact grouping change update "
            "success and token overhead, and how does that change differ from Select?"
        ),
        "primary_endpoint": (
            "Legal Action AND Execution Success AND Structure Valid AND Target Correct "
            "AND Protected Facts Preserved AND No Unsupported Changes"
        ),
        "primary_estimand": dict(PRIMARY_ESTIMAND),
        "key_secondary_estimand": dict(KEY_SECONDARY_ESTIMAND),
        "interpretation_guardrails": [
            "S > 0 only means the grouping effect differs between interfaces; it does not establish Patch as a better replacement.",
            "Similar observed success rates do not establish equivalence or noninferiority.",
            "Token differences are measured overhead, not a utility or worth threshold.",
            "Grouping load changes unit size and unit count together; estimates are configuration effects.",
            "Models are analyzed separately; no primary cross-model pooling.",
            "No additional models, prompts, or cases are added based on observed outcomes.",
        ],
        "cluster_unit": "case",
        "bootstrap_reps": ANALYSIS_BOOTSTRAP_REPS,
        "bootstrap_seed": ANALYSIS_BOOTSTRAP_SEED,
        "unit_load_master_seed": UNIT_LOAD_MASTER_SEED,
        "num_cases": UNIT_LOAD_NUM_CASES,
        "targets_per_case": UNIT_LOAD_TARGETS_PER_CASE,
        "unit_loads": list(UNIT_LOADS),
        "conditions": list(UNIT_LOAD_CONDITIONS),
        "models": list(MODELS),
        "expected_total_calls": EXPECTED_TOTAL_CALLS,
        "generation_config": {
            "temperature": TEMPERATURE,
            "seed": GENERATION_SEED,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "context_window": CONTEXT_WINDOW,
        },
        "ollama_base_url": OLLAMA_BASE_URL,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "baseline_freeze_manifest_sha256": baseline_reference["manifest_sha256"],
        "baseline_cases_file_sha256": baseline_reference["cases_file_sha256"],
        "same_prompt_as_baseline": True,
        "same_model_snapshots_as_baseline": True,
        "fresh_semantic_cases_vs_baseline": True,
        "cases_sha256": sha256_json(list(cases)),
        "cases_file_sha256": sha256_file(CASES_PATH),
        "prompt_schema_binding_sha256": validation["prompt_schema_binding_sha256"],
        "response_schema_sha256_by_condition": schema_hashes_for_freeze(cases),
        "condition_order_policy": (
            "six-condition cyclic rotation over 40 target instances; each condition "
            "appears 6 or 7 times in each execution position per model"
        ),
        "condition_position_balance": position_balance,
        "execution_plan_sha256": sha256_json(list(execution_plan)),
        "core_sha256": sha256_file(CORE_PATH),
        "unit_load_core_sha256": sha256_file(UNIT_LOAD_CORE_PATH),
        "backend_sha256": sha256_file(BACKEND_PATH),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "analysis_script_sha256": sha256_file(ANALYSIS_SCRIPT_PATH),
        "model_snapshots": {
            model: snapshot_fingerprint(model_snapshots[model])
            for model in MODELS
        },
        "deterministic_validation": dict(validation),
        "failure_labels_overlap": True,
        "cost_measurement_scope": (
            "input tokens, output tokens, and wall-clock generation latency; "
            "no storage, retrieval, monetary, or downstream-reading cost"
        ),
        "no_prompt_rescue": True,
        "outcome_dependent_protocol_changes_allowed": False,
    }
    immutable["freeze_fingerprint_sha256"] = sha256_json(immutable)
    return immutable


def freeze_or_verify(immutable: Mapping[str, Any]) -> str:
    if FREEZE_MANIFEST_PATH.exists():
        existing = read_json(FREEZE_MANIFEST_PATH)
        if existing.get("immutable_protocol") != dict(immutable):
            raise RuntimeError(
                "Unit-load freeze mismatch. Protocol/code/model/cases changed after freeze."
            )
        return "VERIFIED_EXISTING"

    if RESULTS_PATH.exists():
        raise RuntimeError("Results exist but unit-load freeze manifest is missing")

    manifest = {
        "status": "UNIT_LOAD_PROTOCOL_FROZEN",
        "created_at_utc": utc_now_iso(),
        "immutable_protocol": dict(immutable),
    }
    write_json_atomic(FREEZE_MANIFEST_PATH, manifest)
    return "CREATED"


# ============================================================================
# Result validation and summaries
# ============================================================================

def expected_key_set(
    cases: Sequence[Mapping[str, Any]],
) -> Set[Tuple[str, str, str, str]]:
    return {
        (
            str(case["case_id"]),
            str(update["update_id"]),
            condition,
            model,
        )
        for case in cases
        for update in case["updates"]
        for condition in UNIT_LOAD_CONDITIONS
        for model in MODELS
    }


def load_and_validate_existing_results(
    cases: Sequence[Mapping[str, Any]],
    immutable: Mapping[str, Any],
) -> Tuple[List[dict], Set[Tuple[str, str, str, str]]]:
    expected = expected_key_set(cases)
    if not RESULTS_PATH.exists():
        return [], set()

    rows = read_jsonl(RESULTS_PATH)
    seen = set()
    case_by_id = {str(case["case_id"]): case for case in cases}

    for row in rows:
        key = result_key(row)
        if key not in expected:
            raise RuntimeError("Unexpected unit-load result key: {}".format(key))
        if key in seen:
            raise RuntimeError("Duplicate unit-load result key: {}".format(key))
        seen.add(key)

        case = case_by_id[key[0]]
        if row.get("case_sha256") != case.get("case_sha256"):
            raise RuntimeError("Stored case hash mismatch for {}".format(row.get("request_id")))
        if row.get("system_prompt_sha256") != immutable["system_prompt_sha256"]:
            raise RuntimeError("Stored system prompt hash mismatch")
        expected_schema_hash = immutable["response_schema_sha256_by_condition"][key[2]]
        if row.get("response_schema_sha256") != expected_schema_hash:
            raise RuntimeError("Stored response schema hash mismatch")

    return rows, seen


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict:
    e2e_success = 0
    failure_counts = {}
    for row in rows:
        if bool(row.get("evaluation", {}).get("end_to_end_success")):
            e2e_success += 1
        for reason in row.get("failure_reasons", []):
            reason = str(reason)
            failure_counts[reason] = failure_counts.get(reason, 0) + 1

    return {
        "completed_calls": len(rows),
        "expected_calls": EXPECTED_TOTAL_CALLS,
        "remaining_calls": EXPECTED_TOTAL_CALLS - len(rows),
        "e2e_success_count": e2e_success,
        "e2e_failure_count": len(rows) - e2e_success,
        "failure_reason_row_counts": failure_counts,
        "failure_reason_count_warning": (
            "Failure labels overlap; do not sum them as independent failed calls."
        ),
    }


def write_run_summary(
    rows: Sequence[Mapping[str, Any]],
    status: str,
    error: Optional[str] = None,
) -> None:
    summary = summarize_rows(rows)
    summary.update(
        {
            "status": status,
            "protocol_version": PROTOCOL_VERSION,
            "results_path": str(RESULTS_PATH),
            "freeze_manifest_path": str(FREEZE_MANIFEST_PATH),
            "updated_at_utc": utc_now_iso(),
        }
    )
    if error is not None:
        summary["error"] = str(error)
    if RESULTS_PATH.exists():
        summary["results_file_sha256"] = sha256_file(RESULTS_PATH)
    write_json_atomic(RUN_SUMMARY_PATH, summary)


def diagnostic_failure_reasons(
    response_json_valid: bool,
    locator: bool,
    result: Mapping[str, Any],
) -> List[str]:
    reasons = []
    if not response_json_valid:
        reasons.append("response_json_invalid")
    if locator is False:
        reasons.append("locator_failure")
    for reason in result["evaluation"].get("failure_reasons", []):
        reason = str(reason)
        if reason not in reasons:
            reasons.append(reason)
    return reasons


def primary_failure_stage(
    failure_reasons: Sequence[str],
    result: Mapping[str, Any],
) -> str:
    if bool(result["evaluation"]["end_to_end_success"]):
        return "success"
    if failure_reasons:
        return str(failure_reasons[0])
    return "other_failure"


# ============================================================================
# Formal calls
# ============================================================================

def run_calls(
    backend: OllamaBackend,
    cases: Sequence[Mapping[str, Any]],
    immutable: Mapping[str, Any],
    execution_rows: Sequence[Mapping[str, Any]],
    existing_rows: List[dict],
    seen: Set[Tuple[str, str, str, str]],
) -> None:
    case_by_id = {str(case["case_id"]): case for case in cases}
    configs = {
        model: GenerationConfig(
            model=model,
            temperature=TEMPERATURE,
            seed=GENERATION_SEED,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            context_window=CONTEXT_WINDOW,
        )
        for model in MODELS
    }

    print()
    print("=" * 78)
    print("UNIT-LOAD EXTENSION — RUN / RESUME")
    print("=" * 78)
    print("Completed: {}/{}".format(len(seen), EXPECTED_TOTAL_CALLS))
    print("Remaining: {}".format(EXPECTED_TOTAL_CALLS - len(seen)))
    print("Primary: G_Patch = Patch(L1) - Patch(L3)")
    print("Secondary: S = [Select(L1)-Select(L3)] - [Patch(L1)-Patch(L3)]")
    print()

    if len(seen) == EXPECTED_TOTAL_CALLS:
        write_run_summary(existing_rows, "UNIT_LOAD_COMPLETE")
        print("All 720 unit-load calls are already complete. No new calls.")
        return

    rows = list(existing_rows)
    write_run_summary(rows, "UNIT_LOAD_RUNNING")

    try:
        for plan_row in execution_rows:
            key = (
                str(plan_row["case_id"]),
                str(plan_row["update_id"]),
                str(plan_row["condition"]),
                str(plan_row["model"]),
            )
            if key in seen:
                continue

            case = case_by_id[key[0]]
            update = next(
                candidate
                for candidate in case["updates"]
                if str(candidate["update_id"]) == key[1]
            )
            condition = key[2]
            model = key[3]
            spec = unit_load_condition_spec(condition)
            initial_memory = initial_memory_for_unit_load_condition(case, condition)
            schema = unit_load_action_schema(condition, initial_memory)
            schema_hash = sha256_text(canonical_json(schema))
            if schema_hash != immutable["response_schema_sha256_by_condition"][condition]:
                raise RuntimeError("Runtime schema hash drift before call: {}".format(key))

            user_prompt = build_unit_load_user_prompt(
                condition,
                initial_memory,
                update["evidence_text"],
            )
            lowered = user_prompt.lower()
            for hidden in HIDDEN_PROMPT_KEYS:
                if hidden.lower() in lowered:
                    raise RuntimeError(
                        "Hidden metadata leaked before call: {} {}".format(key, hidden)
                    )

            request_id = "unit_load__{}__{}__{}__{}".format(
                model_slug(model),
                case["case_id"],
                update["update_id"],
                condition,
            )

            print(
                "[{}/{}] {} | {} | {} | {}".format(
                    len(seen) + 1,
                    EXPECTED_TOTAL_CALLS,
                    model,
                    case["case_id"],
                    update["update_id"],
                    condition,
                )
            )

            call_start = time.perf_counter()
            response = backend.generate(
                request_id=request_id,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                config=configs[model],
                response_schema=schema,
            )
            latency_seconds = time.perf_counter() - call_start

            response_json_valid = True
            parse_error = None
            try:
                action = parse_action_response(response.text)
            except Exception as exc:
                response_json_valid = False
                parse_error = "{}: {}".format(type(exc).__name__, str(exc))
                action = {}

            result = execute_and_evaluate_unit_load(
                case,
                update,
                condition,
                action,
            )
            locator = unit_load_locator_success(
                case,
                update,
                condition,
                action,
            )
            reasons = diagnostic_failure_reasons(
                response_json_valid,
                locator,
                result,
            )
            stage = primary_failure_stage(reasons, result)
            gold_target_unit_id = target_unit_id_for_condition(
                case,
                update,
                condition,
            )

            row = {
                "protocol_version": PROTOCOL_VERSION,
                "planned_call_index": int(plan_row["planned_call_index"]),
                "model_block_index": int(plan_row["model_index"]),
                "target_instance_index": int(plan_row["target_instance_index"]),
                "condition_execution_position": int(
                    plan_row["condition_execution_position"]
                ),
                "request_id": request_id,
                "case_id": str(case["case_id"]),
                "case_sha256": str(case["case_sha256"]),
                "representation_hash": str(case["representation_hash"]),
                "update_id": str(update["update_id"]),
                "target_fact_id": str(update["target_fact_id"]),
                "condition": condition,
                "unit_load": int(spec["unit_load"]),
                "interface": str(spec["interface"]),
                "expected_action_type": str(spec["expected_action_type"]),
                "model": model,
                "generation_config": configs[model].to_dict(),
                "evidence_text": str(update["evidence_text"]),
                "initial_memory": public_unit_load_memory(initial_memory),
                "gold_target_unit_id": gold_target_unit_id,
                "system_prompt_sha256": immutable["system_prompt_sha256"],
                "user_prompt": user_prompt,
                "user_prompt_sha256": sha256_text(user_prompt),
                "response_schema": schema,
                "response_schema_sha256": schema_hash,
                "raw_response": response.text,
                "response_json_valid": response_json_valid,
                "response_parse_error": parse_error,
                "parsed_action": action if response_json_valid else None,
                "locator_success": bool(locator),
                "execution": result["execution"],
                "evaluation": result["evaluation"],
                "failure_stage": stage,
                "failure_reasons": reasons,
                "usage": dict(response.usage),
                "latency_seconds": latency_seconds,
                "response_id": response.response_id,
                "response_metadata": dict(response.metadata),
            }

            append_jsonl_fsync(RESULTS_PATH, row)
            rows.append(row)
            seen.add(key)
            write_run_summary(rows, "UNIT_LOAD_RUNNING")

            print(
                "    locator={} exec={} e2e={} reasons={} in_tok={} out_tok={} latency={:.2f}s".format(
                    locator,
                    result["execution"]["execution_success"],
                    result["evaluation"]["end_to_end_success"],
                    reasons,
                    response.usage.get("input_tokens", 0),
                    response.usage.get("output_tokens", 0),
                    latency_seconds,
                )
            )

        if len(seen) != EXPECTED_TOTAL_CALLS:
            raise RuntimeError(
                "Unit-load loop ended with {}/{} calls".format(
                    len(seen),
                    EXPECTED_TOTAL_CALLS,
                )
            )

        write_run_summary(rows, "UNIT_LOAD_COMPLETE")

    except Exception as exc:
        write_run_summary(
            rows,
            "UNIT_LOAD_INTERRUPTED",
            error="{}: {}".format(type(exc).__name__, str(exc)),
        )
        raise

    print()
    print("=" * 78)
    print("UNIT-LOAD EXTENSION COMPLETE")
    print("=" * 78)
    print("Completed calls: {}/{}".format(len(seen), EXPECTED_TOTAL_CALLS))
    print("Raw results:     {}".format(RESULTS_PATH))
    print("Freeze manifest: {}".format(FREEZE_MANIFEST_PATH))
    print("Run summary:     {}".format(RUN_SUMMARY_PATH))
    print()
    print(
        "Do not add models, prompts, or cases based on these outcomes. "
        "Next step is the already-frozen offline analysis script."
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    paths.require_output_stage(paths.UNIT_LOAD, Path(__file__))
    print("=" * 78)
    print("REPRESENTATION x EDIT INTERFACE — UNIT-LOAD EXTENSION")
    print("=" * 78)
    print("Design: 20 cases x 2 targets x 6 conditions x 3 models = 720 calls")
    print("Loads: 1 / 2 / 3 facts per unit")
    print("Interfaces: Select-unit rewrite / Exact patch")
    print("Primary: G_Patch = Patch(L1) - Patch(L3)")
    print("Key secondary: S = [Select(L1)-Select(L3)] - [Patch(L1)-Patch(L3)]")
    print("Cluster unit: case")
    print("Master seed: {}".format(UNIT_LOAD_MASTER_SEED))
    print()

    if UNIT_LOAD_MASTER_SEED in (20260926, 20261001, 20261004):
        raise RuntimeError("Unit-load formal seed reuses a known development/formal seed")

    print("[1/6] Verifying completed baseline reference...")
    baseline_reference = load_baseline_reference()
    print("  PASS | same frozen baseline prompt and model list")

    print("[2/6] Building/verifying 20 fresh unit-load cases...")
    cases, validation, case_status = build_or_verify_cases(baseline_reference)
    print(
        "  {} | {} cases | {} target instances | {} oracle-success checks | {} wrong-ID execution checks".format(
            case_status,
            validation["case_count"],
            validation["target_instance_count"],
            validation["oracle_correct_action_checks"],
            validation["wrong_existing_id_execution_checks"],
        )
    )

    print("[3/6] Building/validating balanced 720-call execution plan...")
    execution_rows, position_balance = model_major_execution_rows(cases)
    print("  PASS | {} planned calls | each position count is 6 or 7 per condition/model".format(len(execution_rows)))

    print("[4/6] Checking the three frozen Ollama model snapshots...")
    backend = OllamaBackend(
        base_url=OLLAMA_BASE_URL,
        timeout_seconds=OLLAMA_TIMEOUT,
    )
    model_snapshots = preflight_models(backend, baseline_reference)
    print("  PASS | model/runtime snapshots match the completed baseline")

    print("[5/6] Writing preflight and freezing protocol + analysis plan...")
    preflight = {
        "status": "UNIT_LOAD_PREFLIGHT_PASS",
        "created_at_utc": utc_now_iso(),
        "protocol_version": PROTOCOL_VERSION,
        "case_validation": validation,
        "condition_position_balance": position_balance,
        "model_snapshots": model_snapshots,
        "primary_estimand": dict(PRIMARY_ESTIMAND),
        "key_secondary_estimand": dict(KEY_SECONDARY_ESTIMAND),
        "expected_total_calls": EXPECTED_TOTAL_CALLS,
    }
    write_json_atomic(PREFLIGHT_PATH, preflight)

    immutable = build_immutable_freeze_payload(
        cases=cases,
        validation=validation,
        baseline_reference=baseline_reference,
        model_snapshots=model_snapshots,
        execution_plan=execution_rows,
        position_balance=position_balance,
    )
    freeze_status = freeze_or_verify(immutable)
    print("  {} | protocol and offline analysis script frozen before calls".format(freeze_status))

    print("[6/6] Loading completed rows and running/resuming formal calls...")
    existing_rows, seen = load_and_validate_existing_results(cases, immutable)
    run_calls(
        backend=backend,
        cases=cases,
        immutable=immutable,
        execution_rows=execution_rows,
        existing_rows=existing_rows,
        seen=seen,
    )


if __name__ == "__main__":
    main()
