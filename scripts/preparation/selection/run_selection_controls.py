#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run or resume the candidate-selection reader's frozen qualification controls."""

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

# ============================================================================
# Frozen paths
# ============================================================================

OUTPUT_DIR = paths.SELECTION

CASES_PATH = OUTPUT_DIR / "formal_cases.jsonl"
TASKS_PATH = OUTPUT_DIR / "profile_tasks.jsonl"
UPDATER_PLAN_PATH = OUTPUT_DIR / "updater_execution_plan.jsonl"
CONTROL_PLAN_PATH = OUTPUT_DIR / "reader_control_plan.jsonl"
PREFLIGHT_PATH = OUTPUT_DIR / "formal_preflight.json"
FREEZE_PATH = OUTPUT_DIR / "freeze_manifest.json"

RESULTS_PATH = OUTPUT_DIR / "reader_control_results.jsonl"
RUN_SUMMARY_PATH = OUTPUT_DIR / "reader_control_run_summary.json"

EXPECTED_PROTOCOL_VERSION = "profile-selection-layer2-formal-v1"
EXPECTED_LOGICAL_ROWS = 500
RUNNER_REVISION = "prompt-order-fix-2"

OLLAMA_BASE_URL = paths.OLLAMA_URL
OLLAMA_TIMEOUT = 600.0


# ============================================================================
# Helpers
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
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(
                    "{}:{} must be a JSON object".format(
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


def public_candidate_view(
    task: Mapping[str, Any],
) -> List[dict]:
    # profile_tasks.jsonl was written with sort_keys=True. That can reorder
    # nested attribute-dict keys on disk, while the frozen prompt hash was
    # computed before serialization using task["attributes"] schema order.
    # Reconstruct the exact frozen attribute order explicitly.
    attribute_order = [
        str(attribute)
        for attribute in task["attributes"]
    ]

    result = []
    for candidate in task["candidates"]:
        raw_attributes = candidate["attributes"]
        ordered_attributes = {
            attribute: raw_attributes[attribute]
            for attribute in attribute_order
        }
        result.append(
            {
                "profile_id": str(candidate["profile_id"]),
                "attributes": ordered_attributes,
            }
        )
    return result


def build_reader_user_prompt(
    *,
    memory: Optional[Sequence[Mapping[str, Any]]],
    subject: str,
    candidates: Sequence[Mapping[str, Any]],
) -> str:
    if memory is None:
        memory_text = "(no memory state provided)"
    else:
        memory_text = json.dumps(
            [
                {
                    "memory_id": str(unit["memory_id"]),
                    "text": str(unit["text"]),
                }
                for unit in memory
            ],
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


def parse_reader_response(
    text: str,
    response_schema: Mapping[str, Any],
) -> Dict[str, Any]:
    value = json.loads(text.strip())
    if not isinstance(value, dict):
        raise ValueError("Reader response must be a JSON object.")
    if set(value.keys()) != {"profile_id"}:
        raise ValueError(
            "Reader response must contain exactly profile_id."
        )

    allowed = (
        response_schema
        .get("properties", {})
        .get("profile_id", {})
        .get("enum", [])
    )
    profile_id = value.get("profile_id")
    if profile_id not in allowed:
        raise ValueError(
            "Unexpected profile_id: {!r}".format(profile_id)
        )
    return value


# ============================================================================
# Frozen-input verification
# ============================================================================

def verify_frozen_design() -> Tuple[
    dict,
    dict,
    List[dict],
    Dict[str, dict],
]:
    required = (
        CASES_PATH,
        TASKS_PATH,
        UPDATER_PLAN_PATH,
        CONTROL_PLAN_PATH,
        PREFLIGHT_PATH,
        FREEZE_PATH,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                "Missing frozen Layer-2 artifact: {}".format(path)
            )

    preflight = read_json(PREFLIGHT_PATH)
    freeze = read_json(FREEZE_PATH)
    control_plan = read_jsonl(CONTROL_PLAN_PATH)
    tasks = read_jsonl(TASKS_PATH)

    if preflight.get("status") != "PASS":
        raise RuntimeError("formal_preflight.json is not PASS.")

    if freeze.get("status") != "FORMAL_PROTOCOL_FROZEN":
        raise RuntimeError(
            "freeze_manifest.json is not FORMAL_PROTOCOL_FROZEN."
        )

    if freeze.get("protocol_version") != EXPECTED_PROTOCOL_VERSION:
        raise RuntimeError(
            "Unexpected protocol version: {!r}".format(
                freeze.get("protocol_version")
            )
        )

    if len(control_plan) != EXPECTED_LOGICAL_ROWS:
        raise RuntimeError(
            "Expected {} control rows, found {}.".format(
                EXPECTED_LOGICAL_ROWS,
                len(control_plan),
            )
        )

    if len(tasks) != 100:
        raise RuntimeError(
            "Expected 100 profile tasks, found {}.".format(
                len(tasks)
            )
        )

    frozen_hashes = freeze.get("frozen_file_sha256", {})
    actual_hashes = {
        "formal_cases.jsonl": sha256_file(CASES_PATH),
        "profile_tasks.jsonl": sha256_file(TASKS_PATH),
        "updater_execution_plan.jsonl": sha256_file(
            UPDATER_PLAN_PATH
        ),
        "reader_control_plan.jsonl": sha256_file(
            CONTROL_PLAN_PATH
        ),
    }
    for name, actual in actual_hashes.items():
        expected = frozen_hashes.get(name)
        if expected != actual:
            raise RuntimeError(
                "Frozen hash mismatch for {}: expected {}, got {}."
                .format(name, expected, actual)
            )

    task_by_id = {
        str(task["task_id"]): task
        for task in tasks
    }
    if len(task_by_id) != 100:
        raise RuntimeError("Duplicate task_id in profile_tasks.jsonl.")

    control_ids = [
        str(row["control_id"])
        for row in control_plan
    ]
    if len(control_ids) != len(set(control_ids)):
        raise RuntimeError(
            "Duplicate control_id in reader_control_plan.jsonl."
        )

    expected_counts = {
        ("oracle", "g1"): 100,
        ("oracle", "g3"): 100,
        ("stale", "g1"): 100,
        ("stale", "g3"): 100,
        ("no_memory", "None"): 100,
    }
    counts = Counter(
        (
            str(row["arm"]),
            str(row["grouping"]),
        )
        for row in control_plan
    )
    if dict(counts) != expected_counts:
        raise RuntimeError(
            "Control-plan arm counts changed: {}".format(
                dict(counts)
            )
        )

    return freeze, preflight, control_plan, task_by_id


def verify_existing_results(
    existing_rows: Sequence[Mapping[str, Any]],
    plan_by_id: Mapping[str, Mapping[str, Any]],
    reader_model: str,
    generation_config: Mapping[str, Any],
) -> Dict[str, dict]:
    by_control_id: Dict[str, dict] = {}

    for row in existing_rows:
        control_id = str(row.get("control_id", ""))
        if not control_id:
            raise RuntimeError(
                "Existing reader control row is missing control_id."
            )
        if control_id in by_control_id:
            raise RuntimeError(
                "Duplicate saved control result: {}".format(
                    control_id
                )
            )
        if control_id not in plan_by_id:
            raise RuntimeError(
                "Unexpected saved control_id: {}".format(
                    control_id
                )
            )

        plan = plan_by_id[control_id]

        fields = (
            "task_id",
            "case_id",
            "update_id",
            "arm",
            "grouping",
            "reader_input_sha256",
        )
        for field in fields:
            if row.get(field) != plan.get(field):
                raise RuntimeError(
                    "Saved result differs from plan for {} field {}."
                    .format(control_id, field)
                )

        if str(row.get("reader_model")) != reader_model:
            raise RuntimeError(
                "Saved reader model mismatch for {}.".format(
                    control_id
                )
            )

        if row.get("generation_config") != dict(generation_config):
            raise RuntimeError(
                "Saved generation config mismatch for {}.".format(
                    control_id
                )
            )

        by_control_id[control_id] = dict(row)

    return by_control_id


# ============================================================================
# Gate summary
# ============================================================================

def summarize_gate(
    rows: Sequence[Mapping[str, Any]],
    frozen_gates: Mapping[str, Any],
) -> dict:
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["arm"]),
            str(row["grouping"]),
        )
        groups.setdefault(key, []).append(row)

    def gold_correct(arm: str, grouping: str) -> int:
        return sum(
            1
            for row in groups.get((arm, grouping), [])
            if bool(row["gold_correct"])
        )

    def stale_selected(arm: str, grouping: str) -> int:
        return sum(
            1
            for row in groups.get((arm, grouping), [])
            if bool(row["stale_profile_selected"])
        )

    oracle_g1 = gold_correct("oracle", "g1")
    oracle_g3 = gold_correct("oracle", "g3")
    stale_g1 = stale_selected("stale", "g1")
    stale_g3 = stale_selected("stale", "g3")
    no_memory_gold = gold_correct("no_memory", "None")

    oracle_min = int(
        frozen_gates["oracle_min_correct_per_grouping"]
    )
    stale_min = int(
        frozen_gates["stale_expected_profile_min_per_grouping"]
    )
    no_memory_max = int(
        frozen_gates["no_memory_max_gold_correct"]
    )

    gates = {
        "oracle_g1": oracle_g1 >= oracle_min,
        "oracle_g3": oracle_g3 >= oracle_min,
        "stale_g1": stale_g1 >= stale_min,
        "stale_g3": stale_g3 >= stale_min,
        "no_memory": no_memory_gold <= no_memory_max,
    }

    return {
        "status": (
            "READER_CONTROL_GATE_PASS"
            if all(gates.values())
            else "READER_CONTROL_GATE_FAIL"
        ),
        "frozen_gate_thresholds": dict(frozen_gates),
        "observed": {
            "oracle_g1_gold_correct": oracle_g1,
            "oracle_g3_gold_correct": oracle_g3,
            "stale_g1_expected_profile_selected": stale_g1,
            "stale_g3_expected_profile_selected": stale_g3,
            "no_memory_gold_correct": no_memory_gold,
            "valid_json_rows": sum(
                1
                for row in rows
                if bool(row["response_json_valid"])
            ),
            "logical_rows": len(rows),
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
        },
        "gate_checks": gates,
        "next_step": (
            "Formal updater stage may begin."
            if all(gates.values())
            else (
                "Do not start formal updater calls. Inspect the failed "
                "reader-control gate first."
            )
        ),
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    paths.require_output_stage(paths.SELECTION, Path(__file__).with_name("build_selection.py"))
    print("=" * 78)
    print("PROFILE SELECTION LAYER-2 — FORMAL READER CONTROLS")
    print("Runner revision:     {}".format(RUNNER_REVISION))
    print("=" * 78)

    freeze, preflight, plan, task_by_id = verify_frozen_design()

    reader = freeze["reader"]
    reader_model = str(reader["model"])
    generation_config_dict = {
        "model": reader_model,
        "temperature": float(reader["temperature"]),
        "seed": int(reader["seed"]),
        "max_output_tokens": int(
            reader["max_output_tokens"]
        ),
        "context_window": int(reader["context_window"]),
    }

    config = GenerationConfig(
        model=reader_model,
        temperature=generation_config_dict["temperature"],
        seed=generation_config_dict["seed"],
        max_output_tokens=generation_config_dict[
            "max_output_tokens"
        ],
        context_window=generation_config_dict[
            "context_window"
        ],
    )

    system_prompt = str(freeze["reader_system_prompt"])
    response_schema = dict(freeze["reader_response_schema"])

    if sha256_text(system_prompt) != str(
        freeze["reader_system_prompt_sha256"]
    ):
        raise RuntimeError("Frozen reader system-prompt hash mismatch.")

    if sha256_text(
        canonical_json(response_schema)
    ) != str(freeze["reader_response_schema_sha256"]):
        raise RuntimeError("Frozen reader response-schema hash mismatch.")

    print("Frozen protocol:     PASS")
    print("Reader:              {}".format(reader_model))
    print("Temperature:         {}".format(config.temperature))
    print("Logical controls:    {}".format(len(plan)))
    print(
        "Gate: oracle >= {}/100 per grouping | stale >= {}/100 "
        "per grouping | no-memory <= {}/100 gold".format(
            freeze["reader_gates"][
                "oracle_min_correct_per_grouping"
            ],
            freeze["reader_gates"][
                "stale_expected_profile_min_per_grouping"
            ],
            freeze["reader_gates"][
                "no_memory_max_gold_correct"
            ],
        )
    )
    print()

    backend = OllamaBackend(
        base_url=OLLAMA_BASE_URL,
        timeout_seconds=OLLAMA_TIMEOUT,
    )
    snapshot = backend.describe(reader_model)
    print(
        "Reader snapshot:     {} | {} | {}".format(
            snapshot.get("parameter_size"),
            snapshot.get("quantization_level"),
            snapshot.get(
                "model_digest",
                snapshot.get("digest"),
            ),
        )
    )
    print()

    plan_by_id = {
        str(row["control_id"]): row
        for row in plan
    }

    existing_rows = read_jsonl(RESULTS_PATH)
    completed_by_id = verify_existing_results(
        existing_rows,
        plan_by_id,
        reader_model,
        generation_config_dict,
    )

    # Persisted cache: identical frozen reader inputs can reuse a completed
    # response after a safe resume.
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
        "Completed:           {}/{}".format(
            len(completed_by_id),
            EXPECTED_LOGICAL_ROWS,
        )
    )
    print(
        "Remaining:           {}".format(
            EXPECTED_LOGICAL_ROWS - len(completed_by_id)
        )
    )
    print()

    new_calls = 0
    cache_hits = 0

    for logical_index, plan_row in enumerate(plan, start=1):
        control_id = str(plan_row["control_id"])
        if control_id in completed_by_id:
            continue

        task = task_by_id[str(plan_row["task_id"])]
        candidates = public_candidate_view(task)

        memory = plan_row.get("memory")
        user_prompt = build_reader_user_prompt(
            memory=memory,
            subject=str(plan_row["subject"]),
            candidates=candidates,
        )

        user_prompt_sha = sha256_text(user_prompt)
        if user_prompt_sha != str(
            plan_row["user_prompt_sha256"]
        ):
            raise RuntimeError(
                "Rebuilt reader prompt hash mismatch for {}.".format(
                    control_id
                )
            )

        reader_input_payload = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "generation_config": generation_config_dict,
            "response_schema": response_schema,
        }
        input_hash = sha256_text(
            canonical_json(reader_input_payload)
        )
        if input_hash != str(plan_row["reader_input_sha256"]):
            raise RuntimeError(
                "Reader-input hash mismatch for {}.".format(
                    control_id
                )
            )

        cache_reused = input_hash in cache

        if cache_reused:
            cache_hits += 1
            call_record = dict(cache[input_hash])
            print(
                "[{}/{}] {} | {} | {} | CACHE".format(
                    logical_index,
                    EXPECTED_LOGICAL_ROWS,
                    plan_row["task_id"],
                    plan_row["arm"],
                    plan_row["grouping"],
                )
            )
        else:
            new_calls += 1
            print(
                "[{}/{}] {} | {} | {}".format(
                    logical_index,
                    EXPECTED_LOGICAL_ROWS,
                    plan_row["task_id"],
                    plan_row["arm"],
                    plan_row["grouping"],
                )
            )

            request_id = (
                "layer2_reader_control__{:04d}__{}".format(
                    logical_index,
                    input_hash[:12],
                )
            )
            start = time.perf_counter()
            response = backend.generate(
                request_id=request_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                config=config,
                response_schema=response_schema,
            )
            latency = time.perf_counter() - start

            response_json_valid = True
            response_parse_error: Optional[str] = None
            parsed_response: Optional[dict]

            try:
                parsed_response = parse_reader_response(
                    response.text,
                    response_schema,
                )
            except Exception as exc:
                response_json_valid = False
                response_parse_error = str(exc)
                parsed_response = None

            call_record = {
                "reader_raw_response": response.text,
                "response_json_valid": response_json_valid,
                "response_parse_error": response_parse_error,
                "parsed_response": parsed_response,
                "reader_usage": dict(response.usage),
                "reader_latency_seconds": latency,
                "reader_response_id": response.response_id,
                "reader_response_metadata": dict(response.metadata),
                "reader_model_snapshot": snapshot_fingerprint(
                    snapshot
                ),
            }
            cache[input_hash] = dict(call_record)

        parsed = call_record.get("parsed_response")
        selected_profile_id = (
            parsed.get("profile_id")
            if isinstance(parsed, Mapping)
            else None
        )

        result_row = {
            "logical_control_index": int(
                plan_row["logical_control_index"]
            ),
            "control_id": control_id,
            "task_id": str(plan_row["task_id"]),
            "case_id": str(plan_row["case_id"]),
            "update_id": str(plan_row["update_id"]),
            "arm": str(plan_row["arm"]),
            "grouping": plan_row["grouping"],
            "subject": str(plan_row["subject"]),
            "gold_profile_id": str(
                plan_row["gold_profile_id"]
            ),
            "stale_profile_id": str(
                plan_row["stale_profile_id"]
            ),
            "selected_profile_id": selected_profile_id,
            "gold_correct": (
                selected_profile_id
                == str(plan_row["gold_profile_id"])
            ),
            "stale_profile_selected": (
                selected_profile_id
                == str(plan_row["stale_profile_id"])
            ),
            "reader_input_sha256": input_hash,
            "cache_reused": bool(cache_reused),
            "reader_model": reader_model,
            "generation_config": dict(
                generation_config_dict
            ),
            **call_record,
        }

        append_jsonl_fsync(RESULTS_PATH, result_row)
        completed_by_id[control_id] = result_row

        print(
            "    selected={} gold={} stale={} gold_correct={}".format(
                selected_profile_id,
                plan_row["gold_profile_id"],
                plan_row["stale_profile_id"],
                result_row["gold_correct"],
            )
        )

    final_rows = read_jsonl(RESULTS_PATH)
    final_by_id = verify_existing_results(
        final_rows,
        plan_by_id,
        reader_model,
        generation_config_dict,
    )
    if len(final_by_id) != EXPECTED_LOGICAL_ROWS:
        raise RuntimeError(
            "Reader-control run ended incomplete: {}/{}.".format(
                len(final_by_id),
                EXPECTED_LOGICAL_ROWS,
            )
        )

    # Reorder by frozen plan before summary.
    ordered_results = [
        final_by_id[str(plan_row["control_id"])]
        for plan_row in plan
    ]

    # All actual calls in this formal control stage should share the same
    # reader snapshot. Cached rows inherit their originating snapshot.
    snapshot_fingerprints = {
        canonical_json(row["reader_model_snapshot"])
        for row in ordered_results
    }
    if len(snapshot_fingerprints) != 1:
        raise RuntimeError(
            "Reader model/runtime snapshot changed within the control run."
        )

    summary = summarize_gate(
        ordered_results,
        freeze["reader_gates"],
    )
    summary.update(
        {
            "protocol_version": EXPECTED_PROTOCOL_VERSION,
            "source_freeze_sha256": sha256_file(FREEZE_PATH),
            "reader_model": reader_model,
            "reader_model_snapshot": snapshot_fingerprint(
                snapshot
            ),
            "new_reader_calls_this_invocation": new_calls,
            "cache_hits_this_invocation": cache_hits,
            "results_path": str(RESULTS_PATH),
            "results_sha256": sha256_file(RESULTS_PATH),
        }
    )
    write_json_atomic(RUN_SUMMARY_PATH, summary)

    observed = summary["observed"]

    print()
    print("=" * 78)
    print("FORMAL READER CONTROL STAGE COMPLETE")
    print("=" * 78)
    print(
        "Oracle g1:          {}/100".format(
            observed["oracle_g1_gold_correct"]
        )
    )
    print(
        "Oracle g3:          {}/100".format(
            observed["oracle_g3_gold_correct"]
        )
    )
    print(
        "Stale g1 expected:  {}/100".format(
            observed["stale_g1_expected_profile_selected"]
        )
    )
    print(
        "Stale g3 expected:  {}/100".format(
            observed["stale_g3_expected_profile_selected"]
        )
    )
    print(
        "No-memory gold:     {}/100".format(
            observed["no_memory_gold_correct"]
        )
    )
    print(
        "Valid JSON:         {}/500".format(
            observed["valid_json_rows"]
        )
    )
    print(
        "Actual reader calls: {}".format(
            observed["actual_reader_calls"]
        )
    )
    print(
        "Cache reused rows:   {}".format(
            observed["cache_reused_rows"]
        )
    )
    print("GATE: {}".format(summary["status"]))
    print()
    print("Results: {}".format(RESULTS_PATH))
    print("Summary: {}".format(RUN_SUMMARY_PATH))

    if summary["status"] != "READER_CONTROL_GATE_PASS":
        print()
        print(
            "STOP: do not start the 1,200 formal updater calls."
        )
        raise SystemExit(2)

    print()
    print(
        "PASS: the frozen formal updater stage may now begin."
    )


if __name__ == "__main__":
    main()
