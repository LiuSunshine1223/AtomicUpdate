#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run or resume the frozen direct-extraction reader calibration."""

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

PROTOCOL_VERSION = "profile-extraction-reader-calibration-v1"

OUTPUT_DIR = paths.CALIBRATION

TASKS_PATH = OUTPUT_DIR / "extraction_tasks.jsonl"
CONTROL_PLAN_PATH = OUTPUT_DIR / "reader_control_plan.jsonl"
PREFLIGHT_PATH = OUTPUT_DIR / "calibration_preflight.json"
FREEZE_PATH = OUTPUT_DIR / "freeze_manifest.json"

RESULTS_PATH = OUTPUT_DIR / "reader_control_results.jsonl"
RUN_SUMMARY_PATH = OUTPUT_DIR / "reader_control_run_summary.json"

EXPECTED_LOGICAL_ROWS = 500
UNKNOWN_VALUE = "UNKNOWN"

OLLAMA_BASE_URL = paths.OLLAMA_URL
OLLAMA_TIMEOUT = 600.0


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
                    "{}:{} must be a JSON object.".format(
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


# ============================================================================
# Parsing and scoring
# ============================================================================

def parse_reader_response(
    text: str,
    response_schema: Mapping[str, Any],
) -> Dict[str, str]:
    value = json.loads(text.strip())
    if not isinstance(value, dict):
        raise ValueError("Reader response must be a JSON object.")

    required = [
        str(key)
        for key in response_schema.get("required", [])
    ]
    if set(value.keys()) != set(required):
        raise ValueError(
            "Reader response keys must be exactly {}.".format(
                required
            )
        )

    parsed: Dict[str, str] = {}
    for key in required:
        raw = value.get(key)
        if not isinstance(raw, str):
            raise ValueError(
                "Reader field {!r} must be a string.".format(key)
            )
        parsed[key] = raw.strip()

    return parsed


def exact_profile_match(
    parsed: Optional[Mapping[str, str]],
    expected: Mapping[str, str],
) -> bool:
    if parsed is None:
        return False

    if set(parsed.keys()) != set(expected.keys()):
        return False

    return all(
        str(parsed[key]) == str(expected[key])
        for key in expected
    )


def all_unknown(
    parsed: Optional[Mapping[str, str]],
    required_attributes: Sequence[str],
) -> bool:
    if parsed is None:
        return False

    attrs = [str(attribute) for attribute in required_attributes]
    if set(parsed.keys()) != set(attrs):
        return False

    return all(
        str(parsed[attribute]) == UNKNOWN_VALUE
        for attribute in attrs
    )


def attribute_match_map(
    parsed: Optional[Mapping[str, str]],
    expected: Mapping[str, str],
) -> dict:
    return {
        str(attribute): (
            parsed is not None
            and str(parsed.get(attribute)) == str(value)
        )
        for attribute, value in expected.items()
    }


# ============================================================================
# Frozen-design verification
# ============================================================================

def verify_frozen_design() -> Tuple[
    dict,
    dict,
    List[dict],
    Dict[str, dict],
]:
    required_paths = (
        TASKS_PATH,
        CONTROL_PLAN_PATH,
        PREFLIGHT_PATH,
        FREEZE_PATH,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                "Missing frozen calibration artifact: {}".format(path)
            )

    tasks = read_jsonl(TASKS_PATH)
    plan = read_jsonl(CONTROL_PLAN_PATH)
    preflight = read_json(PREFLIGHT_PATH)
    freeze = read_json(FREEZE_PATH)

    if preflight.get("status") != "PASS":
        raise RuntimeError("Calibration preflight is not PASS.")

    if not bool(preflight.get("formal_updater_calls_blocked")):
        raise RuntimeError(
            "Calibration preflight no longer blocks updater calls."
        )

    if freeze.get("status") != "CALIBRATION_PROTOCOL_FROZEN":
        raise RuntimeError(
            "Calibration protocol is not frozen."
        )

    if freeze.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(
            "Unexpected calibration protocol version: {!r}".format(
                freeze.get("protocol_version")
            )
        )

    if not bool(
        freeze.get("fresh_formal_batch_required_after_calibration_pass")
    ):
        raise RuntimeError(
            "Fresh-formal-batch requirement is missing."
        )

    if len(tasks) != 100:
        raise RuntimeError(
            "Expected 100 extraction tasks, found {}.".format(
                len(tasks)
            )
        )

    if len(plan) != EXPECTED_LOGICAL_ROWS:
        raise RuntimeError(
            "Expected {} control rows, found {}.".format(
                EXPECTED_LOGICAL_ROWS,
                len(plan),
            )
        )

    frozen_hashes = freeze.get("frozen_file_sha256", {})
    actual_hashes = {
        "extraction_tasks.jsonl": sha256_file(TASKS_PATH),
        "reader_control_plan.jsonl": sha256_file(
            CONTROL_PLAN_PATH
        ),
    }
    for filename, actual in actual_hashes.items():
        expected = str(frozen_hashes.get(filename, ""))
        if expected != actual:
            raise RuntimeError(
                "Frozen hash mismatch for {}: expected {}, got {}."
                .format(filename, expected, actual)
            )

    counts = Counter(
        (str(row["arm"]), str(row["grouping"]))
        for row in plan
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
            "Frozen control-plan counts changed: {}".format(
                dict(counts)
            )
        )

    control_ids = [str(row["control_id"]) for row in plan]
    if len(control_ids) != len(set(control_ids)):
        raise RuntimeError(
            "Duplicate control_id in frozen control plan."
        )

    task_by_id = {
        str(task["extraction_task_id"]): task
        for task in tasks
    }
    if len(task_by_id) != 100:
        raise RuntimeError(
            "Duplicate extraction_task_id detected."
        )

    # Exact-input caching is valid only if identical frozen reader inputs have
    # identical schemas and, for these calibration controls, identical scoring
    # targets. Verify that invariant before first model call.
    input_groups: Dict[str, List[dict]] = {}
    for row in plan:
        input_groups.setdefault(
            str(row["reader_input_sha256"]),
            [],
        ).append(row)

    for input_hash, rows in input_groups.items():
        first = rows[0]
        for row in rows[1:]:
            if row["user_prompt"] != first["user_prompt"]:
                raise RuntimeError(
                    "Same input hash maps to different user prompts."
                )
            if canonical_json(row["response_schema"]) != canonical_json(
                first["response_schema"]
            ):
                raise RuntimeError(
                    "Same input hash maps to different response schemas."
                )
            if canonical_json(row["expected_values"]) != canonical_json(
                first["expected_values"]
            ):
                raise RuntimeError(
                    "Same input hash maps to different expected values."
                )

    return freeze, preflight, plan, task_by_id


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
                "Existing result is missing control_id."
            )

        if control_id in by_control_id:
            raise RuntimeError(
                "Duplicate saved result: {}".format(control_id)
            )

        if control_id not in plan_by_id:
            raise RuntimeError(
                "Unexpected saved control_id: {}".format(control_id)
            )

        planned = plan_by_id[control_id]
        for field in (
            "extraction_task_id",
            "source_task_id",
            "case_id",
            "update_id",
            "arm",
            "grouping",
            "reader_input_sha256",
        ):
            if row.get(field) != planned.get(field):
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
    gates: Mapping[str, Any],
) -> dict:
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["arm"]),
            str(row["grouping"]),
        )
        groups.setdefault(key, []).append(row)

    def exact_count(arm: str, grouping: str) -> int:
        return sum(
            1
            for row in groups.get((arm, grouping), [])
            if bool(row["exact_profile_match"])
        )

    oracle_g1 = exact_count("oracle", "g1")
    oracle_g3 = exact_count("oracle", "g3")
    stale_g1 = exact_count("stale", "g1")
    stale_g3 = exact_count("stale", "g3")

    no_memory_rows = groups.get(("no_memory", "None"), [])
    no_memory_unknown = sum(
        1
        for row in no_memory_rows
        if bool(row["all_unknown"])
    )

    valid_json = sum(
        1
        for row in rows
        if bool(row["response_json_valid"])
    )

    checks = {
        "oracle_g1": oracle_g1 >= int(
            gates["oracle_exact_profile_min_per_grouping"]
        ),
        "oracle_g3": oracle_g3 >= int(
            gates["oracle_exact_profile_min_per_grouping"]
        ),
        "stale_g1": stale_g1 >= int(
            gates["stale_exact_profile_min_per_grouping"]
        ),
        "stale_g3": stale_g3 >= int(
            gates["stale_exact_profile_min_per_grouping"]
        ),
        "no_memory_all_unknown": no_memory_unknown >= int(
            gates["no_memory_all_unknown_min"]
        ),
        "valid_json": valid_json >= int(
            gates["valid_json_min_rows"]
        ),
    }

    passed = all(checks.values())

    return {
        "status": (
            "READER_CALIBRATION_GATE_PASS"
            if passed
            else "READER_CALIBRATION_GATE_FAIL"
        ),
        "frozen_gate_thresholds": dict(gates),
        "observed": {
            "oracle_g1_exact_profile": oracle_g1,
            "oracle_g3_exact_profile": oracle_g3,
            "stale_g1_exact_profile": stale_g1,
            "stale_g3_exact_profile": stale_g3,
            "no_memory_all_unknown": no_memory_unknown,
            "valid_json_rows": valid_json,
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
        "gate_checks": checks,
        "formal_updater_calls_allowed_from_this_batch": False,
        "fresh_formal_batch_required": True,
        "next_step": (
            (
                "Calibration passed. Do not run updater calls on this batch. "
                "Build a fresh non-overlapping formal batch next."
            )
            if passed
            else (
                "Calibration failed. Do not run updater calls. Diagnose the "
                "reader/extraction errors without lowering the frozen gates."
            )
        ),
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    paths.require_output_stage(paths.CALIBRATION, Path(__file__).with_name("build_calibration.py"))
    print("=" * 78)
    print("PROFILE EXTRACTION READER CALIBRATION — RUN")
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
        "context_window": int(
            reader["context_window"]
        ),
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

    print("Frozen protocol:        PASS")
    print("Reader:                 {}".format(reader_model))
    print("Temperature:            {}".format(config.temperature))
    print("Logical controls:       {}".format(len(plan)))
    print(
        "Unique frozen inputs:   {}".format(
            len(
                {
                    str(row["reader_input_sha256"])
                    for row in plan
                }
            )
        )
    )
    print("Formal updater calls:    BLOCKED")
    print(
        "Gate: Oracle >= {}/100 each | Stale >= {}/100 each | "
        "No-memory UNKNOWN >= {}/100 | Valid JSON >= {}/500".format(
            freeze["reader_gates"][
                "oracle_exact_profile_min_per_grouping"
            ],
            freeze["reader_gates"][
                "stale_exact_profile_min_per_grouping"
            ],
            freeze["reader_gates"][
                "no_memory_all_unknown_min"
            ],
            freeze["reader_gates"]["valid_json_min_rows"],
        )
    )
    print()

    backend = OllamaBackend(
        base_url=OLLAMA_BASE_URL,
        timeout_seconds=OLLAMA_TIMEOUT,
    )
    snapshot = backend.describe(reader_model)

    print(
        "Reader snapshot:        {} | {} | {}".format(
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

    # Cache deterministic responses by exact frozen reader input.
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
        "Completed:              {}/{}".format(
            len(completed_by_id),
            EXPECTED_LOGICAL_ROWS,
        )
    )
    print(
        "Remaining:              {}".format(
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

        user_prompt = str(plan_row["user_prompt"])
        response_schema = dict(plan_row["response_schema"])
        expected_values = {
            str(key): str(value)
            for key, value in plan_row["expected_values"].items()
        }

        reader_input_payload = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "generation_config": generation_config_dict,
            "response_schema": response_schema,
        }
        input_hash = hashlib.sha256(
            canonical_json(reader_input_payload).encode("utf-8")
        ).hexdigest()

        if input_hash != str(plan_row["reader_input_sha256"]):
            raise RuntimeError(
                "Frozen reader-input hash mismatch for {}.".format(
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
                    plan_row["source_task_id"],
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
                    plan_row["source_task_id"],
                    plan_row["arm"],
                    plan_row["grouping"],
                )
            )

            request_id = (
                "profile_extraction_calibration__{:04d}__{}".format(
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
        parsed_mapping = (
            dict(parsed)
            if isinstance(parsed, Mapping)
            else None
        )

        exact_match = exact_profile_match(
            parsed_mapping,
            expected_values,
        )
        unknown_match = all_unknown(
            parsed_mapping,
            plan_row["required_attributes"],
        )

        result_row = {
            "logical_control_index": int(
                plan_row["logical_control_index"]
            ),
            "control_id": control_id,
            "extraction_task_id": str(
                plan_row["extraction_task_id"]
            ),
            "source_task_id": str(
                plan_row["source_task_id"]
            ),
            "case_id": str(plan_row["case_id"]),
            "update_id": str(plan_row["update_id"]),
            "arm": str(plan_row["arm"]),
            "grouping": plan_row["grouping"],
            "subject": str(plan_row["subject"]),
            "required_attributes": [
                str(attribute)
                for attribute in plan_row[
                    "required_attributes"
                ]
            ],
            "expected_values": expected_values,
            "parsed_response": parsed_mapping,
            "attribute_match": attribute_match_map(
                parsed_mapping,
                expected_values,
            ),
            "exact_profile_match": bool(exact_match),
            "all_unknown": bool(unknown_match),
            "reader_input_sha256": input_hash,
            "cache_reused": bool(cache_reused),
            "reader_model": reader_model,
            "generation_config": dict(
                generation_config_dict
            ),
            **{
                key: value
                for key, value in call_record.items()
                if key != "parsed_response"
            },
        }

        append_jsonl_fsync(RESULTS_PATH, result_row)
        completed_by_id[control_id] = result_row

        print(
            "    parsed={} expected={} exact={} unknown={}".format(
                parsed_mapping,
                expected_values,
                exact_match,
                unknown_match,
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
            "Calibration ended incomplete: {}/{}.".format(
                len(final_by_id),
                EXPECTED_LOGICAL_ROWS,
            )
        )

    ordered_results = [
        final_by_id[str(plan_row["control_id"])]
        for plan_row in plan
    ]

    snapshots = {
        canonical_json(row["reader_model_snapshot"])
        for row in ordered_results
    }
    if len(snapshots) != 1:
        raise RuntimeError(
            "Reader model/runtime snapshot changed within calibration."
        )

    summary = summarize_gate(
        ordered_results,
        freeze["reader_gates"],
    )
    summary.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "reader_model": reader_model,
            "reader_model_snapshot": snapshot_fingerprint(
                snapshot
            ),
            "new_reader_calls_this_invocation": new_calls,
            "cache_hits_this_invocation": cache_hits,
            "results_path": str(RESULTS_PATH),
            "results_sha256": sha256_file(RESULTS_PATH),
            "calibration_preflight_sha256": sha256_file(
                PREFLIGHT_PATH
            ),
            "freeze_manifest_sha256": sha256_file(
                FREEZE_PATH
            ),
        }
    )

    write_json_atomic(RUN_SUMMARY_PATH, summary)

    observed = summary["observed"]

    print()
    print("=" * 78)
    print("PROFILE EXTRACTION READER CALIBRATION COMPLETE")
    print("=" * 78)
    print(
        "Oracle g1 exact:       {}/100".format(
            observed["oracle_g1_exact_profile"]
        )
    )
    print(
        "Oracle g3 exact:       {}/100".format(
            observed["oracle_g3_exact_profile"]
        )
    )
    print(
        "Stale g1 exact:        {}/100".format(
            observed["stale_g1_exact_profile"]
        )
    )
    print(
        "Stale g3 exact:        {}/100".format(
            observed["stale_g3_exact_profile"]
        )
    )
    print(
        "No-memory all UNKNOWN: {}/100".format(
            observed["no_memory_all_unknown"]
        )
    )
    print(
        "Valid JSON:            {}/500".format(
            observed["valid_json_rows"]
        )
    )
    print(
        "Logical rows:          {}".format(
            observed["logical_rows"]
        )
    )
    print(
        "Unique reader inputs:  {}".format(
            observed["unique_reader_inputs"]
        )
    )
    print(
        "Actual reader calls:   {}".format(
            observed["actual_reader_calls"]
        )
    )
    print(
        "Cache reused rows:     {}".format(
            observed["cache_reused_rows"]
        )
    )
    print("GATE: {}".format(summary["status"]))
    print()
    print("Results: {}".format(RESULTS_PATH))
    print("Summary: {}".format(RUN_SUMMARY_PATH))
    print()
    print("Formal updater calls from this batch: BLOCKED")

    if summary["status"] != "READER_CALIBRATION_GATE_PASS":
        print(
            "STOP: diagnose this calibration. Do not lower the frozen gates."
        )
        raise SystemExit(2)

    print(
        "PASS: build a fresh non-overlapping formal Layer-2 batch next."
    )
    print(
        "Do NOT run updater calls on this reused calibration batch."
    )


if __name__ == "__main__":
    main()
