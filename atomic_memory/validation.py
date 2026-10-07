"""Construct and audit the fixed, paired endpoint-validation study (experiment 5).

State-property labels come from explicit current assertions, never from the
strict text parser or from reader answers. This module makes no model calls.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Mapping, Optional

from atomic_memory import paths
from atomic_memory.core import (
    ATTRIBUTE_SPECS, CURRENT_TEMPLATE, canonical_json, evaluate_final_memory,
    generate_cases, sha256_json,
)
from atomic_memory.reader import (
    build_reader_user_prompt, exact_profile_match, parse_reader_response,
    response_schema_for_attributes,
)

OUT = paths.OUTPUT_ROOT / "exp5_validation"
VERSION = "controlled-endpoint-validation-v1"
CASE_SEED = 20261013
SCHEDULE_SEED = 20261014
BOOTSTRAP_SEED = 20261015
BOOTSTRAP_REPLICATES = 20000
N_CASES = 50
VARIANTS = (
    "clean", "surface", "contradiction", "deletion", "misbinding", "other_damage",
)
SNAPSHOT_KEYS = (
    "provider", "ollama_version", "model", "model_digest", "parameter_size",
    "quantization_level",
)
CONFIG = dict(model="llama3.1:8b-instruct-q4_K_M", temperature=0.0, seed=42,
              max_output_tokens=64, context_window=8192)
SOURCE_FILES = (
    "exp1_baseline/formal_cases.jsonl",
    "exp3_unit_load/unit_load_cases.jsonl",
    "exp4_profile/formal_cases.jsonl",
    "preparation/selection/formal_cases.jsonl",
    "exp4_profile/freeze_manifest.json",
    "exp4_profile/reader_control_results.jsonl",
    "exp4_profile/reader_control_run_summary.json",
    "exp4_profile/reader_control_plan.jsonl",
)
CODE_FILES = (
    "atomic_memory/validation.py", "atomic_memory/core.py",
    "atomic_memory/backend.py", "atomic_memory/paths.py",
    "atomic_memory/reanalysis.py", "atomic_memory/reader.py",
    "scripts/exp5_validation/build_validation.py",
    "scripts/exp5_validation/run_validation.py",
    "scripts/exp5_validation/analyze_validation.py", "plot/plot_validation.py",
    "plot/style.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def records(path):
    return [json.loads(line) for line in Path(path).read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]


def write_text_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    os.replace(str(temp), str(path))


def write_json_atomic(path, value):
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, sort_keys=True,
                                      indent=2) + "\n")


def write_jsonl_atomic(path, rows):
    write_text_atomic(path, "".join(canonical_json(row) + "\n" for row in rows))


def now():
    return datetime.now(timezone.utc).isoformat()


def normalize_snapshot(snapshot):
    return {key: snapshot[key] for key in SNAPSHOT_KEYS}


def initial_signature(case):
    return sha256_json(sorted((f["subject"], f["attribute"], f["value"])
                              for f in case["facts"]))


def source_context(root):
    """Read qualification evidence and independently check prior case overlap."""
    root = Path(root)
    for name in SOURCE_FILES:
        require((root / name).is_file(), "Required prior evidence missing: " + name)
    freeze = load(root / "exp4_profile/freeze_manifest.json")
    summary = load(root / "exp4_profile/reader_control_run_summary.json")
    require(summary["status"] == "FORMAL_READER_CONTROL_GATE_PASS",
            "The primary reader control gate must already have passed")
    require(all(summary["gate_checks"].values()) and summary["observed"]["logical_rows"] == 500,
            "The qualified reader-control evidence is incomplete")
    require(all(summary["gate_checks"].values()) and summary["observed"]["logical_rows"] == 500,
            "The qualified reader-control evidence is incomplete")
    require(summary["freeze_manifest_sha256"] == digest(
        root / "exp4_profile/freeze_manifest.json"), "Reader gate freeze mismatch")
    require(summary["results_sha256"] == digest(
        root / "exp4_profile/reader_control_results.jsonl"), "Reader gate data mismatch")
    control_plan = records(root / "exp4_profile/reader_control_plan.jsonl")
    require(digest(root / "exp4_profile/reader_control_plan.jsonl") ==
            freeze["frozen_file_sha256"]["reader_control_plan.jsonl"], "Reader control plan mismatch")
    config = control_plan[0]["generation_config"]
    require(config == CONFIG, "The qualified reader configuration changed")
    require(len(control_plan) == 500 and all(r["generation_config"] == CONFIG for r in control_plan),
            "Incomplete or mixed reader calibration configurations")
    prior = {}
    for name in SOURCE_FILES[:4]:
        prior[name] = {initial_signature(c) for c in records(root / name)}
    # Retain exclusion of removed development/feasibility batches by their
    # deterministic generating rules; no historical model calls are performed.
    for name, count, seed in (("historical_feasibility", 200, 20261007),
                              ("historical_development", 20, 20260926),
                              ("integration_smoke", 1, 2026092501)):
        prior[name] = {initial_signature(c) for c in generate_cases(count, seed, 2)}
    return dict(system_prompt=freeze["reader_system_prompt"], config=config,
                qualified_snapshot=normalize_snapshot(summary["reader_model_snapshot"]),
                prior_signatures=prior)


def render(assertion):
    label = ATTRIBUTE_SPECS[assertion["attribute"]]["label"]
    if assertion.get("realization") == "surface":
        return "At present, {}'s {} is {}.".format(
            assertion["subject"], label, assertion["value"])
    return CURRENT_TEMPLATE.format(subject=assertion["subject"], label=label,
                                   value=assertion["value"])


def memory_from_assertions(assertions):
    return [dict(memory_id=mid, text=" ".join(render(a) for a in assertions
                                             if a["memory_id"] == mid))
            for mid in ("m00", "m01")]


def current_values(assertions, subject, attributes):
    return {attr: sorted({a["value"] for a in assertions
                         if (a["subject"], a["attribute"]) == (subject, attr)})
            for attr in attributes}


def state_properties(assertions, reference, subject, attributes, expected):
    values = current_values(assertions, subject, attributes)
    complete = all(values[attr] for attr in attributes)
    unique = False if any(len(v) > 1 for v in values.values()) else (True if complete else None)
    tuples = lambda rows: Counter((a["subject"], a["attribute"], a["value"]) for a in rows)
    actual, gold = tuples(assertions), tuples(reference)
    expected_membership = {mid: sorted(a["fact_id"] for a in reference
                                     if a["memory_id"] == mid) for mid in ("m00", "m01")}
    slots = {(a["subject"], a["attribute"]): a["fact_id"] for a in reference}
    actual_membership = {mid: sorted(slots.get((a["subject"], a["attribute"]),
                                             "__unsupported_slot__:" + a["subject"] + "|" + a["attribute"])
                                    for a in assertions if a["memory_id"] == mid)
                         for mid in ("m00", "m01")}
    competitions = []
    for s, attr in sorted({(a["subject"], a["attribute"]) for a in assertions}):
        vals = current_values(assertions, s, [attr])[attr]
        if len(vals) > 1:
            competitions.append(dict(subject=s, attribute=attr, values=vals))
    return dict(queried_current_values=values, queried_profile_complete=bool(complete),
                queried_profile_unique_and_noncontradictory=unique,
                oracle_profile_supported=all(expected[a] in values[a] for a in attributes),
                global_reference_exact=actual == gold,
                all_reference_assertions_present=all(actual[k] >= v for k, v in gold.items()),
                assertion_count=len(assertions),
                per_unit_counts={mid: sum(a["memory_id"] == mid for a in assertions)
                                 for mid in ("m00", "m01")},
                constructed_membership_preserved=actual_membership == expected_membership,
                competing_current_slots=competitions)


def answer_support(parsed: Optional[Mapping[str, str]], values):
    """Support is distinct from singleton support, abstention, and exact match."""
    if parsed is None:
        return dict(answer_supported=None, attribute_support={})
    fields = {a: None if parsed[a] == "UNKNOWN" else parsed[a] in current
              for a, current in values.items()}
    support = False if False in fields.values() else (None if None in fields.values() else True)
    return dict(answer_supported=support, attribute_support=fields)


def make_case(base, index):
    update = deepcopy(base["updates"][0])
    order = base["shared_fact_order"]
    reference_by_id = {f["fact_id"]: f for f in update["expected_current_facts"]}
    reference = [dict(fact_id=fid, subject=reference_by_id[fid]["subject"],
                      attribute=reference_by_id[fid]["attribute"],
                      value=reference_by_id[fid]["value"], memory_id="m{:02d}".format(pos // 3),
                      realization="canonical") for pos, fid in enumerate(order)]
    subject = update["new_fact"]["subject"]
    attributes = [f["attribute"] for f in base["facts"] if f["subject"] == subject]
    expected = {a["attribute"]: a["value"] for a in reference if a["subject"] == subject}
    other = next(f["subject"] for f in base["facts"] if f["subject"] != subject)
    # Fixed, outcome-independent cycling among the other subject's three slots.
    damage_fact = [a for a in reference if a["subject"] == other][index % 3]
    vocab = ATTRIBUTE_SPECS[damage_fact["attribute"]]["values"]
    damage_value = vocab[(vocab.index(damage_fact["value"]) + 1) % len(vocab)]
    return dict(case_id="exp5_case_{:04d}".format(index), case_index=index,
                master_seed=CASE_SEED, case_seed=base["case_seed"], facts=base["facts"],
                shared_fact_order=order, initial_signature=initial_signature(base), update=update,
                reference_assertions=reference, subject=subject, attributes=attributes,
                expected_values=expected, other_subject=other,
                damaged_fact_id=damage_fact["fact_id"], damaged_value=damage_value)


def make_variant(case, variant, system_prompt, config):
    require(variant in VARIANTS, "Unknown variant")
    assertions = deepcopy(case["reference_assertions"])
    target_id = case["update"]["target_fact_id"]
    pos = next(i for i, a in enumerate(assertions) if a["fact_id"] == target_id)
    if variant == "surface":
        assertions[pos]["realization"] = "surface"
    elif variant == "contradiction":
        old = deepcopy(assertions[pos])
        old["value"] = case["update"]["old_fact"]["value"]
        # New assertion precedes the added old assertion, at the same unit.
        assertions.insert(pos + 1, old)
    elif variant == "deletion":
        assertions.pop(pos)
    elif variant == "misbinding":
        assertions[pos]["subject"] = case["other_subject"]
    elif variant == "other_damage":
        next(a for a in assertions if a["fact_id"] == case["damaged_fact_id"])["value"] = case["damaged_value"]
    memory = memory_from_assertions(assertions)
    membership = {mid: sorted(a["fact_id"] for a in case["reference_assertions"]
                              if a["memory_id"] == mid) for mid in ("m00", "m01")}
    strict = evaluate_final_memory(memory, case["update"], True, True,
                                   expected_unit_fact_counts={"m00": 3, "m01": 3},
                                   expected_unit_membership=membership)
    attrs = case["attributes"]
    payload = dict(system_prompt=system_prompt,
                   user_prompt=build_reader_user_prompt(memory=memory, subject=case["subject"],
                       attributes=attrs, labels={a: ATTRIBUTE_SPECS[a]["label"] for a in attrs}),
                   generation_config=dict(config), response_schema=response_schema_for_attributes(attrs))
    return dict(row_id=case["case_id"] + "/" + variant, case_id=case["case_id"], variant=variant,
                subject=case["subject"], attributes=attrs, expected_values=case["expected_values"],
                memory=memory, constructed_assertions=assertions,
                state_properties=state_properties(assertions, case["reference_assertions"],
                                                 case["subject"], attrs, case["expected_values"]),
                strict_evaluation=strict, input_sha256=sha256_json(payload), payload=payload)


def build_design(context):
    cases = [make_case(c, i) for i, c in enumerate(generate_cases(N_CASES, CASE_SEED, 1))]
    signatures = {c["initial_signature"] for c in cases}
    require(len(signatures) == N_CASES, "Duplicate new initial fact sets")
    overlap = {name: len(signatures & prior) for name, prior in context["prior_signatures"].items()}
    require(not any(overlap.values()), "New cases overlap a prior batch: " + str(overlap))
    plan = [make_variant(c, v, context["system_prompt"], context["config"])
            for c in cases for v in VARIANTS]
    random.Random(SCHEDULE_SEED).shuffle(plan)
    for i, row in enumerate(plan):
        row["execution_index"] = i
    require(len(plan) == N_CASES * len(VARIANTS), "Incomplete paired design")
    return cases, plan, dict(prior_case_counts={k: len(v) for k, v in context["prior_signatures"].items()},
                             overlap_counts=overlap, duplicate_new_initial_fact_sets=0)


def verify(out=OUT, check_code=True):
    out = Path(out)
    require((out / "freeze_manifest.json").is_file(),
            "Run scripts/exp5_validation/build_validation.py first")
    freeze = load(out / "freeze_manifest.json")
    require(freeze["protocol_version"] == VERSION, "Unknown frozen protocol")
    for name, expected in freeze["source_sha256"].items():
        require(digest(paths.OUTPUT_ROOT / name) == expected, "Prior evidence changed: " + name)
    for name, expected in freeze["artifact_sha256"].items():
        require(digest(out / name) == expected, "Frozen artifact changed: " + name)
    if check_code:
        for name, expected in freeze["code_sha256"].items():
            require(digest(paths.PROJECT_ROOT / name) == expected, "Frozen code changed: " + name)
    cases, plan, overlap = build_design(source_context(paths.OUTPUT_ROOT))
    require(cases == records(out / "cases.jsonl"), "Cases differ from deterministic construction")
    require(plan == records(out / "reader_plan.jsonl"), "Plan differs from deterministic construction")
    require(overlap == freeze["overlap_audit"], "Overlap audit differs")
    require(freeze["num_cases"] == len(cases) == N_CASES and freeze["logical_rows"] == len(plan)
            and freeze["variants"] == list(VARIANTS), "Frozen design dimensions differ")
    require(freeze["unique_inputs"] == len({r["input_sha256"] for r in plan}), "Frozen cache count differs")
    require((freeze["case_seed"], freeze["schedule_seed"], freeze["bootstrap_seed"],
             freeze["bootstrap_replicates"]) ==
            (CASE_SEED, SCHEDULE_SEED, BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES), "Frozen analysis settings differ")
    require(freeze["generation_config"] == CONFIG, "Frozen reader configuration differs")
    require(freeze["num_cases"] == len(cases) == N_CASES and freeze["logical_rows"] == len(plan)
            and freeze["variants"] == list(VARIANTS), "Frozen design dimensions differ")
    require(freeze["unique_inputs"] == len({r["input_sha256"] for r in plan}), "Frozen cache count differs")
    require((freeze["case_seed"], freeze["schedule_seed"], freeze["bootstrap_seed"],
             freeze["bootstrap_replicates"]) ==
            (CASE_SEED, SCHEDULE_SEED, BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES), "Frozen analysis settings differ")
    require(freeze["generation_config"] == CONFIG, "Frozen reader configuration differs")
    return freeze, plan


def validate_raw(raw, plan, snapshot, complete=False):
    payloads = {r["input_sha256"]: r["payload"] for r in plan}
    by_key = {}
    for row in raw:
        key = row["input_sha256"]
        require(key in payloads, "Response belongs to an unknown reader input")
        require(key not in by_key, "Duplicate cached response: " + key)
        require(row["payload"] == payloads[key] and sha256_json(row["payload"]) == key,
                "Cached response input mismatch")
        require(any(r["row_id"] == row["first_row_id"] and r["input_sha256"] == key for r in plan),
                "Response provenance row mismatch")
        require(any(r["row_id"] == row["first_row_id"] and r["input_sha256"] == key for r in plan),
                "Response provenance row mismatch")
        require(row["model_snapshot"] == snapshot and row["model"] == CONFIG["model"],
                "Cached response snapshot mismatch")
        require(normalize_snapshot(row["metadata"]) == snapshot, "Response metadata snapshot mismatch")
        require(isinstance(row["response_text"], str), "Missing raw response text")
        require(row["response_sha256"] == sha256_json(row["response_text"]), "Response text hash mismatch")
        by_key[key] = row
    if complete:
        require(set(by_key) == set(payloads), "Reader calls are incomplete; continue run_validation.py")
    return by_key


def score(plan, by_key):
    results = []
    for row in plan:
        raw = by_key[row["input_sha256"]]
        parsed, error = None, None
        try:
            parsed = parse_reader_response(raw["response_text"], row["payload"]["response_schema"])
        except (ValueError, TypeError) as exc:
            error = str(exc)
        support = answer_support(parsed, row["state_properties"]["queried_current_values"])
        results.append(dict(row_id=row["row_id"], case_id=row["case_id"], variant=row["variant"],
                            input_sha256=row["input_sha256"], parsed_response=parsed, parse_error=error,
                            valid_json=parsed is not None,
                            profile_exact_match=exact_profile_match(parsed, row["expected_values"]),
                            any_unknown=parsed is not None and "UNKNOWN" in parsed.values(),
                            all_unknown=parsed is not None and all(v == "UNKNOWN" for v in parsed.values()),
                            **support, state_properties=row["state_properties"],
                            strict_evaluation=row["strict_evaluation"]))
    return results
