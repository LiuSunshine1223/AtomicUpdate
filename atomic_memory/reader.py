"""Shared deterministic profile-reader interface and scoring helpers.

Prompts and schemas preserve the frozen direct-extraction protocol. These
functions perform no model calls and write no experimental artifacts.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from atomic_memory.core import canonical_json

UNKNOWN_VALUE = "UNKNOWN"


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


def response_schema_for_attributes(
    attributes: Sequence[str],
) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            str(attribute)
            for attribute in attributes
        ],
        "properties": {
            str(attribute): {"type": "string"}
            for attribute in attributes
        },
    }


def build_reader_user_prompt(
    *,
    memory: Optional[Sequence[Mapping[str, Any]]],
    subject: str,
    attributes: Sequence[str],
    labels: Mapping[str, str],
) -> str:
    # This function intentionally mirrors the already-frozen builder exactly.
    if memory is None:
        memory_text = "(no memory state provided)"
    else:
        memory_text = json.dumps(
            public_memory(memory),
            ensure_ascii=False,
            indent=2,
        )

    attribute_spec = [
        {
            "key": str(attribute),
            "memory_label": str(labels[str(attribute)]),
        }
        for attribute in attributes
    ]
    attribute_text = json.dumps(
        attribute_spec,
        ensure_ascii=False,
        indent=2,
    )
    example_shape = {
        str(attribute): UNKNOWN_VALUE
        for attribute in attributes
    }

    return (
        "MEMORY STATE:\n"
        + memory_text
        + "\n\nTARGET PERSON:\n"
        + str(subject)
        + "\n\nREQUIRED ATTRIBUTES:\n"
        + attribute_text
        + "\n\nExtract the target person's CURRENT value for every required "
          "attribute.\n"
          "Use the exact value text supported by the memory.\n"
          "If a value cannot be determined from the memory, use "
          "\"UNKNOWN\" for that attribute.\n"
          "Return exactly one JSON object with this key shape:\n"
        + json.dumps(
            example_shape,
            ensure_ascii=False,
        )
    )


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
    attributes: Sequence[str],
) -> bool:
    if parsed is None:
        return False
    attrs = [str(attribute) for attribute in attributes]
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


def classify_profile(
    *,
    parsed: Optional[Mapping[str, str]],
    oracle: Mapping[str, str],
    stale: Mapping[str, str],
    attributes: Sequence[str],
) -> str:
    if parsed is None:
        return "invalid_json"
    if exact_profile_match(parsed, oracle):
        return "oracle_current"
    if exact_profile_match(parsed, stale):
        return "stale_profile"
    if all_unknown(parsed, attributes):
        return "all_unknown"
    return "other_profile"
