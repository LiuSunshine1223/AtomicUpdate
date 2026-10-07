"""Explanatory query-scoped audit; never changes the frozen state endpoint.

The restricted grammar relaxation is explicitly post hoc. It is a sensitivity
analysis of this controlled language, not a general semantic or human judge.
"""
import re
from dataclasses import asdict
from .core import (
    ATTRIBUTE_SPECS, SUBJECT_POOL, parse_fact_sentence, parse_memory,
)

FIELDS = ('queried_profile_complete',
          'queried_profile_unique_and_noncontradictory',
          'answer_supported_by_queried_subject_memory')


def sentences(text):
    return [s.strip() for s in re.split(r'(?<=\.)\s+', text.strip()) if s.strip()]


def relaxed_facts(memory):
    strict = parse_memory(memory)
    if strict['contract_valid']:
        return strict['parsed_facts'], [], []
    facts, recovered, unresolved = [], [], []
    ids = [u.get('memory_id') for u in memory]
    if len(ids) != len(set(ids)) or not all(ids):
        return [], [], ['invalid_memory_ids']
    for unit in memory:
        if not isinstance(unit.get('text'), str):
            unresolved.append('non_string_text')
            continue
        for sentence in sentences(unit['text']):
            try:
                fact = asdict(parse_fact_sentence(sentence))
            except ValueError:
                fact = None
                # Full-string matching and a closed lexicon reject qualifications,
                # negation, extra clauses and unsupported value normalization.
                for attr, spec in ATTRIBUTE_SPECS.items():
                    pattern = (r"(?P<subject>[A-Za-z][A-Za-z0-9_-]*)'s "
                               r'(?:current )?' + re.escape(spec['label']) +
                               r' is (?:now )?(?P<value>' +
                               '|'.join(re.escape(v) for v in spec['values']) + r')\.?')
                    match = re.fullmatch(pattern, sentence)
                    if match and match['subject'] in SUBJECT_POOL:
                        fact = dict(subject=match['subject'], attribute=attr,
                                    value=match['value'], status='current',
                                    source_sentence=sentence)
                        recovered.append(dict(memory_id=unit['memory_id'], sentence=sentence,
                                              rule='present_assertion_optional_current_now_period'))
                        break
                if fact is None:
                    unresolved.append(dict(memory_id=unit['memory_id'], sentence=sentence))
                    continue
            facts.append(dict(memory_id=unit['memory_id'], **fact))
    return facts, recovered, unresolved


def audit(memory, subject, attributes, answer, allow_relaxation=False):
    strict = parse_memory(memory)
    if allow_relaxation:
        facts, recovered, errors = relaxed_facts(memory)
    else:
        facts, recovered, errors = strict['parsed_facts'], [], strict['errors']
    base = dict(parser='canonical' if strict['contract_valid'] else
                ('restricted_relaxation' if allow_relaxation and not errors else 'unresolved'),
                recovered_sentences=recovered, unresolved=errors)
    if errors:
        return dict(base, **{f: None for f in FIELDS}, current_values=None,
                    attribute_support=None)
    values = {a: sorted({f['value'] for f in facts if f['subject'] == subject
                        and f['attribute'] == a and f['status'] == 'current'})
              for a in attributes}
    complete = all(values.values())
    unique = False if any(len(v) > 1 for v in values.values()) else (True if complete else None)
    support = {a: (None if answer is None or answer.get(a) == 'UNKNOWN'
                   else answer.get(a) in values[a]) for a in attributes}
    supported = (False if any(v is False for v in support.values()) else
                 True if all(v is True for v in support.values()) else None)
    return dict(base, queried_profile_complete=complete,
                queried_profile_unique_and_noncontradictory=unique,
                answer_supported_by_queried_subject_memory=supported,
                current_values=values, attribute_support=support)


def reverse_presentation(memory):
    """Reverse units and sentences within units; keep IDs and memberships."""
    if not parse_memory(memory)['contract_valid']:
        raise ValueError('Order probes require canonical, fully parsed input.')
    return [dict(memory_id=u['memory_id'], text=' '.join(reversed(sentences(u['text']))))
            for u in reversed(memory)]


def fact_multiset(memory):
    from collections import Counter
    parsed = parse_memory(memory)
    if not parsed['contract_valid']:
        raise ValueError('Unparseable probe memory.')
    return Counter((f['memory_id'], f['subject'], f['attribute'], f['value'], f['status'])
                   for f in parsed['parsed_facts'])
