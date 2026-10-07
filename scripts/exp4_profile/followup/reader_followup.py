"""Build/freeze, run, or analyze the bounded explanatory follow-up.

Usage: python scripts/exp4_profile/followup/reader_followup.py freeze|run|analyze
No generation occurs during freeze. Original experimental files are read-only.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "atomic_memory").is_dir())
sys.path.insert(0, str(ROOT))
from atomic_memory import paths
from atomic_memory.backend import GenerationConfig, OllamaBackend
from atomic_memory.profile_state_audit import audit, FIELDS, reverse_presentation, fact_multiset
from atomic_memory.reader import (
    canonical_json, build_reader_user_prompt, parse_reader_response, exact_profile_match,
)
from scripts.exp4_profile.run_profile_reader import (
    write_json_atomic, write_jsonl_atomic, read_jsonl,
)

SOURCE = paths.PROFILE
OUT = paths.FOLLOWUP
FIRST = 'llama3.1:8b-instruct-q4_K_M'
SECOND = 'qwen2.5:7b-instruct-q4_K_M'
MODELS = [FIRST, SECOND, 'gemma3:12b-it-q4_K_M']
CONDITIONS = ['L1_select', 'L1_patch', 'L3_select', 'L3_patch']
URL = paths.OLLAMA_URL
SOURCES = ['formal_cases.jsonl', 'freeze_manifest.json', 'updater_results.jsonl',
           'actual_reader_plan.jsonl', 'actual_reader_results.jsonl',
           'reader_control_plan.jsonl', 'reader_control_results.jsonl', 'formal_analysis.json']
FROZEN_CODE = ['atomic_memory/profile_state_audit.py',
               'scripts/exp4_profile/followup/reader_followup.py', 'atomic_memory/backend.py',
               'atomic_memory/core.py', 'atomic_memory/reader.py',
               'scripts/exp4_profile/run_profile_reader.py']


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_key(payload):
    return hashlib.sha256(canonical_json(payload).encode('utf-8')).hexdigest()


def input_payload(row, reader, system, memory=None, rebuild=False):
    config = dict(row['generation_config'], model=reader)
    user = row['user_prompt']
    if rebuild:
        user = build_reader_user_prompt(memory=memory, subject=row['subject'],
                                       attributes=row['required_attributes'], labels=row['memory_labels'])
    return dict(system_prompt=system, user_prompt=user, generation_config=config,
                response_schema=row['response_schema'])


def baseline_cache():
    cache = {}
    for filename in ['actual_reader_results.jsonl', 'reader_control_results.jsonl']:
        for r in read_jsonl(SOURCE / filename):
            key = r['reader_input_sha256']
            item = dict(input_sha256=key, reader_model=FIRST,
                        response_text=r['reader_raw_response'], metadata=r['reader_response_metadata'],
                        model_snapshot=r['reader_model_snapshot'], source='frozen/' + filename,
                        usage=r['reader_usage'], response_id=r['reader_response_id'])
            if key in cache and cache[key]['response_text'] != item['response_text']:
                raise RuntimeError('Frozen identical inputs have different responses; reuse is ambiguous.')
            cache[key] = item
    return cache


def freeze():
    paths.require_output_stage(paths.FOLLOWUP, Path(__file__))
    if (OUT / 'freeze_manifest.json').exists():
        verify()
        print('Existing frozen protocol verified; no overwrite.')
        return
    if (OUT / 'responses.jsonl').exists():
        raise RuntimeError('Cannot freeze after generation.')
    updaters = {r['request_id']: r for r in read_jsonl(SOURCE / 'updater_results.jsonl')}
    actual = read_jsonl(SOURCE / 'actual_reader_plan.jsonl')
    original = {r['source_updater_request_id']: r for r in read_jsonl(SOURCE / 'actual_reader_results.jsonl')}
    controls = read_jsonl(SOURCE / 'reader_control_plan.jsonl')
    oldfreeze = load(SOURCE / 'freeze_manifest.json')
    assert len(updaters) == len(actual) == len(original) == 1200
    assert len(controls) == 500
    system = oldfreeze['reader_system_prompt']
    backend = OllamaBackend(URL)
    snapshots = {m: backend.describe(m) for m in [FIRST, SECOND]}
    baseline = baseline_cache()
    for r in baseline.values():
        if r['model_snapshot']['model_digest'] != snapshots[FIRST]['model_digest']:
            raise RuntimeError('Primary-reader model digest changed.')
    # Verify original prompt hashes before copying any input.
    for row in actual + controls:
        assert input_key(input_payload(row, FIRST, system)) == row['reader_input_sha256']
    rows, audit_rows = [], []

    def add(row, reader, stage, memory=None, rebuild=False, **extra):
        payload = input_payload(row, reader, system, memory, rebuild)
        item = dict(plan_id='followup_{:04d}'.format(len(rows)), stage=stage,
                    reader_model=reader, case_id=row['case_id'], task_id=row['task_id'],
                    update_id=row['update_id'], subject=row['subject'],
                    required_attributes=row['required_attributes'],
                    expected_values=row.get('oracle_expected_values', row.get('expected_values')),
                    input_sha256=input_key(payload), payload=payload, **extra)
        rows.append(item)

    for c in controls:
        add(c, SECOND, 'control', arm=c['arm'], grouping=c['grouping'], control_id=c['control_id'])
    contradiction_rows = []
    for p in actual:
        uid = p['source_updater_request_id']
        u, r = updaters[uid], original[uid]
        assert p['reader_memory'] == u['executed_public_memory']
        common = dict(source_updater_request_id=uid, updater_model=p['updater_model'],
                      condition=p['condition'], strict_state_success=r['strict_state_success'],
                      primary_profile_pass=r['functional_task_pass'])
        add(p, SECOND, 'actual', **common)
        strict_audit = audit(p['reader_memory'], p['subject'], p['required_attributes'], r['parsed_response'])
        relaxed_audit = audit(p['reader_memory'], p['subject'], p['required_attributes'], r['parsed_response'], True)
        values = relaxed_audit['current_values']
        oracle_agrees = (None if values is None else
                         all(values[a] == [p['oracle_expected_values'][a]] for a in p['required_attributes']))
        audit_rows.append(dict(common, case_id=p['case_id'], update_id=p['update_id'],
                               task_id=p['task_id'], canonical_audit=strict_audit,
                               sensitivity_audit=relaxed_audit,
                               singleton_values_equal_oracle=oracle_agrees))
        e = u['evaluation']
        if e['target_new_current'] is True and e['target_old_not_current'] is False:
            contradiction_rows.append(p)
    assert len(contradiction_rows) == 30
    assert sum(original[p['source_updater_request_id']]['functional_task_pass'] for p in contradiction_rows) == 14
    c_lookup = {(c['task_id'], c['arm'], c['grouping']): c for c in controls}
    for p in contradiction_rows:
        uid = p['source_updater_request_id']
        for state in ['actual', 'oracle', 'stale']:
            base = p if state == 'actual' else c_lookup[p['task_id'], state, p['grouping']]
            mem = base.get('reader_memory', base.get('memory'))
            swapped = reverse_presentation(mem)
            assert fact_multiset(swapped) == fact_multiset(mem)
            assert reverse_presentation(swapped) == mem
            for order, view in [('original', mem), ('reversed', swapped)]:
                for reader in [FIRST, SECOND]:
                    add(base, reader, 'order_probe', memory=view, rebuild=True,
                        source_updater_request_id=uid, updater_model=p['updater_model'],
                        condition=p['condition'], probe_state=state, presentation=order,
                        original_primary_profile_pass=original[uid]['functional_task_pass'])
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(OUT / 'reader_plan.jsonl', rows)
    write_jsonl_atomic(OUT / 'query_state_audit.jsonl', audit_rows)
    protocol = '''# Bounded profile-evaluation follow-up

This is an exploratory follow-up motivated by the already observed primary
results. The protocol is frozen before new reader generations. It is not a
prospective preregistration of the original study. Original Y/Z scores and
frozen F/G contrasts remain unchanged. No result-dependent model, prompt,
sample, parsing, ordering, or scoring changes are permitted.

## Query-scoped state audit
All 1,200 executed states are audited, including failures and successes.
The three separate fields report completeness of current values, uniqueness
and noncontradiction, and explicit current-value support for the reader's
answer by the queried subject. They are explanatory fields, not a primary
endpoint. Repeated identical facts are redundant, not contradictory.
Completeness does not imply that the stored value is the updated value.
Support of an answer does not imply a uniquely supported answer.
UNKNOWN is an abstention, not an asserted fact: its support is undetermined.
If any concrete answer value lacks support, whole-answer support is false;
otherwise any abstention or invalid response leaves it undetermined.
If a profile is incomplete but has no competing values, uniqueness is
undetermined. Any competing current values make uniqueness false.

Canonical parsing is applied first. Without a complete parse, all three
canonical-audit fields are null. A separate, explicitly post-hoc restricted
grammar sensitivity analysis accepts present assertions with optional
`current`, optional `now` after `is`, and an optional final period. It uses
only the benchmark's fixed subject/attribute/value lexicon and full-string
matches. A present-tense assertion without `current` is interpreted as
current under this explicit convention. Negations, extra clauses, and all
other unrecognized text leave the entire state unresolved. Rules were
chosen after viewing the 27 grammar deviations; their recovered labels are
not independently validated semantics. There are no human or LLM labels.
Canonical and relaxed results are both retained. A separate diagnostic
compares singleton stored values to the existing oracle; it is not a new
reported success endpoint. All original updater and reader files are hashed.

## Reader robustness
The second reader is the installed Qwen 2.5 7B Q4_K_M model. The original
Llama 3.1 8B reader is retained for paired order probes. Prompts, schemas,
temperature 0, seed 42, context 8192, and output budget 64 match the frozen
primary reader. Model digests and backend version are recorded.
The second reader receives all 1,200 logical rows and all 500 existing clean
controls (oracle/stale in g1/g3 and no-memory). Qualification is at least
95/100 correct in each arm and 500/500 schema-valid responses. All planned
rows are retained even if qualification fails; failure limits interpretation
and does not trigger a replacement model or tuned prompt.

All 30 canonically parsed states retaining both old and new target values
are included, irrespective of original success (14 passes, 16 failures).
Presentation reversal reverses the memory-unit list and the sentence order
within each unit; it preserves all memory IDs, sentence text, memberships,
fact counts and content. It jointly probes presentation-order sensitivity,
not the separate causal contribution of unit versus sentence ordering.
Each identity has matched oracle and stale clean-state controls from the
same task and grouping. Both original and reversed forms go to both readers.
Duplicate task controls remain paired but are not independent evidence.
An order-invariant exact match does not establish uniquely grounded support.

The cache key includes the complete system/user prompts, response schema,
and generation configuration, including model. All keys are bound by this
manifest to a verified model digest. Exact frozen Llama responses may be
reused only after checking source hashes and matching model digests. Every
logical row is scored independently against its own oracle. New calls are
saved before expansion to logical rows; resumption reuses completed calls.
No automatic retry after a returned answer, even if invalid or unexpected.
Transport errors stop the runner; resumption repeats only unfinished calls.

## Frozen analyses and stopping condition
Report all three audit fields by original Y/Z, canonical coverage and the
separate relaxed sensitivity results. Report all 12 second-reader cell
rates, both readers' g1-minus-g3 contrasts, paired reader changes and
reader-by-grouping differences. Use 20,000 whole-case bootstrap resamples
of the same 50 cases, fixed NumPy seed 20260928, retaining both targets and
all configurations/readers together. Intervals are exploratory percentile
95% intervals; no equivalence or multiplicity-controlled claims are made.
Order probes report transition counts and changed response counts for all
30 identities, with original-pass/fail strata and distinct case/input counts;
they are descriptive and not 30 independent samples. Clean controls are
reported alongside. All planned analyses are retained. After these two
additions, synthetic evidence expansion stops and the manuscript is updated.
Natural-language validation is deferred: the available LongMemEval oracle
file is conversational QA and still requires curated update-state/schema
alignment. This follow-up does not create or validate that new pipeline.
'''
    (OUT / 'protocol.md').write_text(protocol, encoding='utf-8')
    manifest = dict(status='FROZEN_BEFORE_NEW_GENERATIONS', frozen_utc=datetime.now(timezone.utc).isoformat(),
                    backend_url=URL, model_snapshots=snapshots,
                    source_sha256={f: digest(SOURCE / f) for f in SOURCES},
                    code_sha256={f: digest(ROOT / f) for f in FROZEN_CODE},
                    artifact_sha256={f: digest(OUT / f) for f in
                                     ['reader_plan.jsonl', 'query_state_audit.jsonl', 'protocol.md']},
                    logical_rows_by_stage=dict(Counter(r['stage'] for r in rows)),
                    unique_new_calls=len({r['input_sha256'] for r in rows} - set(baseline)),
                    second_reader_actual_unique_inputs=len({r['input_sha256'] for r in rows if r['stage']=='actual'}),
                    contradiction_identities=[p['source_updater_request_id'] for p in contradiction_rows],
                    bootstrap_replicates=20000, bootstrap_seed=20260928)
    write_json_atomic(OUT / 'freeze_manifest.json', manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['source_sha256','artifact_sha256','code_sha256','contradiction_identities']},indent=2))


def verify(check_code=True):
    """Check frozen data; generation additionally requires the exact source version."""
    manifest = load(OUT / 'freeze_manifest.json')
    fields = [('source_sha256', SOURCE), ('artifact_sha256', OUT)]
    if check_code:
        fields.append(('code_sha256', ROOT))
    for field, base in fields:
        for name, expected in manifest[field].items():
            path = base / name
            if not path.is_file() or digest(path) != expected:
                if field == 'code_sha256':
                    raise RuntimeError('Frozen code version differs: ' + str(path)
                                       + '. Use the matching source version or a new study directory.')
                raise RuntimeError('Frozen file changed: ' + str(path))
    return manifest


def run():
    paths.require_output_stage(paths.FOLLOWUP, Path(__file__))
    manifest = verify()
    plans = read_jsonl(OUT/'reader_plan.jsonl')
    cache = baseline_cache()
    generated = read_jsonl(OUT/'responses.jsonl')
    for r in generated:
        if r['input_sha256'] in cache:
            raise RuntimeError('Duplicate response cache entry.')
        cache[r['input_sha256']] = r
    backend = OllamaBackend(URL, timeout_seconds=180)
    for m, snapshot in manifest['model_snapshots'].items():
        if backend.describe(m) != snapshot:
            raise RuntimeError('Backend/model snapshot differs from freeze.')
    pending = {}
    for p in plans:
        if p['input_sha256'] not in cache:
            pending.setdefault(p['input_sha256'], p)
    # Finish one model before loading the other; this order is fixed.
    pending = sorted(pending.values(), key=lambda p:(p['reader_model'] != SECOND, p['plan_id']))
    print('Resuming with {} cached new calls; {} pending.'.format(len(generated),len(pending)),flush=True)
    with (OUT/'responses.jsonl').open('a',encoding='utf-8',newline='\n') as handle:
        for i,p in enumerate(pending,1):
            start = time.monotonic()
            payload = p['payload']
            response = backend.generate(request_id=p['plan_id'],system_prompt=payload['system_prompt'],
                        user_prompt=payload['user_prompt'],config=GenerationConfig(**payload['generation_config']),
                        response_schema=payload['response_schema'])
            result = dict(input_sha256=p['input_sha256'], reader_model=p['reader_model'],
                          response_text=response.text,metadata=dict(response.metadata),
                          model_snapshot=manifest['model_snapshots'][p['reader_model']],
                          usage=dict(response.usage),response_id=response.response_id,source='new_generation',
                          started_from_plan_id=p['plan_id'],latency_seconds=time.monotonic()-start)
            handle.write(canonical_json(result)+'\n'); handle.flush()
            cache[p['input_sha256']] = result
            if i % 20 == 0 or i==len(pending):
                print('Completed {}/{} pending calls; reader {}'.format(i,len(pending),p['reader_model']),flush=True)
    results=[]
    for p in plans:
        response=cache[p['input_sha256']]
        error=None
        try:
            parsed=parse_reader_response(response['response_text'],p['payload']['response_schema'])
        except (ValueError,TypeError) as exc:
            parsed=None;error=str(exc)
        r={k:v for k,v in p.items() if k!='payload'}
        r.update(parsed_response=parsed,response_parse_error=error,
                 profile_pass=exact_profile_match(parsed,p['expected_values']),
                 response_source=response['source'])
        results.append(r)
    write_jsonl_atomic(OUT/'reader_results.jsonl',results)
    summary=dict(status='COMPLETE',logical_rows=len(results),new_generation_calls=len(read_jsonl(OUT/'responses.jsonl')),
                 rows_by_stage=dict(Counter(r['stage'] for r in results)),
                 frozen_source_hashes_verified=True,completed_utc=datetime.now(timezone.utc).isoformat())
    write_json_atomic(OUT/'run_summary.json',summary)
    print(json.dumps(summary,indent=2))


def analyze():
    import numpy as np
    manifest=verify(check_code=False)
    results=read_jsonl(OUT/'reader_results.jsonl')
    assert len(results)==2060 and len({r['plan_id'] for r in results})==2060
    primary=read_jsonl(SOURCE/'actual_reader_results.jsonl')
    audits=read_jsonl(OUT/'query_state_audit.jsonl')
    plans={r['source_updater_request_id']:r for r in read_jsonl(SOURCE/'actual_reader_plan.jsonl')}
    cases=sorted({r['case_id'] for r in primary})
    assert len(cases)==50
    sample=np.random.default_rng(manifest['bootstrap_seed']).integers(0,50,size=(20000,50))
    def interval(values):
        values=np.asarray(values,dtype=float)
        assert values.shape==(50,)
        lo,hi=np.percentile(values[sample].mean(axis=1)*100,[2.5,97.5])
        return dict(estimate_pp=float(values.mean()*100),ci_low_pp=float(lo),ci_high_pp=float(hi))
    by_reader={FIRST:primary,SECOND:[r for r in results if r['stage']=='actual']}
    arrays={};cell_rows=[]
    for reader,rows in by_reader.items():
        for updater in MODELS:
            for cond in CONDITIONS:
                group=[r for r in rows if r['updater_model']==updater and r['condition']==cond]
                assert len(group)==100
                vec=[]
                for case in cases:
                    pair=[r for r in group if r['case_id']==case];assert len(pair)==2
                    vec.append(sum(r.get('profile_pass',r.get('functional_task_pass')) for r in pair)/2)
                arrays[reader,updater,cond]=np.array(vec)
                cell_rows.append(dict(reader_model=reader,updater_model=updater,condition=cond,
                                     successes=round(sum(vec)*2),n=100,**interval(vec)))
    contrasts=[];reader_changes=[]
    for updater in MODELS:
        for interface in ['patch','select']:
            vectors={reader:arrays[reader,updater,'L1_'+interface]-arrays[reader,updater,'L3_'+interface]
                     for reader in [FIRST,SECOND]}
            for reader in [FIRST,SECOND]:
                contrasts.append(dict(reader_model=reader,updater_model=updater,interface=interface,
                                      **interval(vectors[reader])))
            reader_changes.append(dict(updater_model=updater,scope='grouping_contrast',interface=interface,
                                       **interval(vectors[SECOND]-vectors[FIRST])))
        for cond in CONDITIONS:
            reader_changes.append(dict(updater_model=updater,scope='cell',condition=cond,
                                       **interval(arrays[SECOND,updater,cond]-arrays[FIRST,updater,cond])))
    controls=[]
    for arm,grouping in [('oracle','g1'),('oracle','g3'),('stale','g1'),('stale','g3'),('no_memory',None)]:
        group=[r for r in results if r['stage']=='control' and r['arm']==arm and r['grouping']==grouping]
        assert len(group)==100,(arm,grouping,len(group))
        controls.append(dict(arm=arm,grouping=grouping,n=len(group),passes=sum(r['profile_pass'] for r in group),
                             valid=sum(r['parsed_response'] is not None for r in group)))
    audit_summary=[]
    for mode in ['canonical_audit','sensitivity_audit']:
        for y,z in [(True,True),(True,False),(False,True),(False,False)]:
            group=[r for r in audits if r['strict_state_success']==y and r['primary_profile_pass']==z]
            audit_summary.append(dict(mode=mode,Y=y,Z=z,n=len(group),
                fields={f:dict(Counter('undetermined' if r[mode][f] is None else str(r[mode][f]).lower()
                                      for r in group)) for f in FIELDS},
                parser_counts=dict(Counter(r[mode]['parser'] for r in group)),
                singleton_values_equal_oracle=sum(r['singleton_values_equal_oracle'] is True for r in group)))
    second_audits=[]
    for r in by_reader[SECOND]:
        p=plans[r['source_updater_request_id']]
        second_audits.append(dict(source_updater_request_id=r['source_updater_request_id'],
            profile_pass=r['profile_pass'],audit=audit(p['reader_memory'],p['subject'],p['required_attributes'],r['parsed_response'],True)))
    write_jsonl_atomic(OUT/'second_reader_answer_audit.jsonl',second_audits)
    probes=[]
    order_rows=[r for r in results if r['stage']=='order_probe']
    for reader in [FIRST,SECOND]:
        for state in ['actual','oracle','stale']:
            pairs=defaultdict(dict)
            for r in order_rows:
                if r['reader_model']==reader and r['probe_state']==state:
                    pairs[r['source_updater_request_id']][r['presentation']]=r
            assert len(pairs)==30 and all(len(p)==2 for p in pairs.values())
            for stratum in ['all','primary_pass','primary_fail']:
                group=[p for p in pairs.values() if stratum=='all' or
                       p['original']['original_primary_profile_pass']==(stratum=='primary_pass')]
                probes.append(dict(reader_model=reader,state=state,stratum=stratum,n=len(group),
                    cases=len({p['original']['case_id'] for p in group}),
                    unique_original_inputs=len({p['original']['input_sha256'] for p in group}),
                    original_pass=sum(p['original']['profile_pass'] for p in group),
                    reversed_pass=sum(p['reversed']['profile_pass'] for p in group),
                    changed_answers=sum(p['original']['parsed_response']!=p['reversed']['parsed_response'] for p in group),
                    transitions=dict(Counter('{}->{}'.format(int(p['original']['profile_pass']),int(p['reversed']['profile_pass'])) for p in group))))
    summary=dict(exploratory=True,cluster_unit='case',n_cases=50,bootstrap_replicates=20000,bootstrap_seed=manifest['bootstrap_seed'],
                 source_sha256=manifest['source_sha256'],freeze_manifest_sha256=digest(OUT/'freeze_manifest.json'),
                 reader_results_sha256=digest(OUT/'reader_results.jsonl'),audit_summary=audit_summary,
                 controls=controls,second_reader_qualified=all(c['passes']>=95 and c['valid']==100 for c in controls),
                 cell_summary=cell_rows,grouping_contrasts=contrasts,paired_reader_changes=reader_changes,order_probes=probes,
                 second_reader_answer_support=dict(Counter(str(r['audit'][FIELDS[2]]) for r in second_audits)),
                 second_reader_exact_match_unsupported=sum(r['profile_pass'] and r['audit'][FIELDS[2]] is False for r in second_audits))
    write_json_atomic(OUT/'analysis.json',summary)
    print(json.dumps({k:summary[k] for k in ['second_reader_qualified','controls','audit_summary','second_reader_exact_match_unsupported']},indent=2))
    print('Grouping contrasts:',json.dumps(contrasts,indent=2))
    print('Order probes:',json.dumps([r for r in probes if r['stratum']=='all'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','run','analyze'])
    parser.add_argument('--check', action='store_true', help='Recompute analyze outputs in memory without writing')
    args=parser.parse_args()
    if args.check:
        if args.stage != 'analyze':
            parser.error('--check is only valid for analyze')
        from atomic_memory.reanalysis import check_analysis
        check_analysis(analyze, globals())
    else:
        paths.require_output_stage(paths.FOLLOWUP, Path(__file__))
        {'freeze':freeze,'run':run,'analyze':analyze}[args.stage]()
