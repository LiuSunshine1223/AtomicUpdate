# Bounded profile-evaluation follow-up

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
