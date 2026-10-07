# Exploratory adjacent-swap check after the original 300 responses

This check was selected after seeing the original constructed-state results
(including 6/50 exact matches on New->Old contradictions). It was not part of
the initial 300-input plan and is not publicly preregistered. The input list,
source/code hashes, actual model snapshot and analysis rules are locally locked
before the added calls; freeze_manifest.json supplies the timestamp and hashes.

## Fixed design

Use all 50 original exp5 contradiction cases, irrespective of original success.
They are the same new base cases used by the original six-arm constructed study,
not 50 additional independent cases and not the main updater-study case batch.
Each original new/old pair must be adjacent in the same memory unit. Exchange
only these two complete sentences. Preserve unit IDs and unit order, all other
sentences and their positions, assertion multiset, state-property labels, strict
scores, query, oracle, system instructions, JSON schema and generation settings.
The original pair is New->Old; the added pair is Old->New. Both remain contradictory.
Project the original shuffled schedule onto its 50 contradiction rows. No new
case selection, target selection, updater, reader or prompt tuning is allowed.

## Calls and caching

Read the original New->Old responses as the comparison baseline; never regenerate
them. Added inputs have different complete-prompt hashes, and all 50 must be
unique and absent from the original 300-input cache. Make at most one response
per added input under the original qualified Llama 3.1 8B Instruct Q4_K_M model
and runtime (temperature 0, seed 42, context 8192, output 64). A transport failure
stops; resume only unfinished inputs. Keep invalid JSON and unexpected answers;
do not retry based on the output. No old file, endpoint or score is overwritten.

## Analysis committed before added responses

Pair by case, retaining all 50 cases including original failures. Report counts
out of 50 for both orders, parsed-answer changes, 1->0, 0->1, 1->1 and 0->0 exact
match transitions, valid JSON, UNKNOWN and answer support. Changed means a change
in parsed three-field response, as in the existing supplement order-probe table.
Estimate Old->New minus New->Old exact-match rate, with a descriptive 95% paired
percentile interval from 20,000 whole-case resamples, numpy RandomState seed
20261016. No significance test, multiplicity claim or equivalence claim is added.
Zero/full outcomes can produce degenerate empirical bootstrap intervals; present
counts and do not treat such intervals as certainty about population probabilities.

## Interpretation written before added responses

A larger recovery rate under Old->New, with failures becoming passes, shows that
presentation order affects this fixed reader's oracle recovery while the same
contradiction remains. It does not identify the reader's internal mechanism.
A small net rate change with many opposing transitions can still conceal strong
case-level order sensitivity. A small change with few transitions/answer changes
provides only limited sensitivity in this reader and construction; it establishes
neither equivalence nor general order robustness. If recovery falls, interpret
the observed direction without changing the oracle or selecting a favorable order.
This check estimates neither naturally generated contradiction prevalence nor
performance on arbitrary natural language or other readers/models.

## Accounting and stopping

The original 300 evaluations stay intact. Add 50 reader inputs/calls, using the
same 50 case clusters; the constructed study then has 350 logical evaluations
and 350 physical calls, but the order comparison has 50 pairs (100 assessments,
50 of which were already among the 300). The existing profile follow-up remains
2,060 logical rows and 605 new calls. Together these two supplemental batches
have 2,410 logical rows and 955 new calls; their case sets and dependence differ.
The 3,720 updater calls over 90 updater cases are unchanged. After this check,
update the figure, manuscript and supplement once, and stop adding experiments.
