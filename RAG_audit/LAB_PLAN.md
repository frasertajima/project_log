# rag_audit — an audit trail for every RAG answer we already trust

*Started 2026-09-24. Subjects: `RQ` (`ragstash/rag_query.py`), `SP` (`sparsebridge/sb_query.py`),
`CT` (`rstash`, pure Rust). Prompted by
["Beyond RAGs: Building Actually Truthful AI Harnesses"](https://towardsdatascience.com/beyond-rags-building-actually-truthful-ai-harnesses/)
(Towards Data Science), taken selectively — see §0.*

**What this lab is:** infrastructure. A separate module that every RAG surface hands the same
bundle to (query, retrieved chunks, prompt, answer, index state) and that returns an
inspectable, replayable record of *what the answer rests on*, plus checks of whether it
actually rests there.

**What this lab is not:** a retrieval improvement. measurement_lab already proved retrieval
coverage is not the bottleneck at this corpus size (+55% answer-bearing coverage, zero answer
gain across four tests). Nothing here changes what the LLM sees.

---

## 0. The thesis, recorded before any code

The article's central claim: *retrieval gives the model candidates for evidence; it does not
prove that a retrieved passage supports a claim.* It proposes a six-layer "truth harness"
(source capture → span retrieval → claim ledger → support checks → human review → release
manifest). Most of that is built for regulated publishing and does not fit a one-person tool.
What survives:

1. **The claim ledger** — atomic claim ↔ exact evidence span ↔ support status
   (supported / partial / insufficient / conflicting / unsupported).
2. **Deterministic code before LLM judges.** Code checks that citations exist, resolve and are
   well formed; an LLM judge is used only for "does this span support this claim?", and only
   after it has been calibrated against human labels.
3. **The citation failure taxonomy** — *incorrect* (source exists, doesn't support the
   sentence), *incomplete* (material claim, no evidence), *weak* (stale / off-scope source).
   "Citation present" alone is close to useless as a metric.
4. **A manifest** — which index, which chunks, which model, which prompt produced this answer,
   so it can be replayed when any of those change.

The user's framing, which this lab adopts as its spine: **accuracy alone is not what gives RAG
its value — the infrastructure for auditing and double-checking it is.** An 81%-correct system
you can audit is more useful than an 85%-correct one you can't, because you can find and
attribute the other 19%.

## 1. The precise gap (from measurement_lab, not from the article)

- **Zero refusals in 80 hand-labelled answers.** The prompt's "if the excerpts don't answer the
  question, say so" never fired, including on 36 retrieval failures. The model always answers.
  There is no refusal path in practice — an audit is the missing one.
  *(Qualified in P1: zero answers were **labelled** REFUSED, but 3 refusal-**shaped** answers
  exist. They are the 3 WRONG ones, refusals of questions the excerpts could answer.)*
- **Confidently wrong: 4.4%** (population-weighted, live config), PARTIAL 14.4%. The costly
  failure is the confident error, not the retrieval miss (RESULTS_HANDLABELS §3).
- **Nothing is recorded per answer.** `log_query()` logs retrieval score shape only — no
  context hashes, no answer, no model, no index fingerprint. A bad answer seen last week cannot
  be reconstructed.
- A real example already in the labelled set (`labels_hand_batch.json` item 1): asked about
  **CF**'s 10-Q, the answer cites `[RGTI, IBM, CSCO, CSCO #2, BA]` — every label resolves, every
  one is the wrong company, and all five sit in one bracket. Exactly the article's "incorrect
  citation", and invisible to a "citation present" check.

## 2. Seams that already exist (why this is cheap)

- All three surfaces hold the identical data at one point: after `call_llm`, before
  `linkify_citations` (`rag_query.py` `answer_query`, `sb_query.py` `answer_query`,
  `rstash-cli/src/main.rs` `answer_query`). Same prompt text in all three.
- Chunks are content-addressed (FNV-1a u64, shared with seedverify) — evidence identity is free.
- `score_shape()` (std30) exists in RQ; SP/rstash don't compute it yet. The audit computes it
  from the ranked scores, so all three get it.
- seedverify's `Judge::{Match, NoMatch, Error}` three-way pattern and verdict cache
  (`judge_<model>.bin`) — the template for Tier 1.
- Judge calibration already measured (`measurement_lab/results/judge_preference*.json`):
  gemma4:12b agrees with the human 86.7% (26/30, 6 order flips); gemma4:e4b flips 19 times —
  unusable as a judge.
- 80 stored, hand-labelled answers (`labels_hand_batch.json` + `labels_hand_results.jsonl`) —
  a ready validation set for P1.

## 3. Architecture

One Rust crate, `rag_audit` (lib + bin), at `~/machine_learning/rag_audit/`.

- **rstash** links it directly (path dependency).
- **RQ / SP** pipe a JSON bundle to the `rag_audit record` binary on stdin via
  `python/rag_audit_client.py` — the same subprocess-bridge pattern as `gq_search`. One
  implementation, so Python and Rust cannot drift (the lesson of the TICKER_BOOST port).
- **Best-effort, never fatal.** An audit failure prints one dim line and the answer still
  prints. `RAG_AUDIT=off` disables it entirely.
- The audit sees the **raw** answer (before `linkify_citations` rewrites it).

**Bundle (`rag_audit.bundle/1`)** — what each surface sends:
`system, retrieval_mode, query, tickers[], ranked_scores[], context[{label, source, hash, sim,
text, url, tag, sec_period_end, sec_form}], prompt, provider, answer, gen_error, retrieval_ms,
gen_s, embstore_path, env{RSTASH_*}`.

**Record (`rag_audit.record/1`)** — what the audit stores:
- `~/.cache/rag_audit/traces/<audit_id>.json` — the full bundle plus derived fields
  (timestamp, host, embstore fingerprint `path:mtime_ns:size`, prompt/answer FNV,
  `context_fp` = FNV over the ordered context hashes, score shape) and **empty slots for the
  later tiers** (`tier0`, `tier1`, `review.claude`, `review.human`). Written atomically
  (tmp + rename).
- `~/.cache/rag_audit/audit_log.jsonl` — one compact line per answer pointing at the trace.

### Tier 0 — deterministic, every query (P1)
1. **Citation resolution** — every `[label]` (including multi-label brackets) is a label that was
   actually in context; count uncited sentences.
2. **Numeric grounding** — every $/%/date/figure in a sentence, normalised ("$2.55 billion" ≈
   "2,550 million"), is found in the *cited* passage / in *some other* passage (mis-citation) /
   *nowhere* (fabricated). The highest-value check for SEC questions.
3. **Evidence span** — best trigram/token-overlap span in the cited passage. That span *is* the
   ledger's evidence span; low overlap routes the claim to Tier 1.
4. **Entity mismatch** — `detect_ticker` knows who was asked about; cited SEC rows that are all
   other companies are flagged (the CF example).
5. **Weak source / weak retrieval** — std30 spread, SEC period age, staleness/source-health at
   query time.

### Tier 1 — LLM entailment judge, opt-in `RAG_AUDIT=judge` (P2)
- Claims = sentences, split by code (the v1 policy for "atomic claim", stated up front).
- Per (claim, evidence span): `Supported / Partial / NotSupported / Error`. Error is never
  counted as NotSupported.
- Verdict cache keyed by (claim FNV, chunk hash, judge model).
- Judge = gemma4:12b or cloud, and not the model that wrote the answer where possible.
- Only Tier-0-flagged claims are judged (~1–3 calls/query, not 6+).

### Tier 2 — Claude triage, then human review (P3)

Human review is the most expensive tier and is itself unreliable (fatigue, anchoring, the
labeller being the tool's author). So: **Claude reviews every entry first and ranks what the
human should look at; the human reviews the ranked queue; everything stays browsable.**

- **Offline batch, not in the query path.** A `rag_audit triage` pass over un-triaged traces
  (e.g. the day's log) calls Claude (`claude -p`, several traces per call) with the trace, the
  Tier 0/1 results and the policy. Per trace it writes `review.claude`:
  `{per_claim[{claim, verdict, span_ref, note}], needs_human: bool, priority: 0-3, reason,
  reviewer_model, prompt_version}`.
- **Why Claude and not the local judge here:** Tier 2 is judging *answers as a whole* (did it
  answer the question, did it overclaim, did it miss a conflict) — harder than Tier 1's
  span entailment. It runs offline, so latency doesn't matter.
- **Self-review guard:** when the answer was generated by Claude (`LLM_PROVIDER=claude`), the
  triage record says so and those entries get a higher human sampling rate — self-preference is
  a known judge bias.
- **Catching Claude's misses:** the human queue always includes a **random sample of entries
  Claude did *not* flag** (the article's "random low-risk sampling"). Claude's miss rate on that
  sample is the number that says whether triage can be trusted.
- **Catching the human's misses:**
  - *Blind first look* on a sample: the human records a verdict before Claude's is revealed —
    measures agreement without anchoring.
  - *Disagreement is a flag.* Where human and Claude disagree, the entry is marked for a
    second look, not silently resolved either way.
  - *Test–retest:* occasionally re-present an already-reviewed entry, blind; self-consistency
    is the human's reliability number.
- **Surface:** an `RA` RUSTMM menu item — "Audit review queue": show the queue by priority,
  show each claim beside its evidence span, record approve / wrong / partial / unsure + note
  into `review.human`. Every human verdict grows the calibration set for Tiers 1 and 2.

### Replay (P4)
Because each trace stores the context text, hashes, model and embstore fingerprint, logged
queries can be **re-run after an `RX` re-embed, a model swap or a prompt change**, and the
claim ledgers diffed — "what do we do when the evidence changes".

## 4. Phases

### P0 — bundle + provenance record, wired into RQ / SP / rstash
- `rag_audit` crate: `record` (stdin bundle → trace + log line + one-line footer),
  `show <id|last>`, `compare <id> <id>`.
- Python client; rstash path dependency.
- **Pass:** the same question through all three surfaces produces three records with the same
  schema and every field populated, and `compare` shows where their context sets agree/differ.
  RQ vs rstash GPU-exact should mostly agree (recall@30 0.951).
- No verdicts. The footer says only "logged".

### P1 — Tier 0 checks + footer
- **Pre-registered test:** run Tier 0 over the 80 stored hand-labelled answers. Does a flag
  catch the WRONG / PARTIAL answers at a tolerable flag rate (≤15% of queries)?
- **Stop rule:** if not, keep the provenance log and drop the footer — report usefulness at an
  operating point, not AUC (measurement_lab lesson 2).
- **Known limitation:** 4.4% × 80 ≈ 2–4 WRONG examples. P1 needs a harder, company-named,
  numeric question set (L4-style) to produce enough errors to measure against.

### P2 — Tier 1 judge
- Hand-check ~100 (claim, span) verdicts **before** trusting it. Also closes
  RESULTS_GROUNDTRUTH §7.3 ("no human validation of the judge").

### P3 — Tier 2 triage + `RA` review queue
- **Pass:** Claude's miss rate on the unflagged random sample, and human–Claude agreement on the
  blind sample, are both measured and reported. Neither is assumed.
- **Stop rule:** if the unflagged-sample miss rate isn't clearly below the flagged-set hit rate,
  triage isn't ranking anything — keep the queue, drop the ranking.

### P4 — replay + diff

## 5. What this lab will not claim
- No check proves truth. No schema proves entailment, no citation list proves completeness, no
  hash proves relevance (the article's own caveats, adopted).
- Tier 0 flags are *routing* signals, not verdicts.
- A reviewed answer is "reviewed", not "correct". The number that matters is the measured error
  rate of each reviewer, Claude and human alike.

## 6. Traps, from prior labs
- **Validate the proxy before building on it** (measurement_lab lesson 1): "citation present",
  "number found", "Claude says supported" are all proxies until checked against hand labels.
- **Judge position/order bias** — judge_preference had to swap order to find it.
- **Hash-drift bugs** — twice now ragstash's readers produced text that no longer hashed to what
  seedverify embedded (byte vs char offsets). The trace stores the text *and* the hash, so a
  replay can detect that the two have drifted apart.

## Status
- 2026-09-24 — plan written; P0 started.
- **2026-09-24 — P0 PASSED.**
  - Built: `rag_audit` crate (`src/{lib,bundle,store,util,main}.rs`) with 13 failure-mode tests
    (roundtrip, contract rejections, truncation at every 7th byte, 2000 random bit-flips with
    no panic, tampered answer/chunk text/hash/id/schema → `Corrupt`, torn log line skipped).
  - Python client: `python/rag_audit_client.py`.
  - Wired into all three surfaces; backups `*.bak_ragaudit_20260924_134752` for
    `rag_query.py`, `sb_query.py`, rstash-cli `main.rs` + `Cargo.toml`. rstash rebuilt.
  - **Live test** — "What was Tesla's automotive revenue in its most recent quarterly
    filing?" through RQ / SP / rstash (GPU-exact): all three traces have the **same context
    fingerprint `fc2c60f2`** (identical 6 chunks, identical order), same prompt hash, same
    embstore fingerprint. `compare` shows jaccard 1.00 on both pairs. The answers differ
    only in wording. This is the first direct evidence that the three surfaces hand the LLM
    the same context, not just "similar retrieval".
  - Hand preview of the P1 numeric-grounding check on that trace: "$21,205 million" occurs in
    the cited `[TSLA #3]` chunk and in no other chunk → *grounded, correctly cited*.
  - Failure paths verified live: missing binary, invalid bundle, `RAG_AUDIT=off`, malformed
    stdin (exit 1). None of them breaks the query.
  - **Caveat found:** scores are "as ranked" per surface, so they aren't directly comparable.
    RQ/rstash `sim` includes the +0.20 TICKER_BOOST (TSLA 0.771), while SP's exact-SEC `sim`
    is unboosted (0.571) and SP's `ranked_scores` is its ANN list (top1 0.667, a non-TSLA
    chunk). `compare`'s spread line is only meaningful within a surface. If P1 needs
    cross-surface score comparisons, add a `boost` field per chunk.
- Next: P1 — Tier 0 checks, pre-registered against the 80 hand-labelled answers.

### P1 pre-registration (written 2026-09-24, before any Tier 0 code ran on the labelled set)

**Validation set:** `measurement_lab/labels_hand_batch.json`: 40 queries × 2 configs = 80
answers (gemma4:e4b-it-qat), with each answer's excerpts (label + text; no source/hash/SEC
metadata). Labels come from `labels_hand_results.jsonl`: CORRECT 64, PARTIAL 13, WRONG 3,
REFUSED 0. Population weights come from the `weight` field, the same reweighting as
score_labels.py.

**Disclosure, which matters:** before writing this I looked at the 3 WRONG answers. **All 3 are
refusal-shaped** ("The excerpts do not mention/provide/specify…"), and the labeller counted
them WRONG because the answer was available. So in this set the costly failure is **false
abstention, not fabrication**. This also qualifies measurement_lab's "zero refusals": zero
answers were *labelled* REFUSED, but refusal-*shaped* answers exist and were labelled WRONG.
The refusal check (F6) was added **after** seeing this, so its score on this set is
**contaminated** and cannot count as validation. F1–F5 were in the plan (§3) beforehand.

**Flags (an answer is FLAGGED if any fires):**
- **F1 unknown_citation**: a bracketed label that wasn't in context.
- **F2 number_absent**: a checkable number found in no excerpt and not in the query.
- **F3 number_miscited**: a checkable number found only in excerpts the sentence doesn't
  cite, when the sentence does cite something.
- **F4 uncited_number**: a sentence with a checkable number and no citation, even after
  counting citations anywhere in its paragraph.
- **F5 entity_mismatch**: a company was detected in the query, but every cited SEC chunk
  belongs to other companies. **N/A on this set**, which has no ticker metadata.
- **F6 refusal** *(post-hoc, see above)*: the answer contains an insufficient-evidence phrase.

**Checkable number:** $/%/unit-bearing values, or plain numbers ≥ 10. Years match exactly
(excerpt text, SEC period metadata, or the query). Matching tolerance is set by the answer's
own precision ("$21.2 billion" ⇒ ±$0.05B). A unitless excerpt number may be in units,
thousands, millions or billions (SEC tables).

**Report:** for each flag and for the union, the weighted and raw flag rate among CORRECT /
PARTIAL / WRONG. **Pass rule (from §4):** the union catches WRONG+PARTIAL while flagging ≤15%
of queries (weighted). Report the result both with and without F6. Whatever the numbers, **3
WRONG answers is too few to validate anything**. A P1b set of fresh, harder numeric questions
with fresh labels is the real test, and the only clean one for F6.

### P1 result (2026-09-24) — **stop rule FIRED**

Run: `eval/p1_validate.py` → `eval/results_p1.json`. Weighted rates, raw counts in parentheses.

| pooled A+B (n=80) | CORRECT (64) | PARTIAL (13) | WRONG (3) | flag rate |
|---|---|---|---|---|
| F1 unknown_citation | 0.0% (0) | 8.5% (1) | 0% (0) | 1.4% |
| F2 number_absent | 0 | 0 | 0 | **0%** |
| F3 number_miscited | 0 | 0 | 0 | **0%** |
| F4 uncited_number | 11.4% (7) | 15.2% (2) | 33.3% (1) | 12.8% |
| F6 refusal *(contaminated)* | 1.4% (1) | 8.5% (1) | **100% (3)** | 5.8% |
| **UNION F1–F5 (pre-registered)** | 11.4% | 23.8% (3) | **33.3% (1)** | **14.2%** |
| UNION F1–F6 | 11.4% | 23.8% | 100% | **16.4%** |

Config A (live) alone: F1–F5 union 12.8% flag rate / 1 of 2 WRONG; F1–F6 union exactly 15.0% /
2 of 2 WRONG. Config B: 15.6% / 17.8%.

**Verdict under the pre-registered rule: FAIL.** F1–F5 stays under the 15% flag-rate budget but
catches 1/3 WRONG, and that one only by coincidence (an uncited year in a refusal). Adding F6
catches 3/3 but goes over budget (16.4%) and is contaminated anyway. Per §4 the footer is
demoted: **default footer = provenance line only; `RAG_AUDIT_FOOTER=flags` opts in.** Tier 0
still runs and is stored in every trace, because P3 triage and P1b both need it.

What the numbers actually say:
1. **The number checks (F2/F3) never fired, on any label.** Zero false alarms on 64 CORRECT
   answers, which is worth knowing, but also zero true hits, because this set has **no
   fabricated or mis-cited numbers at all**. They are **unvalidated, not refuted**. The set
   can't test them.
2. **F4 is pure noise, and all of it is years.** Every one of the 10 `uncited_number` flags is
   a year ("2025", "2022") in a sentence without a bracket. A candidate `tier0/2` would exclude
   years from F4. That tuning comes from this set, so it must be judged on P1b, not here.
3. **The failure mode in this data is false abstention.** All 3 WRONG answers are refusals of
   answerable questions. F6 routes them, but a refusal flag cannot tell a *correct* refusal from
   a false one. That call needs someone to read the excerpts, i.e. Tier 1/Tier 2 work. It is the
   strongest argument yet for P3's Claude triage.
4. **Honest scale:** 3 WRONG answers; every rate above has an enormous CI.

### F7 stale_period (pre-registered 2026-09-24 for P1b; found in live use, not the labelled set)

Two live rstash traces from the same afternoon, both on "latest" questions, both scored
*clean* by F1–F6:
- "what is LLY's latest earnings?" cites a 2024-06-30 10-Q; a 2025-03-31 10-Q was in context.
- "NVIDIA's data center revenue in its most recent quarter?" answers "$22.6B, Q1 FY2025" citing
  the 2024-04-28 10-Q, while the 2025-10-26 10-Q and 2026-01-25 10-K were in context. The number
  is correctly grounded **and the answer is still wrong**. This is the article's *weak source*
  failure, which F1–F6 are blind to.

**F7:** the query contains a recency word (latest / most recent / current / last quarter / …),
and for some company, the newest cited SEC period is older than that company's newest period
in context. Deterministic, metadata only. Fires on exactly those 2 of the 5 live traces, and
not on the 3 Tesla traces that cited the newest filing. The live code is now `tier0/1+f7`.

### P1b — the clean test (next; needs human labelling time)
- ~40 fresh questions: company-named, numeric, **with ~⅓ "latest/most recent" phrasing**
  (for F7). Queries drawn from `measurement_lab/labels_l4_ticker.json`'s construction.
- Run live through RQ/rstash (traces are recorded automatically). Labels are CORRECT /
  PARTIAL / WRONG / REFUSED, **collected blind to the Tier 0 flags**, plus a "was the answer
  actually in the excerpts?" field so F6 refusals can be scored as correct vs false.
- Score F1–F7 and the `tier0/2` candidate (years excluded from F4). Pass rule unchanged: catch
  WRONG+PARTIAL at ≤15% weighted flag rate. If F7 alone passes, it can go in the default footer
  on its own.

#### P1b pre-registration (2026-09-24, written before any P1b answer was generated)

**Question set:** `eval/p1b_questions.json`, 40 questions drafted by Claude from the SEC
inventory (64 tickers, filings through 2026-03):
- 14 **latest**: "most recent / latest quarter"
- 16 **period**: a named quarter end date
- 6 **compare**: year-over-year change
- 4 **unanswerable by design**: 2 companies absent from the corpus (PFE, XOM), 2 future
  periods. These give a clean measure of *correct* refusals, which P1 had none of.

**Pipeline:** `eval/p1b_run.py` runs every question through **RQ in-process** (the reference
surface; P0 showed RQ/SP/rstash give identical context). Traces go to a separate audit dir,
`~/.cache/rag_audit_p1b/`. The only QA done before labelling is on ticker detection; **nobody
looks at Tier 0 flags before labels exist.**

**Labels:** `eval/p1b_label.py`, blind to the flags and to the question's `kind`. Per answer:
- **verdict**: CORRECT / PARTIAL / WRONG / REFUSED (label.py's definitions). For a "latest"
  question, presenting an older period as the latest counts as WRONG when a newer period of
  the same metric is in the excerpts.
- **in_excerpts**: yes / partly / no. Was the answer actually available? This separates a
  *correct* refusal from a *false* one.
- **newest_used** (recency questions only): yes / no / n/a.

**Primary analysis:** `tier0/2`, the union of F1–F7 with years excluded from F4. Secondary
analyses: each flag alone, tier0/1 (F1–F6 as run in P1), and F7 alone on the 14 latest
questions against `newest_used`. Scored **unweighted**: this set isn't stratified, so there
are no sampling weights.

**Pass rule, one declared change:** P1's "≤15% of all queries flagged" budget assumed the live
population's ~20% bad-answer rate. This set is deliberately hard, so bad answers may well
exceed 15%, and then the rule would fail any flag that works. For P1b the rule becomes
**recall ≥ 50% on WRONG+PARTIAL *and* false-alarm rate ≤ 15% on CORRECT**, i.e. the same
false-alarm budget P1 was implicitly spending. Refusals get their own table: **F6 on REFUSED
split by in_excerpts** (false refusals, where in_excerpts = yes/partly, vs correct refusals,
where it's no). F6 routes both; P3 has to tell them apart.

**Consequence:** if tier0/2 passes, `RAG_AUDIT_FOOTER=flags` becomes the default. If only F7
passes, only F7 goes in the default footer. If nothing passes, the footer stays
provenance-only and Tier 0 is triage input for P3.

CLI: `rag_audit list [N]` · `show [last|last~N|id|substring]` · `compare A B` · `record` (stdin)
· `tier0` (stdin bundle → Tier 0 JSON, nothing stored) · `check [SPEC|all]` (re-run Tier 0 into
stored traces).
Data: `~/.cache/rag_audit/{audit_log.jsonl,traces/}`; override with `RAG_AUDIT_DIR`.

#### P1b status (2026-09-24)
- **Answers generated:** all 40 via `eval/p1b_run.py` (RQ, gemma4:31b-cloud) → traces in
  `~/.cache/rag_audit_p1b/`, map in `eval/p1b_run.jsonl`. Nobody has looked at answers or
  flags. The only QA done was ticker detection.
- **Detector finding (live RQ/rstash bug, out of P1b scope, left as-is on purpose):** "Eli
  Lilly" is never detected (q11, q34), so those two questions ran without the company boost.
  `build_company_terms()` keys each company only on the first significant word of its SEC name,
  and only if that word is at least 4 letters. For "ELI LILLY AND COMPANY" that word is "eli"
  (3 letters), so LLY has **no name entry at all**, and only the symbol "LLY" works. The questions
  were not rephrased, so the audit sees real behaviour. Candidate fix: also index the second
  significant word, and check the spurious-firing rate on the measurement_lab 200-query set
  (the reason the strict detector exists). rstash's `corpus::detect_ticker` port would need the
  same change.
- **Tools tested:** labeller smoke-tested in a scripted pty (saves, asks Q3 on recency
  questions, no flag text reaches the screen). Scorer run on *random fake* labels in a scratch
  copy, with output discarded; only the exit code and result structure were checked.
- **Waiting on:** Fraser's blind labels → `python3 eval/p1b_label.py`, then
  `python3 eval/p1b_score.py`.

### P1b result (2026-09-24) — **every flag set FAILS the pre-registered rule**

40 blind labels: CORRECT 25 · PARTIAL 8 · WRONG 7 · REFUSED 0. Full output: `eval/results_p1b.json`.

| flag set | recall W+P (15) | false alarm on CORRECT (25) | pass |
|---|---|---|---|
| tier0/2 PRIMARY (F1–F7, no years in F4) | 60% | **56%** | fail |
| F6 refusal | 47% | 56% | fail |
| F7 stale_period | 27% (4) | **0%** | fail (recall) |
| F1/F2/F3/F5 | 0 | 0 | never fired |

**Label convention, as observed:** REFUSED was never used. **Correct refusals were labelled
CORRECT** (10 answers with in_excerpts = no and verdict CORRECT, including all 4
designed-unanswerable questions). So 9 of F6's 14 "false alarms" are correct refusals being
routed. That's defensible routing, but under this rule it counts as noise. Not re-scored post
hoc; recorded.

**F7 against the "used newest" label:** caught 4, missed 6, **0 false alarms**. All 6 misses
share a cause that F7 can't see by design: **the newest filing never reached the excerpts.**

**The finding that matters is about retrieval, not the audit flags** (measured from the traces
after labelling; `in_ctx` = the target period appears among the company's excerpts):
- **All 7 WRONG answers are "latest" questions.** The corpus-newest period reached the excerpts
  on only **5 of 14** "latest" questions. Similarity search has no concept of date.
- **10-K-only context on quarterly questions:** q04 (AMZN latest), q05 (COST latest; newest
  excerpt was a *2024-09-01 10-K* while a 2026-02-15 10-Q is in the corpus), q18 (AMZN AWS
  June quarter). Fraser spotted this while labelling.
- **Right filing, wrong chunk:** on period questions the target period reached context 13/15,
  yet Fraser marked the figure absent (in_excerpts = no) on 6 of those 13. Similarity picked MD&A
  chunks from the right 10-Q that don't contain the asked-for figure. Open question: does
  SEC10Q.DAT (MD&A + risk-factor text via export_sec10q.py) even contain every financial-statement
  figure?
- Eli Lilly: the detector miss (q11, q34) gave 0–1 LLY chunks. That's why q11 was WRONG.

**Consequences:**
1. Per the rule, **no flags go in the default footer.** But P1b shows what *should* go under
   the answer: **facts, not unvalidated judgments.** For each company: the periods and forms in
   the evidence, the newest period in the corpus, the newest on EDGAR. Facts can't false-alarm,
   and they would have shown all 7 WRONG answers at a glance.
2. **Next priority moves from the audit to retrieval:** period/form-targeted retrieval. It's
   measurable automatically (target-period-in-context, currently 5/14 · 13/15 · 5/6) before any
   new labels are collected.
3. Fraser's proposal, a feedback field after each answer (good / wrong / retry + note) that
   writes `review.human`, turns P3's human review into labels collected at the point of use.
   Retry must **change the evidence**, e.g. apply the period filter or the newest filing.
   Regenerating from the same excerpts would only produce a confident rewrite.

### Step 1 — period-targeted retrieval (2026-09-24, LIVE in RQ / SP / rstash)

**What changed (retrieval, not audit; backups `*.bak_periodfilter_20260924_175439`):**
- `ragstash/period_intent.py` (new) + Rust port `rstash-core/src/period_intent.rs`. Rules:
  explicit date (±7 days) → fiscal year N → recency word (+quarter ⇒ newest 10-Q, +annual ⇒
  newest 10-K, else newest filing; plus any newer filing, since a fiscal Q4 lives only in the
  10-K) → otherwise none.
- `rag_query.select_context()`: retrieval split out from generation. When a target exists,
  **all context slots come from the target filing(s)** (exact cosine over that slice). Remaining
  slots fill from the general list **minus the same company's other filings**. SP applies the
  same rule to its company-exact pass. rstash has a twin `select_context()` plus
  `rstash --retrieve-only IN OUT`.
- Every trace records the decision (`bundle.period_filter`, shown by `rag_audit show`).
  `RAG_PERIOD_FILTER=off` restores the old behaviour on all three surfaces.
- **Detector:** second-name-word fallback ("Eli Lilly" → LLY) plus a full-name phrase entry
  ("advanced micro devices"). "business" and "micro" were added to the generic words. The name
  table is now **multi-valued** ("alphabet" → GOOG *and* GOOGL).

**Measured** (`eval/period_metric.py`, P1b's 36 date questions, retrieval only):

| | latest | named quarter | compare | older-quarter noise | 10-K-only quarterly |
|---|---|---|---|---|---|
| filter off | 5/14 | 13/16 | 6/6 | 159 chunks | 3 |
| **filter on** | **14/14** | **16/16*** | 6/6 | **0*** | **0** |

\* q30 (KO fiscal 2024) shows as a miss in the script only because its metric parses literal
dates. Checked by hand: all 6 chunks are the FY2024 10-K. Compare shows 6/6 even with the filter
off because the detector fix alone fixed q34 (LLY).

Detector on the measurement_lab sets: L2 spurious 2/300 (unchanged), L3 13/250 (unchanged),
**L4 recall 192 → 200/200**. L4 "extra" detections go 1 → 5: four are GOOG alongside GOOGL
(intended), and one is the pre-existing COUR-on-a-UDMY-question.

**Python vs Rust verification** (Fraser: keep both until stable, then finalise in Rust):
- `eval/period_intent_cases.json`, 18 shared cases: Python 18/18 (`period_parity.py`), Rust
  18/18 (`cargo test -p rstash-core --test period_intent_cases`).
- End to end (`period_rust_parity.py`): **36/36 questions give identical detection, intent and
  filtered context rows, in order**. The first run was 35/36, and the one difference was a real
  bug. rstash's single-valued name map with `or_insert` over HashMap iteration picked GOOG *or*
  GOOGL at random for "Alphabet", while Python took the last written. Both are now
  multi-valued, and Rust's detector output is sorted. The same check found that Rust's
  `GENERIC_NAME_WORDS` had silently dropped 9 of Python's 49 words; the lists are identical now.
- Live: "Apple iPhone net sales, most recent quarter" gets **the same context fingerprint
  `5346a4dd` on RQ, SP and rstash**, and the answer is $56,994M for the quarter ended
  2026-03-28 [AAPL #4]. P1b's pre-filter answer was fiscal Q2 2025 with no figure.

**Still open:** the corpus itself is a quarter behind EDGAR (Apple's 2026-06-27 10-Q, filed
2026-07-31, isn't in SEC10Q.DAT). That's step 2 (EDGAR freshness in the audit section) and
step 3 (SEC top-up). The P1b answers were generated before the filter; re-generating them and
re-labelling is the answer-level test of this change.

### Step 2 — the audit section (2026-09-24, LIVE in RQ / SP / rstash)

Under every answer, a divided block rendered by **one** Rust renderer (`src/section.rs`),
identical on all three tools:
```
── audit ────────────────────────────────── 20260925T011228Z-RQ-ecb58382
evidence  AAPL 10-Q 2026-03-28 ×6
          period filter AAPL → 2026-03-28 (most recent 10-Q)
freshness AAPL newest indexed 10-Q 2026-03-28 · EDGAR 10-Q 2026-06-27 (filed 2026-07-31)
          ⚠ corpus is behind EDGAR: 10-Q 2026-06-27 (filed 2026-07-31) is not indexed yet — …   [orange]
answer    1 claim, 1 cited · 1 figure: 1 in the cited excerpt · citations match excerpt labels
retrieval brute-force cosine (cupy GPU, f32) · 6 chunks · spread 0.034 · ctx 5346a4dd
```
- **Facts, not flags** (the P1b rule). The only default judgment is staleness, shown in
  **orange**. It's a date comparison: *corpus behind EDGAR*, or *evidence older than the
  newest indexed filing*. Neither ever fires on an explicitly dated question, where an old
  filing is what was asked for.
- Bare years are not counted as "figures" (P1's F4 noise; a refusal restating "2025" read as
  "1 number").
- Freshness facts come from the surfaces: `rag_query.freshness_facts()` and rstash's twin,
  newest indexed (period, form) vs EDGAR's newest 10-Q/10-K by report date. They're carried in
  `bundle.freshness`. **rstash now shares ragstash's EDGAR cache** (`~/.cache/ragstash/edgar`;
  identical format, checked), so both tools see one snapshot.
- `RAG_AUDIT_SECTION=off` gives the one-line footer; `RAG_AUDIT_FOOTER=flags` adds the Tier 0
  flags inside the section. `rag_audit section [SPEC]` re-renders any stored trace.
- Tests: `tests/section.rs` (6), covering orange on/off, dated questions never orange, years
  not figures, and grouping. Parity: `period_rust_parity.py` now also requires identical
  freshness facts: **36/36**.

**Finding: 28 of 28 companies named in P1b are behind EDGAR**, mostly by one quarter. ORCL,
NVDA and AMD are further behind, and GOOG's own records stop at a 2022 10-K. The orange line
will therefore show on nearly every company question until the corpus is topped up. That's
accurate, not noise, and it makes **step 3 (SEC top-up)** the next priority.

**Seen in live use (not fixed here):** "Intel gross margin, quarter ended 2025-06-28" now gets
six chunks from the right 10-Q, and the answer still says the figure isn't there. That's the
*right filing, wrong chunk* problem from P1b: the top 6 by similarity miss the chunk that holds
the figure.

### Step 3 — SEC top-up, menu `SU` (2026-09-24, LIVE; manual by design)

`ragstash/sec_topup.py` (wrapper `sec_topup.sh`, menu **SU** directly under RX). The orange
audit line now ends "menu SU tops it up". It is **manual on purpose** (Fraser agreed): it fetches
from SEC, appends to SEC10Q.DAT / SEC10K.DAT (read by the COBOL tools too) and re-embeds.
1. **Plan:** EDGAR (1-hour cache, not 7-day) vs the newest indexed 10-Q/10-K per company. It
   fetches every newer 10-Q (max 4) and the latest 10-K if it's newer. Each filer (CIK) is
   processed once (GOOG is skipped as GOOGL's share class). Tickers missing from SEC's current
   ticker map (BFB, UDMY) are skipped and named. `--dry-run` stops after the plan.
2. **Confirm**, then **back up both DATs**, then for each company: the existing fetchers
   (`sec_mda` / `sec_risk` `fetch_filing`), then the existing `export_sec10q/10k.py --append`,
   which is idempotent and gemma4:e2b-quality-checked. **After every append, both files must
   still be whole 8131-byte records, or both are restored and the run stops.**
3. **Re-check freshness** against EDGAR and name any company still behind (quality-check
   rejects, no MD&A or Item 1A found).
4. **Confirm, then embed** (seedverify `embed_all.sh` + `staleness_report --after-embed`, the
   same as RX). A per-machine lock (`~/.cache/ragstash/sec_topup.lock`) prevents concurrent runs.
   Every run appends a line to `~/.cache/ragstash/sec_topup_log.jsonl`.

Supporting changes: `ragstash/freshness.py` (GPU-free; shared by `rag_query.freshness_facts`
and the top-up), and `edgar_lookup.get_filings(..., ttl=)`.

**Verified end to end on AAPL:**
- plan: 10-Q 2026-06-27 plus AAPL's first 10-K (2025-09-27)
- append: fetched 2/2, appended 12 records (3 MD&A, 9 risk factors), both passed the quality check
- EDGAR re-check: AAPL current, 1/1
- embed: 728 chunks, including 618 web chunks that were already waiting, in 34 s; "everything on disk is embedded and searchable"
- re-ask: RQ and rstash both answer **$54,252M for the quarter ended 2026-06-27**, with **no
  orange line** and an identical ctx fp `2736bbde`

Failure paths, induced on purpose: a concurrent second run is refused ("another SEC top-up is
already running"); a torn append (1000 bytes, not a whole record) is detected and restored
byte-identical from backup (on scratch copies, not the live DATs). `SU` parses at rust_menu's
fixed MENU.DAT columns (type I, cmd at col 72, active Y), and the key is unique.

**Full plan as of today:** 60 companies behind EDGAR, 115 filings. Budget roughly **1–2 hours**:
the AAPL run took 2 min for 2 filings, most of it the per-filing gemma4:e2b quality check. Run
it on **one machine**. The DATs sync via Synology, but embeddings (`~/.cache/seedverify`) are
per machine, so the other machine then needs **RX** (not SU). Never run SU on both at once;
the lock is per machine. Backups (`*.DAT.bak_topup_<stamp>`, ~40 MB per run) sit beside the
DATs; delete old ones by hand.

#### Step 3b — `u` = update this company, from inside rstash (2026-09-24, Fraser's request)
When the audit section shows a company behind EDGAR, rstash's prompt becomes
`Question (q=quit, u=update TSLA from EDGAR):`. Pressing `u`:
1. runs `sec_topup.py --tickers TSLA --yes` for just that company (fetch → quality check →
   backed-up append → embed; seedverify's output is filtered to progress lines);
2. if new vectors landed in the store, drops gq_search cleanly, **re-execs rstash** (it and
   gq_search hold corpus + vectors in memory, so a reload is the only honest way to see new
   data), and **re-asks the last question** via `RSTASH_REASK`. If nothing new was embedded,
   nothing reloads.

**Live test (pty-driven, real top-up):** "latest TSLA earnings?" answered from the 2026-03-31
10-Q with the orange line and the `u` offer → `u` appended 5 records for the 2026-06-30 10-Q
and embedded 44 chunks → reloaded in **10.8 s** (GPU-exact) → re-asked. The answer was now
Q2 2026 (revenue $28,236M), with 24/24 figures in the cited excerpts, no orange line, and the
prompt back to `(q=quit)`. Note: the CPU ANN path (rstash launched without `config.env`) has
to rebuild its HNSW graph after any corpus change, which takes ~3 min.

### Step 4 — inline-XBRL fact sheets (2026-09-24; Rust production + Python reference)

**Why:** "LLY latest earnings and interest cost" got a correct refusal. LLY's interest expense
($345M Q2 2026) exists only in a segment table in the financial-statement notes, and the 10-Q
pipeline indexes **only MD&A**: 29.7K of that filing's 129K characters (~23%). This is also the
likely cause of P1b's "right filing, figure absent" answers (6 of 13 period questions). SEC's
company-facts feed doesn't help either: it omits dimensional (segment) facts, and that interest
expense is one. But every number in a filing's HTML is an inline-XBRL `ix:nonFraction` tag with
an exact concept, period and scale, and the HTML is already in the fetch caches (407 filings,
all well-formed XML).

**What:** each cached 10-Q/10-K becomes a set of period-labelled passages, e.g. `Interest expense
(nonoperating) [segment: Reportable segment]: 3 months ended 2026-06-30 $345 million; 6 months
ended 2026-06-30 $677 million; 3 months ended 2025-06-30 $249 million; …`. They include
revenue by product and geography, full balance sheet, cash flow, EPS.
- **Files:** `sec_10q/data/SEC10QXBRL.DAT`, `sec_riskfactors/data/SEC10KXBRL.DAT`. Same 126-byte
  header as the SEC DATs, one passage (≤690 bytes) per line, so every existing reader's 700-byte
  chunker returns each line unchanged. The COBOL tools never read them.
- **Writer:** `~/machine_learning/sec_xbrl` (Rust, production; lib + `sec_xbrl
  rebuild|append|show`). **Python reference:** `ragstash/xbrl_facts.py`. The spec was made
  explicitly portable: a hand-written camel-case tokenizer instead of a lookahead regex, integer
  round-half-up months, strict ISO dates, ASCII-digit regexes, lxml `itertext` semantics
  (comments/PIs excluded) matched by the Rust streaming parser.
- **Readers** (all four SEC files via `ALL_SEC_PATHS`): `ragstash/sec_reader.py`, rstash-core
  `corpus/sec.rs`, seedverify `concept` defaults. `gq_search` was rebuilt in `fedora43-oxide`.
  RQ's combined-index cache key now covers all four files.
- **Top-up:** `SU` / rstash `u` run `sec_xbrl append` for the companies just fetched (Python
  fallback) and embed the new passages too.
- **Audit section:** the evidence line counts XBRL passages, e.g. `LLY 10-Q 2026-06-30 ×6 (4 XBRL)`.

**Verification:**
- **Parity gate** `eval/xbrl_parity.py`: Python and Rust write **byte-identical** files for all
  407 filings (43,455 + 23,908 passages, ~50 MB), the first time it was run. Rust 4.4 s vs Python 14.4 s.
- **Reader contract:** every line reads back as exactly one passage (67,363 lines → 67,314
  distinct hashes; the 49 duplicates are identical texts, deduped identically everywhere).
- **Rust tests** (`sec_xbrl/tests`, 6): a synthetic iXBRL filing with a comment inside a fact,
  duplicate facts, nil, dash-for-zero, negative sign, divide units (per share), two dimensions,
  unparseable number-words, an orphan context, plus truncated/garbage/missing files (never panic)
  and the 690-byte passage bound.
- **Scale:** 301 10-Q + 101 10-K fact sheets = **~67K passages (+31% corpus)**; one-time embed ≈ 45–50 min.

**Results after the one-time embed** (67,314 passages in 982 s; "everything on disk is embedded"):
- `eval/xbrl_eval.py` re-asked, through RQ's real answer path, **all 18 P1b questions Fraser had
  labelled "answer not (fully) in the excerpts"**. Now: **18/18 have XBRL passages in context, 18/18
  answer without refusing, 18/18 have ≥1 figure found in the excerpt they cite** (a fact from the
  audit, not a correctness label). Spot-read against the excerpts:
  - LLY interest expense **$345M** Q2 2026, $677M H1 (was a refusal);
  - Intel Q2 2025 gross margin **27.55%**, which the model *computed* from XBRL gross profit
    $3,542M / revenue $12,859M (P1b: "not in excerpts");
  - AWS operating income **$10,160M** Q2 2025;
  - NVIDIA Data Center FY2026 **$193,737M** (segment fact).
- **Gap found: multi-concept trend questions.** "How has LLY's interest expense changed relative to
  revenue over recent quarters?" got the interest-expense lines (with year-ago comparison) but **no
  revenue lines**. Six similarity-ranked slots, no period word, so all six went to interest
  passages, and the model correctly said revenue wasn't there. The fix is not more similarity. It's
  a **structured series step**: detect "trend / over time / vs" plus the named concepts, then pull
  those concepts' values across the company's fact sheets **deterministically** into one compact
  time-series passage. Proposed as step 5.
- Parity with the new corpus: `period_rust_parity.py` **36/36** (detection, intent, filtered
  context, freshness), with gq_search's row-count guard satisfied. Period metric, filter on:
  latest 14/14. Filter *off* fell 5/14 → 2/14, because older filings' fact sheets add more
  same-company competition, which is more evidence the filter is load-bearing.
- rstash live: `evidence LLY 10-Q 2026-06-30 ×6 (6 XBRL)` · 8/9 figures in the cited excerpt.

### Step 5 — deterministic XBRL trend series (2026-09-24, LIVE in RQ / SP / rstash)

**Why:** "How has LLY's interest expense changed relative to revenue over recent quarters?"
got interest lines but no revenue. Similarity fills six slots with whatever reads most like
the question, and a two-metric, many-period comparison doesn't fit that shape. Fraser: this kind
of question will come up often in the SEC setting.

**What (`sec_xbrl/src/series.rs`, ONE implementation):** rstash links it; RQ and SP call `sec_xbrl
series --tickers … --query …` (JSON). When a question names a company, a figure and a trend word
(trend/trended, over time, last N quarters/years, vs/relative to, growth, a year earlier, …):
1. **Figures:** a curated map of ~23 metrics (revenue, interest expense, net/operating income,
   margins, EPS, R&D, SG&A, cash flow, capex, debt, cash, …). Each has phrases (longest match
   first, so 'cost of sales' isn't 'sales' and 'earnings per share' isn't 'earnings') and XBRL
   concepts in priority order. Composites: gross/operating/net margin = ratio to revenue; free
   cash flow = operating cash flow − capex. Two figures plus "vs / relative to / % of" gives a
   ratio row. Everything is **computed here, not by the LLM**.
2. **Values from every cached filing of the company** (same inline-XBRL parser as the fact
   sheets): latest-filed value per period. **Q4 and 10-Q cash-flow quarters are derived** from
   consecutive year-to-date totals (Q4 = fiscal year − nine months) and marked `*`. Dimensional
   (segment) tags are used only as a fallback. **When a metric uses more than one tag, every value
   carries its source `[1]`/`[2]`**, so a substituted measure is visible per period.
3. **One passage** in context slot 1 (quarterly by default, 8 quarters; "years" gives annual;
   "last N" is honoured; an explicit date anchors the window's end; YoY % when asked). The other
   5 slots come from normal (period-filtered) retrieval. A filing index (`~/.cache/sec_xbrl/
   index.tsv`) makes repeat queries ~0.13 s.
- **Audit section** `series` line: company, frequency, window, `k/N complete`, filings, derived
  count, metrics. **Orange when the window has gaps** (older filings not cached).
- **History back-fill:** `sec_topup.py --history N` fetches older 10-Qs/10-Ks in the window that
  aren't cached, then fact sheets and embed. It skips the ~1 min/filing MD&A quality check; the
  next normal top-up appends their MD&A. **rstash `h`** runs this for short series, reloads,
  re-asks.
- `RAG_XBRL_SERIES=off` disables it on all three tools.

**Verification:**
- **Independent oracle** `eval/series_oracle.py`: SEC's company-facts API, which SEC parses from
  the same filings through a completely separate path. 10 companies × revenue / net income /
  operating income, **232 values: 231 agree to $1M, including every derived Q4, 1 disclosed tag
  substitution, 0 disagreements**. The substitution: Ford Q3 2024's only cached source (the
  2025 10-Q's comparative) tags ProfitLoss $896M, which includes $4M of noncontrolling interest,
  not NetIncomeLoss $892M. That case is what motivated the per-value `[k]` source markers.
  A first oracle run showed 32 false misses; that bug was in the *oracle* (it didn't merge tags
  per period like the series does), and it's fixed.
- **Unit tests** (`sec_xbrl/tests/series.rs`, 7): trend detection and metric order, composites,
  annual/quarterly/count/anchor, Q4 = FY − 9M on synthetic filings, YoY, substitution marking,
  unknown company. The tests caught "trended" not being recognised.
- **Parity** `period_rust_parity.py`: **42/42**, the 36 P1b questions plus 6 trend questions.
  Detection, intent, filtered rows, freshness **and series passages** are identical in Python and Rust.
- **Side effect checked:** all 6 P1b *compare* questions now also get a series. That's intended:
  anchored at the asked quarter, their window contains the year-ago quarter. One was wrong: Etsy
  "gross merchandise **sales**" matched bare "sales" → revenue. Bare "sales" was removed ("net
  sales" / "total sales" remain), and "a year earlier / prior year / year ago" now request YoY.
- **Live:** the LLY trend question on RQ and rstash gets an identical table (ctx `c9257616`):
  interest expense as % of revenue per quarter, 1.7% → 1.9% → 1.0% → 1.5%; 8/8 quarters complete,
  4 derived; 18/18 figures in the cited excerpt (RQ run).
- **Live `h` on Coca-Cola** (the hard case: only 10-Ks cached). Before: a short quarterly series,
  orange gap line, `h` offered. `h` fetched 6 older filings plus the 4 new 10-Qs (the normal
  top-up's MD&A append also picked up the history filings' MD&A, 313 records, which is why it
  took longer) → embedded. After: **KO quarterly 2024-09-27 → 2026-07-03, 8/8 complete, 14 filings**,
  and 16/16 answer figures in the cited excerpt. Values match KO's reported revenue ($11,854M Q3
  2024, $12,535M Q2 2025). The pty harness itself missed the prompt (an ANSI code sits inside
  `Question (q=quit`), so the result was verified directly afterwards.
- **False alarm fixed:** the orange "evidence is older than the corpus" fired on a trend question
  whose series already ran through the newest indexed quarter. It now stands down when a series
  for that company covers the newest period.

**Overview document for the whole day: [`README.md`](README.md).**
- **Ratio convention fixed (Fraser):** "LLY revenue *versus* interest expense" rendered "Revenue as
  % of interest expense: 9800%", which followed word order and is unorthodox for financial
  reporting. Auto-inferred ratios are now oriented **smaller-as-%-of-larger by median magnitude**
  over the window, whatever the word order: "Interest expense as % of revenue: 1.6% … 1.5%".
  Named ratios (margins) keep their definition.
  - **Interest coverage** ("interest coverage", "times interest earned") = operating income ÷
    interest expense, shown as a **multiple**. When the company reports no operating-income line
    (LLY, typical in pharma), EBIT is approximated as **pre-tax income + interest expense**, marked
    `†` with the method stated. LLY: 28.2× → 41.3× → 38.1× → 27.7× → 27.8×.
  - A metric the company doesn't report in the window is listed as "not reported" instead of a
    row of n/a, and **no longer counts against completeness**. Otherwise it would offer an `h`
    history fetch that can't help.
  - Tests: 10 series tests, including word-order independence, coverage as a multiple, and the
    EBIT fallback. Parity 42/42 and oracle 231/232 are unchanged.

### Step 6 — feedback, retry and replay (2026-09-25, LIVE in rstash / RQ / SP)

At the prompt after any answer:
- **`g` / `w` / `p`** = good / wrong / partial, with an optional note after a colon (`w: I wanted
  the March quarter`). The colon keeps questions like "p and g revenue?" as questions. It writes
  `review.human` into the trace (atomic) and a line to `~/.cache/rag_audit/reviews.jsonl`; the
  latest review of an answer supersedes earlier ones. These are labels collected in normal use.
- **`r` / `r: note` = retry with CHANGED evidence**, never a re-roll over the same excerpts (P1b:
  failures come from missing evidence). `rag_audit::feedback::plan`, one implementation, rules
  in order:
  1. a note **leads** the question, so its dates and "latest" win period parsing;
  2. refusal + a period filter → twice the excerpts from the same filing;
  3. company named, no filter → company-only scope, twice the excerpts;
  4. otherwise → twice the excerpts.

  The retry trace carries `retry_of` / `retry_reason` / `retry_prev_ctx`, and its audit section
  says **"evidence changed: ctx a → b"**, or, in **orange**, **"same evidence as the original"**.
- **`rstash --replay [wrong,partial]`** re-asks every answer with that latest verdict, links each
  new trace to the original, and prints a summary (evidence same/CHANGED, answer same/changed).
  This is the regression check after any pipeline change.
- CLI: `rag_audit review ID g|w|p [note]` · `rag_audit reviews [wrong|partial|good]` ·
  `rag_audit retry-plan ID [note]` (JSON; RQ/SP use it) · `rag_audit record --print-id`.

**Verification:**
- Tests: command grammar, verdict supersede, retry-plan rules (rag_audit, 34 tests total).
- **Parity 45/45**: the 3 new retry plans (wider context, company scope, company scope + series)
  are identical in Python and Rust.
- **Live (pty, rstash):**
  1. "Tesla total revenue, most recent quarter" → $28,236M (Q2 2026).
  2. `w: I wanted the March 2026 quarter` → recorded.
  3. `r: quarter ended March 31, 2026` → the period filter switched to the 2026-03-31 filing;
     *evidence changed f3f2cbcc → 1002dc31*; answer $22,387M.
  4. `g` → recorded.
- `rstash --replay wrong` then re-asked the original and correctly reported
  **evidence same, answer same**, orange in the audit. Nothing in the pipeline had changed.

### Step 7 — Claude triage + blind human review queue (P3, 2026-09-25)

- **`rag_audit triage`** (offline batch; menu **`AQ`** runs it, then the queue): Claude reviews every
  untriaged answer's full trace (question, excerpts, answer, audit facts) and returns a verdict
  (supported / partially_supported / unsupported / correct_refusal / false_refusal / stale),
  per-claim support, needs_human, priority and a reason → `review.claude` + `triage.jsonl`.
  - **Safety:** `claude -p --tools "" --strict-mcp-config`, run from a temp dir with stdin
    closed. The excerpts are untrusted text; the call can judge but not act. Output is drained on
    threads (no pipe deadlock) and there's a 240 s timeout.
  - **Code overrides force a human whatever Claude says:** answer written by Claude (self-review),
    corpus behind EDGAR, series gaps, **evidence older than the newest indexed filing** (the audit's
    orange date comparison, now one shared function), figures found in no excerpt, unmatched citations.
- **`rag_audit review-queue` (blind):** Claude-flagged answers by priority, plus a deterministic
  ~20% of *unflagged* answers (to measure Claude's misses), plus a few rechecks of answers already
  reviewed (test-retest), interleaved so the reviewer can't tell which is which. **Claude's
  verdict is shown only after the human's.** Disagreements go on a second-look list. `e` = full
  excerpts, `s` = skip.
- **`rag_audit triage-stats`:** agreement, flag precision, the miss rate on the unflagged sample,
  recheck consistency, and the second-look list.

**First P3 measurement, with no new labels:** real Claude triage of the 40 P1b answers vs Fraser's
blind P1b labels (`eval/triage_vs_p1b.py`):

| | |
|---|---|
| exact agreement (good/partial/wrong) | 28/40 (70%) |
| flag precision (flag confirmed wrong/partial) | **6/7 (86%)** |
| false alarms on answers judged good | **1/25 (4%)** |
| recall of wrong/partial answers | **6/15 (40%)** |
| miss rate unflagged vs flag hit rate | 27% vs 86% → **P3 rule PASS** |

**Reading it honestly: the flags are trustworthy, but recall is weak, and the misses are a
definition difference, not carelessness.**
- 6 of the 9 misses are refusals that Claude judged "correct refusal: not in these excerpts".
  That's exactly its instruction. Fraser judged them *partial* because the figure should have been
  findable, and the XBRL fact sheets later proved it was.
- 1 miss is Tesla "latest" (q01): Claude can't see that the corpus held a newer filing. The live
  audit has that fact now, and it's wired in as the stale-evidence override above. It can't be
  scored on P1b, whose traces predate the freshness facts.

Post-hoc check (same labels, so disclosed as such): **routing every refusal to a human** would lift
recall to 12/15 but raise false alarms to 15/25, because many refusals were correct. **Not
adopted.** The next real P3 numbers come from `AQ` over live answers, which now carry freshness,
period-filter and series facts.

Tests: 3 triage tests with a fake Claude (`RAG_AUDIT_CLAUDE_CMD`), covering JSON extraction,
overrides, the end-to-end flow, the queue (flagged / sample / recheck), blind recording and
agreement, rechecks never overwriting the first verdict, and stats. The queue UI was pty-tested on
a scratch copy: Claude hidden until after the verdict, then revealed; the disagreement landed
on the second-look list.
- **False alarm fixed (Fraser, 2026-09-25):** "cash flow versus revenue for TSLA during the last 3
  years" got an annual series (FY2023–FY2025, current) but the orange line still said *evidence is
  older than the corpus*. The check compared the series' end (the FY2025 10-K, 2025-12-31) with
  the newest filing of ANY form (a 2026 10-Q, a partial year). A series always ends at the newest
  period available **at its frequency** (only a dated question anchors it earlier, and those are
  exempt), so any series for the company now means "current".
  - The same fix applies to triage's override through the shared `section::stale_evidence`.
  - Annual tables now add *"Note: the latest filing is a 10-Q for the period ending 2026-06-30;
    the fiscal year after 2025-12-31 is still in progress"*, so FY2025 isn't mistaken for the
    latest data of any kind.
  - Tests: +1 in section, +1 in series. Parity 45/45.
- **Segment-subset bug fixed (found from Fraser's TSLA D&A table, 2026-09-25):** the model's
  column said "Depreciation and Amortization (Automotive segment)". It was right: Tesla tags its
  consolidated D&A as `DepreciationAmortizationAndImpairment`, which wasn't in the metric's tag
  list, so the series fell back to a segment-dimension tag and showed **Automotive-only D&A as
  Tesla's total**.
  - (a) Added the tag. Its own label says "…and impairment".
  - (b) **A dimensional value may stand in for the total only when it's the sole member of its
    axis** (a one-segment company, like LLY's "Reportable segment"). With several members, any one
    is a subset, so the metric is reported as "not reported" instead. TSLA D&A is now $1,348M
    Q3 2024 (consolidated; was $938M Automotive).
  - Tests: subset rejected, single segment allowed. Oracle 231/232, parity 45/45, unchanged.
  - Also fixed a **flaky test harness**: tests shared temp files across parallel threads; each
    call now gets a fresh directory, 5/5 clean runs.
- **Table alignment (Fraser):** the model often writes Markdown tables, and the terminal prints
  them raw, so a `*` or a wider figure misaligned the columns. `rag_audit::format::align_tables`
  (rstash links it; RQ/SP call `rag_audit align`) pads columns to the widest *visible* cell,
  right-aligns amounts (also when a citation sits inside the cell), and left-aligns dates so a
  footnote `*` trails them. It runs before citation linkification: links add no visible width.
  The audit still records the raw answer. 4 format tests.
- **Series table display (Fraser):** trend answers came back as bullets that re-typed every figure
  with the same sentence frame. Now the code prints the series itself (`sec_xbrl::table::render`;
  rstash links it, RQ/SP get it as `display` from `sec_xbrl series [--color]`): a boxed table,
  periods as rows **newest first** (Fraser's choice), unit once in the title ($ millions), period
  labels by month ("Jun 2026", not "Q2": fiscal ≠ calendar quarters), markers in their own slot
  (° derived from YTD, ¹² tag substitution, † EBIT fallback), and a trend row under each column,
  oldest → newest, **scaled from zero** (min-max drew D&A's +15% as ▁▇█▆▇; zero-based shows it
  nearly flat, which is honest; a column with a negative value gets no bar). When a series is in
  context the prompt says the table is already shown: commentary only, ≤4 sentences, no Markdown.
  `format::for_terminal` tidies any remaining Markdown (bullets → •, bold). Found on the way: the
  citation linker treated the `[` of an ANSI code as a citation and linked "(TSLA)" in the title
  — fixed in term_links.rs and term_links.py. The series *text* passage is unchanged, so the
  oracle and parity gates are unaffected (parity 45/45 re-run). Tests: sec_xbrl 21, rag_audit 42.

- **Missing companies (Fraser, SP):** "how has JNJ earnings and expenses been during the last five
  years?" — JNJ isn't in the corpus, so detection (corpus names only) found nothing, retrieval
  returned IBM/COUR/TEAM, the model refused, and nothing offered a download; `r: you should
  download the SEC filings needed` only re-searched (the note led the query and pulled LLY).
  Now: `freshness.missing_companies` / `rstash_core::edgar::missing_companies` flag a company the
  question names — an ALL-CAPS SEC ticker (minus a stoplist of acronyms that are also tickers:
  AI, IT, EPS…) or a full 2+-word SEC name ("Johnson & Johnson") — that EDGAR lists but the corpus
  lacks. It shows as an orange line before the answer and an orange `coverage` fact in the audit
  (bundle field `missing`; Claude triage is told and routes it to a human). The `a` key runs
  `sec_topup.py --add --history 20` (new: `--add` accepts companies not in the corpus), then reloads
  and re-asks; RQ and SP gained `u`/`h`/`a` too (they re-exec themselves). A `r:`/`w:` note asking
  to download points at the right key. Verified: 13 shared cases on both sides; 90/90 identical
  on the real SEC map; 0 false alarms on all 71 real questions so far; retrieval parity 47/47.
  Known limit: one-word names (Merck, Costco) aren't matched by name — use the ticker.
- **rstash JSON escape bug (found on the way):** `extract_json_string` pushed the bare `u` of
  `\uXXXX`, so Ollama answers showed "Ru0026D" for R&D; decoded properly now (incl. surrogate
  pairs), unit-tested. 1 of 34 stored rstash traces was affected (the test question itself).

## Status — 2026-09-25: daily-driving phase

Steps 0–7 are live in rstash / RQ / SP (audit trace and section, period filter, SEC top-up and
`u`/`h`, XBRL fact sheets, trend series, feedback / retry / replay, Claude triage and the blind
queue), plus the day-2 fixes from Fraser's own use: ratio convention, interest coverage, the
annual-series false alarm, segment subsets, table alignment, series table display.
- **Fraser now daily-drives rstash** to surface bugs and suggestions.
- **The full `SU` run is deferred**; the per-company `u` path is being judged in real use first.
- **Open items:** `AQ` over live answers (22 queued) for the first live P3 numbers ·
  "right filing, wrong chunk" for narrative MD&A questions · P1c re-label only when things settle.
- **Gates** (re-run after any change): xbrl_parity (byte-identical), period_parity + rstash-core
  cases (18), period_rust_parity (45), series_oracle (231/232 vs SEC), cargo tests (rag_audit 41,
  sec_xbrl 18), then `rstash --replay`.
