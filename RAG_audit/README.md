# The audited RAG system — RQ / SP / rstash (built 2026-09-24 → 25)

**Start here.** This is the overview of what changed in two days, how the pieces fit, how to use
them day to day, and what the evidence says. The phase-by-phase lab record (pre-registrations, every
result, including the failures) is [`LAB_PLAN.md`](LAB_PLAN.md).

Prompted by *"Beyond RAGs: Building Actually Truthful AI Harnesses"* (Towards Data Science). Its
thesis is that retrieval gives a model *candidates* for evidence, not proof. The day started as
an audit module and ended as a different RAG system. The thesis we adopted (Fraser's): **accuracy
alone isn't what gives RAG its value; the infrastructure for auditing and double-checking it
is.** Every improvement below was found *by* that infrastructure.

---

## 1. Before → after

| | Morning | Evening |
|---|---|---|
| What an answer rests on | unrecorded | a replayable trace per answer: chunks + hashes, prompt, model, index fingerprint, retrieval decisions |
| "Latest / most recent" questions | newest filing in context **5/14** (all 7 WRONG answers were these) | **14/14**, and the excerpts come only from the target filing |
| Specific-quarter questions | right filing, but the figure absent in 6/13 | figures present: **18/18** previously-failing questions now answered from grounded figures |
| What was indexed from a 10-Q | MD&A text only (~23% of the filing) | MD&A **plus every inline-XBRL figure** (statements, notes, segments), 69K passages |
| Trend / "X vs Y over time" | impossible: 6 similarity slots, one metric | a **deterministic time series** (Q4 derived, ratios computed) in slot 1 |
| Stale data | silent | **orange** in the audit section; `SU` tops up; rstash `u` / `h` do one company |
| Corpus freshness vs EDGAR | 28/28 sampled companies behind, unknown to anyone | checked per question; AAPL / TSLA / LLY / KO topped up live |
| Python vs Rust | two ports that had quietly diverged | held to **identical output** by parity gates, which found 3 real bugs |
| Corpus | ~219K chunks | **291,487** chunks, all embedded |
| When an answer is wrong | nothing to do but re-ask | `w: note` records it · `r: note` **retries with changed evidence** · `rstash --replay` re-checks every wrong answer after a change |
| Who checks the answers | nobody | **Claude triages every answer** (tool-less); you review a **blind** queue (`AQ`); agreement and misses are measured, not assumed |
| Ratios and tables | model's own arithmetic and raw Markdown | computed in code, **reporting conventions** (smaller as % of larger; coverage as a multiple ×); tables **column-aligned** in the terminal |

## 2. How a question flows

```
question
  │ company detection  (ragstash rag_query.detect_ticker / rstash-core detect_ticker)
  │    fixed today: "Eli Lilly" (second-word fallback), full names ("advanced micro devices"),
  │    share classes ("alphabet" → GOOG + GOOGL)
  ├─ period intent     (ragstash/period_intent.py ↔ rstash-core period_intent.rs)
  │    "latest quarter" → newest 10-Q · "quarter ended Sep 27, 2025" → that filing · "fiscal 2024" → that 10-K
  │    → ALL context slots from the target filing(s); that company's other filings dropped
  ├─ trend intent      (sec_xbrl/src/series.rs, one Rust implementation)
  │    "interest expense vs revenue over recent quarters" → XBRL time series across every cached
  │    filing → context slot 1
  ├─ retrieval         RQ: brute-force f32 (GPU) · SP: sparsebridge ANN · rstash: GPU-exact int8
  ├─ series table      (sec_xbrl::table) trend figures printed BY CODE: boxed, newest quarter first,
  │                    ° derived / ¹² tag markers, zero-based trend bars (old → new) under each column
  ├─ LLM answer        commentary only when a series table is shown (prompt: don't restate the figures,
  │                    ≤4 sentences, plain prose); Claude CLI or Ollama
  ├─ terminal tidy     (rag_audit::format::for_terminal) * / - bullets → •, **bold** / headings → bold,
  │                    Markdown tables padded, amounts right-aligned
  └─ rag_audit         trace + Tier 0 facts + freshness vs EDGAR → the audit section under the answer
                       → you:  g/w/p = rate · r = retry with changed evidence
                               u = update this company from EDGAR · h = fetch its filing history
                               a = add a company the corpus doesn't have (rstash, RQ and SP alike)
                       → later, offline: Claude triage → your blind review queue (AQ) → triage-stats
```

Data side:
```
EDGAR ──fetch──▶ HTML caches (sec_10q/cache, sec_riskfactors/cache)
                   ├─ export_sec10q/10k.py --append ─▶ SEC10Q.DAT / SEC10K.DAT   (MD&A, risk factors; COBOL reads these)
                   └─ sec_xbrl (Rust) ────────────────▶ SEC10QXBRL.DAT / SEC10KXBRL.DAT (every tagged figure)
                                                          │
                     seedverify embed (RX) ◀──────────────┘   readers: ragstash sec_reader.py · rstash-core sec.rs
                                                                        (gq_search inherits) · seedverify
```

## 3. Daily driving

**The loop:**
1. **Ask** in rstash (`rstash` in a shell, or `CT` from the menu).
2. **Read the audit section** under the answer.
   - Orange means a date problem: the corpus is behind EDGAR (press **`u`**), a "latest" answer
     came from old filings, or a trend table has gaps (press **`h`**).
   - Orange `coverage` means the company isn't in the corpus at all (press **`a`**).
   - `series` lines tell you a computed table was used.
3. **Rate it when you notice something**: `g`, `w: what was wrong`, `p: what was missing`. Every
   rating becomes a label; that's how the system gets measured without hours of labelling.
4. **Wrong? Retry with a hint**, e.g. `r: quarter ended March 31, 2026` or `r: use the 10-K`. The
   note steers retrieval, and the audit says whether the evidence actually changed.
5. **Now and then**, run `AQ`. Claude triages everything new, and you review what it flagged plus
   a hidden sample of what it didn't, blind. `rag_audit triage-stats` shows whether Claude's flags
   can be trusted.
6. **After any change to the pipeline**, run `rstash --replay` to re-ask everything you marked wrong.

**Keeping SEC data fresh (Fraser's choice, 2026-09-25):** no full `SU` run for now (1–2 h). When
an orange "corpus is behind EDGAR" appears, press **`u`** to top up just that company (a minute
or two, then reload and re-ask). Note what that process is like; it's the thing to judge before
deciding whether a periodic full `SU` is worth it.

**Asking trend questions:** name the company, the figure(s) and a trend word. The series step
then builds a computed table from every cached filing:
- *"LLY revenue versus interest expense over the last 5 quarters"* → both series plus **interest
  expense as % of revenue**. Ratios are always the smaller figure as % of the larger, whatever
  the word order.
- *"How has NVIDIA's gross margin trended?"*, *"operating margin"*, *"net margin"* → the ratio to revenue.
- *"Eli Lilly interest coverage over the last 5 quarters"* → operating income ÷ interest, as a
  **multiple (×)**. Where a company reports no operating-income line, EBIT is approximated as
  pre-tax income + interest and marked **†**.
- *"Microsoft free cash flow over the past two years"* → operating cash flow − capex.
- *"… over the last 3 years"* / *"annual"* → a **fiscal-year** table (it notes when a newer
  partial-year filing exists). *"… last 12 quarters"* → quarterly. *"… up to the quarter ended
  March 31, 2025"* ends the window there. *"growth" / "a year earlier"* adds **YoY %**.
- **Figures it knows:** revenue, cost of revenue, gross profit, operating income, pre-tax income,
  net income, interest expense, R&D, SG&A, operating expenses, income tax, operating cash flow,
  capex, dividends, buybacks, D&A, EPS, diluted shares, cash, long-term debt, total assets,
  liabilities, equity, inventory.
- **Reading the table:**
  - `*` = derived from year-to-date totals (Q4 = fiscal year − 9 months).
  - `[1]`/`[2]` = which XBRL tag a value came from, when a figure needed more than one.
  - *"Not reported"* = the company doesn't tag that figure (a segment subset is never
    substituted for a company total).

## 4. Using it: reference

**Menu (RUSTMM / COBOLMM):** `RQ` ask (Python, brute force) · `SP` ask (sparsebridge ANN) ·
`CT` rstash · `RX` embed anything new · **`AQ` audit review queue** (Claude triages, then you review
blind) · **`SU` top up SEC filings from EDGAR** (plan → confirm →
backed-up appends → fact sheets → embed; a full run is ~60 companies / ~115 filings / 1–2 h;
run it on one machine, then `RX` on the other).

**rstash** (`rstash` in a shell = the menu's `CT`; it reads `~/machine_learning/rstash/config.env` itself):
- prompt `Question (q=quit, g(ood)/w(rong)/p(artial)[: note]=rate, r(etry)[: note], u=update TSLA from EDGAR, h=fetch KO filing history):`
- `g` / `w` / `p` [`: note`]: rate the last answer (labels collected in normal use → `rag_audit reviews`)
- `r` [`: note`]: **retry with changed evidence** (your note leads the question, e.g. `r: quarter ended
  March 31, 2026`; with no note: more excerpts, or that company's filings only). The audit says whether
  the evidence changed.
- `u`: top up just the named company when it's behind EDGAR → reload (~11 s) → re-ask
- `h`: back-fill older filings when a trend series has gaps → reload → re-ask
- `a`: add a company the question names but the corpus lacks (a ticker like `JNJ`, or a full
  name like "Johnson & Johnson"): latest 10-K + up to 4 10-Qs of text, 5 years of XBRL → embed →
  reload → re-ask (`sec_topup.py --tickers JNJ --add --history 20` by hand)
- A `r:`/`w:` note that asks to *download* filings points you at `a` / `u` / `h` instead — a retry
  only re-searches what is already indexed.
- u / h / a work the same in rstash, RQ and SP (Python re-executes itself after the top-up).
- `rstash --replay [wrong,partial]`: re-ask everything you marked wrong/partial. Run it after any change.

RQ and SP have the same `g/w/p/r` keys (same rules, via the `rag_audit` CLI).

**Switches** (env or `rstash/config.env`): `RAG_AUDIT=off` · `RAG_AUDIT_SECTION=off` (one-line
footer) · `RAG_AUDIT_FOOTER=flags` (show the unvalidated Tier 0 flags) · `RAG_PERIOD_FILTER=off` ·
`RAG_XBRL_SERIES=off`.

**Tools:**
- `rag_audit list | show [last|id] | compare A B | section [id] | check [id|all]`
- `rag_audit review ID g|w|p [note] | reviews [wrong] | retry-plan ID [note]`
- `rag_audit triage [--limit N] | review-queue | triage-stats`: Claude triage (tool-less), the blind queue, the P3 numbers
- `sec_xbrl show TICKER [PERIOD]`: a company's fact sheet · `sec_xbrl series --tickers LLY --query "…"` · `sec_xbrl rebuild`
- `python3 ragstash/sec_topup.py --tickers LLY [--history 12] [--dry-run]`

## 5. Reading the audit section

```
── audit ────────────────────────────── 20260925T044024Z-rstash-2259a3d0
evidence  KO 10-Q 2025-09-26 ×1 (1 XBRL) · … · xbrl-series ×1     what the answer was built from
          period filter KO → 2026-07-03 (most recent 10-Q)        what the period filter chose, and why
freshness KO newest indexed 10-Q 2026-07-03 · EDGAR 10-Q 2026-07-03 (filed 2026-07-29)
series    KO quarterly 2024-09-27 → 2026-07-03 · 8/8 quarters complete · 14 filings · 2 derived (*)
          Revenue
answer    9 claims, 8 cited · 16 figures: 16 in the cited excerpt · citations match excerpt labels
retrieval rstash GPU-exact · 6 chunks · spread 0.006 · ctx 10f3207b
```
- **Facts, not verdicts.** Every line can be checked. The flags the lab tested (P1, P1b) failed
  their pre-registered bars, so they stay opt-in.
- **Orange = a date comparison:**
  - the corpus is behind EDGAR (→ `SU` / `u`);
  - a "latest" question was answered from filings older than the newest indexed one;
  - a trend series has gaps (→ `h`).
- **Orange `coverage`:** a named company isn't in the corpus at all (→ `a`). The refusal that
  follows is a corpus gap, not a model error; Claude triage routes it to you.

  An explicitly dated question never goes orange: an old filing is what was asked for.
- `*` in a series = derived from year-to-date totals (Q4 = fiscal year − 9 months). `[1]`/`[2]`
  = which XBRL tag each value came from, shown when a metric had to use more than one.
- `series` = a computed XBRL table was used (company, quarterly/annual, window, `k/N complete`,
  derived count). Orange if the window has gaps (→ `h`). A series always ends at the newest period
  *for its frequency*, so an annual table ending at the last full fiscal year is current.
- `retry` = this answer re-asked an earlier one: "evidence changed: ctx a → b", or orange "same
  evidence as the original".
- `ctx` is a fingerprint of exactly which chunks the LLM saw. The same fingerprint on RQ and
  rstash means they answered from identical evidence.

## 6. The evidence

| Check | Result |
|---|---|
| Same question on RQ / SP / rstash (P0) | identical context fingerprint on all three |
| Tier 0 flags, 80 hand labels (P1) | **FAIL** (pre-registered): 1/3 WRONG caught. All 3 WRONG answers were *false refusals* |
| Tier 0 flags, 40 fresh blind labels (P1b) | **FAIL**: every flag set. Found the real problem: retrieval ignores dates |
| Period filter (36 date questions, no LLM) | latest 5/14 → **14/14** · older-quarter noise 159 → 0 chunks · 10-K-only-for-quarterly 3 → 0 |
| Detector fix (measurement_lab sets) | company recall 192 → **200/200**, no new false firings on 550 ordinary questions |
| XBRL fact sheets, Python vs Rust | **byte-identical** on all 407 filings, first run |
| XBRL on the 18 P1b "figure absent" questions | **18/18** now answered with ≥1 figure in the cited excerpt (spot-checked: LLY interest $345M, Intel GM 27.55% computed, AWS $10,160M) |
| Trend series vs SEC company-facts (independent oracle) | **231/232 agree to $1M** incl. every derived Q4; 1 disclosed tag substitution; **0 disagreements** |
| Python vs Rust retrieval parity | **42/42** questions: detection, intent, filtered rows, freshness, series |
| Table alignment | 4 format tests (aligned widths, amounts right incl. cited cells, dates left, multi-byte) |
| Series table + tidy | 3 table tests (newest first, every row the same visible width with/without colour, zero-based sparkline, number/period formats) · 1 tidy test · linker ignores ANSI `[` (Rust + Python test) · parity still **45/45** · live on rstash and RQ (TSLA, LLY) |
| Missing-company detection | 13 shared cases (Python + Rust) · **90/90** questions identical on the real SEC map (10,412 tickers), **0 false alarms** on the 71 real questions asked so far · retrieval parity now **47/47** (+2 missing-company questions) |
| Live loops | `u` on TSLA (reload 10.8 s, Q2 2026 answer) · `h` on KO (10-K-only → 8/8 quarters) · `w`→`r:` on TSLA (evidence changed, Q1 answer) |
| Claude triage vs the 40 blind P1b labels (P3) | flag precision **6/7**, false alarms **1/25**, recall 6/15, miss 27% vs flag hit 86% → **PASS** (recall weak; see LAB_PLAN step 7) |

**Bugs the verification infrastructure found** (none were visible from the answers alone):
- *Ported Python ↔ Rust behaviour, caught by parity:*
  - rstash picked GOOG *or* GOOGL at random for "Alphabet" (HashMap order);
  - rstash's generic-word list was missing 9 of Python's words;
  - rstash in a shell silently ran the slow CPU path (it never read its own config).
- *Retrieval, caught by the audit:*
  - dates ignored by similarity search;
  - quarterly questions answered from 10-K-only excerpts;
  - "Eli Lilly" never detected;
  - the MD&A-only corpus.
- *Correct refusals that looked like failures:* the LLY interest-expense figure was genuinely not in the index.
- *Series, caught by tests and the oracle:*
  - "trended" wasn't recognised;
  - bare "sales" turned "gross merchandise sales" into revenue;
  - Ford's substituted net-income tag.
- *Caught by Fraser while daily driving (day 2):*
  - "revenue *versus* interest" printed a 9,800% ratio (word order) → reporting convention;
  - interest coverage was n/a for LLY (no operating-income line) → EBIT fallback;
  - an annual table was flagged stale against a partial-year 10-Q → a series is current at its
    frequency;
  - **Tesla's "D&A" was Automotive-segment D&A** (a subset standing in for the total) → segment
    values are allowed only when they're the company's only segment;
  - misaligned tables → aligned;
  - a JNJ question (JNJ not in the corpus) got unrelated excerpts and a refusal, with no way to
    fetch JNJ; `r: download the filings` just re-searched → missing-company detection, the `a` key
    in all three tools, and a hint when a note asks for a download;
  - rstash's JSON reader decoded `\u0026` as `u0026` ("R&D" → "Ru0026D" in Ollama answers) → fixed;
  - bullet answers that re-typed every figure ("Revenue was … and D&A was …" ×5, `*` everywhere) →
    code prints the series as a boxed table (newest first, zero-based trend bars) and the model
    writes a short commentary; Markdown bullets/bold tidied for the terminal.
- *A flaky test* (parallel tests sharing temp files) → a fresh directory per test.

## 7. Lessons worth keeping

1. **A retrieval win is not an answer win; an answer failure is often a retrieval (or indexing)
   failure.** The biggest gains came from *what was indexed and which filing was chosen*, not
   from the LLM or the prompt.
2. **Show facts, not unvalidated judgments.** Two pre-registered flag evaluations failed. The
   date comparisons and the evidence breakdown are what made problems visible at a glance.
3. **Similarity search has no concept of date, of "latest", or of "a series over time".** Those
   need deterministic steps (period filter, XBRL series) in front of or beside it.
4. **Compute, don't ask the model to, and follow reporting conventions when you do.** Q4
   derivation, ratios and YoY are done in code, and the LLM reads them. Ratios are stated
   smaller-as-%-of-larger ("interest expense 1.5% of revenue", never "revenue 6,659% of interest").
   Interest coverage is a multiple (EBIT ÷ interest; EBIT ≈ pre-tax income + interest where no
   operating line exists, marked †).
5. **Parity gates find real bugs.** Byte-level and decision-level comparisons between the two
   implementations caught behaviour differences no answer ever revealed.
6. **Verify against an independent source** (SEC company-facts), not only against a twin of your own logic.
7. **Human labelling is expensive** (P1b took hours) → automatic metrics wherever possible
   (target-period-in-context, figures-in-cited-excerpt, oracles); labels only for what nothing
   else can judge, and collected **during normal use** (`g/w/p`) rather than in sessions.
8. **Daily use finds what tests don't.** Four of day 2's fixes came from reading real answers
   (a 9,800% ratio, a subset standing in for a total). The audit section made each one visible
   enough to question.
9. **A second reviewer is only useful if you measure it.** Claude's flags were 86% precise but
   caught only 40% of bad answers, mostly because "correctly refused" and "should have been
   findable" are different questions. Blind review plus a hidden sample of unflagged answers is
   what exposes that.

## 8. Where things live

| Piece | Path |
|---|---|
| Audit module | `~/machine_learning/rag_audit/` (Rust; `python/rag_audit_client.py` for RQ/SP): `store` trace · `tier0` checks · `section` audit section · `feedback` g/w/p + retry plans · `triage` Claude triage + blind queue · `format` table alignment · `review_queue.sh` (menu `AQ`) |
| XBRL fact sheets + trend series | `~/machine_learning/sec_xbrl/` (Rust, production) · `ragstash/xbrl_facts.py` (Python reference) |
| Period filter | `ragstash/period_intent.py` · `rstash-core/src/period_intent.rs` · shared cases `rag_audit/eval/period_intent_cases.json` |
| Freshness vs EDGAR | `ragstash/freshness.py` · rstash `freshness_facts()` (shared EDGAR cache `~/.cache/ragstash/edgar`) |
| SEC top-up | `ragstash/sec_topup.py` (menu `SU`, `sec_topup.sh`) |
| RQ / SP / rstash | `ragstash/rag_query.py` · `sparsebridge/sb_query.py` · `sparsebridge/rstash-rs/` (run from `~/machine_learning/rstash/`) |
| Rebuild on the other machine | `rag_audit/rebuild.sh`: builds sec_xbrl + rag_audit + rstash, runs their tests, prints a TSLA table (build outputs don't sync) |
| Eval scripts | `rag_audit/eval/`: `period_metric.py`, `period_rust_parity.py`, `xbrl_parity.py`, `series_oracle.py`, `xbrl_eval.py`, `triage_vs_p1b.py`, `p1b_*` |
| Your ratings / Claude's reviews | `~/.cache/rag_audit/reviews.jsonl`, `triage.jsonl` (and inside each trace: `review.human`, `review.claude`) |
| Data | `SEC10Q.DAT`, `SEC10K.DAT` (+ `…XBRL.DAT`) under `COBOL/main_menu/sec_*/data/` · traces `~/.cache/rag_audit/` · series index `~/.cache/sec_xbrl/index.tsv` |

**Re-run the gates after any change:** `python3 eval/xbrl_parity.py` · `python3 eval/period_parity.py`
+ `cargo test -p rstash-core` · `period_metric.py --dump py_period_ctx.jsonl` then
`period_rust_parity.py` (45 questions) · `series_oracle.py` (232 values vs SEC) · `cargo test` in
`rag_audit/` (42 tests) and `sec_xbrl/` (21) · then `rstash --replay` for your wrong answers.

**Desktop:** code and DAT files sync (Synology); embeddings and caches don't. Rebuild `sec_xbrl`,
`rag_audit`, `seedverify`, rstash (and `gq_search` in `fedora43-oxide`, or use the synced
binary), then `RX`.

## 9. Status and next (2026-09-25)

**Phase: daily driving.** Fraser uses rstash day to day to surface bugs and suggestions; fixes
land as they're found (four already on day 2).

- ~~Feedback / retry / replay~~ **done (step 6)**. Labels accumulate in `reviews.jsonl`.
- ~~Claude triage~~ **done (step 7)**: first P3 measurement vs the P1b labels. Flags 86% precise,
  4% false alarms, recall 40% (misses are mostly "correct refusal vs should-have-been-findable").
  **Next real numbers:** `AQ` over live answers. **22 answers are waiting in the queue.**
- **Full `SU` run: deferred by choice.** Use `u` per company when orange appears, and judge that
  experience first. The other machine still needs `RX` after anything is topped up here.
- **"Right filing, wrong chunk"** for MD&A-only questions (Intel gross margin was fixed by XBRL,
  but narrative questions can still miss the right passage).
- **P1c re-label** of fresh answers (only once the above settle; labelling is costly).
