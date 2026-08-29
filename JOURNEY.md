# The Journey — from a slow radix sort to an uncertainty-aware workstation

*Written 2026-08-29, at Fraser's request, as a narrative counterpart to `PROJECTS.md`
(which is a map, not a story) and to `hybrid_resilience_lab/ARC.md` (which tells one
project's story in depth). This one is the whole arc, at low resolution.*

---

## 1. The origin, and why it mattered

It started with **radix sort** — months of it, hand-written, learning GPU programming the
hard way, and at the end it was **correct and slow**, comprehensively beaten by Thrust.

*(Sourced from Fraser's own account: this predates `PROJECTS.md`, which begins around
July 2026 and has no record of it. The gap is real and worth knowing about — the map
misses the whole first era. Noted also in the graph_lab memory.)*

It is tempting to file that as a failure. It isn't, and the rest of this document is the
evidence. Two things came out of those months that every later project spends:

- **A working mental model of the machine** — memory hierarchy, occupancy, launch
  overhead, where a GPU actually loses. Every later win is an application of that model.
- **The calibration to lose to a library and keep going.** Being destroyed by Thrust is
  the correct outcome for a first attempt at a solved problem, and knowing what that feels
  like is what makes it possible to later ask "is beating the library actually plausible
  here?" — which is the question the whole portfolio now runs on.

## 2. The turn: the Tensor Core Engine

Then the **Tensor Core Engine**, over 15-plus iterations, to v5.1. This is where the
approach changed from "implement the standard thing well" to **"re-think the standard
approach for an advantage the standard approach structurally cannot have."**

The differentiator is not tuning. It is **Ozaki/Dekker split-precision GEMM** — running on
TF32 tensor cores, then recovering FP32→near-FP64 accuracy (~1e-6…1e-7) from the splits.
Stock CuPy cannot do that, because it isn't a matter of doing the same computation faster;
it is a different decomposition of the arithmetic. Plus fused cuBLASLt NN epilogues.
21/21 smoke tests vs CuPy/NumPy on two cards.

And, just as important, **a stated weakness**: FFT-bound and sparse workloads, where
CuPy/cuFFT already win. The engine came with a documented sweet spot instead of a claim
of general superiority.

## 3. Acceleration: MPDOK, and the four-point litmus test

**MPDOK** (GMRES-IR / LU-IR) generalized the same trick from multiplication to *solving*:
bulk work in low precision, a cheap high-precision correction on top. This is where the
portfolio started producing results that were **both faster and more accurate than the
SciPy/FP64 baseline** — not a speed/accuracy trade, which is the usual shape, but a win on
both axes at once. That is only available if you re-derive the algorithm rather than
optimize the standard one.

Then the crucial piece of infrastructure — not code, but a **decision rule**. `gp_engine/PLAN.md`'s
**four-point litmus test**, which a project must pass before the engine is worth reaching for:

1. dominant cost is dense GEMM / dense solve;
2. compute-bound, not bandwidth/launch-bound (n ≳ few thousand);
3. needs better than TF32 accuracy;
4. GPU-resident across many ops.

Evidenced by the portfolio itself: BEM scattering felt great, **cryo-ET (FFT) was the trap**.
This is the radix-sort lesson, formalized. Roughly 30 MPDOK application labs followed, and
the test is why so many of them landed.

## 4. Breadth: the labs

`gp_engine` — exact mixed-precision GP regression, **12.6× vs FP64 cuSOLVER at n=27k**, and
**n=100,000 exact GP fit in 5.4 minutes on an 8 GB card**, where the 40 GB kernel matrix
never exists at all. `tiny_pointers` and its descendants — `hash_join_engine`, `gpustash`,
`sparsebridge` — turning a paper's succinct-data-structure idea into working engines.
`rbfx`, `decision_engine`, `seedverify`, the `stash` family. Physics and ML labs by the
dozen.

The **soft-EM regime-mixture** mechanism recurred across four labs, and — the same pattern
as the litmus test — earned **its own** two-condition checklist: the regime must genuinely
*recur*, and must be *rare/imbalanced*. `shm_lab` is the recorded counterexample (a
one-time permanent retrofit is a change-point problem, not a latent-class one — the wrong
tool, not a broken implementation).

## 5. The maturation nobody plans for: the results got more honest

The genuinely interesting shift in the last year is not that the engines got faster. It is
that **the labs started producing trustworthy negative results, and acting on them.**

- **`vol_regime_lab`** — a domain that passed every checklist (real, recurring, 7.95% base
  rate, separation 2.68) still failed a walk-forward replication and a block bootstrap.
  Verdict recorded precisely: soft-EM beats a *misspecified* baseline everywhere, but
  "beats a plain empirical quantile" was **not established**. A clean point estimate was
  killed by its own replication check.
- **`hybrid_resilience_lab`** — set out to fix a GNN's collapse with a cheap-correct base;
  found **no such base exists there**, and shipped a *monitoring design* instead of a model.
  Its deepest finding is a structural one: **label-free drift monitoring cannot see concept
  drift, by construction** — the standard dashboards show green while the ranking dies.
- **`graph_lab` Phase 6** — the headline collapse was **corrected by a factor of ~3** by our
  own follow-up work, and both the original claim and the correction were left visible.
- **`sparsebridge` S0** (2026-08-29) — scoped a streaming-index phase, then measured the
  premise and found the corpus hadn't changed in 38 days. The real cost turned out to be a
  *missing cache* from a machine move. Stopped, and wrote down why.

The through-line: **the portfolio now spends effort finding out whether an idea is worth
building before building it**, and treats a well-measured "no" as a deliverable. That is a
larger capability gain than any single speedup.

## 6. Where it all lands: RUSTMM

The part that makes this a portfolio rather than a collection: **RUSTMM/COBOLMM is in daily
use.** ~90 menu items — SEC 10-K/10-Q scoring, CVaR portfolio, charts, RSS, transcription,
OCR, half a dozen search engines, RAG (`RQ`), sparsebridge ANN retrieval (`SP`), the
decision engine (`DE`/`DD`), MPDOK semantic search and dashboards.

This matters more than it looks. Research code that is used daily gets **corrected by
reality**: the sparsebridge signed/unsigned hash bug and the missing-hits bug were both
found by using the tool, not by testing it. A daily-use surface is the portfolio's best
bug-finder and its only real validation.

---

## 7. What the arc is actually building toward

Two themes, and the second one snuck up:

**Theme 1 — re-think the standard approach for a structural advantage.** Split-precision,
iterative refinement, tiny pointers, distillation probes. Consistent, and it works.

**Theme 2 — systems that know when they are not confident.** This was never declared as a
goal, and it is now most of the portfolio:

| piece | what it contributes |
|---|---|
| `gp_engine` | a calibrated **posterior variance**, not just a point prediction |
| `decision_engine` / `decision_harness_lab` | k-state **Bayes decision** over actions + costs, plus `voi.py` (value of information) |
| `OUTCOME_SHAPE_TAXONOMY.md` | which outcome shapes break naive decision rules, and why |
| `hybrid_resilience_lab` §4.6–4.8 | what monitoring **can** and **cannot** see, and the ~25-labels/period design |
| `vol_regime_lab` | when a confident-looking model result should **not** be trusted |
| MPDOK / IR | the same idea in arithmetic: cheap estimate + **known-quality correction** |

Iterative refinement and Bayesian decision-making are the same instinct at two levels of
the stack: **compute cheaply, know your error, and pay for precision only where it changes
the answer.**

## 8. The gap, stated plainly

**The uncertainty machinery has never been connected to a real recurring decision.**

`decision_harness_lab` stopped, deliberately and correctly, at synthetic toy examples —
Phase 3 (a real domain with a fitted posterior) was left as a separate lab and never
started. `vol_regime_lab` auditioned financial volatility for that role and returned "not
warranted." `decision_engine` is wired into the menu as `DE`/`DD` but **runs specs a human
writes by hand**; nothing feeds it.

Meanwhile RUSTMM makes real, costly, repeated decisions every day — and makes them with
hand-written rules.

See §9 for the specific one.

## 9. The application this points at: escalation in `GF`/`QX`

The SEC fast-scoring probe (`FAST_SCORING.md`) already has the exact shape of the problem:

- **GL/QL** = the expensive path (LLM on every sub-chunk, thousands of calls, hours).
- **GF/QX** = the cheap path (ridge probe on cached embeddings, minutes, zero LLM calls),
  distilled from GL/QL's own past scores, reproducing them at mean Spearman **0.840 / 0.810**.
- A **decision** between them, made per filing, with genuinely asymmetric costs.

That decision is currently made by: a **novelty gate**, a **hand-written rule table**
("a laggard dimension is decision-critical", "a genuinely new kind of filing"), and a
periodic 1-in-100 audit that retrains the probe.

Three things are already known to be wrong with that, from our own documents:

1. **The novelty gate is a covariate-shift detector.** `ARC.md` §4.6 established that this
   class of monitor is *structurally incapable* of seeing concept drift — the filing looks
   normal, the score looks normal, the relationship has changed. It will show green.
2. **One global gate covers 14 dimensions of very different reliability.** The doc's own
   caveat: the ρ≈0.84 is *in-sample*; held-out is **0.4–0.84 by dimension**. `guidance_tone`
   and `management_confidence` are not `liquidity`, and are not treated differently.
3. **The confidence signal is a distance, not a posterior.** There is no per-filing,
   per-dimension uncertainty to threshold on, so no principled place to put the threshold.

Every one of those has a part already built for it:

- **`gp_engine`** replaces the ridge probe with a GP probe: same distillation, but it
  returns **mean + variance per dimension per filing**. The variance is the missing signal.
- **`decision_engine` + `voi.py`** replaces the rule table: escalate when the **expected
  value of the information** exceeds the cost of the LLM call. That is precisely the
  question the rule table is guessing at, and `voi.py` already answers it.
- **`ARC.md` §4.8's monitoring design** replaces the audit loop's role: the 1-in-100 audit
  is *already a label budget*. Score it as precision/lift per period and alarm on lift, and
  it becomes drift detection at no extra cost.

The measurable objective is concrete and honestly falsifiable: **how few LLM calls can be
made while still matching GL/QL's rankings** — escalation rate vs. fidelity, against the
current gate as the baseline to beat.

**One honest caveat on the "uses the engines" claim.** At filing level this is *n ≈ 31
(10-K) / 270 (10-Q)* — a GP that size runs instantly anywhere and **fails the four-point
litmus test on size (point 2)**. The GP *statistics* are what earn their place here, not
the GPU work. The version that does clear the bar is **chunk level**: thousands of embedded
sub-chunks per filing across the watchlist is n in the 10⁵–10⁶ range, which is exactly
`gp_engine` Phase 2's out-of-core territory (n=100k exact, 5.4 min, 8 GB card). Worth being
clear about which claim is being made before starting.

---

## 10. Where to look

| what | where |
|---|---|
| The map (paths, status) | `claude_hub/PROJECTS.md` |
| One project's story in full depth | `hybrid_resilience_lab/ARC.md` |
| The four-point engine litmus test | `gp_engine/PLAN.md` §1 |
| The soft-EM two-condition checklist | `gp_engine/PLAN.md` §7 |
| Decision machinery | `gp_engine/decision_harness_lab/`, `decision_engine/` |
| The escalation surface in §9 | `COBOL/main_menu/SEC_gemma_analysis/FAST_SCORING.md` |
| Latest measured stop-decision | `sparsebridge/RESULTS_S0.md` |
