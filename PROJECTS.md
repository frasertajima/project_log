# Project Map

Single source of truth for where things live. Keep entries terse (path + one-liner +
status + what it's reusable for). When a project moves or a new one is born, update it here.

> **Drive note:** paths under `/run/media/fraser/ows/` live on a **removable drive** —
> may be unmounted. Paths under `/var/home/fraser/` are on the stable home partition.

## Quick lookup (keyword → path)

| You say… | It's here |
|---|---|
| tensor core engine / v5.1 / matmul engine | `~/machine_learning/fortran/examples/collected_examples/matrix_dot/tensor13/tensor_core_engine_v5/` |
| MPDOK / mixed-precision solver / GMRES-IR | `…/tensor_core_engine_v5/MPDOK/` |
| cryo-EM / denoising / v28f | `/run/media/fraser/ows/v28f_cryo_em/` |
| RBF / point cloud / spatial | `…/tensor_core_engine_v5/rbf_pointcloud/`, `…/rbf_spatial/` |
| climate / LES / downscaling | `/run/media/fraser/ows/v28d_streaming/`, `/run/media/fraser/ows/v30e_les_full/` |
| GP engine / kernel engine (new project) | `~/machine_learning/gp_engine/` (PLAN.md) |
| stash / pdfstash / mdstash / concept search | `~/stash/`, `~/pdfstash/`, `~/mdstash/`, `~/concept/` |
| COBOL menu / COBOLMM | `~/machine_learning/COBOL/main_menu/` |
| seedverify | `~/machine_learning/seedverify/` |
| ragstash / RAG glue / gpustash pipeline | `~/ragstash/` |
| sparsebridge / sparse-ID graph bridge / ANN search bench | `~/machine_learning/sparsebridge/` |
| streaming index / incremental rebuild / rebuild cost breakdown (S0 run, verdict STOP) | `~/machine_learning/sparsebridge/RESULTS_S0.md`, `STREAMING_INDEX_PLAN.md` |
| hybrid resilience / cheap base + learned correction / model drift | `~/machine_learning/hybrid_resilience_lab/` |
| the whole arc / GNN lessons / monitoring design / concept drift | `~/machine_learning/hybrid_resilience_lab/ARC.md` |
| the journey / the whole portfolio narrative / what it's building toward | `~/machine_learning/claude_hub/JOURNEY.md` |
| escalation lab / GF-QX repair / when to pay for the LLM / VoI | `~/machine_learning/escalation_lab/LAB_PLAN.md` |
| SimplePINN / v41 / zero-parameter PINN / gas dispersion | `~/machine_learning/CIFAR-10/v41_pinn_production/` |
| emergency response / leak detection / source inversion | `~/machine_learning/CIFAR-10/v36_source_inversion/`, `v37`-`v40` |
| SGS turbulence / LES / latent dynamics / PDE surrogate | `~/machine_learning/CIFAR-10/v30*_*/` |
| hybrid physics-ML solver / long-term stability | `~/machine_learning/CIFAR-10/v31_physics_hybrid/`, `v32_long_term_stability/` |
| cuDNN optimization sweep / NHWC / graph API / TF32 | `~/machine_learning/CIFAR-10/v28g_turbulence/` … `v28s_graph_api/` |
| turbulence spectral lab / neural operator correction | `~/machine_learning/turbulence_spectral_lab/` |
| quantum chemistry / FNO / LDA density correction | `~/machine_learning/quantum_chemistry/` |
| recsys / DLRM / embedding lookup lab | `~/machine_learning/sparsebridge/recsys_lab/` |
| recsys_router-rs / native Rust+CUDA router (callable from Python) | `~/machine_learning/sparsebridge/recsys_router-rs/` |
| graph_lab / GPU GNN on dynamic sparse graphs | `~/machine_learning/sparsebridge/graph_lab/` |
| backup service / FastAPI automation | `~/backup_service/` |
| cufolio / portfolio optimization (standalone) | `~/quantitative-portfolio-optimization/` |
| rbfx / rbf rust library / interpolation crate | `~/machine_learning/rbfx/` |
| Bayesian CVaR / cvar_gp_lab / GP portfolio | `~/machine_learning/gp_engine/cvar_gp_lab/` |
| climate cat risk / reinsurance / climate_cat_lab | `~/machine_learning/gp_engine/climate_cat_lab/` |
| grid reserve sizing / renewable reserve margin / grid_reserve_lab | `~/machine_learning/gp_engine/grid_reserve_lab/` |
| bridge structural health monitoring / SHM / shm_lab / KW51 | `~/machine_learning/gp_engine/shm_lab/` |
| hydrology / reservoir sizing / water security / hydro_reserve_lab | `~/machine_learning/gp_engine/hydro_reserve_lab/` |
| VoI dispatch pattern / decision.py+voi.py reuse template | `~/machine_learning/gp_engine/VOI_DISPATCH_PATTERN.md` |
| home solar/battery/HVAC / home_energy_lab / energy dispatch | `~/machine_learning/gp_engine/home_energy_lab/` |
| SVP / shortest vector / lattice / mid-point Hessian | `~/machine_learning/svp_hessian_lab/` |

(`~` = `/var/home/fraser`; `…/tensor_core_engine_v5` = the full path in row 1.)

---

## Foundations (reusable engines — build new work on these)

### Tensor Core Engine v5.1
- **Path:** `~/machine_learning/fortran/examples/collected_examples/matrix_dot/tensor13/tensor_core_engine_v5/`
- **What:** CUDA Fortran shared lib (`cuda_matlib.so`) exposing GPU matrix ops to Python
  via ctypes + CuPy arrays.
- **Differentiator vs plain CuPy:** (1) Ozaki/Dekker **split-precision GEMM** —
  TF32 tensor-core speed at FP32→near-FP64 accuracy (~1e-6…1e-7); (2) **fused cuBLASLt
  NN epilogues** (bias / bias+ReLU). Tiers: fast TF32 · split-3 · split-5 · exact FP64 · fused NN.
- **Sweet spot:** workloads dominated by **dense/batched GEMM** you can push to tensor cores.
  Weak fit: FFT-bound or sparse workloads (CuPy/cuFFT already win there).
- **Status:** v5.1 refactor complete, 21/21 smoke tests vs CuPy/NumPy (RTX 4060 + A1000).
  Key docs: `CODE_REVIEW_TENSOR_CORE_ENGINE_V5.md`, `V51_CHANGES.md`, `PLAN_V5.md`.

### MPDOK — Mixed-Precision Dense-Operator Krylov solver
- **Path:** `…/tensor_core_engine_v5/MPDOK/`
- **What:** Standalone CUDA Fortran solver (`mpdok.so`) — **GMRES-IR** and **LU-IR**
  (iterative refinement): bulk work in low precision, cheap high-precision correction.
  Independent of the tensor-core engine's `.so` (links cuBLAS/cuSOLVER directly).
- **Sweet spot:** dense linear systems where a naive CuPy FP64 solve is the baseline to beat.
- **Status:** core reviewed, all B/R/P1 findings fixed. Doc: `MPDOK/CODE_REVIEW_MPDOK_SOLVER.md`.
- **Application labs (~30):** see below.

---

## SVP / lattice algorithms lab
- **svp_hessian_lab** — `~/machine_learning/svp_hessian_lab/` (started 2026-08-13) — a "Tiny
  Pointers"-style research→working-code→reusable-engine treatment of arXiv:2608.02478v2
  ("Solving the Shortest Vector Problem in 2^0.6039n Time via Mid-point Hessian," Hhan/KAIST):
  cheap random Gaussian-sampled Hessian estimate reveals a shortest lattice vector's *direction*,
  then an expensive exact decode step recovers it — structurally the same shape as MPDOK's
  GMRES-IR (cheap bulk + expensive correction), which is why this was worth building even with no
  immediate application in this codebase (no crypto/lattice work currently active; see
  `research/RESEARCH.md`). Built module-by-module in Python first (learning-first, per Fraser's
  direction), Rust port only once the full paper pipeline (§3-§6) is validated end-to-end — plan:
  `/var/home/fraser/.claude/plans/curried-percolating-breeze.md`, phase log: `LAB_PLAN.md`.
  **Module 1 DONE (2026-08-13, §3/Thm 3.7 "one Hessian is enough if we know where to look")**:
  `prototype/common/` (hand-rolled LLL reduction, GPV/Klein discrete Gaussian sampler, brute-force
  SVP oracle, Babai nearest-plane decoder) + `module1_core_hessian/` (`hessian_direction.py`,
  `full_pipeline.py`). **Real bug caught**: dual-sampling Gaussian width guessed ad hoc twice (both
  wrong, near-chance-level signal, no crash) before reading §3.1-3.3 in full for the exact `ξ_t`
  formula — after the fix, 144/144 eigenvector-alignment trials (n=4,6,8) hit cosine similarity
  ≥0.999 and 48/48 full-pipeline trials (n=4,6,8,10) exactly recovered the true shortest vector.
  Full results: `prototype/module1_core_hessian/RESULTS.md`; reusable-technique notes:
  `REUSE_NOTES.md`. **Module 2 DONE (2026-08-13, §4 Walsh-Hadamard batching)**: removes Module 1's
  "known parity class" assumption entirely — `module2_batch_hessian/walsh_hadamard.py` (matrix-
  valued FWHT, Lemma 4.1) evaluates the Hessian estimator for all 2^n classes from one shared
  sample batch. Batched result matches a naive per-class sum to ~1e-15 (exact reindexing, not
  approximate); batching beats naive by 4.4x-**101.7x** (n=10, N_t=40k), growing with n and N_t as
  predicted. **18/18 trials (n=4,6,8) solved SVP with zero oracle information about the parity
  class** — the lab's first fully unassisted solver. `module2_batch_hessian/RESULTS.md`/
  `REUSE_NOTES.md`. **Module 3 DONE (2026-08-13, §5 random sublattice coset Hessians)**: shrinks
  Module 2's full 2^n search to 2^ell=2^(1-chi)n by sampling from one random coset of an index-2^h
  sublattice of L* instead of the whole dual lattice — worked through and implemented the paper's
  Eq. 21 identity (a coset-restricted estimate still "sees" the shortest vector's contribution with
  an unknown sign regardless of which coset was picked, so no need to guess the right one; check
  both extreme eigenvalues instead). New `common/f2linalg.py` (GF(2) linear algebra). **19/19
  trials (n=6,8,10) exactly recovered the shortest vector**, with both the search-space reduction
  (2^h) and the coset-sampling rejection overhead (~2^h) matching theory — both sides of the
  trade-off measured, not just the win. `module3_coset_hessian/RESULTS.md`/`REUSE_NOTES.md`.
  **Module 4 DONE (2026-08-15, §6.1-6.7 importance sampling + sparsification)**: samples from a
  wider, cheaper Gaussian and reweights by the importance ratio (reusing Module 3's coset sampler
  + Module 2's FWHT batching unchanged), then Bernoulli-sparsifies with inverse-probability
  reweighting to stay unbiased. **Honest finding**: the paper's efficiency motivation for wide
  sampling (their DGS degrades below a width threshold) doesn't transfer to this lab's own
  GPV/Klein sampler — no wall-clock win observed, stated plainly. What holds: accuracy comparable
  to Module 3 (cosine sim 0.95-0.99), **14/14 full solves** with importance sampling alone,
  sparsification empirically unbiased (≤1.5% relative Frobenius diff at aggressive 10%-target
  sparsification), **14/14 full solves** on sparsified samples with a real ~2x measured sample
  reduction. `module4_importance_sampling/RESULTS.md`/`REUSE_NOTES.md`. **This completes Modules
  1-4 — the paper's full non-quantum mechanism (§3-§6.7).** Module 5 (quantum, §6.8) stays
  notes-only (no quantum hardware/simulator budget). **`pipeline_full.py` DONE (2026-08-16)**:
  benchmarks Modules 2/3/4 head-to-head (Module 1 as a labeled reference floor only) plus an
  empirical chi sweep. **Honest negative finding**: at this lab's toy scale, with its practical
  GPV/Klein sampler, Modules 3-4 are SLOWER than Module 2's plain full search (n=8: 4.19s full
  search vs. 7.24s coset vs. 7.65s +importance/sparsify), and time grows monotonically with chi
  (2.96s→28.74s) instead of the paper's predicted balance point — root-caused to the coset
  sampler's real rejection overhead (Module 3's own finding) and the GPV sampler's lack of a
  narrow-width efficiency cliff (Module 4's own finding) compounding rather than paying off at
  small n. All 52 solves in this phase still succeeded — a wall-clock finding, not a correctness
  regression. `prototype/RESULTS_PIPELINE_FULL.md`. **Module 5 DONE (2026-08-17,
  `module5_dgs_sampler/`, lab-motivated, not a numbered paper section)**: Fraser asked for a real
  efficient sampler rather than rushing to Rust. Targeted the SPECIFIC bottleneck diagnosed above
  (coset rejection overhead, not Theorem 2.7's narrow-width issue) by constructing an explicit
  integer basis for the target sublattice directly — every sample generated is in-coset by
  construction, zero rejection (verified: determinant=2^h, 200/200 samples re-checked in-coset).
  **Caught a real bug along the way**: Module 3's parity-label formula (`B.T @ X`) was wrong for
  this lab's basis convention (should be `X @ B.T` — Module 2 had it right); fixed with one
  canonical `common/lattice.parity_label`. Investigated and confirmed this didn't invalidate
  Module 3/4's search-based results (a real argument: random `P` composed with any fixed
  relabeling is still random) — but Module 5's exact, non-search-based construction had no such
  robustness, which is exactly why it caught the bug. **Second independent finding**: toy-scale
  "did any of many candidates succeed" is a weak signal (512+ random directions, no Hessian, also
  succeeded 8/8 at n=8); the confound-free single-shot control (random 20% vs. Hessian-derived
  100%) confirms the mechanism carries real signal regardless. **Headline result**: re-running
  `pipeline_full.py`'s own chi sweep with the fixed sampler REVERSES the earlier trend — time now
  *decreases* with chi (1.76s→1.35s) instead of increasing (2.96s→28.74s), confirming the Phase 5
  root-cause diagnosis with a working fix. `module5_dgs_sampler/RESULTS.md`/`REUSE_NOTES.md`.
  **Module 6 DONE, partial (2026-08-18, `module6_dgs_combination/`)**: attempted Theorem 2.7's
  efficient DGS itself (no source paper available locally — reconstructed the mechanism from
  general knowledge, flagged explicitly). Core identity (average two independent same-coset
  samples → exactly narrows width by sqrt(2)) verified via closed-form brute-force enumeration
  before writing any code. **Two distinct real bugs found and fixed/documented**: bucketing a pool
  by coset and pairing within each bucket is WRONG (weights cosets linearly, needs quadratic —
  confirmed ~2x mass error on a Z^2 test); the fix (random-pair-then-reject) matches theory
  exactly. A second, Module-5-inspired "zero rejection" attempt was ALSO found wrong by a
  different mechanism. **Module 6 initially concluded a practical zero-rejection sampler would need
  "a cleverer construction"** — and a **review pass (2026-08-14) proved that conclusion wrong and
  fixed it**: the two bugs were the SAME bias (cosets weighted linearly p_c where the identity
  needs quadratically p_c^2), and a bias nameable that precisely is not a barrier but a density
  ratio you divide out. Importance-weighting each sample by its coset mass p_c makes the
  zero-rejection version exactly correct, discarding nothing; the apparent obstacle (p_c is a
  theta-function sum) dissolved because **Klein's algorithm already computes it internally** and
  just wasn't returning it (`klein_sample(..., return_log_normalizer=True)`). New
  `weighted_combination.py` + **committed `test_weighted.py` (5 tests)**: Klein normalizer vs
  closed-form coset mass 2.2e-16; weighted round vs theory 3.4e-4; unweighted regression guard
  catches the old bias; 2 chained rounds 7.1e-4; real n=6 Hessian alignment 0.9996-1.0000 vs direct
  GPV 0.9991-0.9999. **Efficiency: rejection acceptance 0.22-0.45 (n=4) → 0.03-0.06 (n=6) →
  0.006-0.012 (n=8) vs weighted ESS/N 0.66-0.79 → 0.48-0.74 → 0.33-0.73, i.e. 1.6x → 8.6-23.6x →
  26.8-119.4x, growing with n because it removes an exponential factor.** Not claimed: a
  reproduction of Theorem 2.7's 2^{n/2+o(n)} (one exponential factor removed; base pool still
  GPV/Klein, inherits Klein's approximation — exact on Z^2, unquantified at larger n). Also fixed:
  research note misattributed ref [2], which the bibliography confirms is ADRS STOC'15.
  `module6_dgs_combination/RESULTS.md` § Resolution / `REUSE_NOTES.md`.
  **Follow-up (2026-08-14) closed the review's last two items and overturned a Module 4
  conclusion**: (a) the n=5 covariance spread WAS small-pool noise as originally claimed (verified
  via a direct-vs-direct null model: noise floor 0.639 at N=68 vs the 0.498 in question) — but the
  pool-size sweep exposed an n=5 error *plateau* (0.417→0.256) that noise can't explain; (b)
  computing the EXACT covariance by enumeration as a neutral referee showed **direct Klein was the
  biased one** — n=5/seed 205: direct 0.3247±0.0290 vs combination 0.0487±0.0079, a **6.7x accuracy
  advantage** for the combination sampler (also 3.1x, 1.9x on two other lattices; comparable on 6).
  This is exactly the wide-sample-then-narrow-exactly effect the technique exists for, and it
  **refines Module 4's "our GPV/Klein sampler needs no narrow-width fix"** (erratum added there):
  direct Klein at narrow width is usually fine but *sometimes* systematically biased,
  lattice-dependently, while the combination sampler is consistently accurate — value is
  **robustness**, not uniform improvement. Mechanism unexplained (Gram-Schmidt skew does NOT
  correlate — seed 205 has the lowest skew and worst error). (c) **End-to-end SVP solve** added
  (`solve_end_to_end.py`), the last module to get one: **32/32 solves succeed** for both sources,
  combination ~3.5x slower — reported plainly as saturating, i.e. the solve task cannot
  discriminate at toy n, so the accuracy win shows only in the covariance-vs-exact test.
  **Root-caused (2026-08-14, `module6_dgs_combination/FINDING_klein_bias.md`)**: why does
  narrow-width Klein fail on some lattices, and why didn't GS skew predict it? Skew is
  SCALE-INVARIANT and Klein's difficulty isn't — Klein draws coordinate i at width
  s_i=sigma/||b*_i||, so uniform-but-large GS norms starve every coordinate (seed 205, the worst
  lattice, has the LOWEST skew). Real mechanism: Klein's law is exactly rho(v-t)/prod_i Z_i, so it
  is exact iff prod_i Z_i is path-independent — and **std(log prod Z_i) IS Klein's error, Pearson
  r=+0.955** (vs skew -0.338, min_i s_i -0.524), a diagnostic needing no ground truth. **The fix
  falls out**: the density ratio is prod_i Z_i itself, which Klein already computes and discards —
  importance-weighting by it makes direct Klein **unbiased** (mean err 0.070→0.040, **worst
  0.322→0.054**; decays 1/sqrt(N) where raw plateaus at 0.32). Productized as
  `common/gaussian.sample_discrete_gaussian_lattice_weighted`, now the lab's accuracy-preferred
  sampler — **cheaper than the combination sampler (~1/3 the Klein calls) and equally accurate**,
  so Module 6's combination sampler keeps its role as the faithful implementation of the paper's
  technique rather than as the practical tool. Modules 1-4 used raw Klein (slight bias in
  principle, didn't affect their eigenvector-direction conclusions; flagged in their write-ups).
  Third instance in this lab of "diagnose a bias precisely enough and it becomes a correction
  term", second where the needed quantity was already computed and thrown away.
  **Modules 1-4 retested with the correction (2026-08-14, `RETEST_modules_1_to_4.md`)**: opt-in
  `klein_weighted=` flags added (default False, prior results reproducible). Test designed to avoid
  the saturation trap (lattices picked by Klein hostility, angular-error scoring, reduced sample
  budgets). **Surprise**: scored against the true shortest vector v the correction looked WORSE
  (0.00126 raw vs 0.00494 weighted). Ruled out variance (bias-only decomposition) and ratio bias
  (N-sweep: both plateau), then computed the **exact Hessian by enumeration** — which sits
  **0.0044 from v** on that lattice (Lemma 3.6's residual, irremovable). The weighted estimator's
  plateau IS that value, i.e. it converges correctly; raw Klein's bias just happens to err TOWARD
  v. **Scored against what it actually estimates (the exact Hessian), weighted wins 5/6 lattices by
  14-46x** on the hostile ones. **For the SVP task: no difference — 216/216 solves succeed either
  way down to N=25** (Babai + scale grid absorb ~1e-3 angular error). So Modules 1-4's conclusions
  stand, now measured not asserted; and Phase 7d's over-broad "accuracy-preferred sampler" claim was
  **corrected** to "use it when the ESTIMATE must be unbiased (moments/covariance), skip it in the
  solve path". Methodological lesson recorded: scoring an estimator against what you WANT (v) rather
  than what it ESTIMATES (the exact Hessian) can invert the ranking.
  **RUST ENGINE DONE (2026-08-14, `engine/`)** — std-only crate, ZERO dependencies (keeps the PRNG
  reproducible for cross-language parity, avoids a LAPACK dep for 20x20 eigenproblems, zero
  supply-chain surface). Scope chosen BY MEASUREMENT: ports §3+§4 (the configuration our own
  benchmarks found fastest-correct); §5/§6 deliberately excluded as measured-slower here, staying
  in Python for study. **31-49x faster than the Python prototype** on identical lattices, both
  brute-force verified (n=6: 2.58s → 0.05s), raising the practical ceiling to n=14 in 14s/28MB
  where Python strained past n=8. **Parity fixtures caught a real cross-language bug on first
  run**: Python `round()`/numpy `rint` use banker's rounding, Rust `f64::round` rounds half away
  from zero — and LLL's size-reduction sits exactly on `mu=±0.5`, so the two silently produced
  different valid bases (fixed: `linalg::round_half_even`). Error harness applied
  (`rust-error-harness` skill): `Lattice::new` is the single integrity gate, the load-bearing check
  being **dimension** (the solver allocates 2^n matrices, so an unvalidated n is an uncatchable
  multi-GB abort — now rejects n=22 before allocating); no unwrap/expect/panic!/`let _ =` in
  `src/`; `#![forbid(unsafe_code)]`; every gate fired for real, not just written. **25 tests**:
  cross-language parity, brute-force-oracle equivalence, seed reproducibility, truncation+bit-flip
  corruption sweep, degenerate-input contracts. Docs: `engine/README.md`.
  **PyO3 BINDINGS + DEMO NOTEBOOK DONE (2026-08-14)** — bindings behind an OPTIONAL `python`
  feature so the core crate stays zero-dependency by default (`engine/src/python.rs`, maturin).
  Exports solve/brute_force/hessian_direction/reduce_basis. **Panic fence** with two honest caveats
  documented rather than oversold: PyO3 already fences (ours is defence-in-depth + a better
  message), and `panic="abort"` silently no-ops BOTH fences — so that is rejected at COMPILE time
  via `#[cfg(panic="abort")] compile_error!`. `_panic_for_testing()` exists so the fence is FIRED,
  not asserted (verified: RuntimeError raised, interpreter survives). 14 Python-side tests.
  **Notebook `demo/svp_hessian_demo.ipynb`** (script-generated by `build_notebook.py`, 22 cells,
  executed, 0 errors, 2 figures): paper citation + the maths briefly (periodic Gaussian, both
  Hessian representations, mid-point trick, Lemma 3.6, xi_t, parity classes); live examples on the
  Rust engine (alignment 0.9998 at 5k samples; unassisted solve n=2..8 all 7 verified vs oracle;
  the 2^n work curve); Rust-vs-Python table; **a findings section covering all nine discoveries**
  from this project; the guarded boundary demonstrated live incl. firing the fence; honest limits;
  and the non-crypto SVP/CVP applications.
  **Module 7 DONE (2026-08-14, `module7_optimal_parameters/`) — LAST GAP CLOSED**: built §6.1's
  Kabatiansky-Levenshtein machinery from the ground up (B_KL → beta → t_0 → g → K_2 → g_2 → iota →
  Eq. 30's objective + all nine Lemma 6.4/6.5/6.7/6.9 constraints), copying nothing from the PDF
  except as assertion targets. beta and t_0 match; the paper's own params verify to 10 dp
  (s=0.1537683785, iota=0.1593947534, objective 0.6038669 < 0.603867). **Independent optimization
  recovers the paper's (r,R,chi) to ~1e-7 without being told them**, and the same code reproduces
  Remark 6.10 (0.6040346 vs 0.604035) and the quantum §6.8 exponent (0.5410518 vs 0.54106) —
  three targets from one implementation. **The computation caught a transcription error in our own
  notes** (iota digits transposed; the code was right — the argument for deriving over copying).
  Fed back in: the illustrative params were VALID but suboptimal (exponent 0.700 vs 0.604 — nothing
  errored, the deficiency was purely asymptotic), and with the paper's params the pipeline solves
  48/48 at n=6 but runs **~2x slower**, the third time this project saw an asymptotically-better
  configuration lose at measurable scale. `module7_optimal_parameters/RESULTS.md`/`REUSE_NOTES.md`;
  notebook §7.9 updated. **The lab is now complete against the paper as written** — §3-§6
  implemented and verified, §6.8 exponent reproduced, parameters derived, Rust engine + PyO3
  bindings + demo notebook. Remaining gaps are the deliberate ones (Babai for the preprocessed BDD
  oracle; Klein for the efficient DGS; verification limited to n<=14 where brute force reaches).

## Cryo-EM family
- **v28f_cryo_em** — `/run/media/fraser/ows/v28f_cryo_em/` — 3-layer CNN cryo-EM denoiser,
  Fortran/cuDNN, matches/exceeds Topaz-Denoise (SSIM 0.86, PSNR 21.6 dB). **Latest.** Mirror at
  `~/machine_learning/CIFAR-10/v28f_cryo_em/`.
- **v28e_cryo_em** — `/run/media/fraser/ows/v28e_cryo_em/` — earlier cryo-EM iteration.
  (symlinked into MPDOK as `MPDOK/Link to v28e_cryo_em`.)
- **cryo_em_denoising2** — `/run/media/fraser/ows/cryo_em_denoising2/` — related denoising work.

## Climate / physics-sim family
- **v28d_streaming** — `/run/media/fraser/ows/v28d_streaming/` — climate temperature downscaling CNN.
- **v30e_les_full** — `/run/media/fraser/ows/v30e_les_full/` — large-eddy simulation.
- **v28e_climate** — `…/MPDOK/v28e_climate/` — climate lab under MPDOK.

## The journey write-up — `claude_hub/JOURNEY.md` (2026-08-29)

The whole-portfolio narrative, at low resolution: radix sort (months, correct, slow, beaten
by Thrust — pre-dates this map) -> the Tensor Core Engine's 15+ iterations and the turn to
**re-thinking the standard approach** (split-precision, not tuning) -> MPDOK, faster *and*
more accurate than the FP64 baseline -> ~30 application labs -> the maturation into
trustworthy **negative** results (`vol_regime_lab`, `hybrid_resilience_lab`, graph_lab's own
3x self-correction, sparsebridge S0) -> RUSTMM as the daily-use surface where it all lands.
Names the second, undeclared theme the portfolio has been building toward: **systems that
know when they are not confident** (gp_engine posteriors + decision_engine/`voi.py` + ARC's
monitoring design + IR's cheap-estimate-with-known-error). States the gap plainly -- that
machinery has **never been connected to a real recurring decision** -- and names the
candidate: the **GF/QX escalation decision** in the SEC fast-scoring probe. See JOURNEY.md
SS9 for why the current novelty gate is the wrong instrument (ARC SS4.6: covariate-shift
detectors cannot see concept drift) and what replaces it.

## The arc write-up — `hybrid_resilience_lab/ARC.md` (2026-08-27)

**Start here** for the whole story: `sparsebridge/graph_lab` Phases 0-6 plus
`hybrid_resilience_lab` Phases 7.0-7.5, in narrative order, including every correction made
to our own results (the Phase 6 headline overstated ~3x by a base-rate confound; the 70/70
partition defect; a hole in our own pre-registered decision rule; seed-dependent ground
truth). Ends with **"What this implies for future GNN work"** -- eight transferable lessons,
the load-bearing ones being: check preprocessing before architecture (standardisation moved
F1 0.35->0.52 while residuals *hurt* and five sampling strategies were indistinguishable);
measure graph structure before designing around it (Elliptic is a disjoint union of 49
per-timestep graphs, which invalidated a whole approach and reframed what the GCN was doing);
reachability is not signal on hub-heavy graphs; evaluate per-period on *lift*, not raw
precision; **label-free drift monitoring cannot see concept drift by construction** -- so
embedding/feature-drift dashboards show green while the ranking dies; and the cheap-base
pattern transfers only where the base is *discriminating* as well as correct.
**The project's deliverable is a monitoring design, not a model**: precision@k (free) + ~25
audited labels/period + alarm on lift.

## hybrid_resilience_lab — cheap-correct base + bounded learned correction
- **hybrid_resilience_lab** — `~/machine_learning/hybrid_resilience_lab/` (started 2026-08-26)
  — successor to `sparsebridge/graph_lab/` Phase 6. Tests whether the
  **cheap-correct-base + learned-correction** pattern — which recurs independently in
  `v31`/`v32` (spectral solver + CNN SGS), `turbulence_spectral_lab` (cheap solver + neural
  operator) and `quantum_chemistry` (LDA + FNO) — buys real resilience on a **non-physics**
  problem, where there is no conservation law to get the base for free. Motivating result:
  Phase 6's GCN collapsed ~70% → ~10% precision at timestep 43 with its confidence scores
  unmoved. Plan in `LAB_PLAN.md`; **Phase 7.0 is a falsification test that can end the lab
  early** (if no regime-invariant signal exists, the honest answer is drift detection +
  retraining, not a cleverer model).
  **Two structural facts measured while scoping it (2026-08-26)**, which overturned the
  lab's original opening plan: the **Elliptic graph is a disjoint union of 49 per-timestep
  graphs** (234,355/234,355 edges are same-timestep, zero cross), so **0.0%** of later
  illicit nodes are within 1–3 hops of any training-era illicit node. Label propagation from
  known-bad — the obvious invariant base — is therefore *structurally impossible* there, and
  the Phase 6 collapse is a pure **feature-distribution shift**, not a connectivity failure.
  The 252M-node Bitcoin graph is not temporally disjoint and has real value fields, so
  propagation- and value-conservation bases are testable only there.
  **Phase 7.0 + 7.0b DONE (2026-08-27) — central hypothesis NOT supported.** 7.0
  (`topo_base.py`, Elliptic): the label-free topological base scored *below random* (lift
  0.13x/0.12x); diagnostics showed 5 of 7 AML priors point the wrong way there, and
  decisively `is_relay` — the one feature with real signal (AUC 0.604) — collapsed 10.0% →
  0.7% at timestep 43, harder than the GCN. 7.0b (`phase70b_bitcoin.py`, 252M Bitcoin
  graph, temporal split at block 436,617): **propagation from known-illicit seeds refuted**
  (PPR AUC 0.471, *below chance*, despite 18.2% cross-period edges — the graph is too
  hub-dense for reachability to carry information, since hop 2 reaches 43% of all nodes);
  **value conservation carries real signal (AUC 0.727, ~2.9x lift)** but cannot rank the top
  of the queue (56.8% of nodes tie at "perfectly conserving", precision@25 = 0) and sits
  at/below chance in the densest test quartile. **The explanation for the asymmetry with the
  physics labs**: value conservation is *correct* but not *informative* — 56.8% of all
  entities satisfy it, criminal or not — and the pattern needs a base that is both.
  **A correction to Phase 6 fell out of this**: comparing raw precision across timestep 43
  is confounded by the illicit base rate (9.16% → 2.53%); on lift the GCN degraded
  7.89x → 3.12x, a 2.5x loss rather than 8x, so "worthless in production" overstated it —
  recorded in `graph_lab/RESULTS.md`. Three options now open (salvage the label-free drift
  detector / one bounded follow-up on value conservation / write up the negative result) —
  see `hybrid_resilience_lab/RESULTS.md`.
  **Phase 7.3 DONE (2026-08-27) — partial success, the one thing in this lab that works.**
  `drift_detector.py`: four label-free signals computed per period over ALL nodes (the real
  deployment situation — scores for everything, labels for almost nothing), validated against
  the GCN's realised lift. Rationale that survives 7.0/7.0b's failure: **a drift detector
  needs its reference to be STABLE, not ACCURATE**, so 7.0's weak topological ranking is
  still a usable reference. Result: `disagreement` (model-vs-structure rank correlation)
  period-AUC **0.795** and `pos_rate_shift` 0.773 genuinely track degradation — and
  decisively beat `feature_shift` (0.545), the conventional input-drift monitor most teams
  deploy, which is useless here and alarms on every timestep. **But it is a lagging,
  partial detector**: at 1.5σ it catches ~2 of 4 damaged periods with 1-2 false alarms and
  **misses timestep 43 itself**, where every label-free signal looked completely normal —
  sharpening Phase 6's lesson from "degraded silently" to "was invisible to standard
  monitoring at the moment it mattered". Honest counter-caveat: ts43's lift 0.00 rests on
  0 hits vs 1.2 expected (P≈30% under Poisson), so the ground truth is itself noisy there.
  Deployable as a dashboard trend line, not an automated alarm. 15 periods / 4 bad = thin.
  **Phase 7.3b DONE (2026-08-27) — rolling-window early warning: NEGATIVE, and the reason is
  the lab's sharpest result.** `early_warning.py`: CUSUM / EWMA / rolling-mean /
  rolling-reference at textbook defaults fixed before running (CUSUM k=0.5σ h=4,5σ; EWMA
  λ=0.2 L=3; W=3). **20 configurations, none early or concurrent**; best is `rolling
  reference W=3` at lagging +2, consistently across three signals. Decisive reason: **at
  ts43 every signal sits BELOW its training baseline** (z = −0.24, −1.25, −0.56) — no
  positive deviations to accumulate, so this is an **information limit, not a method
  limit**. **The mechanism: this is CONCEPT drift** (P(y|x) changed while P(x) and P(score)
  did not), and label-free monitors observe only `x` and `model(x)`, so it is invisible *in
  principle*. Confirmed empirically: `feature_shift` (genuine covariate shift) is elevated
  at z = +4.8 to +9.2 across the ENTIRE test span while lift swings 0×–10.96× — observable
  and completely uninformative. **Practical conclusion: no label-free substitute for labels
  exists for this failure mode; a small continuously-labelled audit sample (fixed random
  slice reviewed every period regardless of score) is the only thing that observes P(y|x).**
  The lab's three phases now compose: cannot prevent (7.0/7.0b), can partially detect the
  aftermath (7.3), cannot detect it early (7.3b) — a coherent negative result with a
  mechanism rather than a shrug.
  **Phase 7.4 DONE (2026-08-27) — audit-sample sizing, and it reframes what labels are FOR.**
  `audit_sample.py`. In deployment analysts adjudicate every case they open, so
  **precision@k is already free**; what is unobserved is the **base rate** — exactly the
  Phase 6 confound (ts43: precision 0.0%/base 1.75% = broken; ts46: precision 2.8%/base
  0.28% = 9.9x and healthy; a precision-only monitor calls both a disaster). Hypergeometric
  draws, Laplace-smoothed base rate, threshold calibrated on healthy validation periods,
  4,000 draws/period. **Result: ts43 is detected 100% at EVERY audit size including zero** —
  precision there is exactly 0.0000, so the free monitor already catches the event. **What
  the audit actually buys is false-alarm reduction: 45% (free) -> 20% (n=25) -> 10% (n=200)
  -> 5% (n=400).** Without it a precision-only monitor fires on ~half of all healthy periods
  and gets ignored — which is how you return to a silently-degrading model nobody watches.
  Recommendation: the knee is **n≈25 (2% of volume)**, halving false alarms for nearly
  nothing beside an already-labelled ~60-case queue; n=200 (17% of volume, 3-4x the
  labelling) for 10%. Most promising untested idea: **pool the audit over a rolling window**,
  since base rates move slower than they are measured. Caveats stated: ts45/ts46 have 5 and 2
  illicit nodes and cannot be classified reliably, and the bad-period set is seed-dependent
  (ts45 was lift 0.00 in 7.3's run, 4.00 here) — ts43 itself is stable across both runs.
  **Phase 7.5 DONE (2026-08-27) — rolling-window audit pooling REJECTED on evidence.**
  `audit_pooling.py`, run over 3 GCN seeds (addressing 7.4's own seed-dependence caveat).
  7.4's most promising idea — pool W periods' audits for ~Wxn effective labels at n/period
  cost — **is false**. No pooling wins at every audit size, and the damage grows with n
  (+1pp at n=10 to +7pp at n=200: the bias-variance signature, since at large n the unpooled
  estimate is already precise and pooling adds only bias). Matched effective labels is
  decisive: 200 labels unpooled = 8% false alarms, but 200 effective via W=2 at n=100 = 18% —
  halving per-period cost more than doubles false alarms. **Mechanism confirmed, not just
  outcome**: at ts36 (true base 1.93%, model doing 12.18x) pooling inflates b_hat
  0.0241 -> 0.0771 by carrying richer earlier periods forward, crushing estimated lift and
  manufacturing a **56% false-alarm rate on an excellent period**. Twist: at ts49 pooling
  HELPED (99%->25%) because stale rates there were lower — so pooling injects an error whose
  **sign depends on which way the base rate recently moved**, worse than a consistent bias
  because it cannot be corrected. Caveat limiting all absolute numbers: false-alarm rates are
  dominated by ts45/ts46 (5 and 2 illicit nodes; a 0.28% base rate cannot be estimated from
  200 draws), so the pooling-vs-no-pooling comparison is valid but the levels are not
  achievable targets. **7.4's recommendation stands: audit each period independently, n~25
  as the cheap knee.**

## CIFAR-10 `v28`–`v41` series — the turbulence → dispersion arc (Dec 2025 – Jan 2026)

**Added to this map 2026-08-26.** This whole arc predates the map (which had dense coverage
only from 2026-07 onward) and was effectively invisible to keyword lookup — the omission was
found while trying to relocate the PINN work below. All live under
`~/machine_learning/CIFAR-10/` unless noted. Listed by arc, not exhaustively by directory.

**`v28g`–`v28s` — Fortran/CUDA 3D-CNN training + a systematic cuDNN optimization sweep**
(2026-01-07). Full Fortran/CUDA training pipeline with cuDNN backward pass and NVIDIA Apex
Adam. Each suffix is one optimization axis, benchmarked in its own docs: `_memory`,
`_tensor`, `_pthread`, `_cudnn`, `_tf32`, `_workspace`, `_tuning_order`, `_nhwc`,
`_data_conversion`, `_pure_nhwc`, `_graph_api` (cuDNN Graph API + NDHWC). Real measured
gains include 4.1×, 1.98×, and 28× at various stages. Also `v28_baseline_plus_export`,
`v28b_managed`, `v28c_warp_shuffle`, `v28d_streaming_v2` (2026-03), `v28e_climate_cnn`
(2026-06), `v29_real_data`.

**`v30`–`v30j` — SGS turbulence parameterization, LES, and PDE surrogates** (2026-01-07/08).
CNN sub-grid-scale models (`v30`, `v30b`, `v30c`), an a-posteriori LES solver with CNN-SGS
(`v30d`, `v30e`), a PDE surrogate (`v30f`), long time-series (`v30g`), and three latent-dynamics
attempts (`v30h`, `v30i`, `v30j`) that are the most instructive part of the arc — see the
failure taxonomy below.

**`v31`–`v32` — hybrid physics+ML solver and long-term stability** (2026-01-10/12).
`v31_physics_hybrid`: a validated hybrid where a spectral physics solver carries the
simulation and ML supplies an SGS correction; Python↔Fortran agreement 99.999%, hybrid
preserves energy better than physics-only (1.510 vs 1.477 over 1000 steps).
`v32_long_term_stability`: pushed to 1000 stable steps via a 10× timestep reduction, then
found instability at ~1600 steps **regardless of ML** — a numerical-stability limit, not a
model-quality one.

**`v33`–`v34` — reacting flow and chemistry surrogates** (2026-01-13/14).

**`v35`–`v41` — pollutant dispersion → emergency response → PINN** (2026-01-15/19).
`v35_pollutant_dispersion` (physics-informed ML), `v36_source_inversion` (real-time inverse
leak localization), `v37`/`v38`/`v39` emergency-response decision support (plain →
CNN-enhanced → cuDNN), `v40_production_release`, then the pivot: **`v41_pinn_production`
(SimplePINN)** — advection-diffusion-reaction solved with upwind finite differences,
**zero learnable parameters, no training required**, 0.8 s for a 128³ / 2.1M-cell domain
(237 Mcps). Its own docs record why: `v35`–`v40` CNN attempts scored **0% on advection**
("peak didn't move" — symmetric kernels cannot represent directional flow), while explicit
physics encoding scored 103%. `v41_pinn_experiments` holds the experimental precursor.

**Standalone, same period**: `fluid_simulator_cuda` (2026-01-20), `fluid_simulation_gemma`
(2026-01-23), `Monte-Carlo-Lattice-Options-Pricing` (2026-01-03), `Kindle clips` (2026-04).

### The model-fragility taxonomy this arc produced (relevant to `graph_lab` Phase 7)

Read together, `v30h`–`v41` is a catalogue of distinct ways a learned model fails, each with
the mitigation that was actually tried — directly useful to any resilience work:

| failure mode | where | what happened | mitigation tried |
|---|---|---|---|
| Explosion / NaN | `v30` baseline, `v32` | 200-step explosion; NaN at ~1600 steps | smaller dt; physics-constraint loss terms |
| **Frozen fixed point** | `v30i` | LSTM latent collapsed to the time-average and **stopped moving** — "t=100, 1000 and 2523 look the same while DNS is moving". It minimized rollout loss exactly as asked: *"the model was doing exactly what we asked, but we were asking the wrong question"* | multi-time-scale training (`v30j`) |
| Stability bought with dynamics | `v30h` | 2524-step stable rollout (98.8% success over a 256-config grid search, 46 h) but "lacks temporal dynamics" — over-damped | architecture change (`v30i`) |
| Learned correlation ≠ mechanism | `v35`–`v40` | CNNs scored 0% on advection regardless of training | drop learned params entirely (`v41`) |
| **Distribution shift** | `graph_lab` Phase 6 | GCN collapsed ~70% → ~10% precision at timestep 43 when a dark-market shutdown changed the illicit population | *(open — this is the Phase 7 question)* |

**The resilience mechanisms already proven in this codebase**, weakest to strongest
constraint: physics-constraint loss terms (`v30h`) → multi-time-scale training (`v30j`) →
**cheap-correct-solver + learned correction** (`v31`/`v32`, and the same pattern reused in
`turbulence_spectral_lab` → `quantum_chemistry`) → **zero learnable parameters** (`v41`).
The middle option is the interesting one for `graph_lab`: it keeps a learned component but
bounds it inside something that stays correct when the regime changes.

## Spectral-correction pattern labs (2026-02/03)
- **turbulence_spectral_lab** — `~/machine_learning/CIFAR-10/turbulence_spectral_lab/` (2026-02-16)
  (path corrected 2026-08-29 — the map said `~/machine_learning/turbulence_spectral_lab/`, which does not exist) —
  established the **cheap-solver + neural-operator-correction** pattern. `PROJECT_REPORT.md`.
- **quantum_chemistry** — `~/machine_learning/quantum_chemistry/` (2026-03-28) — applies that
  same pattern to quantum chemistry: can a 3D Fourier Neural Operator trained on LDA electron
  density fields correct a cheap solver toward a more expensive one? Docs include
  `CLIMATE_PHYSICS_CONNECTION.md`, `DATA_SCALE_ANALYSIS.md`, `COLAB_STRATEGY.md`.

## GP / kernel engine
- **gp_engine** — `~/machine_learning/gp_engine/` — exact mixed-precision Gaussian-process regression
  at scale (FP32 tensor-core factor + IR to FP64) on the tensor-core engine + MPDOK. Detailed plan in
  `PLAN.md`; interface spec `SPEC.md`; engine module `gp_core.py`. Status: **Phase 0 complete + Phase 1
  fused kernels done (2026-07-09, RTX 4060)** — see `RESULTS_PHASE1.md`: **12.6× vs FP64 cusolver at
  n=27k** (31.0s → 2.47s, relres 6e-12) via direct-from-X fused CUDA kernels (no gram tile, d≤16);
  **n=40k in-core** (1.48× capacity, 7.0 GB) where FP64 caps at 27k. κ envelope: clean ≲1e6, fails
  detectably ≳2e8 (`RESULTS_PHASE0.md`). **Phase 1 COMPLETE (2026-07-09):** hyperparameter loop
  (`gp_hyperopt.py`, 80-eval LML fit 15.6s at n=10k = 8× vs FP64, 2.8 min at n=27k; logdet bias harmless
  for fitting — 100% rank agreement); predictive mean+variance (`gp_predict`, var err 5.7e-6);
  **CUDA Fortran port done** (`gp_solver.cuf`/`gp_solver.so` + `gp_fortran.py`, MPDOK-style Makefile;
  `test_parity.py` bit-identical vs the CuPy oracle `gp_core.py`). **Phase 2 OOC streaming Cholesky
  COMPLETE (2026-07-09)** — `gp_ooc.py`: implicit-K panel regeneration (only factor panels cross PCIe),
  validated vs in-core at n=12k; **n=100,000 exact GP fit in 5.4 min on the 8 GB card** (v2: factor
  75.3s @ 4.43 TFLOPS, relres 3.2e-11 — 2.5× the in-core cap; the 40 GB kernel matrix never exists;
  v1 was 19 min) and n=60,000 in 39.1s. v2 = tiered pinned-RAM/NVMe-memmap panels + ping-pong
  double-buffered factor pipeline + VRAM panel cache for IR. Hard-won ops lessons in
  `RESULTS_PHASE2.md`: systemd-oomd saga + zram freeze → memmap panels; run big jobs as detached
  services in `compute.slice`; **CuPy stream-placement traps** (`set(stream=)` and manual
  `cublasSetStream` don't stick — raw cublas calls + async copies must run inside `with stream:`).
  **Phase 2b DONE (2026-07-09): n=200,000 exact GP fit + logdet in 71 min, CONVERGED (relres
  3.4e-10, 13 IR steps)** on the same 8 GB card + 46 GB box (v3 row-chunked streaming,
  `_ChunkStreamer` — VRAM bounded ~5.5 GB at any n; 81.6 GB panels = 1.8× RAM, zero oomd/freeze;
  disk-bound at 1.5 TFLOPS — the wall is LUKS-NVMe ~1 GB/s, not the GPU). High-κ guidance measured:
  at κ≈8e6 IR converges at 0.32×/step → use tol 1e-9 + max_ir≥16 at n≥150k. A 200k kernel matrix
  would be 320 GB in FP64; it never exists. **Demo notebook: `gp_engine_demo.ipynb`** (executed;
  regenerate via `build_demo_nb.py`) — live engine-vs-CuPy comparison, IR traces, hyperopt,
  uncertainty bands, OOC tiers, backend parity, honest limits. **Release preview in `test/`
  (2026-07-10):** binary-only distribution — `gp_solver.so` + self-contained `gp_engine.py`
  wrapper + Fortran-only demo notebook + README (engine-family lineage; Fraser to add links).
  No CUDA-C source, no OOC module (deliberately held back). **Phase 3 planned (2026-07-10,
  PLAN.md §6 + `gp_lab/LAB_PLAN.md`):** lab-led real-data validation on the published exact-GP
  benchmark suite (kin40k/protein/elevators/3droad; RMSE/NLL/coverage vs published baselines),
  pulling kernel extensions in demand order. **Phase 3 items 1–2 DONE (2026-07-10): ARD + Matérn
  3/2, 5/2 in both backends** (generic `Kernel(kind=, ell=scalar|vector)`; parity bit-identical;
  Matérn logdet bias 10× smaller than RBF), hyperopt ARD-capable, `gp_lab/datasets.py` fetches all
  6 suite datasets (shapes verified vs literature). **First real-data fit** (kin40k 10k subsample,
  Matérn32 ARD): RMSE 0.126, 97.2% coverage, relres 6e-12 — and it exposed real issues: kin40k is
  noiseless → nugget floor added (σ_n²=floor²+e^2θ reparam). **M1 DONE (2026-07-10):
  `run_benchmark.py` (staged optimizer: iso-subsample → ARD-subsample → full polish, 80 full-cost
  evals vs ~600 naive) + full kin40k ×3 seeds: RMSE 0.0654±0.002, NLL −1.32, coverage 97.4%,
  ~10 min/seed on the 4060** (`gp_lab/results/*.json`). Key finding: Matérn converges at
  κ-bound ~7e8 — the RBF-calibrated κ≲1e7 envelope is loose for Matérn (spectrum bounded below);
  envelope needs per-kernel refinement (a widening). **M2 DONE (2026-07-10): protein via OOC**
  — `gp_ooc.py` gained `forward_solve`/`ooc_predict` (validated vs in-core oracle), auto in-core/OOC
  dispatch in `run_benchmark.py` at n=38k. protein (n=41,157) fits in ~14s (factor ~10s, 3.7GB
  pinned panels), RMSE 0.5382±0.0063, NLL 0.574±0.026, coverage 93.7%. Two real issues found and
  fixed en route: (1) subsample-only ARD hit a lengthscale/σ_n degeneracy (28× σ_n spread across
  seeds) — fixed by affording one full-data hyperopt polish eval even for OOC-fit datasets
  (decoupled `FULL_POLISH_MAX` from `IN_CORE_MAX`); (2) that fix initially OOM'd because CuPy's
  memory pool doesn't release blocks on `del` — needs explicit `free_all_blocks()`. Median-NLL +
  pathological-point-count now always reported alongside mean NLL (CASP/protein has near-duplicate
  rows that blow up mean NLL on a handful of points — real GP behavior, not a bug).
  **Published baselines verified (2026-07-10)**, cited from Wang et al. 2019 NeurIPS (arXiv:1903.08114,
  Table 3 ARD; PDF cached in `gp_lab/papers/`): **kin40k beats the published exact-GP baseline on both
  RMSE (0.065 vs 0.080) and NLL (−1.32 vs −0.76) on more data, one $300 GPU vs their 8×V100**; protein
  NLL beats theirs (0.57 vs 0.96) but RMSE is ~5% higher (0.538 vs 0.511) — plausibly their gradient-based
  optimizer vs our derivative-free NM, flagged not hidden. **M3 DONE (2026-07-10): d≤32 kernel widening**
  (MAX_D 16→32 both backends; measured zero occupancy regression at low d, d=26 validated bit-identical
  vs a fresh dense reference) **+ elevators/bike/pol**: elevators and bike beat the published baseline on
  both RMSE and NLL; pol shows the same NLL-wins/RMSE-loses pattern as protein (~44% higher RMSE) — a
  cross-run signal now seen in 2/5 datasets, hypothesized to be optimizer choice (derivative-free NM vs
  the paper's gradient-based L-BFGS+Adam), not yet confirmed. Found + fixed two real bugs in the hyperopt
  harness (NM can return a penalty-valued "best" result; a dead-code guard that would've hidden a fully-
  infeasible search) and one real data finding (bike's LML wants near-zero noise — needed `--floor 1e-2`
  vs the 1e-3 default; automatic floor escalation is a natural next step). 5/5 real-data benchmarks now
  done (kin40k, protein, elevators, bike, pol); bike also required a preprocessing fix (re-fetched via
  the `uci_datasets` mirror to match the paper's d=17, was d=12).
  **M4 DONE via full CUDA Fortran OOC port (2026-07-12):** after the Python streaming layer hung
  the n=391k 3droad run five times (root-caused to host zram memory pressure, not a GPU/code
  deadlock), the whole OOC streaming layer was ported to CUDA Fortran per the new language policy
  (PLAN.md §6b): `gp_ooc_solver.cuf` + thin `gp_ooc_fortran.py` ctypes wrapper, validated
  bit-identical vs Python at 12k/60k/100k/200k, then **n=391,387 3droad converged first try:
  3h51m total, relres 6.5e-11 in 3 IR steps** (isolated via systemd-run memory-capped scope).
  Same-day perf investigation found + fixed a 4× transfer-path tax (CUDA managed-memory UVM
  page-residency ping-pong → pinned staging + `cudaMemcpyAsync`, per `gp_ooc.py`'s own
  `_ChunkStreamer` design) — **full technical postmortem incl. failed attempts + CUDA Fortran
  streaming checklist: `gp_engine/CUDA_FORTRAN_STREAMING_LESSONS.md`** (reusable for any
  GPU↔host streaming work). **3droad scored (2026-07-13): RMSE 0.0702 / NLL −1.024 beats the
  published exact-GP baseline (0.110±0.017 / 1.239±0.025) on both metrics** — d=3 preprocessing
  matched via the `uci_datasets` mirror (ARD auto-pruned their OSM-ID column, ℓ=63.9);
  `run_benchmark.py` now has `--ooc-backend {fortran,python}`. **All 6/6 suite datasets scored**
  (4 beat published on both metrics, 2 mixed NLL-better/RMSE-higher). **M5 write-up DONE
  (2026-07-13): `gp_lab/RESULTS_LAB.md`** (full scorecard, every caveat documented, κ-envelope
  finding — kin40k converged inside the RBF failure band at κ~7e8 — and a verifiable-numbers-only
  hardware comparison) **+ companion chart notebook `gp_lab/RESULTS_LAB.ipynb`**
  (`build_results_nb.py`; numbers load live from results/*.json). **Matched-n + hybrid-optimizer
  follow-up DONE (2026-07-13, RESULTS_LAB.md §9):** `datasets.py` gained `protocol="paper"`
  (reproduces the paper's exact published n — its tables use ⌊0.64N⌋, not the 4/9 its text
  claims) and `run_benchmark.py` gained `--hyperopt hybrid` (NM→bounded L-BFGS-B, catching/fixing
  two real gate bugs along the way). Verdict: kin40k's win nearly vanishes at matched n (data
  volume); pol's loss is fixed by hybrid (genuine NM non-convergence); **3droad's M4 win survives
  matched n comfortably** (0.080 vs published 0.110 at n=278,319); protein's gap remains open.
  Remaining perf levers: VRAM panel cache for IR (Python has it, Fortran doesn't yet); faster
  disk for large-n IR. Skorch-vs-gp_engine toy comparison: `skorch_examples/gp_engine_vs_gpytorch_sine.ipynb`.
  **Next planned lab: `gblup_lab/` (below).**

- **gblup_lab** — `gp_engine/gblup_lab/LAB_PLAN.md` (2026-07-14/15, **all 5 phases DONE**) — takes
  MPDOK's `gblup/` exact-vs-APY genomic-BLUP story and adds gp_engine's marginal-likelihood
  hyperparameter fit + nonlinear/RKHS marker kernels. Reads MPDOK's data in place
  (`MPDOK/gblup/data/`: wheat N=599/d=1279, mice N=1814/d=10346, G2F inbreds N=2193/d=48580 —
  all local, no `/OWS` needed); MPDOK going to GitHub separately, link pending. **Engine gap
  found + fixed:** `gp_core.py` gained `PrecomputedKernel` (K = sigma_f2*A_base + sigma_n2*I
  from any externally-built N×N matrix) since the register-resident coordinate kernel hard-caps
  d≤~32, unusable for marker data. **Phase 0** (`RESULTS_PHASE0.md`): solver parity vs numpy
  `dgesv` at 1e-10–1e-14 rel-err; CV r matches MPDOK's own *live* `cv_lambda_sweep` code to float
  precision — but found MPDOK's README numbers (wheat r≈0.45–0.55, mice r=0.280) are stale
  vs. its current code (actual: 0.37–0.45, 0.138), verified by running MPDOK's real function
  fresh. **Phase 1** (`RESULTS_PHASE1.md`): MLE hyperparameter fit (`gblup_hyperopt.py`, NM over
  log(sigma_f,sigma_n), training-fold-only) lands within 0.002–0.016 r of MPDOK's CV-grid
  without ever seeing the validation fold — validates the LML objective. First NLL numbers this
  lab family has had for genomic prediction; **found a real calibration issue** (not a bug):
  wheat/E3 fold 4 gave 46/120 held-out points exactly-zero predictive variance (FP32 cancellation
  floor) with residuals up to 1.38, traced to the wheat panel's related/repeated breeding lines.
  **Phase 2** (`RESULTS_PHASE2.md`): RKHS/Gaussian+Matérn marker kernel via `marker_kernel.py`
  (GEMM-trick distance builder, any d) — r up +0.09 to +0.27 over Phase 1's linear GRM on every
  trait, and wheat/E3's calibration failure is fully resolved (0 pathological points, was 46).
  Verified not an X/y alignment bug (a naive from-X linear kernel already beats the published `A`
  fit). **Found published `A` and a from-X linear kernel only correlate 0.25–0.46** (different
  BGLR construction pipelines, not a bug). **Literature check DONE (2026-07-14):** read de los
  Campos, Gianola, Rosa, Weigel & Crossa (2010, *Genetics Research* 92:295-308) directly — same
  599 CIMMYT wheat lines, same 4 environments. Their RKHS-over-linear gain (Table A2, 10-fold CV
  MSE→r) is +0.01–0.08 r-equivalent; once gp_engine's RKHS is compared against the right baseline
  (linear-on-same-X, not MPDOK's separately-normalized `A`), its own gain is +0.046 to +0.094 r —
  same order of magnitude, agreeing to ~0.01–0.04 r on 3/4 environments. Verified, not assumed.
  Mice has no literature check yet (paper found is wheat-only). **Phase 3** (`RESULTS_PHASE3.md`,
  stretch goal): built `g2f_hybrid.py` — parental→hybrid GRM via averaged-parent marker dosage
  (Bernardo 1994-style), 4,979 unique hybrids from the 161k-row yield table. **Checked
  leaderboard-comparability before running anything**: the real competition scores on a held-out
  2024 season not present in the local 2014-2023 CSV (confirmed by reading the competition paper)
  — ran a random hybrid-combination 5-fold CV instead, explicitly flagged as not leaderboard-
  comparable. **Found + fixed a real engine-envelope issue**: unnormalized linear kernel at
  d=48,580 (diag≈12,000) blew the FP32 ~1e-6-absolute variance floor for 4,979/4,979 held-out
  points; unit-diagonal normalization (VanRaden's own convention) fixed it, 2 pathological pts
  left, r improved 0.658→0.733 as a side effect. Final: linear r=0.733, RBF r=0.775, Matérn
  r=0.777 — RKHS gain +0.042–0.044 r, inside Phase 2's literature-verified range. **Phase 4
  (2026-07-15, `RESULTS_PHASE4.md`):** fetched the real 2024 G2F season from CyVerse (DOI
  10.25739/78mn-4394, found via WebDAV listing) — ground truth yield + the competition's own
  hybrid genotype panel (5,899 hybrids × 2,425 markers, confirms Phase 3's parent-averaging
  approximation was mathematically correct). Trained on 2014-2023 only, predicted the true 2024
  held-out season once (no peeking): linear r=0.271, RBF r=0.408, Matérn r=0.415. Accuracy drop
  from Phase 3's 0.73-0.78 (same-era random CV) to 0.27-0.41 (genuinely new season, marker-only,
  no weather/soil/GxE covariates) is the expected signature of real distribution shift, not a
  regression — 2024 was ~0.9 std above the training-era mean yield. RKHS gain here (+0.14 r) is
  larger than Phase 2/3's literature-anchored range — flagged as plausible but not independently
  verified (different regime: extrapolation vs. within-population interpolation). Data fetched
  and used strictly after Phases 0-3 were already written, at Fraser's request, to keep the
  comparison genuinely held-out. **All 5 phases of gblup_lab now complete.**

- **cvar_gp_lab** — `gp_engine/cvar_gp_lab/PLAN.md` (2026-07-18) — Bayesian-CVaR portfolio
  optimizer: GP posterior mean *and* covariance (not a point-estimate model) feeding a
  CVaR scenario-LP. Reuses `gblup_lab`'s GEMM-trick kernel builder (`marker_kernel.py`,
  repointed at asset return-history descriptors instead of markers) and its `mle_fit_rkhs`
  LML-fitting pattern, extended with a new explicit mean-function term (`gp_return_model.py`'s
  `mle_fit_rkhs_mean` — neither `gp_hyperopt.py` nor `gblup_hyperopt.py` had one). CVaR math
  (Rockafellar-Uryasev scenario-LP) and data come from `~/quantitative-portfolio-optimization`
  (cufolio); `MPDOK/portfolio_studio` (mean-variance + graph-resolvent, NOT CVaR — a finding
  from this lab's own scoping) supplies the eventual backtest baseline numbers.
  **Phase 0 DONE (2026-07-18)** — see `RESULTS_PHASE0.md`. Caught and fixed two real bugs
  en route: (1) target leakage (y computed from the same window as its own descriptor X
  → GP trivially interpolates, sigma_n2->0, NOT evidence of a good kernel — fixed via a
  disjoint 252d-descriptor/63d-target time split); (2) a unit mismatch comparing a
  y-scale-fit sigma_f2-scaled kernel against a standardized sample covariance — fixed by
  comparing correlation-shaped (unit-diagonal) matrices throughout. **Real findings**:
  cross-validated GP mean-shrinkage beats naive grand-mean shrinkage by ~29% RMSE (5-fold
  CV, genuinely held-out — the strongest Phase 0 result); RKHS covariance is far better
  conditioned than the (T<N, rank-deficient) sample correlation matrix but not yet shown
  more *accurate* than Ledoit-Wolf, deferred to Phase 2's backtest; the near-duplicate-asset
  calibration risk gblup_lab found does NOT transfer cleanly — moderate correlation (0.8-0.9,
  financials cluster) mildly favors RKHS's calibration same direction as gblup_lab, but a
  literal duplicate pair (GOOG/GOOGL, corr 0.998) **inverts it** (RKHS collapses to 0.75%
  retained variance, linear retains 18.5%) — flagged as needing an explicit duplicate-pair
  pre-flight check before Phase 1/2, not assumed safe either way.
  **Phase 1 DONE (2026-07-18)** — see `RESULTS_PHASE1.md`. `cvar_lp.py`: standalone
  Rockafellar-Uryasev Mean-CVaR LP (CVXPY/CLARABEL), ported from cufolio's
  `cvar_optimizer.py` without its cuOpt/turnover/cardinality machinery.
  `scenario_gen_gp.py`: Monte Carlo scenarios sampled from the GP's full posterior
  (mean + covariance), replacing cufolio's GBM `ForwardPathSimulator`. Single-period
  sanity check only (both LPs solve to `optimal`, feasible) — not a backtest. **Caught
  a third methodological bug**: the in-sample `fit_gp_returns` posterior mean barely
  shrinks at all (same `sigma_n2->0` interpolation trap as Phase 0's bug #1, recurring
  in a new spot) — fixed via a new `cv_shrunk_mean` (genuinely held-out, k-fold
  out-of-fold prediction, confirmed ~28% tighter std than naive y matching check 4's
  ~29% RMSE finding). **Open question flagged for Phase 2, not resolved**: the GP-CVaR
  portfolio came back with near-zero predicted CVaR (4.6e-5) concentrated in oil & gas
  E&P names (OXY/COP/EOG/APA) — could be genuine low realized volatility in that window,
  or the RKHS covariance underestimating true energy-sector correlation; only an
  out-of-sample backtest can tell, explicitly not claimed as a win here.
  **Phase 2 DONE (2026-07-18)** — see `RESULTS_PHASE2.md`. Reused `MPDOK/portfolio_studio/
  backtest_engine.py` directly (same 2015-2019 train / 2020-2025 test split, same
  29-asset universe) for the equal/cuFolio/MPDOK/smart/smart2 baselines — only
  `gp_cvar`'s weights (one GP fit + one CVaR-LP solve on the training window) are new.
  **`gp_cvar` beats cuFolio (best static baseline) on total return (+330.5% vs +323.2%),
  Sharpe (1.13 vs 1.10), and max drawdown (-29.4% vs -30.4%)** — loses only to the
  regime-switching smart/smart2 hybrids. Added realized CVaR₉₅/VaR₉₅/Sharpe, metrics
  none of `portfolio_studio`'s strategies compute today. **Caught a real discrepancy**:
  this run's baseline numbers don't match `MPDOK_PORTFOLIO_STUDIO.md`'s static table
  (documentation drift in a doc this project doesn't own, not a bug in the harness
  reuse — investigated enough to rule out an obvious date-range cause, not chased
  further). **Partially resolves Phase 1's open question** (no near-zero-CVaR
  pathology on this universe/period) while surfacing a new one: `gp_cvar`'s weights
  concentrate in mega-cap tech (5 of 9 positions pinned at the weight cap) — a strong
  bet on this specific 2020-2025 window, untested on a less tech-favorable regime.
  **Phase 3 (stretch: walk-forward rolling refit) not started.**

- **mnist_gpc_lab** — `gp_engine/mnist_gpc_lab/LAB_PLAN.md` (2026-07-15) — first extension of
  `gp_engine` from regression to *classification*: `../gp_classifier.py`'s `LaplaceBinaryGPC`/
  `OneVsRestLaplaceGPC` (GPML Algorithm 3.1, Newton-Raphson mode-finding). MNIST one-vs-rest vs.
  SVM: accuracy close, GPC's log-loss worse (real OvR-combination miscalibration, not hidden), but
  headline finding **GPC is essentially never confidently wrong** (0.57% of errors carry
  confidence>0.5 vs SVM's 57%, 200-seed bootstrap CI) — confirmed structural across kernel
  family/likelihood (M3) and via MCMC ground truth, not a Laplace-approximation artifact (M4).
  Reusable for any future classification lab needing a calibrated GP classifier.

- **place_gpc_lab** — `gp_engine/place_gpc_lab/LAB_PLAN.md` (2026-07-15) — same confident-wrong
  question as `mnist_gpc_lab`, on a physically real safety-relevant signal instead of MNIST pixels:
  RANSAC-homography verify features (`ratio_matches`, `inliers`, `inlier_ratio`) from
  `visualstash/place`, binary match/no-match (same physical place or not) instead of 10-way OvR —
  a confidently-wrong "yes" is the loop-closure failure mode that silently corrupts a robot's map.
  **Engine addition:** `place pairfeatures <ref_dir> <query_dir> <pairs.csv> <out.csv>` (new
  subcommand in `visualstash/place/src/bin/place.rs`) turns a hand-built frame-correspondence table
  into a labeled feature CSV, decoupling ground truth from feature computation. First pass data:
  `video1`/`video2` (same route, ~2 min apart, both morning overcast, no lighting confound),
  frames 1-73/1-75 only — the segment confirmed by eye to be the same physical path before the
  walk deliberately diverged around a tree at video1 frame ~73-115 (excluded, not mislabeled).
  **Same qualitative finding replicates**: ~equal accuracy (~0.96 both), but GPC's errors carry
  confidence>0.9 only 15.7% of the time vs. SVM's 65.5% (200-seed bootstrap, non-overlapping CIs).
  Notebook: `PLACE_GPC_LAB.ipynb`. **Desktop demo DONE (2026-07-15, `PLACE_DEMO.ipynb`):** deployment
  GPC+SVM (fit on all 219 pairs) replayed live against 3 route pairs (1 simple: `straight_ref`/
  `straight_query`; 2 complex: `video1`/`video2` full route, `hard_route_night`/
  `hard_route_afternoon2`) via a new `build_demo_pairs.py` windowed-RANSAC-search + best-match
  pipeline — confidence drops sharply (to 0.048) exactly inside the model's own held-out route-
  divergence zone, a good sign. Flagged as not yet the true day/night stress test: the hard_route
  demo reused `LIVE_CAMERA_EXPERIMENTS.md`'s existing anchors, which were themselves picked for
  strong RANSAC signal, biasing the windowed search toward already-easy frames. **Unbiased
  full-brute-force follow-up DONE (2026-07-15):** same-weather route confirmed genuinely
  well-matched throughout (not an anchor artifact, median 80 inliers/97% plausible over 35k
  unwindowed pairs); the real dangerous case — `video1` (overcast) vs `hard_route_afternoon2`
  (sunny), same physical route, documented lighting-mismatch condition — confirms both models
  correctly read most of the route as non-matching, but on marginal-evidence frames (14-31
  inliers, clustered at 2 route stretches) **SVM hits >0.85 confidence on 7 frames GPC never
  matches (GPC mean 0.605 on those same frames)** — the confident-wrong pattern, replicated at
  scale on the project's own hardest documented condition. **Retrain-with-cross-weather-negatives
  follow-up DONE (2026-07-16):** added 150 confirmed-far cross-weather negatives to training (369
  pairs total) — did NOT fix SVM's overconfidence (11 flagged frames after vs. 7 before); GPC's
  caution unaffected either way. Evidence the gap is structural (Laplace posterior variance
  shrinkage vs. Platt-sigmoid calibration), not a training-coverage gap "add more data" could
  patch. **Android port DONE (2026-07-16):** new `place::gpc` (pure-Rust, GPU-free Laplace-GPC
  inference, verified bit-for-bit against the Python fit to 6 decimals), `place-jni`'s
  `verifyLockGpc` (RANSAC verify + GPC score in one native call), wired into `android-app`'s HUD as
  a second, independent lock-state banner alongside the original RANSAC-threshold tracker
  (unchanged) — both logged to `trace.csv`. Cross-compiled for `aarch64-linux-android`, Kotlin
  compiles, debug APK assembles with model+lib correctly packaged.
  **Real-device field validation DONE (2026-07-16), with a genuine root-cause find:** live walks on
  pre-recorded hard-route videos (`video1_ref`/`sunny_hard_ref`/a fresh `july16_ref`) all tracked
  poorly (median 5-6 verified inliers, chance-level) despite matching lighting/route — traced to a
  **capture-pipeline mismatch**, not a tracker or GPC bug: (1) `july16_ref` was extracted at
  1080x1920 vs. the live analysis stream's 720x1280 (fixed); (2) the stock Camera app records video
  in HDR (`arib-std-b67`/HLG, confirmed via ffmpeg's stream probe), and desktop `extract_frames.sh`
  naively downconverts 10-bit HDR to 8-bit SDR JPEGs with no real tone-mapping — a second,
  harder-to-fix mismatch against the live SDR analysis stream. Added a **"Record route"** in-app
  live-recording feature (continuous frame capture through the exact same Camera2 pipeline used for
  live tracking, replacing the discrete-snapshot "quick capture" prototype once a route-length bug
  was found: `seq.rs`'s trajectory scorer needs >= `DS=10` buffered frames and a continuous ordered
  sequence, not disconnected snapshots) plus route auto-discovery (`discoverRoutes()`: any bundled
  asset folder + any on-device `custom_*` recording, no hardcoded list) and in-picker delete
  (routed recordings to external app storage, adb/file-manager-reachable without root). **Result on
  the real hard route, recorded and walked entirely on-device: continuous LOCKED through turns and
  the stairs, the best tracking result of the whole investigation** — confirms the capture-pipeline
  mismatch, not the tracker/GPC logic, was the root cause all along. One real mistake made and
  caught mid-session: the initial on-device route discovery filtered on "any folder with .jpg
  files," which matched `DefectScanActivity`'s own `reports`/`defect_scan_refs` folders too, and the
  user deleted both via the picker's Delete button before the fix landed (filter now requires the
  `custom_` recording prefix). Deferred: on-device video import (paused, not needed now that
  recorded-through-the-app routes work this well), bootstrap CI on the crossweather confident-wrong
  gap, isotonic/temperature recalibration of the SVM if fixing it becomes the goal.

- **mining_gpc_lab** — `gp_engine/mining_gpc_lab/LAB_PLAN.md` (sketched 2026-07-17) — same
  confident-wrong question as `mnist_gpc_lab`/`place_gpc_lab`, applied to ore/waste classification
  on `mining_mpdok`'s real Carlin Trend Au geochemistry (NURE-HSSR, 4,106 samples, P95 cutoff).
  **Phase 0+1 DONE (2026-07-17):** `datasets.py` reproduces `01_data.ipynb`'s filter exactly (n=4,106,
  cutoff 0.0109 ppm). Phase 1 found `P(ore)` never crosses 0.5 for either model at ~5.1% prevalence
  (threshold-0.5 metrics degenerate) — fixed by selecting/reporting on average precision, matching
  `mining_mpdok` fig17's own methodology. Result: **AP 0.240±0.053 (GPC) vs 0.082±0.035 (SVM)**,
  confidently-wrong-on-a-miss rate **52.8% [51.7,53.9] (GPC) vs 99.8% [99.7,99.9] (SVM)**, 200-seed
  bootstrap — the sharpest separation of the three GPC-vs-SVM labs so far. Also found a real engine
  gap: `gp_classifier.py`'s `LaplaceBinaryGPC` had no ARD (scalar `ell` only, unlike `gp_core.py`'s
  regression path) — later fixed in Phase 4.
  **Phase 2 DONE (2026-07-17):** rather than a confusion-matrix $ layer (blocked by Phase 1's
  degenerate-threshold finding), reused `mining_mpdok` fig17/fig18's own ranked top-k drilling-
  campaign economic model unchanged ($1M/target, $50M/HG discovery) — needs no threshold, only an
  ordering. **GPC − SVM net advantage at the k=50 headline campaign: $423M [$396M, $450M] (95%
  paired bootstrap CI, 200 seeds)** — larger in absolute terms than fig18's own FRK-vs-MPDOK gap
  ($200M), though a different claim (classifier ranking quality, not regression-then-rank).
  **Phase 3 DONE (2026-07-17) — the lab's one open question, answered:** regenerated
  `mining_mpdok`'s synthetic nested-Matérn field (6% nugget) at the full Carlin Trend set and
  reran Phase 1's methodology. The nugget-masking effect that widened MPDOK's regression advantage
  over FRK as noise dropped (fig19: 13pp→42pp) does **not** transfer to classification calibration
  — the GPC-vs-SVM confident-wrong gap *narrowed* slightly instead (47.0pp real-Au → 40.4pp
  synthetic). Read as a negative result that sharpens the finding: Phase 1/2's advantage looks like
  an intrinsic Laplace-vs-Platt calibration property (consistent with `place_gpc_lab`'s own
  conclusion), not a masking artifact that would inflate under cleaner data the way MPDOK's
  regression edge does. All three core phases (0/1/2/3) now done; `MINING_GPC_LAB.ipynb` built and
  executed. **Phase 4 DONE (2026-07-17):** extended `gp_classifier.py`'s `LaplaceBinaryGPC` to
  support ARD (vector `ell`, pre-scale-then-isotropic-distance identity from `gp_core.py`, verified
  bit-identical to isotropic at length 1) — a real, reusable engine fix, not a lab-local workaround.
  Fit per-pathfinder-element lengthscales (As/Sb/Ag/Cu/Zn/Tl, d=8 with spatial) via LML-maximizing
  Nelder-Mead (same convention as `gp_hyperopt.py`/`gblup_hyperopt.py`). **Aggregate matches Carlin
  pathfinder theory** (As/Sb/Tl mean ell/std 4.08 vs. Ag/Cu/Zn 6.47, shorter=more informative) **but
  the individual ranking doesn't cleanly confirm it** — Cu is actually the single shortest
  lengthscale of all six, driven by Zn's outlier long (pruned) lengthscale dragging the base-metal
  mean up; reported honestly rather than rounded to fit the theory. Adding pathfinders roughly
  doubles AP for both models (GPC 0.240→0.493, SVM 0.082→0.439) and narrows but doesn't close the
  confident-wrong gap (47.0pp→38.5pp, 50-seed bootstrap) — consistent with the gap being
  substantially intrinsic to the Laplace-vs-Platt mechanism. `MINING_GPC_LAB.ipynb` updated with the
  Phase 4 section and re-executed, 0 errors. **Follow-up (2026-07-17, `economic_layer_pathfinder.py`,
  Fraser asked directly whether Phase 4 moves the Phase 2 $ result):** reran Phase 2's exact top-k
  campaign economics on pathfinder-ARD-ranked candidates, 200 seeds. **The economic gap narrows far
  more (7.5x) than AP or confident-wrong would suggest**: $423M [$396M,$450M] spatial-only →
  **$56M [$47M,$66M]** with pathfinders, even though both models' absolute net value roughly
  doubles-to-triples (GPC $565M→$1,132M, SVM $142M→$1,076M). Read: ranking quality at the very top
  of a candidate list converges faster than calibration quality — richer features make the
  "obviously anomalous" points easy for either model to rank correctly (what the top-k economics
  measure), while SVM's confident-wrong rate on its remaining misses stays far worse (77.5% vs.
  39.0%, unchanged) since a ranked campaign doesn't penalize a miss by how confident the wrong call
  was. The two metrics tell genuinely different parts of the story. Only Phase 5 (optional FRK
  baseline) remains.

- **Engine roadmap note (2026-07-17, `gp_engine/PLAN.md` §6c)**: documented (not yet decided)
  whether `LaplaceBinaryGPC`'s new ARD fit (Nelder-Mead over log-hyperparameters) needs a
  gradient-based (L-BFGS-B) port as feature sets grow past this lab's d=8 — checked directly that
  the d=8 NM fit had converged with budget to spare (plateaued by eval ~275/360), and set concrete
  triggers (d>~15-20, multi-start disagreement, prohibitive wall time) rather than porting
  preemptively. Read before starting any future ARD fit with a much larger feature panel.
- **`gp_engine/EXPLORATION_APPLICATIONS_ROADMAP.md` (2026-07-17)**: planning doc generalizing
  `mining_gpc_lab`'s pattern (calibrated GPC beats uncalibrated SVM on binary economic decisions
  gated by expensive verification) to other exploration verticals — ranked candidates: porphyry
  copper (cheapest next step, nearly a direct rerun with new USGS data), critical minerals/REE
  (highest narrative value, harder data/labeling), oil & gas drill/no-drill (most direct reuse of
  the ranked-campaign economic model, hardest public-data situation), CCS site suitability (newest,
  highest-stakes false-negative, least mature public data). Explicitly flags this as one verified
  data point (gold) generalized by hypothesis, not four proven results.

- **bayesian_decision_lab** — `gp_engine/bayesian_decision_lab/LAB_PLAN.md` (sketched 2026-07-17)
  — tests whether keeping the full GP posterior (mean + variance) as an input to a real 3-action
  loss-matrix decision rule (Skip/Probe/Drill) beats thresholding either model's point probability
  — isolating "does variance itself add decision value" from "GPC's mean is just better calibrated"
  via an explicit control condition (GPC mean-only vs. GPC full-posterior vs. SVM). Reuses
  `mining_gpc_lab`'s already-fitted models/data verbatim, no refitting; **no engine change needed**
  — `LaplaceBinaryGPC.predict()` already returns `(mean, var, prob)`. Conceptual precedent:
  `MPDOK/gp_regression/bo_engine.py`'s Expected Improvement. **Phase 0 DONE (2026-07-17):**
  `models.py` reproduces `mining_gpc_lab`'s exact frozen seed-0 fit, verified against that lab's
  stored prediction ranges; `decision.py` implements the payoff matrix + Bayes action rule +
  variance-blind control. **Found and fixed a real bug along the way**: the first illustrative
  Probe payoff ($2.0M gross) made Probe mathematically never the Bayes-optimal action at any
  P(ore) — its breakeven-vs-Skip probability was already past where Drill's leverage overtakes it,
  so there was no probability band where Probe could win. Fixed ($5.0M gross gives a real niche,
  P(ore) in (0.010, 0.021)) and now checked by an explicit `has_probe_niche` invariant rather than
  eyeballing action counts. **Phase 1 DONE (2026-07-17) — a genuinely surprising result, not a
  hypothesis confirmation:** 200-seed comparison ranks **GPC-mean-only ($1,308.8M) >
  GPC-full-posterior ($1,290.5M) > SVM ($1,277.6M)**, all pairwise gaps bootstrap-robust. The
  variance-aware posterior is measurably *worse* than the variance-blind mean control (−$18.3M/seed
  [−$23.8M,−$12.6M]) — opposite of the lab's motivating hypothesis. Mechanism checked directly (not
  guessed): the Laplace/MacKay moment-matching correction shrinks the latent toward the GP prior's
  50%, not the true ~5% ore base rate — a uniform upward nudge on this data (every fitted mean sits
  well below the prior), not a selective lift of genuinely high-upside points, tipping marginal
  points into Drill that the naive mean would have sent to Probe. GPC still clearly beats SVM either
  way — doesn't contradict `mining_gpc_lab`'s calibration findings, which measure a different thing
  (aggregate ranking quality, not a single asymmetric-payoff point decision). **Phase 3 flagged as
  needing re-scoping before proceeding**: as originally planned it would likely just re-observe
  Phase 1's finding rather than test the actual risk-seeking/tail-exploiting mechanism the lab's
  motivating framing described (closer to `bo_engine.py`'s Expected Improvement, which rewards
  variance directly, than to a recalibrated mean) — a decision point for the next session, not
  resolved unilaterally. `BAYESIAN_DECISION_LAB.ipynb` built and executed, 0 errors — realized-$
  bar, action-mix stacked bar, seed-0 spatial decision maps, and the mechanism scatter (`p_full` vs.
  `p_mean` colored by variance, confirming every point sits above the diagonal). **Phase 3
  REDESIGNED (2026-07-17), proposal only:** proved via Jensen's inequality (sigmoid is convex
  everywhere this dataset's fitted latent means sit, all below 0) that a linear-in-state payoff
  mathematically cannot express "explore high variance for hidden upside" at all — `p_full` already
  *is* the Bayes-optimal quantity for that payoff structure, so Phase 1's finding is real but isn't
  a critique of posterior variance in general. Two real alternatives proposed: (1) continuous-grade
  option-value payoffs (reuses `gp_engine`'s regression path + `bo_engine.py`-style convexity, a
  different lab really), (2) sequential value-of-information (Probe generates a real posterior
  update via rank-one GP conditioning, then a second action is chosen — the literal formalization of
  "explore ambiguous evidence instead of committing"). **Confirmed and built (2), 2026-07-17**
  (`voi.py`/`run_voi.py`/`bootstrap_voi.py`): Probe now pays off via a local 1D Gaussian-conjugate
  posterior update (Gauss-Hermite quadrature over the existing mean/var, no engine changes, no full
  refit). **Structural prediction confirmed exactly**: SVM and GPC-mean-only (`var=0` by
  construction) chose Probe zero times across all 200 seeds — not rarely, exactly zero, as the
  theory requires; only GPC's full posterior ever found it worthwhile (mean 69.1 probes/seed).
  **Headline ranking now matches the lab's original hypothesis, reversed from Phase 1**:
  GPC-full-posterior ($1,343.4M) > GPC-mean-only ($1,299.4M) > SVM ($1,276.7M), all pairwise gaps
  robust (+$66.7M/+$43.9M over SVM/mean-only). Read alongside Phase 1, not instead of it: both
  results are correct, and the honest lesson is that "does posterior variance help" depends on
  whether the *decision structure* has anywhere for that information to go, not on the variance or
  model alone. **Cost/benefit accounting added (Fraser's request)**: the $57.3M/seed
  [$51.0M,$64.2M] value of the Probe option comes overwhelmingly from avoiding drilling losses on
  ambiguous ground (~68.2 true-waste probes/seed vs. ~0.9 true-ore) — probed sites would have lost
  money net if drilled directly (−$23.9M/seed) or scored exactly $0 if skipped; probing beats both
  (+$40.9M/seed). Honestly "cheap insurance on ambiguous ground," not "an oracle for hidden
  bonanzas" — worth stating plainly rather than defaulting to the more dramatic framing.
  `BAYESIAN_DECISION_LAB.ipynb` updated with the full Phase 3 section (6 charts total), rebuilt and
  re-executed, 0 errors. **Phase 2 DONE (2026-07-17):** swept Drill's gross payoff 20x (breakeven
  probability 10%→0.5%) on the sequential-VoI framework, neither model ever refit (only the cheap
  decision-rule step is recomputed per sweep point). **GPC's advantage over SVM is regime-dependent,
  not a fixed number**: peaks at +$294.3M around a moderate breakeven probability (5%), compresses
  to single digits (~$9M) when Drill becomes a near-automatic yes for everyone, ticks back up
  slightly at the very extreme end ($19.0M) rather than continuing to shrink monotonically — flagged
  plainly, not smoothed over. Never flips sign anywhere on the grid tested. `BAYESIAN_DECISION_LAB
  .ipynb` now covers all three phases, rebuilt and re-executed, 0 errors, 7 charts total. **Phase 4
  effectively begun (2026-07-18)**: `decision.py`/`voi.py` — the payoff matrix, Bayes action rule,
  and sequential VoI Probe mechanism — were moved out of this lab into `gp_engine/` as genuinely
  shared modules, specifically so `porphyry_cu_gpc_lab` (and future labs) could reuse the whole
  decision framework directly instead of copy-pasting; this lab's own scripts (`models.py`,
  `run_lab.py`, `bootstrap_study.py`, `run_voi.py`, `bootstrap_voi.py`, `cost_ratio_sweep.py`) each
  gained one `sys.path.insert` line to keep resolving after the move, verified to reproduce their
  exact pre-refactor numbers. This is the reusable core Fraser wants to keep extending going
  forward — see `porphyry_cu_gpc_lab`'s entry below for its first reuse.

- **porphyry_cu_gpc_lab** — `gp_engine/porphyry_cu_gpc_lab/LAB_PLAN.md` (sketched 2026-07-18) —
  first lab from `EXPLORATION_APPLICATIONS_ROADMAP.md`'s ranked list (porphyry copper, the cheapest
  validation of whether the mining_gpc_lab pattern generalizes). **Data sourcing done**: found the
  same USGS NURE-HSSR "reanalysis" program `mining_gpc_lab`'s Nevada file came from, but multi-state
  (DOI 10.5066/F7765DHF, AZ/CA/ID/MT/NV/NM/UT/OR, 54,157 records) — downloaded and saved to `data/`.
  Arizona subset (7,633 samples, comparable scale to Carlin Trend) is the classic US porphyry Cu(-Mo)
  province (Morenci, Bagdad, Ray, Miami-Globe, Sierrita, etc.); already sanity-checked (not just
  downloaded) — top-Cu samples cluster exactly at the known Tucson/Mesa-area districts, not randomly.
  Same 51-element ALS ICP-MS suite as before, now including Mo/Re (real porphyry Cu-Mo pathfinders
  per the classic Lowell-Guilbert zoning model — a genuinely different pathfinder theory than
  Carlin's As/Sb/Tl, not just a relabeling). Open decision flagged, not yet resolved: whole-state
  Arizona (simpler, recommended for Phase 0/1) vs. a tighter named district cluster (closer to
  Carlin Trend's bounded-province framing, needs real boundary research). Bonus: the same downloaded
  file covers NV/UT/NM/MT too, all real porphyry Cu states — future verticals need no new data
  hunting. Notes `bayesian_decision_lab`'s framework is now a reusable asset this lab could adopt
  directly rather than rebuilding `mining_gpc_lab`'s simpler economics from scratch.
  **Phase 0 DONE (2026-07-18):** `datasets.py` — whole-state Arizona, Cu P95 cutoff (133.50 ppm,
  5.0% ore, 385/7,633), stratified `spatial`/`spatial_pathfinder` splits. **Real data-quality issue
  found and handled, not glossed over**: Re_ppm is 91.3% below detection (physically sensible — only
  shows up near real molybdenite), Au_sq_ppm 9.9% — both floor-imputed before log-transforming
  rather than dropped, so pathfinder censoring doesn't shrink the primary Cu dataset.
  **Phase 1 DONE (2026-07-18) — the roadmap's "does the pattern generalize" question, answered
  yes:** same degenerate-threshold-at-0.5 pattern as gold (confirmed, not assumed), AP pivot
  applies again. 200 seeds: **GPC AP 0.278±0.042 vs. SVM 0.104±0.046** (~2.7x); confidently-
  wrong-on-a-miss **GPC 46.7% [45.9,47.6] vs. SVM 93.3% [92.1,94.5]** — a ~46.6pp gap, same order
  of magnitude as gold's 47.0pp and just as statistically robust, on a genuinely different
  commodity, geology, pathfinder suite, and ~1.86x larger dataset.
  **Phase 2 DONE (2026-07-18) — skipped straight to `bayesian_decision_lab`'s sequential
  value-of-information framework** (Fraser's direction, wanting that framework generalized rather
  than rebuilding ranked-campaign economics per lab). **Required refactoring `decision.py`/`voi.py`
  out of `bayesian_decision_lab/` into `gp_engine/` as genuinely shared, dataset-agnostic modules
  first** — both were already pure functions of arrays with zero mining-specific code, just coupled
  to that lab's import order; moved and verified the refactor changed nothing (both labs reproduce
  their exact prior numbers). Porphyry-specific payoff constants derived (not reused from gold):
  $2M/$150M drill, $0.1M/$15M probe, same relative ratios as gold since neither was rigorously
  sourced. **Result, 200 seeds: GPC-full $8,820.6M > SVM $8,751.8M > GPC-mean $8,705.8M** — GPC-full
  wins again, but **the runner-up flips from gold's ranking** (there, mean-only beat SVM; here, SVM
  beats mean-only), reported plainly as a real dataset-dependent difference rather than forced to
  match gold's story — the honest reading is that only the *correctly-integrated* posterior
  (GPC-full) robustly generalizes; which naive simplification is second-best does not. Cost/benefit
  accounting confirms the same "cheap insurance against drilling into waste" mechanism as gold
  ($235.8M/seed value added by Probe, ~166.5 true-waste vs. ~1.0 true-ore probes/seed), at a larger
  dollar scale. **`PORPHYRY_CU_GPC_LAB.ipynb` built and executed (2026-07-18)** — 4 charts (Phase 1
  AP/reliability/confident-wrong, Phase 2 sequential-VoI value/action-mix/cost-benefit), 0 errors;
  caught and fixed one bootstrap-rng reuse bug while finalizing (shifted SVM's CI by 0.1pp vs.
  the script's own output — now matches exactly, 92.1%). Lab considered wrapped up at this stage
  per Fraser's direction; Phases 3-4 not started.

- **climate_cat_lab** — `gp_engine/climate_cat_lab/LAB_PLAN.md` (sketched 2026-07-23) — after a
  Claude Security scan of `gp_engine` (see `gp_engine/README.md`'s "Known issues / fixes"),
  brainstormed which end applications the OOC solver (full joint GP posterior at n>200k on one
  consumer GPU) unlocks; four ranked directions written up in that session (reinsurance/cat-risk
  tail aggregation, grid/renewable reserve sizing, fleet-scale structural-health VoI, water/
  hydrology siting). This lab specs out #1 first. **Not "GP beats baseline" like every prior lab
  — the target is the failure mode**: linear/flat-correlation aggregation (the actuarial-capital-
  model shortcut, architecturally the same object as the Gaussian copula blamed for 2008 CDO
  mispricing) has zero tail dependence, and climate-correlated loss is exactly the kind of
  structured tail dependence that breaks it hardest. Builds a synthetic oracle DGP (regime-mixture
  + spatial shock → genuine, checkable tail dependence) and a four-method ladder (independence →
  flat correlation → vanilla spatial GP → GP + fitted regime-mixture) scored against that oracle's
  *actual* achieved survival probability and dollar shortfall/over-capitalization — isolating
  whether "fit a GP" alone closes the gap or whether the tail-dependence mechanism itself must be
  modeled (vanilla GP is still elliptical/Gaussian, same zero-tail-dependence property as the
  naive baseline, just better-shaped correlation). Reuses `cvar_gp_lab/cvar_lp.py`'s Rockafellar-
  Uryasev CVaR machinery (reformulated as a capital/retention decision, not portfolio weights),
  `cvar_gp_lab/scenario_gen_gp.py`'s posterior-sampling pattern, and `gblup_lab/marker_kernel.py`'s
  GEMM-trick kernel builder repointed at exposure lat/lon + hazard covariates. Phase 2 explicitly
  targets a book size (~100k-300k policies) past the ~40k in-core ceiling, so it needs
  `gp_ooc_fortran.py`/`gp_ooc_solver.so` — the scale this lab exists to justify. Entirely synthetic
  (no proprietary claims/treaty data); Phase 3 stretch grounds hazard geography in FEMA National
  Risk Index while keeping losses synthetic. **Premise-verification research pass DONE
  (2026-07-23)**, per Fraser's direction that industry-practice claims need real saved sources,
  not links, and must be general practice not one company: six parallel research forks fetched
  and saved verbatim primary/near-primary sources to `climate_cat_lab/research/*.md` (index
  `RESEARCH.md`), each with an explicit verdict, and `LAB_PLAN.md` was rewritten against the
  findings (full research also folded into `LAB_PLAN.md` as an appendix, not just linked).
  **Strengthened**: the flat-correlation-aggregation premise is now confirmed at 3 independent
  frameworks, not assumed — Solvency II SCR (Directive 2009/138/EC Art.104/Annex IV, exact 5x5
  correlation matrix quoted), A.M. Best BCAR ("square root rule," self-critiqued in A.M. Best's
  own methodology doc), and S&P's Insurance Capital Model (whose pre-2023 methodology carried
  Nat Cat risk at a fixed 100% correlation — zero diversification credit — an unplanned finding
  that lands closer to this lab's naive baseline than the original draft even claimed). Gaussian
  copula's zero tail dependence is settled math (Sibuya 1960; Embrechts/McNeil/Straumann 2002;
  Donnelly & Embrechts 2010). **Corrected**: the "Gaussian copula killed Wall Street" framing was
  too strong — the peer-reviewed source (Donnelly & Embrechts 2010) explicitly rejects that
  popular claim (Salmon 2009, *Wired*) and supports only "partly contributed," now the plan's
  language; the "climate trend is making the tail fatter every year" framing was also too broad —
  Swiss Re Institute sigma data says >80% of the long-term insured-loss increase is exposure
  growth, not climate, with a genuine climate-driven signal clearly documented only for specific
  perils (NA wildfire, +14%/yr) — now the plan's narrower, sourced claim, which conveniently
  matches the lab's own wildfire/regime-shock DGP mechanism. Phase 2's book-size figures
  (100k-300k policies, $300-400k avg insured value) are now labeled by source quality: avg
  insured value well-anchored (4 convergent consumer-data sources), policy count reframed as a
  conservative lower bound (a real regional insurer, Florida Peninsula Holdings, implies
  ~400-470k policies per NAIC 2025 data), "tens of billions aggregate" flagged as unsourced
  derived arithmetic. FEMA NRI confirmed real/public/usable, with one open item (license text
  not read verbatim, fema.gov blocked automated fetch). **Phase 0 DONE (2026-07-23,
  `RESULTS_PHASE0.md`):** oracle DGP (`dgp_simulator.py` — regime-mixture + spatial-RBF
  shock field over a synthetic 500-property book, `exposures.py`) checks out on all 4
  sanity checks. Headline: the literature's own tail-dependence coefficient λᵤ (Donnelly &
  Embrechts 2010 definition) is 0.439 for nearby property pairs at q=0.99 (44x the 0.010
  independence baseline), vs. only 0.198 for a Gaussian model fit to the *exact same mean
  and covariance* — two models statistically identical in their correlation matrix disagree
  2x+ on joint catastrophic-loss probability, the concrete numeric version of this lab's
  whole premise. Caught and fixed a real methodology bug along the way: the first draft of
  the tail check conditioned on total-book-loss years instead of each property's own
  marginal quantile, which is confounded by the regime mixture's shared mean-shift and gave
  a backwards (lower-not-higher) result — diagnosed and corrected to the standard
  joint-exceedance-probability λᵤ definition before anything downstream depended on it.
  **Phase 1 DONE (2026-07-23, `RESULTS_PHASE1.md`):** fit all four methods (independence /
  flat correlation / vanilla spatial GP / GP + regime-mixture) on one realistic 60-year
  historical sample, scored against a 500,000-year oracle resample ($10.2M true 99.5%
  capital). Headline: every method badly under-reserves (~93.3% achieved survival vs. the
  99.5% target) — the core finding holds regardless of method. Genuine surprise: a better-
  shaped correlation ALONE (vanilla spatial GP) is statistically indistinguishable from no
  correlation at all (93.30% vs 93.29%) — vanilla GP is still elliptical/Gaussian, exactly
  as LAB_PLAN.md's own Risks section warned. A sample-size sweep (60→500 years) then an
  oracle-cheat diagnostic (true regime labels, real methods never get this) isolated why:
  the regime-mixture method's first classifier diluted its "systemic" fit by mixing ~3.7
  ordinary years per genuine systemic year (fixed top-25%-quantile split vs a true ~6.7%
  regime frequency) — a bias that doesn't shrink with more data, and which vanished (99.54%
  survival) once fed the true labels, confirming the mechanism. Fixed by sizing the
  partition to the model's own fitted frequency estimate (still oracle-free): the corrected
  regime-mixture method then climbs from 94.8% (60yr) to 97.3% (500yr) survival as data
  grows — the ONLY one of the four methods that improves with more historical data at all,
  since it's the only one that can represent a regime; methods 1-3 sit on a flat ~93.3%
  ceiling regardless of sample size. Phase 2 (scale to a realistic book size past the ~40k
  in-core ceiling, requiring `gp_ooc_solver`) — see `RESULTS_PHASE2.md`/`RESULTS_ROBUSTNESS.md`.
  **Phase 3 (2026-08-03, `RESULTS_PHASE3.md`) — fourth application of `gp_engine/
  VOI_DISPATCH_PATTERN.md`** (after `grid_reserve_lab`'s Phase 4, `shm_lab`'s Phase 2,
  `hydro_reserve_lab`'s Phase 2), completing all three originally-parked ideas. This lab was the
  template's own flagged **weakest-fit candidate** on decision-cycle length — resolved as a real,
  workable annual cadence, not a genuinely weak fit: property-cat reinsurance treaties really do
  renew annually, ~60-70% of the global market at the real January 1 date, with a real pre-renewal
  information-gathering window (September's Monte Carlo Rendez-Vous, October's Baden-Baden) — the
  concrete analog for Probe. State = the oracle's own `regime` (resimulable, since
  `dgp_simulator.py` is entirely synthetic — mirrors `grid_reserve_lab`'s bootstrap convention, not
  `shm_lab`/`hydro_reserve_lab`'s fixed-real-dataset one). Skip/probe/drill = retain risk / delay
  binding past Jan 1 for more information / bind reinsurance cover now. **New modeling piece
  required** (`regime_forecast.py`, a near-direct port of `grid_reserve_lab`'s own module given the
  identical DGP shape): a `LaplaceBinaryGPC` fit on a 1-property early-reporting subset (checked
  empirically — 5+ properties already gave AP=1.000) — `regime_mixture.py`'s existing
  `fit_regime_mixture_soft` gives only a point-probability with no propagated uncertainty, the same
  gap the other two real-model VoI labs hit. **Economics uniquely clean**: real cat-XL Rate-on-Line
  (ROL = premium/limit, real range "well below 1%...up to 35-45%") makes the breakeven P(systemic)
  *exactly* the ROL swept, no derivation needed. **Headline, a fourth genuinely distinct outcome for
  this family** (200 seeds): GPC's calibrated mean robustly beats SVM at every ROL tested
  ($8.9M-$88.6M), but GPC's posterior *variance* is measurably *worse* than GPC-mean at 9 of 12 ROL
  points (CIs excluding zero, $1.3M-$7.5M) — not a null result like `shm_lab`, a genuine small
  negative, the clearest demonstration yet of variance actively hurting, via the same
  MacKay-shrinkage-toward-0.5 mechanism `bayesian_decision_lab`'s own Phase 1 first found (true
  base rate 6.67%, far from 50%). Probe's niche fraction was 0.0000 at every seed tested.
  `gp_engine/VOI_DISPATCH_PATTERN.md` updated with this fourth worked example, a corrected
  condition-3 lesson (this lab's "weakest fit" flag was simply never checked against real cadence
  data), and a new "variance can be a small real negative, not just positive-or-null" lesson. With
  all three originally-parked ideas plus `grid_reserve_lab` itself now done, the honest conclusion
  is that "does posterior variance help" is a per-domain empirical question the template helps you
  check, not one it can predict in advance.

- **grid_reserve_lab** — `gp_engine/grid_reserve_lab/LAB_PLAN.md` (sketched 2026-07-27) — ports
  `climate_cat_lab`'s finding (flat-correlation aggregation understates spatially-correlated tail
  risk) to grid operating-reserve sizing: a synthetic wind/solar fleet with a regime-mixture
  "drought" shock (analogous to a Dunkelflaute event) vs. a five-method ladder (deterministic
  reserve heuristic → independence → flat correlation → vanilla spatial GP → GP + soft-EM
  regime-mixture), scored in achieved-reliability and dollar terms (reserve procurement cost vs.
  value-of-lost-load). Reuses `gblup_lab/marker_kernel.py`'s kernel builder, `cvar_gp_lab/cvar_lp.py`
  reformulated as a reserve-sizing LP, and the soft-EM regime-mixture pattern now live in
  `portfolio_studio` (third domain for the same mechanism). **New:** the traditional-method
  baselines (methods 0-2) are implemented in Rust (rayon-parallel Monte Carlo aggregation) so the
  GP-vs-traditional comparison is fair on wall-clock as well as dollars — `rbfx` may also gain a
  predictive-variance API if Phase 2 wants the GP side Rust-callable too (open decision, not yet
  made). Strongest of the three parked ideas (grid reserve / fleet structural-health / hydrology
  sizing) for eventually becoming a live monitored app, since EIA-930 + NREL data are public and
  continuously updated (unlike claims/sensor data). **Research pass DONE (2026-07-27),
  `research/RESEARCH.md`**, same rigor as `climate_cat_lab/research/`: six claims checked, three
  held up close to drafted (Dunkelflaute/"resource drought" real and quantified — ERCOT 82 events
  2018-2022 worst ~146 GWh; EIA-930/NREL WIND Toolkit/NSRDB confirmed public/reusable; VOLL and
  capacity-cost figures real), three forced corrections folded into `LAB_PLAN.md` (LOLE citation
  was wrong standard — BAL-502 series, not BAL-002; the "3+5" reserve heuristic isn't ERCOT's,
  ERCOT's real rules are fixed-2300MW/N-1/2.5-sigma; and the load-bearing premise itself needed
  softening — real ISO practice (MISO LOLE Study, E3 RECAP) already uses real historical
  correlation, not independence, so the actual gap is "spatially-resolved tail-dependence model vs.
  aggregate zone/fleet-level historical correlation," narrower than first drafted). VOLL updated to
  ERCOT's current $35,000/MWh (was $9,000-30,000 guess); reserve-cost now PJM ($120,150/MW-yr,
  2026/27 BRA) and MISO (~$79,200/MW-yr) real cleared prices. **Phase 0 DONE (2026-07-27,
  `RESULTS_PHASE0.md`):** synthetic 100-site wind fleet (`fleet.py`), regime + spatial-shortfall
  oracle (`dgp_simulator.py`) passes all 4 sanity checks — near-pair tail-dependence coefficient
  λᵤ=0.607 (q=0.99), 61x the 0.010 independence baseline, vs. only 0.082 for a Gaussian model fit
  to the identical mean/covariance (7.4x gap, stronger separation than `climate_cat_lab`'s own 2.2x
  Phase 0 result). One real methodology fix en route: shortfall had to be one-sided
  (`max(expected−actual,0)`) not signed — signed shortfall is ~zero-mean noise on normal days,
  letting the fleet-wide drought regime's jump swamp the spatially-decaying signal (near/far pairs
  came out indistinguishable, 0.19 vs 0.17) until fixed. **Phase 1 DONE (2026-07-27,
  `RESULTS_PHASE1.md`):** five-method ladder scored against a 500,000-day oracle (true required
  reserve 5,059.8 MW). ERCOT N-1 (571 MW) and generic "5% wind" (728.5 MW) rules badly under-reserve
  (2.9%/34.5% achieved reliability) — flagged honestly as rules built for a different risk
  (single-contingency), not a claim ERCOT is under-reserved in real life. **Central finding, exactly
  mirroring `climate_cat_lab`'s own Phase 1**: vanilla spatial GP (2,644 MW, 95.3%) does NOT beat the
  real-ISO-practice aggregate-correlation baseline (3,047 MW, 96.3%) — only GP + soft-EM
  regime-mixture (5,925 MW) clears the target (≥99.9998%), at ~17% over-procurement (~$104M/yr) vs.
  every other method's multi-billion-dollar under-procurement gap. `reserve_baseline` (Rust, rayon)
  beat a NumPy reference by 34.6x-46.2x at 500k scenarios. One real bug caught and fixed: a
  `sys.path.insert(0,...)` cross-lab import shadowed this lab's own `regime_mixture.py` with
  `climate_cat_lab`'s same-named module — fixed by switching to `sys.path.append`. **Phase 2 DONE
  (2026-07-27, `RESULTS_PHASE2.md`) — a real scoping pivot, not the original NREL/OOC plan**:
  EIA-930's bulk CSV endpoint (no API key needed, unlike its Akamai-blocked main site) gives 15 real
  US Balancing Authorities as "sites," genuine 2023-train/2024-test split (no synthetic oracle for
  real data). **Follow-up (same day) found the REAL bug**, per Fraser's two requested fixes
  (harmonic climatology instead of 30-day rolling mean; data-driven `p_hat_bounds` instead of
  copied from `climate_cat_lab`): both fixes were correct but, applied alone, made the symptom
  *worse* (a 99.5%/0.5% split) — which pinpointed the actual bug, summing already-CLIPPED per-site
  shortfall for regime detection (confirmed via an exact match to the fraction of near-zero-clipped
  days). Fixed by feeding the GMM the fleet-wide SIGNED, unclipped total deviation instead: a
  genuine ~50/50 split resulted (symmetric `gmm_means`, flat month-by-month — checked, not
  seasonal), and **Method 4 (soft-EM) now clearly beats Method 3 (vanilla GP) again** ($4.95B vs
  $5.27B/yr), matching Phase 1 and reversing the earlier "statistical tie." Open finding that
  survived the fix: the real regime is a persistent ~50/50 above/below-trend split, not the
  synthetic world's rare ~5-7% drought — a genuinely different characterization, not the same
  story transplanted. Also found: one real EIA-930 data bug (SWPP, 3.59M MW on one date) inflating
  a nameplate proxy 10x, fixed via per-BA percentile winsorization; and at 15 real BA-level sites
  this domain never reaches the ~40k OOC ceiling — an honest finding about real decision resolution
  (BA/zone-level, not per-turbine), not unfinished work. **`GRID_RESERVE_LAB.ipynb` built
  (2026-07-27)** — math + all charts, 0 errors — including a direct soft-vs-hard-partition
  experiment (same GMM, different downstream GP-fitting) answering Fraser's "does soft-EM win by
  not throwing away data?" question: yes, but conditionally — a real ~9% win on the synthetic
  oracle's rare (~5%) regime, a statistical tie on real data's balanced (~50%) regime where hard
  isn't data-starved to begin with. Confirms `climate_cat_lab`'s mechanism from a second domain.
  **Second research pass (2026-07-27)** verified a mid-session "why isn't this adopted" narrative
  (SERVM/MARS/GE-MAPS, real-time vs. planning timescales, market transparency, regulatory
  asymmetry) — found real but overstated: GE-MAPS isn't actually a Monte Carlo tool (it's
  deterministic); ERCOT's ORDC already proves real-time PROBABILISTIC reserve mechanisms are
  viable, contradicting "real-time is always deterministic"; market-transparency/legal-defensibility
  concerns are real norms but unconfirmed as applied to regime-mixture models specifically (none has
  ever been proposed to test the objection against). Net honest read: adoption gap looks more like
  inertia + an unbuilt scenario-generator than a hard regulatory/computational barrier.
  **Phase 4 — sequential-VoI dispatch layer, DONE (2026-08-02, `RESULTS_PHASE4.md`)**: a first,
  cheap test (Fraser's direction: "test the waters") of whether `gp_engine/decision.py`/`voi.py`'s
  Skip/Probe/Drill sequential value-of-information framework (`bayesian_decision_lab` →
  `porphyry_cu_gpc_lab`) adds value stacked on top of this lab's own static annual reserve-sizing
  number, rather than only being a mining/classification pattern. Reframed as a day-ahead dispatch
  decision (state = `dgp_simulator.py`'s own per-day drought/normal `regime`; skip = do nothing;
  probe = pay for an expedited early-telemetry nowcast; drill = commit `delta_mw` of extra reserve
  for that day), with a new 1-site "fast-reporting" GPC/SVM nowcast classifier
  (`regime_forecast.py` — deliberately tiny, since more early-reporting sites trivially separate
  the regime and leave nothing for Probe to resolve) and economic constants derived from
  `reserve_calc.py`'s own already-sourced VOLL/PJM figures (no new sourcing). **Two honest findings,
  200 seeds**: (1) GPC-full-posterior beats variance-blind SVM/GPC-mean by a real, robust
  $10.44M/yr [$9.67M,$11.31M] — a third independent confirmation of this codebase's recurring VoI
  mechanism (posterior variance only pays off when the decision structure has somewhere for it to
  go); (2) day-by-day dispatch at this lab's own derived VOLL/reserve-cost ratio (breakeven
  P(drought)=0.0016, 40x below the true ~6.5% base rate) costs **~$173M/yr more** than the existing
  static annual buffer ($277-288M/yr vs. $104M/yr) — every model ends up committing reserve on
  ~93%+ of days regardless of forecast quality, an "always commit" regime the lopsided VOLL/
  reserve-cost ratio creates structurally, flagged as depending on an unsourced pricing assumption
  (PJM's annual capacity rate divided by 365, treated as a day-ahead price) rather than a proven
  economic conclusion. **New shared template**: `gp_engine/VOI_DISPATCH_PATTERN.md` generalizes
  this reuse (a four-condition checklist plus a "check the breakeven probability against the true
  base rate first" lesson from finding #2 above) for testing on `shm_lab` (best next candidate —
  condition 4's $ sourcing is the open item, not the method) and `hydro_reserve_lab`
  (condition 3, decision-cycle length, is the open question there); `climate_cat_lab`'s reinsurance
  retention decision flagged as the weakest fit (annual/multi-year commitment cycle).
  Lab considered feature-complete at this stage.

- **shm_lab** — `gp_engine/shm_lab/LAB_PLAN.md` (sketched 2026-07-28, **research pass DONE
  2026-07-28**, nothing fit yet) — fourth port of the soft-EM regime-mixture mechanism
  (`climate_cat_lab` → `cvar_gp_lab` → `grid_reserve_lab` → here), applied to bridge
  structural-health monitoring: classical temperature/EOV-correction vs. vanilla spatial GP vs. GP
  + soft-EM regime-mixture, scored on detecting a **real** structural-state change (not a synthetic
  oracle) — the KW51 railway bridge (Leuven, Belgium) was physically retrofitted 2019-05-15 to
  2019-09-27 (a real construction-defect correction, confirmed from the dataset owner, not routine
  planned work) partway through its 15-month monitoring campaign. Dataset: Zenodo DOI
  10.5281/zenodo.3745914, direct download today (CC-BY-NC-SA 4.0), chosen after a real dead end —
  BCSIMS (BC's own bridge-monitoring network, including Ironworkers Memorial and Port Mann Bridge)
  turned out to gate actual SHM sensor data behind an "authorized" (scientist/engineer) account
  tier, confirmed directly from its own design paper (Kaya/Ventura/Huffman/Turek draft, p.9) —
  public registration only unlocks earthquake shake-maps. Fraser's call: use a genuinely open
  benchmark now, revisit BC access only after the lab is complete. **Carries a prominent, explicit
  disclaimer (Fraser, 2026-07-28): this lab is theoretical/educational only, not to be relied on
  for any real-structure decision, and any real qualified-engineer review of where its GP soft-EM
  approach differs from certified methods is a starting point for questioning assumptions, not an
  answer in itself** — required in the app UI too, not just the docs. **Distinguishing feature vs.
  every prior lab in this family: the application layer (FastAPI, ingest a chosen public dataset →
  run both calculations → present side by side with inline math documentation) is the stated point
  of this lab, not an optional Phase 3+.** **Research pass (2026-07-28, `research/RESEARCH.md`)
  found one real, load-bearing correction**: GP regression for SHM EOV removal (including
  heteroscedastic GP) and regime-switching cointegration are BOTH already established, published
  techniques — this lab's own contribution narrowed from "does GP/regime-awareness help SHM"
  (already answered, by others) to "does the specific soft-EM mechanism already validated three
  times in this codebase transfer competitively to this domain and this real event." Real,
  sourced SHM cost figures found (~$29,000 Florida scour-monitoring system; ~$11,900/pier
  cathodic-protection monitoring) but no figure yet for the false-negative/missed-damage cost
  side — the economic-layer stretch stays explicitly unconfirmed. **Phase 0 DONE (2026-07-28,
  `RESULTS_PHASE0.md`)** — downloaded and directly inspected `trackedmodes.zip` (not assumed from
  a page summary this time): 11,328 hourly samples, 14 tracked modes, 11 environmental covariates,
  real pre/during/post-retrofit split. Both sanity checks pass on real data: the EOV/temperature
  confound is real and mode-dependent in KW51 specifically (correlation -0.103 to **-0.738** across
  well-identified modes), and a real retrofit signal exists — mode 5 (weakest temperature
  confound) shows a clean **+2.07%** frequency rise post-retrofit, physically consistent with the
  strengthening work, while mode 8 (strongest confound, -0.738) shows an ambiguous shift that could
  be retrofit or could just be the pre/post windows' different seasonal temperature mix — **the
  exact confound this lab exists to disentangle, now observed directly rather than only argued
  from citations.** **Phase 1 DONE (2026-07-28, `RESULTS_PHASE1.md`)** — fair-fight ladder
  (classical regression → vanilla GP → GP + soft-EM regime-mixture) on daily-aggregated data,
  train/held-out/during/post split, matched 11.5% false-alarm-rate calibration. **Headline finding
  is a genuine mixed/negative result, reported as found**: the soft-EM regime-mixture shows no
  clear advantage over the simpler methods — well-calibrated but no earlier-detecting on the two
  least temperature-confounded modes (5, 12), and poorly calibrated (23-85% false-alarm rate vs.
  the other methods' matched 11.5%) on the more confounded modes, including mode 8 — the single
  case this lab most wanted regime-awareness to help with. Classical regression vs. vanilla GP was
  itself a wash, mode-dependent, no consistent winner — consistent with the research pass's
  finding that GP alone isn't a novel advantage here. **One real data-leakage bug found and
  fixed**: an earlier version fit regime B's EM directly on the same held-out-normal points later
  used to measure its own false-alarm rate, producing a spurious ~1.0 false-alarm reading on every
  mode; fixed by restricting the EM fit to during+post only and scoring held-out days post-hoc
  against the frozen, converged mixture. A smaller earlier bug (chronological train/held-out split
  put winter in train and spring in held-out, causing pure extrapolation error) was also caught and
  fixed via a random split. **Phase 1b/1c DONE (2026-07-28, `RESULTS_PHASE1B.md`)**, triggered by
  Fraser's own structural hypothesis after Phase 1's muted result: prior soft-EM wins in this
  codebase came from a recurring regime AND cross-sectional pooling across correlated units, and
  Phase 1's per-mode fits had neither. Phase 1b built a **joint** model sharing one
  regime-responsibility trajectory across all 5 modes (a one-time retrofit can't be made
  recurring, but pooling across modes was restorable) — false-alarm rate dropped to 5.8% with
  detection just 6 days after the true retrofit start, a dramatic improvement over any single
  mode. **But Phase 1c's honest control (a naive joint chi-squared statistic summing the same five
  per-mode z-scores, no soft-EM at all) matched that exact same detection speed and flag rate**, at
  11.5% false-alarm rate — meaning pooling across modes, not the soft-EM mechanism, did almost all
  of the work; soft-EM's own isolated contribution is real but modest (roughly half the
  false-alarm rate at matched detection speed, on a small enough sample — 3 vs. 6 flagged days —
  to be suggestive rather than decisive). **Sharpened, transferable rule of thumb for the rest of
  this codebase's soft-EM family**: check whether plain pooling across correlated units already
  gets most of the benefit before reaching for soft-EM's added complexity — soft-EM is worth it
  when regimes are genuinely rare/imbalanced or the within-regime relationship is nonlinear enough
  that a likelihood-ratio test beats a sum-of-squares, not merely whenever multiple correlated
  signals exist. **`SHM_LAB.ipynb` built (2026-07-28)** — single reference notebook (math + all
  charts, 0 errors), consolidating the disclaimer, the EOV/retrofit findings, the mixed per-mode
  result, and the pooling-vs-soft-EM comparison. **Lab considered feature-complete at Phase 1c** —
  Fraser's call to stop there rather than proceed to Phase 2-original (the FastAPI app), since
  soft-EM's contribution turned out modest and precisely characterized rather than a clear win
  worth building an app around.
  **Phase 2 (2026-08-02, `RESULTS_PHASE2.md`) — a different, later addition, not the FastAPI app**:
  second test of `gp_engine/VOI_DISPATCH_PATTERN.md` (after `grid_reserve_lab`'s Phase 4), per
  Fraser's direction to try this lab next with the same template, anticipating cost-sourcing might
  be "a separate problem" (confirmed — `research/06_inspection_and_failure_cost.md` found a
  well-corroborated I-35W collapse cost anchor ($234M rebuild) but only an unverified-against-
  primary-source inspection-cost figure, so the breakeven-probability sweep was made the primary
  result instead of one unsourced headline number). Reframed Skip/Probe/Drill as inspect/wait/
  remediate on KW51's real retrofit label (not a synthetic oracle); required real new modeling
  (`damage_classifier.py`, a genuine `LaplaceBinaryGPC` fit on the five modes' existing z-scores,
  since every model in this lab was GP regression with no native `(mean,var,prob)` triple) and a
  bootstrap convention closer to `bayesian_decision_lab`/`porphyry_cu_gpc_lab` (fresh splits of one
  fixed real dataset) than `grid_reserve_lab`'s synthetic-Monte-Carlo one. **Three findings novel
  to this lab family**: (1) the real retrofit is detected almost perfectly by the classifier
  (AP≈0.999–1.000, checked not engineered — verified to persist even with a single mode or an
  8-day training set); (2) GPC's posterior *variance* therefore adds nothing measurable across a
  full breakeven sweep (GPC-full bit-identical to GPC-mean in all 200 seeds at every point tried)
  — the null counterpart to `grid_reserve_lab`'s modest-but-real positive result; (3) GPC's
  calibrated *mean* itself doesn't uniformly beat SVM here — SVM wins by a real, robust margin at
  moderate-to-high breakeven probabilities (this dataset's positive class is the *majority*, 82.4%,
  a first for this lab family), traced to the same MacKay moment-matching shrinkage-toward-0.5
  mechanism `bayesian_decision_lab`'s own Phase 1 found, miscalibrating in the opposite direction
  since the true base rate sits far above 50% rather than far below it. `gp_engine/
  VOI_DISPATCH_PATTERN.md` updated with both worked examples and three new checklist lessons
  (recurring-regime-vs-change-point; regression-only source models need a new classifier stage;
  check for near-perfect separability before trusting a null variance result).

- **hydro_reserve_lab** — `gp_engine/hydro_reserve_lab/LAB_PLAN.md` (litmus-test pre-check DONE
  2026-07-28, **research pass DONE 2026-07-28**, `research/RESEARCH.md`, same rigor as
  `climate_cat_lab`/`grid_reserve_lab`/`shm_lab`) — the third parked idea from `grid_reserve_lab`'s
  own text ("grid reserve / fleet structural-health / hydrology sizing"), never scoped beyond that
  one-line mention until this session. Tested directly against `gp_engine/PLAN.md` §7's cross-lab
  soft-EM litmus test (the checklist `shm_lab`'s own muted result forced into existence) BEFORE any
  code was written — **passes both required conditions, better-grounded than `shm_lab` ever was.**
  Target basin chosen: the **Colorado River Basin**, in its documented 23-year "megadrought," the
  worst in 1,200 years, with two real dated shortage-tier escalations (Lake Mead Tier 1, August
  2021; Tier 2, August 2022) — condition 1 (regime recurs) verified via multiple independent real
  mechanisms (ENSO, PDO, a 2025 paper on increasing-frequency Western-US "hydrological whiplash").
  Condition 2 (rare/imbalanced) verified with a real number — **5.5% historical extreme-drought
  likelihood** (U.S. Drought Monitor) — **plus a real, flagged complication**: by 2022 nearly the
  whole basin was in extreme drought simultaneously, a possible sign of a nonstationary (not
  fixed-rate) regime, to be checked directly in Phase 0, not assumed either way. **The load-bearing
  correction this lab makes in advance, not after the fact**: the Bureau of Reclamation's own CRSS
  planning model already uses historical/paleo-record ensemble scenario resampling (30-1,000+
  traces) — genuinely correlation-aware, NOT the naive-independence strawman `grid_reserve_lab`
  first assumed and had to walk back. This lab's hypothesis is written already narrowed: does an
  explicit fitted regime-mixture add value over resampling-based real practice, specifically where
  the regime may be shifting nonstationarily or in representing a rare regime's tail more sharply —
  **with a mandatory non-mixture pooling control run from the start** (per `shm_lab`'s own Phase
  1c lesson), not bolted on after an initial result. Real, sourced (if fragmented by user
  class/sub-basin) dollar figures found: $20.6B/yr total basin value; conservation ~$417/AF (as
  low as $69.89/AF) vs. new-supply projects >$2,400/AF; municipal ($512/AF) vs. agricultural
  ($30/AF) price disparity; crop value $814/AF (Lower Basin) vs. $131/AF (Upper Basin). Real
  reliability-standard analogue found: "Firm Yield" (the standard technical quantity) and
  Seattle's real 98%/"1-in-50-year" standard (illustrative, not yet confirmed basin-specific).
  **USGS data access directly verified open, twice** — both the legacy `waterservices.usgs.gov`
  API and its documented 2027 successor `api.waterdata.usgs.gov` returned real data via live,
  unauthenticated `curl` requests in this session, resolving the one risk flagged at the pre-check
  stage. **Phase 0 DONE (2026-07-28, `RESULTS_PHASE0.md`)** — pulled real daily discharge for five
  Upper Colorado River Basin gauges (Lees Ferry AZ, Green River UT, Colorado River near Cisco UT,
  Gunnison River CO, San Juan River UT), 97 complete water years (1928-2025). **Every flagged
  mechanism confirmed directly in real data**: strong spatial correlation across gauges (mean
  pairwise log-flow correlation 0.764, supporting the pooling lever); a real extreme-drought rate
  of 6.2% empirically matching the cited 5.5% literature figure almost exactly; and — the most
  important finding — **the nonstationarity complication flagged in the research pass is real,
  not just a documented worry**: the moderate-drought rate more than doubled, from 19.7%
  (pre-2000) to 42.3% in the real 2000-2025 megadrought period. This is now a firm, evidence-driven
  design input for Phase 1: Method 2's regime-mixture should use a **time-varying regime
  probability** by default, a real departure from every prior lab in this family's fixed-rate
  design. A further honest nuance found: the real 2021 (Tier 1) shortage year ranks as the 4th
  percentile of the full 97-year record by raw inflow, while 2022 (Tier 2, the more severe
  declared shortage) ranks only the 13th percentile — shortage tiers reflect cumulative reservoir
  storage deficit, not single-year inflow rank alone, meaning Phase 1's Firm Yield scoring needs
  to track storage state across years, not just annual inflow. **Phase 1 DONE (2026-07-28,
  `RESULTS_PHASE1.md`)** — fit all four methods on 71 pre-2000 water years, scored against the
  real, held-out 2000-2025 megadrought via a lumped-reservoir Firm Yield simulation. **A genuinely
  humbling finding, not a clean method-ranking story**: every method over-committed demand
  relative to the true hindsight-optimal Firm Yield — including Method 2 (the time-varying soft-EM
  regime-mixture this lab exists to test), whose fitted drought probability rose only from 2.8% to
  12.1% across the real test years, far short of the real 42.3% rate Phase 0 measured, because the
  pre-2000 training data itself never showed a clean ramping-toward-drought pattern to extrapolate
  from. The mandatory non-mixture trend control (Method 3) scored numerically best (92.3% real
  achieved reliability and $7.9B dollar consequence, vs. 46.2% and $34-41B for the other three) —
  **but its own fitted trend is not statistically significant on the training data (p=0.645,
  r²=0.003)**, so this is not evidence a simple trend beats a regime-mixture; it is evidence that
  **detecting a real acceleration after it happened is a fundamentally different, easier problem
  than forecasting one from data that precedes it**, regardless of method sophistication. This
  prompted a real, new addition to `gp_engine/PLAN.md` §7's cross-lab litmus test: passing the
  recurring/rare-regime conditions does not imply a pre-acceleration training window contains
  enough signal to extrapolate a coming acceleration forward — a separate, checkable question
  (via a real significance test on the fitted rate/trend) that this lab is the first in the family
  to have needed to ask. **`HYDRO_RESERVE_LAB.ipynb` built (2026-07-28)** — single reference
  notebook (math + all charts, 0 errors), consolidating the disclaimer, the 97-year real-data
  charts (including the megadrought-shaded flow series), and the key "forecast vs. reality" chart
  showing exactly where the statistical assumptions broke. **Lab considered feature-complete at
  Phase 1.**
  **Phase 2 (2026-08-03, `RESULTS_PHASE2.md`) — third application of `gp_engine/
  VOI_DISPATCH_PATTERN.md`** (after `grid_reserve_lab`'s Phase 4 and `shm_lab`'s Phase 2), resolving
  the template's own open condition-3 question for this lab: the Bureau of Reclamation's real
  annual August decision (the mechanism behind the real Tier 1/2 declarations) is a genuine,
  defensible recurring cadence, just a much longer one than `grid_reserve_lab`'s daily dispatch.
  State = whether Lees Ferry itself is a moderate-drought-or-worse water year (Phase 0's own
  25th-percentile threshold); skip/probe/drill = proceed with planned allocation / wait for an
  updated in-season hydrologic outlook / curtail immediately. **New modeling piece required**
  (`drought_classifier.py`, a `LaplaceBinaryGPC` fit on the other four gauges' z-scores against the
  Lees Ferry label) — none of this lab's four methods produce a native `(mean,var,prob)` triple,
  the same gap `shm_lab` hit. **A real bug caught and fixed during the build**: the first design
  used `phase0_run.py`'s own `basin_index` (mean z-score across all 5 gauges) as both label and,
  via all 5 gauges' z-scores, the classifier's features — circular (SVM's test AP came back
  exactly 1.000, the giveaway); fixed by predicting Lees Ferry's status from the other four gauges
  only. Economic constants reused, not invented, from `phase1_run.py`'s own already-sourced
  $417/AF and $2,400/AF figures — the best-sourced economics of the three VoI labs so far.
  **Headline: the largest, most robust positive VoI result of the three labs tested** — 200 seeds,
  GPC-full beats both SVM and GPC-mean at every point on a 12-point breakeven sweep, $15.89B
  [$14.55B,$17.19B] ahead of SVM at the derived breakeven (0.174, vs. a true base rate of 25.8%),
  with GPC-mean statistically no better than SVM — the cleanest isolation yet of "the advantage is
  entirely posterior variance, not a better mean," traced to this lab having both genuine per-year
  classification ambiguity (AP≈0.87-0.93) and economics landing the breakeven probability in a
  genuinely decision-relevant range, unlike `grid_reserve_lab`'s "always commit" extreme or
  `shm_lab`'s near-total separability. `gp_engine/VOI_DISPATCH_PATTERN.md` updated with this third
  worked example and two new lessons: annual cadence is a real, workable decision cycle (resolving
  the condition-3 question, though with far fewer real decision instances than a daily domain);
  and watch for label circularity when a convenient aggregate index is tempting to use as both
  label and feature source. `climate_cat_lab`'s reinsurance retention is now the last open
  candidate from the original three parked ideas.

- **VoI family wrap-up (2026-08-04, `gp_engine/VOI_DISPATCH_PATTERN.md`'s closing section)**: after
  all four labs (`grid_reserve_lab`, `shm_lab`, `hydro_reserve_lab`, `climate_cat_lab`), Fraser's own
  synthesis — the deciding variable across all four was **how hard the classification problem
  already was for a plain GP mean**: genuine ambiguity (`hydro_reserve_lab`, AP≈0.87-0.93) → the
  largest win; near-total separability (`shm_lab`, AP≈0.999-1.000) → nothing to resolve; an already-
  adequate mean (`climate_cat_lab`) → a small net negative from the added machinery. **Sequential-VoI
  is a niche tool, reach for specifically when a plain GP/GPC mean is already struggling — not a
  default upgrade** — now stated plainly in the pattern doc as the family's one-sentence takeaway.

- **home_energy_lab** — `gp_engine/home_energy_lab/LAB_PLAN.md` (scoped 2026-08-04, research pass
  DONE, no code yet) — the fifth application of the same reservoir/regime-mixture/VoI-dispatch stack,
  and the first candidate found in this session's search for an end-user-shaped application that
  doesn't depend on gov/corporate-gated data: home solar/battery/HVAC dispatch. Real, live-verified
  free data (Open-Meteo Historical Weather API, no signup, confirmed via a real `curl` test) drives a
  synthetic, EIA-anchored (not raw-ingested) load model — HVAC ~42-52% of real home electricity use,
  a real documented summer AC peak (RECS 2020) — plus real 2026 solar/battery cost anchors (Tesla
  Powerwall 3 $11,500-$16,500/13.5kWh, $2.55-$3.45/W solar, 30% federal credit through 2032). A real
  per-building alternative load dataset (NREL ResStock, live-verified downloadable, no signup) is
  flagged as a future cross-check rather than the primary driver, matching Fraser's own preferred
  design (synthetic parameters informed by real research, not a raw ingested dataset). **Two
  genuinely separate questions scoped, not one**: (1) does the same GP+soft-EM+VoI dispatch stack
  earn its keep for battery/HVAC charge-or-wait-or-commit decisions (reuses `hydro_reserve_lab/
  reservoir_sim.py`'s storage mechanics relabeled, and expects to need a new classifier stage again,
  per `VOI_DISPATCH_PATTERN.md`'s own recurring lesson); (2) a `hydro_reserve_lab`-style capacity-
  sizing solver — given a load pattern and the real weather record, what solar+battery size
  minimizes total cost at a target reliability, a genuinely practical "how much do you need"
  deliverable Fraser specifically asked to include. **Deliberately scoped as one instance, not a
  premature generic engine** — Fraser's own EV-charging idea (structurally the same reservoir shape:
  SOC↔battery, driving demand↔load, range-anxiety↔stress-regime) is noted as the natural second
  instance (Phase 4/stretch) that would justify extracting a shared engine afterward, mirroring how
  `decision.py`/`voi.py` themselves were extracted from one lab only after they existed, not designed
  generically up front. Phases 0-3 scoped (data/sanity checks → method ladder 0-3 → VoI dispatch
  layer → capacity-sizing solver); Phase 4 (EV) explicitly deferred. Real TOU tariff and battery
  efficiency/degradation specs flagged as not yet sourced (Phase 0's job), not invented.
  **Real calibration case added (2026-08-04, `research/04_vancouver_real_calibration_case.md`)**:
  Fraser's own BC Hydro bill — a real ~72-day billing period (Mar 20-May 30), tier-split by a
  rate-year boundary: 421 kWh/35.1 per day (Mar 20-31), 1,756 kWh/29.3 per day (Apr 1-May 30),
  summing to the originally-quoted 2,177 kWh exactly. **A real inconsistency found, then resolved
  with more real data in the same session**: the initial "70 kWh/day for March" reading was a units
  error (treating a ~2.4-month bill as one calendar month), not a real anomaly — no EV-charging-
  spike hypothesis needed; the real signal is a genuine seasonal decline (35.1→29.3 kWh/day, spring
  warming), consistent with the originally recalled seasonal range. Real BC Hydro rate structure
  verified: a **tiered threshold** (10.97¢/kWh first 675 kWh/month, 14.08¢/kWh above), not a flat
  TOU price as first assumed, plus real optional Time-of-Day pricing (±5¢/kWh peak/off-peak) —
  closes the "TOU tariff not yet sourced" gap with a real, current figure; a real bill can straddle
  a rate-year boundary, which Phase 0's loader needs to handle. Location fixed to Vancouver, BC
  (not an arbitrary
  illustrative city); the stress-regime hypothesis narrowed to a single-sided winter heating case,
  matching Vancouver's real mild oceanic climate (negligible AC load) rather than the original
  two-sided winter/summer framing.
  **Phase 0 DONE (2026-08-04, `RESULTS_PHASE0.md`)**: real 10-year (2016-2025) Vancouver weather
  pulled (Open-Meteo, 87,672 hourly rows, 0 missing). Load model exactly calibrated to Fraser's real
  2-point bill data (base=24.175 kWh/day, heating=0.9006 kWh/degree-day at an 18°C base) — cross-
  checked, not fit, against the recalled seasonal range and landed within ~1 kWh/day. Solar model
  (8kW illustrative) gives 999 kWh/yr/kW, in the real plausible range, with a real 5.6x summer/
  winter seasonal swing. **The stress-regime hypothesis confirmed directly on real data**: low-
  solar/high-heating-demand co-occurs 1.78x more than independence predicts, with real 4.5x
  multi-day persistence (P(stress tomorrow|stress today)=0.502 vs. marginal 0.111) — a fifth
  confirmation, in a new domain, of this codebase's recurring regime-mixture premise. Battery
  simulator built and verified energy-conserving to floating-point precision after a real bug in
  its own first self-test was caught and fixed (a naive aggregate energy-balance check that didn't
  account for round-trip efficiency losses correctly, replaced with a rigorous per-timestep
  AC-bus power-balance check).
  **Phase 1 DONE (2026-08-04, `RESULTS_PHASE1.md`)**: the four-method dispatch ladder, fit on one
  real training year (2016), scored on the real held-out 2017-2025 record (9 years), real 8kW/
  13.5kWh system, real BC Hydro tiered+optional-TOD rates (`rate_model.py`, self-test reproduces
  Fraser's real Mar 20-31 bill to the cent). **Method 2 (plain GP forecast) wins ($624/yr) over
  both naive baselines (Method 0 $689/yr, Method 1 $661/yr) — Method 3 (GP + regime-mixture) is a
  small real negative on top of Method 2 ($632/yr)**, a third instance of this codebase's
  "regime-awareness doesn't automatically help" finding (after `climate_cat_lab`'s Phase 3 and part
  of `shm_lab`'s Phase 2). A genuinely counter-intuitive real finding: Method 0 has the HIGHEST
  self-sufficiency (52.5%) but is the MOST expensive — self-sufficiency doesn't capture *when* grid
  energy is bought, and Methods 1-3 trade higher total kWh (round-trip battery losses) for a much
  lower average $/kWh by shifting consumption to the real off-peak window. **Three real bugs caught
  and fixed during the build**: `battery_sim.py`'s own first energy-balance self-test was itself
  wrong; `dispatch_sim.py`'s first draft let the same-hour reactive step immediately discharge what
  the proactive charging step had just added (fixed: off-peak deficits now served directly from
  grid); `gp1d.py`'s exact GP was too slow at 3 training years (20-36s/fit), training set reduced to
  one real year matching `shm_lab`'s own established scale.
  **Phase 2 DONE (2026-08-04, `RESULTS_PHASE2.md`)** — the fifth application of the sequential-VoI
  mechanism this codebase has now built five times. State = a real, data-derived high-demand-day
  label (net load top 25%); new classifier stage (`stress_classifier.py`, `LaplaceBinaryGPC` on
  yesterday's net load + temp) checked for separability first (val AP≈0.80, real variance range
  0.014-0.66). Economics needed **zero new sourcing** — derived entirely from this lab's own
  already-verified BC Hydro rates, the best-sourced VoI economics so far. **Headline: GPC's
  calibrated mean robustly beats SVM (+$5.94-$338/seed depending on breakeven, 200 seeds), but
  posterior variance adds nothing on top, across the entire cost-ratio range tested** — a clean
  null result, and a *second, distinct mechanism* for that null this family has now found
  (`shm_lab`'s was too-easy-to-separate; here the classifier has real, checked ambiguity, but the
  payoff structure never rewards resolving it). Two independent layers of this lab (Phase 1's
  regime-mixture margin, Phase 2's VoI variance) now agree: for this real dispatch problem, the
  plain forecast already does essentially all the useful work.
  **Phase 3 DONE (2026-08-04, `RESULTS_PHASE3.md`) — all core phases of this lab now complete.**
  The capacity-sizing solver: a 2D (solar, battery) grid search minimizing real annualized cost.
  **A real error caught while scoping this phase**: the earlier-sourced solar/battery economics
  were US market data including the US federal 30% tax credit, which does not apply to Fraser's
  real Vancouver, BC household — corrected with real BC Hydro rebates verified directly from
  bchydro.com ($1,000/kW solar capped $5,000; $500/kWh battery capped $1,500, battery-only-with-
  solar) and real CAD installed costs (`research/05_bc_solar_battery_rebates_corrected.md`).
  **Headline: the cost-minimizing system is 4kW solar + NO battery ($1,359/yr) — 47% cheaper than
  the 8kW/13.5kWh reference system used throughout Phases 1-2 ($2,560/yr)**; battery capacity never
  earns back its capital cost anywhere on the tested grid, a real property of BC Hydro's low rates
  and modest battery rebate, not a modeling artifact. A genuinely counter-intuitive confirmed
  finding: 0kW solar + a battery shows **negative self-sufficiency (−2.8%)** — round-trip losses on
  pure grid-arbitrage cycling increase total kWh purchased even while *reducing* the dollar bill,
  the starkest confirmation yet of Phase 1's own "self-sufficiency ≠ cost-effectiveness" finding.
  Explicitly flagged as $-optimization only — a battery's real backup-power/Peak-Saver/EV-
  integration value isn't priced by this model, so this isn't a blanket "don't buy a battery"
  conclusion.
  **Notebooks DONE (2026-08-04)**: `HOME_ENERGY_LAB.ipynb` (consolidated Phase 0-3 write-up, real
  charts, 0 execution errors) and `SCENARIO_BUILDER.ipynb`, an editable companion built on a new
  `scenario_engine.py` — generalizes Phase 3's hardware/rebate/rate constants into plain-dict
  catalogs a user can override in one cell (rather than editing engine code) as prices/rebates
  change. Adds two real 2026 hardware trends Fraser flagged: Anker SOLIX (cheaper battery-only
  arbitrage than Powerwall-class, $700-$1,300 USD/kWh) and balcony solar ("Balkonkraftwerk", ~0.8kW
  plug-in kits, DE/US, `research/06_alternative_hardware_options.md`). Delivers real return-of-
  capital (simple payback-year) figures per hardware scenario plus a 20-year cumulative-savings
  chart, and two optimizers (cheapest system; cheapest system reaching a self-sufficiency target,
  both "most of the time" energy-weighted and "all of the time" zero-grid-import-every-day).
  **Headline real payback findings**: balcony solar (Germany, subsidized) pays back in ~1.2 years,
  the unsubsidized US balcony kit in ~5.9 years, the Phase-3-optimal 4kW/no-battery system in 17.5
  years, and the 8kW+battery systems (Powerwall-class or Anker SOLIX) do not pay back within 30
  years at real BC Hydro rates — battery-only arbitrage pays back fastest at small scale, not large.
  **A real bug caught while building the notebook**: the first payback draft compared `total_annual`
  (grid + amortized capital) instead of `grid_annual` alone against the no-hardware baseline, which
  double-counts the capital cost already being recouped in the numerator and buried the real 4kW
  system's $434/yr savings down to $130/yr — fixed to compare grid-cost-only savings, the correct
  convention for a payback calculation.

- **decision_harness_lab** — `gp_engine/decision_harness_lab/LAB_PLAN.md` (2026-08-06, **Phases
  0-2.5 DONE**) — a generic Bayesian decision harness on top of `decision.py`/`voi.py` (an array of
  actions/costs/outcome distributions, Monte Carlo simulation, chart-first evaluation) built as a
  deliberate prerequisite step before any real GP soft-EM application, after the EV-trip-optimizer
  idea stalled on needing real telemetry data first. `decision.py` gained additive `expected_value_nstate`/
  `bayes_action_nstate`/`realized_value_nstate`/`oracle_value_nstate` (k-state generalization of the
  existing 2-state functions, every prior lab's call sites unchanged). `harness.py`'s Monte Carlo
  layer is cross-checked against those closed-form functions on every toy example, not trusted
  blindly. Three toy examples (`newsvendor`, `insure_vs_self_insure`, `invest_decision`), each
  engineered to produce a different outcome shape — multi-lumped, bimodal (the flagship case,
  needing a symlog x-axis or the near-$0 cluster is invisible next to the $200k tail), right-skewed
  — all rendered in `DECISION_HARNESS_LAB.ipynb` (executed, 0 errors). **Explicitly scoped to stop
  before any GP/soft-EM work**: synthetic toy data only, since a real soft-EM win has never come from
  synthetic data in this codebase (`gp_engine/PLAN.md` §7's litmus test needs a real, recurring,
  rare/imbalanced regime) — Phase 3 (picking a real domain and wiring in a fitted GPC posterior) is
  a deliberately separate, not-yet-started next lab. **`OUTCOME_SHAPE_TAXONOMY.md` (Phase 2.5)**: a
  taxonomy of outcome shapes and their generative mechanisms (discrete-state mixture, payoff
  nonlinearity, noise-family skew), grounded in `shape_diagnostics.py`'s real numbers, not asserted
  from charts — caught a real methodological gap along the way: a height-based KDE mode counter
  misses `self_insure`'s rare disaster cluster entirely (peak too short despite 8% real probability
  mass), fixed with a mass-based counter instead. Doc ends with a Phase 3 readiness checklist —
  narrows the search, but a real domain still needs its own history checked against the litmus test
  directly, not inferred from shape alone. **Sharpened 2026-08-09 after `vol_regime_lab`**: added "A
  sharpened rule: 'beats what?' is not one question" — a real domain (financial volatility) cleared
  every check in this doc (real, recurring, rare, separation ratio 2.68) and still only showed an
  established soft-EM win over a misspecified parametric baseline, not over the plain empirical
  quantile; checklist now has a 5th step (run the 3-condition, multi-window/bootstrap-replicated
  comparison, not just the shape check) before trusting any "Yes, if real" cell in the table.
  **`soft_em_illustration/` + `SOFT_EM_ILLUSTRATION.md`
  (2026-08-06, DONE)**: the first place in this lab an actual model gets fit (a 2-component
  `sklearn.mixture.GaussianMixture`, `grid_reserve_lab/regime_mixture.py`'s core mechanism stripped
  of its spatial-fleet layer) to synthetic data with controllable separation/imbalance/sample-size
  knobs, scored via reserve/quantile sizing against `grid_reserve_lab/reserve_calc.py`'s own
  reliability-and-cost convention. **Caught a real confound mid-build**: a first 2-condition design
  (empirical vs. soft-EM) showed soft-EM "winning" even with no real regime present, because any
  parametric fit smooths a noisy deep-quantile order statistic regardless of structure — fixed with a
  third `one_component` control (mirroring `bayesian_decision_lab`'s own isolate-what's-doing-the-work
  design), which produced a clean separation crossover (soft-EM ties below ~2σ, wins clearly above
  ~3σ, on both reliability and cost). **Two more real findings, not the single hypothesis each sweep
  started with**: reliability alone is misleading at high imbalance (`one_component` "wins" only by
  grossly overpaying, caught by reading cost alongside it); soft-EM's edge over `empirical` shrinks
  with sample size (a variance fix) while its edge over `one_component` stays flat (a bias/
  misspecification fix no amount of data resolves) — two structurally different reasons soft-EM wins.
  Negative control (a single skewed process, no real regime) still shows a modest soft-EM edge,
  correcting the taxonomy's implied claim to "helps when the truth isn't Gaussian," not "only helps
  when there's a real regime."

- **vol_regime_lab** — `gp_engine/vol_regime_lab/LAB_PLAN.md` (2026-08-09, **Phase 0-2 DONE**) — a
  deliberate "detour" lab (Fraser's framing): tests whether multi-regime financial volatility (Fraser's
  own domain proposal — a mixture-of-GP-experts, soft E-step responsibilities + per-regime weighted
  M-step) is a viable real Phase 3 domain for `decision_engine`, kept separate so it stands on its own
  merits. `research/RESEARCH.md` grounds it in 5 verified citations (Jacobs/Jordan/Nowlan/Hinton 1991;
  Tresp 2000; Rasmussen & Ghahramani 2002; Hamilton 1989; Hamilton & Susmel 1994 SWARCH) and traces the
  mechanism directly to `grid_reserve_lab/regime_mixture.py`'s own already-proven E-step/M-step
  design — no new mechanism needed, just a new domain. **Phase 0 litmus pre-check on live FRED VIX
  data** (`VIXCLS`, free, no API key, 9,246 real trading days 1990-2026): both conditions hold at
  VIX>30 — 94 recurring episodes, 7.95% base rate, separation ratio 2.68. **Phase 1**: a synthetic
  sanity check (reusing `soft_em_illustration`'s own oracle/fit_compare code, VIX-calibrated) predicted
  `soft_em` would beat a misspecified `one_component` but NOT cleanly beat plain `empirical` at
  realistic training windows; the real 1990-2015→2016-2026 split then found `soft_em` achieving
  near-exact calibration at the deep 0.99 tail — looked like a clean win. **Phase 2 stress-tested that
  single-split finding, and it did not hold up**: a 4-window walk-forward replication check found
  `soft_em` closest-to-target in only 1 of 4 real eras (one window — training on the calm 1990s,
  scored against the dotcom crash + 2008 GFC — saw ALL THREE conditions fail badly, a genuine
  structural break no model here defends against); a 63-day block bootstrap (respecting VIX's real
  serial correlation) found all three conditions' calibration-gap CIs overlapping heavily — Phase 1's
  clean point estimate was not statistically significant. **The final, precise verdict**: use `soft_em`
  over a naive single-Gaussian fit (justified, consistently, everywhere tested — it matched or beat
  `one_component` in all 4 windows); do NOT assume it beats a plain empirical quantile at this domain's
  real separation/rarity level (2.68-3.21σ, short of `soft_em_illustration`'s own ≥4-6σ "decisively
  wins" zone) — not established. `VOL_REGIME_LAB.ipynb` (executed) is the full write-up: mechanism,
  literature, every number, the verdict. Phase 3 (folding into `decision_engine`) not warranted as an
  unconditional win; the sharpened lesson (separation predicts "beats misspecified," not "beats
  nonparametric") is worth keeping in `OUTCOME_SHAPE_TAXONOMY.md` regardless.

## rbfx — Rust interpolation library (standalone, wraps gp_engine)
- **Path:** `~/machine_learning/rbfx/`
- **What:** Compact Rust crate (`rbfx-core`) + PyO3 wheel (`rbfx-py`, `import rbfx`) wrapping
  `gp_engine/gp_solver.so` (FP32 Cholesky factor + FP64 iterative refinement) as a generic
  dense RBF/kriging interpolation kernel — no new numerics, an FFI/packaging layer over the
  same `.so` `gp_engine`/`MPDOK/kriging`/`rbf_pointcloud`/`rbf_spatial` already call via
  ctypes. Kernels: rbf/matern32/matern52 (isotropic `ell`); mean-only `predict` (no variance
  yet); GPU required (no CPU fallback, by design).
- **Status (2026-07-18): v1 built and validated.** `bench/parity_test.py` confirms `rbfx`
  matches `gp_fortran.py`'s oracle to float precision (alpha rel-diff ~1e-10, logdet exact)
  across rbf/matern32/matern52 at n=500/2000. `bench/benchmark.py` benchmarks time+accuracy
  vs `scipy.interpolate.RBFInterpolator` (existing MPDOK benchmarks only compared time, not
  accuracy) — matched RMSE at equal N, rbfx ~10x+ faster by n=8-10k as scipy's dense CPU
  solve becomes the bottleneck (see `README.md`'s measured table). Reusable for **any**
  dense-interpolation use case, not just the geostatistics labs — e.g. the planned
  Bayesian-CVaR portfolio-optimizer extension (covariance-kernel swap in `MPDOK/portfolio_studio`)
  is a candidate consumer.
- **Reusable for:** any project needing fast dense scattered-data interpolation from
  Python/Jupyter without hand-rolling a ctypes wrapper.
- **Binary-only distribution (2026-07-18):** `test/` — a `pip install`-able wheel and an
  unpacked drop-in flavor, no Rust source in either (`test/build_dist.sh` regenerates both
  from `rbfx-py`; `test/README.md` has the how-to). Both bundle `libgp_solver.so` and use an
  `$ORIGIN`-relative runtime path (not this dev machine's `rbfx/.native` symlink), so they're
  portable to any machine with the same CUDA/NVIDIA HPC SDK toolchain already installed — not
  a zero-dependency public release (CUDA/cuBLAS/cuSOLVER/nvfortran runtime aren't bundled).

## decision_engine — Rust decision engine (standalone, wraps decision_harness_lab)
- **Path:** `~/machine_learning/decision_engine/`
- **What:** Cargo workspace (`decision-engine-core` + `decision-engine-cli` + `decision-engine-py`),
  a faithful Rust port of `gp_engine/decision_harness_lab`'s `decision.py` (k-state Bayes decision
  core) and `harness.py` (Monte Carlo outcome-shape engine) — no CUDA, no GP fitting, consumes an
  already-known state distribution exactly like the Python originals do. Callable two ways at once:
  `import decision_engine` in Jupyter (PyO3, `decision-engine-py`) and a structured JSON spec via
  `decision-engine run --input <path|->` (`decision-engine-cli`) — the CLI path is what a local LLM's
  tool-calling or the COBOLMM/rustmm menu module uses, following that codebase's own
  `claude_knowledge/API.md`/`alert_scan.sh` precedent for LLM/external-program-facing structured
  input (no JSON-schema module existed there before this).
- **Status (2026-08-06): Stage 1 DONE.** `cargo test -p decision-engine-core` passes (k=2 parity,
  k=3 smoke test, Monte-Carlo-vs-closed-form cross-check). `tests/parity/run_parity.py` runs all 3 of
  `decision_harness_lab/toy_examples/` through the real Python `harness.py` and the Rust CLI on the
  equivalent JSON spec — all 3 pass within 5% relative tolerance on the mean (cross-RNG Monte Carlo
  noise, not a bit-parity target) and exact `best_action` agreement. `decision_engine_demo.ipynb`
  executed, 0 errors. **A real bug caught by the parity tests, not before**: an early draft folded
  `invest_decision.py`'s Lognormal `scale` parameter into an inflated `sigma`, which is mathematically
  wrong (scaling a `Lognormal(0,sigma)` draw by `c` is `Lognormal(ln(c), sigma)`, not
  `Lognormal(0, c·sigma)`) — fixed by adding `scale` as its own parameter, documented in `README.md`.
- **`"mode": "diagnose"` (2026-08-09, DONE)** — not the blind "activate soft-EM" flag Fraser
  considered and rejected (rightly — this codebase's own research says that would often be a worse
  default than doing nothing), but a **diagnostic** mode: given raw historical outcome data (no known
  state labels), fits a 1-2 component Gaussian-mixture via from-scratch EM (pure Rust, no CUDA, no
  external ML library — the exact mechanism `soft_em_illustration/fit_compare.py` and `vol_regime_lab`
  validated by hand, ported not wrapped), runs the same `empirical`/`one_component`/`soft_em`
  comparison across several splits (generic position-based expanding-window or random-holdout
  generators, domain-agnostic unlike `vol_regime_lab`'s hand-picked calendar eras), and recommends
  one with a verbose plain-language `narrative` grounded in the actual numbers — explicitly built so
  a non-expert doesn't need to already know this session's research to get the right answer. New
  `--narrative` CLI flag prints just that text. **Real findings while building it, not hidden**: (1)
  the decision logic originally ranked by win-rate, which is too coarsely quantized at a practical
  (~5-10) split count to be useful — switched to ranking by mean-gap magnitude, with "high" confidence
  requiring separation from the closest competitor only, documented as a genuinely harder bar to clear
  here than the 150-trial Python bootstraps this mode's methodology came from (this splitting scheme's
  CI is a cross-validation-fold spread, not a shrinking standard error). (2) A median-split EM init
  badly mis-fit real VIX log-data (separation ratio 1.61 vs. `vol_regime_lab`'s labeled 2.68) — fixed
  with multiple percentile-based inits, kept whichever converges to the highest log-likelihood
  (standard EM practice) — but this did NOT close the gap: unsupervised MLE is structurally dominated
  by the bulk of a distribution and doesn't reliably rediscover a genuinely rare, domain-defined
  regime from raw values alone, a real, documented scope boundary (`tests/parity/run_diagnose_check.py`),
  not a bug — even so, the tool's final recommendation on real VIX data stayed safely conservative
  (`"empirical"`, confidence `"none"`), the correct non-overclaiming behavior either way. Full
  continuous-covariate GP regression (wrapping `gp_solver.so` via FFI, `rbfx-core`'s CUDA-dependent
  pattern) stays out of scope — deliberately, to keep this engine lightweight/portable ("for wide
  use"). Wired into the COBOLMM menu's `example_diagnose_spec.json` bundled demo and `API.md`.
  **Dedicated `DD` menu entry (2026-08-09)**: initially only reachable through `DE`'s generic prompt
  (type `example_diagnose_spec.json`) — Fraser wanted a menu option that reads as its own capability
  at a glance. Added `decision_engine_diagnose.sh` (reuses `decision_engine.sh`'s exact interactive
  flow via two env vars — default spec, header — rather than duplicating it), a `make diagnose`
  target, and a new `MENU.DAT` row (`DD`, "Run a diagnostic (recommends the best method)") right
  after `DE`. Same careful process as the original `DE` row: backed up first
  (`MENU.DAT.bak_decision_engine_dd_<timestamp>`), all 144 lines re-verified at exactly 134 bytes,
  live `rustmm` rendered via a real pty — both `DE` and `DD` show correctly, nothing else disturbed.
  **A real gap caught immediately after, not before**: the diagnostic only validated *which* method
  to use, with no way to actually act on the recommendation — `nstate`/`monte_carlo` need a full
  state/payoff spec `diagnose`'s raw-`data` input never asked for, so there was no bridge back.
  Fixed by computing each condition's actual reserve fit on the FULL dataset (not just
  cross-validation folds) — new `reserve` field per condition, `recommended_value` at the top level,
  and a ">>> USE THIS NUMBER <<<" line in the narrative — so the tool's answer is a usable number,
  not just a verdict. `one_component`'s reserve on the bundled demo data (111,868) vs. `empirical`'s/
  `soft_em`'s (~1,700-1,800) is a vivid, real illustration of exactly why it loses.
- **Reusable for:** any decision problem shaped like `decision.py`/`harness.py`'s contract (an array
  of actions, a known or per-instance state distribution, an optional Monte Carlo outcome
  simulation) — callable from any language/tool via the CLI's JSON contract, not just Python/Jupyter.
- **Wired into COBOLMM/rustmm (2026-08-09)**: `COBOL/main_menu/decision_engine/` — a new "DECISION
  ENGINE" section, option `DE`, following `alert_scan`/`concept_search`'s "the engine lives
  elsewhere, this dir is just the front door" convention exactly (`Makefile`'s `run`/`build`
  targets, a `decision_engine.sh` wrapper with the standard `/dev/tty`-pause fix for menu-launched
  non-blocking stdin, and an `API.md` mirroring `claude_knowledge/API.md`'s LLM-facing convention —
  call the `decision-engine` binary directly for programmatic use, not through `make`). `cfg/MENU.DAT`
  backed up first (`MENU.DAT.bak_decision_engine_<timestamp>`), 3 new 134-byte fixed-width rows
  inserted between the RAG QUERY and LLM SETTINGS sections, verified two ways: every one of the
  file's 143 lines re-parsed at exactly 134 bytes with correct field boundaries, and the live
  `rustmm` menu rendered via a real pty — the new section appears correctly, no box/column
  corruption anywhere else in the menu.
- **File-record + batch mode (2026-08-09)**: `decision-engine run --input example2.json` now writes
  `example2_result.json` alongside stdout (override with `--output`), so iterating on a spec leaves a
  matching trail of results. A glob input (e.g. `'input*.json'`, quoted so the shell doesn't expand
  it first) triggers batch mode — every match runs independently, each writing its own
  `<stem>_result.json`, with one `{"batch": [...]}` JSON summary on stdout rather than each file's
  raw result (keeps stdout parseable in every mode). `--output` is rejected in batch mode (multiple
  outputs can't share one explicit path). `decision_engine.sh` (the COBOLMM front door) now prompts
  for a spec file or glob instead of only running the bundled demo. All 4 input shapes (named file,
  `--output` override, stdin, glob) manually smoke-tested; `cargo test --workspace` and
  `tests/parity/run_parity.py` still pass unchanged.

## RBF / geometry (on the tensor-core engine)
- **rbf_pointcloud** — `…/tensor_core_engine_v5/rbf_pointcloud/` — RBF surface reconstruction /
  normal estimation from point clouds.
- **rbf_spatial** — `…/tensor_core_engine_v5/rbf_spatial/` — spatial RBF interpolation.
- **tiny_pointers** — `…/tensor_core_engine_v5/tiny_pointers/`.

## MPDOK application labs
Path prefix: `…/tensor_core_engine_v5/MPDOK/`.

**Finance / economics**
- `fred_rate_predictor` — Fed funds rate predictor; web UI on `:8004`.
- `energy_trader` — energy trading strategy baseline demo.
- `portfolio_studio` — portfolio strategy studio (distinct from the standalone cufolio project below).
  **`gp_cvar` integrated live (2026-07-18)**: a 4th strategy alongside equal/cuFolio/MPDOK, plus
  `gp_smart`/`gp_smart2` — the same drawdown regime-switchers as `smart`/`smart2` but switching
  gp_cvar<->MPDOK instead of cuFolio<->MPDOK (reuses `_backtest_smart`/`_backtest_smart2` unchanged,
  now parameterized with `label_a`/`label_b` so switch-event logs say the right strategy name).
  New file `gp_cvar_strategy.py` bridges to `gp_engine/cvar_gp_lab/` (cross-repo dependency, fenced
  at both import and call time — a broken/missing cvar_gp_lab degrades to omitting gp_cvar/gp_smart/
  gp_smart2 from the response, not a crashed endpoint; verified by deliberately breaking the import).
  `index.html` gained a 4th holdings tab, 3 new stat boxes, matching chart lines/switch markers/
  events-panel entries, and CSV export coverage. `daily_monitor.py`'s `state.json` was redesigned
  from a flat single-switcher schema (which only ever tracked `smart1`, never `smart2`) to a
  namespaced schema tracking all 4 switchers, with a self-migrating loader for the old flat shape —
  verified against a real production `state.json`/`trade_log.csv` (migrated cleanly, old rows kept
  their columns, new columns populated on the new row). Full backtest verified live via the running
  FastAPI server on a fresh (non-Phase-2) date range, not just imported and unit-tested.
  **Signal-board bug fixed (2026-07-18)**: the live `/api/signal` "risk on/off" headline
  only ever answered "did a new floor breach fire today," with no notion of "are we
  already sitting defensive from an earlier switch" — it could (and did) read "risk on"
  for months while the smart/smart2 switcher was genuinely parked in MPDOK. Fixed with a
  layered fix, preference order: (1) the user's own currently-displayed backtest's
  `smart_switches` last entry (matches whatever date range they're actually looking at —
  this is what resolved Fraser's exact complaint), (2) `state.json`'s cron-persisted mode
  (new `backtest_engine._read_switcher_modes()`, read-only/degrades-to-None, avoids a
  circular import with `daily_monitor.py`) as a fallback before any backtest has been run
  this session — found live that this can legitimately disagree with (1), since
  `daily_monitor.py` uses its own fixed 1-year rolling test window, not the user's chosen
  dates, (3) the original today-only floor-breach reading as the final fallback. Verified
  against the real running server including the missing-state.json degradation case.
  **Found the same bug in a second place (2026-07-18)**: `signal.html` (served at `/signal`,
  what Fraser calls the "Studio Board") is a wholly separate standalone page with its own
  independent copy of this exact recommendation logic — the fix above only touched
  `index.html`'s embedded signal panel. `signal.html` has no backtest response to prefer
  (it never calls `/api/backtest`), so its fix uses tier (2)/(3) only, plus a new case
  tier (1) can't reach: cron says cufolio but today's fresh data shows a breach state.json
  hasn't caught yet ("switch pending" messaging, distinct from either steady state).
  Verified via a targeted Node harness (extracted just `renderSignal()`, stubbed the DOM)
  against 5 cases incl. the user's exact reported scenario — the pre-existing code even had
  a comment correctly diagnosing this exact gap ("the backtest uses the actual cuFolio
  portfolio value, which may have already breached its floor") without ever fixing it.
  **Root cause resolved (2026-07-18): `daily_monitor.py` no longer uses its own rolling
  2-year-train/1-year-test window.** Confirmed by direct comparison that its old rolling
  window and the UI's own default backtest (train 2015-2019, test 2020-today) are
  genuinely DIFFERENT portfolios, not the same facts viewed narrower/wider — different
  training data gives different cuFolio/MPDOK weights, hence a different compounding
  path and a different relative "4-month floor" on any given date. Fraser's call: the
  signal board must answer the same question as the portfolio optimizer it's monitoring,
  or the two conflict in a way that's confusing, not informative — if you want a
  shorter/different period, change it in the UI, don't have the monitor roll its own.
  `daily_monitor.py` now hardcodes the same fixed window as `index.html`'s own default
  sliders (`TRAIN_START`/`TRAIN_END`/`TEST_START` = "2015-01-01"/"2019-12-31"/"2020-01-01",
  `test_end` always today, matching the UI's own dynamic default) — verified end to end:
  re-ran it live, `state.json` now shows `smart1`/`smart2` = mpdok (matching the Portfolio
  Studio's own default backtest exactly, switch at the same day-1531 breach), and
  `/api/signal` + `signal.html`'s actual render function (re-tested against the real live
  API response) now correctly show "DEFENSIVE — in MPDOK". Note: this is a deliberate
  discontinuity in `trade_log.csv`'s history — rows before this point reflect the old
  rolling-window convention, not a bug.
  **Stale-data incident fixed (2026-07-22)**: `DATA_PATH` (`quantitative-portfolio-
  optimization/data/stock_data/sp500.csv`) is a static export, not a live feed — it had
  been frozen at 2026-03-27 for ~4 months (nobody had re-run the one-off download script
  that made it) while `daily_monitor.py` kept writing switch events stamped with the
  current wall-clock date, making stale March signals look live. This is what caused
  Fraser's "MPDOK rising but smart portfolios falling" report — the smart curves were
  really replaying an old February/March drawdown. Fixed: new
  `quantitative-portfolio-optimization/scripts/refresh_sp500_data.py` (atomic write,
  backs up the previous file, validates ticker-count and freshness before accepting),
  wired to a systemd --user timer (`portfolio-studio-refresh.timer`, weekdays 17:00
  America/Vancouver, `OnSuccess=` chains straight into `portfolio-studio-monitor.service`
  which runs `daily_monitor.py`; linger enabled so it fires even when logged out).
  `backtest_engine.py` gained `_data_freshness()`, surfaced as `data_as_of`/`data_stale`
  in both `run_custom_backtest` and `compute_live_signal` (so `/api/backtest` and
  `/api/signal` responses — and downstream, `index.html`/`signal.html` — carry it), plus
  `test_data_freshness.py` (6 tests, incl. one that guards the live file on disk directly).
  Same `sp500.csv` is also read by `network_influence`'s `backtest_engine.py`/
  `shock_engine.py` and `gp_engine/cvar_gp_lab/data.py` — each got the same
  print-on-stale guard (lighter touch than portfolio_studio's full API wiring since
  they're not live-monitored daily products). Re-running `daily_monitor.py` against
  fresh data (through 2026-07-21) found a real signal the stale run had hidden:
  `smart2`/`gp_smart2` switched OUT of MPDOK back to cuFolio/gp_cvar on 2026-07-22
  — `smart1`/`gp_smart` remain in MPDOK.
- `macro_contagion` / `macro_contagion_signed` — macro shock contagion propagation; signed variant.
- `network_influence` — network influence/contagion propagation with portfolio backtest (k=3 blind-spot study).

**EM / acoustic scattering (BEM)**
- `acoustic_scattering` (+ `_v2` / `_v3` / `_v4`) — GPU boundary-element Helmholtz/Laplace scattering;
  v2 built on radar learnings, v4 adds Robin (impedance) BC.
- `acoustic_lab` — acoustic scattering lab (v4) with a web server front end.
- `radar_scattering` — bistatic radar cross-section, staged pipeline, mixed-precision iterative refinement.
- `radar_scattering_3d` — 3D RCS, full bistatic scattering matrix on a consumer GPU.
- `bem_cobol` — BEM engine with a COBOL layer; roadmap to 3D RCS.

**Quantum**
- `quantum` — quantum dynamics (Krylov matrix-exponential of Hamiltonians) on consumer hardware.
- `quantum_mbl` — many-body-localization finite-size scaling (N=20→24→26).
- `quantum_ml` — quantum kernel ridge regression (QKRR).

**ML / DL foundations**
- `ntk_hessian` — Neural Tangent Kernel + Hessian-vector products (deep-learning math foundations).
- `nas_sweep` — neural architecture search sweep.
- `ML_static_blindspot` — MPDOK on standard datasets (titanic/housing/energy) demonstrating a static blindspot.
- `full_covariance_online` — full-covariance online learning.
- `SPD_matrices_vehicle_tracking` — SPD-matrix fleet/vehicle tracker via GMRES-IR.

**Geostatistics / genomics / GP**
- `kriging` — kriging solver on NOAA data, out-of-core path.
- `gp_regression` — Gaussian-process regression + Bayesian-optimization engine (`server.py`).
- `gblup` — exact genomic BLUP at scale (GRM build); includes a critique of approximate GBLUP.
- `mining_mpdok` — fixed-rank kriging for mining; "masking effect" study.

**Data assimilation**
- `enkf_mpdok` — ensemble Kalman filter vs MPDOK for atmospheric turbulence detection.

## Search & tooling — the "stash" family (pure Rust)
Fast local search tools + a COBOL/Rust menu system that ties them together. See `STASH_FAMILY.md`
(present in each stash dir) for the shared layout.
- **stash** — `~/stash/` — instant NAS file-name search (pure Rust, sub-millisecond trigram index).
- **pdfstash** — `~/pdfstash/` — instant full-text PDF search (pure Rust).
- **mdstash** — `~/mdstash/` — instant full-text markdown search (pure Rust).
- **concept** — `~/concept/` — CLI wrapper (`concept`) for the **seedverify** concept/semantic-search
  engine (`~/machine_learning/seedverify/`); auto-starts ollama, runs LLM-verified search.
- **COBOL/main_menu (COBOLMM)** — `~/machine_learning/COBOL/main_menu/` — COBOL command-line menu
  launcher (transitioning COBOL→Rust); front end for the stash family + catalyst_search, book_library,
  chart_downloader, concept_search. Docs: `COBOL_TO_RUST_V2.md`, `CODE_CENSUS_2026-06-10.md`.

## Services
- **backup_service** — `~/backup_service/` — FastAPI automation service (`backup.py`, ostree upgrade
  automation, ollama OCR shortcuts). Doc: `adding fastAPI automations.md`.

## Other
- **quantitative-portfolio-optimization (cufolio)** — `~/quantitative-portfolio-optimization/` —
  standalone NVIDIA-CUDA portfolio optimization (Mean-CVaR, large-scale simulation); an NVIDIA
  developer example. Not MPDOK-based — distinct from `MPDOK/portfolio_studio`.
- **kmerstash** / **kmerstash2** — `/run/media/fraser/ows/kmerstash*/` — k-mer / bioinformatics.
- **emergency_response** — `/run/media/fraser/ows/emergency_response/`.
- **cudf-rapidsai-notebooks** — `/run/media/fraser/ows/cudf-rapidsai-notebooks/` — RAPIDS demos.

---

## Cross-project ideas (speculation parking lot)

### The "will it feel the engine?" litmus test (why some past labs hit Amdahl's law)
For the tensor-core engine to produce a *felt* wall-clock win over stock CuPy, the workload must pass
**all four**:
1. **Dominant cost is dense GEMM / dense solve** — not FFT, not sparse, not elementwise.
2. **Compute-bound, not bandwidth/launch-bound** — matrices big enough (n ≳ few thousand) that tensor
   cores saturate; not a loop of tiny per-call GEMMs.
3. **Needs > TF32 accuracy** — so the Ozaki/Dekker split-precision tiers differentiate from stock CuPy
   TF32 (which already does the easy case).
4. **GPU-resident across many ops** — no per-op PCIe transfer tax.
Portfolio evidence: **BEM scattering (dense operator) + MPDOK IR = felt great**; **cryo-ET (FFT-bound)
= Amdahl trap**. Dense-operator problems win; FFT/sparse/bandwidth-bound don't.

### Ranked candidate directions (2026-07-09 discussion)
1. **Gaussian-process / kernel-method engine at scale** (build on `gp_regression`, `kriging`, `gblup`).
   O(n³) dense on ill-conditioned kernel matrices → passes all four tests; most demonstrable "felt" win.
   **PRESSURE-TESTED 2026-07-09 → GO.** De-risked: MPDOK's own test suite already benchmarks GP/RBF/SPD
   solves via GMRES-IR & LU-IR vs SciPy cho_solve/CuPy to relres 1e-11. Key synergy: the kernel matrix is
   *implicit* (generated from X, n×d, tiny) → never store K in FP64, regenerate tiles for the IR residual
   → in-core n≈38k FP32 vs ≈27k FP64 on 8GB (~1.4×), ~3–5× wall-clock at IR-recovered FP64 accuracy.
   Gaps to build: SPD/Cholesky-IR path **with log-determinant** (marginal likelihood — MPDOK has LU-IR/
   GMRES-IR only, no POTRF/logdet); the hyperparameter-optimization loop (the "keep resident, factor many
   times" driver = the felt demo); predictive variance (LU-IR "factor once, solve many" already fits);
   OOC tiling for n≥50k (reuse `kriging/kriging_ooc.py`). Limit: FP32-IR needs κ(K)≲1e7 → works with a real
   noise nugget, degrades for near-noiseless stiff interpolation (fall back to FP64 / more IR steps).
2. **Second-order ML** (NTK / Hessian / K-FAC / Gauss-Newton natural gradient) on `ntk_hessian`.
   The honest "ML" answer — GEMM-bound, research-novel; avoids the LLM-inference trap.
3. **Scale the proven dense-operator physics** — BEM radar/acoustic to larger / 3D (known win).
4. **Compute-bound retrieval core for concept/seedverify** — motivating (daily driver) *only if* made
   compute-bound (heavy kernel/reranking), not single-pass cosine (that's bandwidth-bound).

### Traps to avoid (fail the litmus test)
LLM token-generation inference (memory-bound, quantized — llama.cpp's turf); cryo-ET / anything
FFT-bound; sparse solvers; loops of small per-call GEMMs.

### Candidate next engine #1 — sparse mixed-precision preconditioning (2026-07-29 scouting, NOT pressure-tested)
`MPDOK/TERRAIN.md` (May 2026) scoped this as "Stage 3" — extending the dense mixed-precision
IR idea to sparse systems via an FP16/TF32 approximate factorization used only as a
*preconditioner* inside an FP64 outer Krylov solve — and explicitly deferred it, noting
"not packaged anywhere; research papers only" at the time, in favor of MPDOK's dense focus.
**A fresh check (2026-07-29) found that gap has narrowed in just two months, not widened**:
real, recent papers now report strong benchmarked results in exactly this space — a
mixed-precision Block-ISAI tensor-core preconditioner claiming 6x over cuSPARSE and 11.2x over
PETSc's built-in preconditioner; a 2024 mixed-precision Block-Jacobi study (arXiv 2407.15973); a
2024/2025 study of low-precision incomplete-Cholesky preconditioners for sparse least-squares
(arXiv 2504.07580); and a very recent (2026) mixed-precision randomized-preconditioning
Cholesky-QR paper (arXiv 2606.18411). **Honest read: this is no longer an empty field to walk
into — it's an active, competitive research area with real, strong published numbers.** Any
pursuit here needs a genuine differentiation angle beyond "first to try it": candidates worth
reading toward before committing — (a) a packaged, general-purpose, SciPy/CuPy-ergonomic library
the way MPDOK itself aims to be, vs. these one-off research benchmarks tied to specific problem
classes; (b) a specific connection back to this codebase's own domain (sparse structure in
inducing-point/sparse GP approximations, linking to `gp_engine`) rather than a generic sparse-CFD
target already well-served. Not pressure-tested; a real candidate to read toward, not a commitment.

### Candidate next application space #2 — dynamic GPU hash tables for the tiny-pointers mechanism (2026-07-29 scouting, reread in detail 2026-07-29, NOT pressure-tested)
Prior brainstorming already exists and predates the current `stash`/`pdfstash`/`mdstash`/
`kmerstash` family (`SynologyDrive/Adversa 2022/tiny pointers lab.md`, `tiny pointers ideas.md`,
and `OK, so the tiny pointer's application is going very well.md`, 2022) — reread in full, not
re-derived. **The core mechanism, precisely**: a "load-balancing dereference table" — each key
hashes to a small bucket of size B; the pointer only records its position *within* that bucket
(`log₂ B` bits), and `DEREFERENCE(k,p) = hash(k)·B + p` gives O(1) lookup with **zero probing**,
not just shorter probing. Measured in the original GPU lab (CUDA Fortran/nvfortran, RTX A1000,
capacity 4.19M slots, B=16): insertion retains ~50% throughput from 50%→99% load (259.87M→133.54M
elem/s) vs. traditional open addressing's 10x collapse (865.23M→86.08M); traditional max-probe
explodes to 75,652 steps at 99% load vs. tiny-pointer's constant 1 step; dereference throughput
stays flat (~2.5-2.7B lookups/sec) regardless of load factor; elements never move on growth, so
100% of handles survive resizing. A real, non-obvious risk flagged in the same notes, worth
remembering before indexing any non-sequential key space: a Fibonacci/golden-ratio hash gave an
exceptionally low 0.26% spill rate for sequential keys (TINYJOIN, COBOL prototype) — but that's an
artifact of sequential keys specifically; random/UUID-like keys revert to the paper's ~6% baseline,
so backup-table sizing should assume 6%, not the lucky 0.26%.

**Every still-unbuilt application idea from that reread, kept with fidelity rather than
compressed, so none get lost to busy-work** (bioinformatics/DNA k-mer matching is the one idea
from these notes already built, as `kmerstash`/`kmerstash2`, `/run/media/fraser/ows/kmerstash*/`
— not listed again below):

1. **GPU vector/ANN proximity-graph index** — replace HNSW/IVF-PQ's 32/64-bit global neighbor
   pointers with tiny local ones inside each bucket; original notes claimed 3-5x larger embedding
   index fitting entirely in VRAM, warp-parallel dereference with zero divergence (every thread in
   a warp resolves its neighbor lookup in the same single step).
2. **Lock-free, zero-copy GPU hash-join engine** for relational analytics — build phase inserts
   the smaller ("inner") table via the existing tiny-pointer insert path; probe phase batch-looks-
   up the "outer" table's rows; the *absolute stability of elements* (no rehash-driven pointer
   invalidation, even approaching 99% load) is what makes this concurrency-safe without locks.
   Flagged in the original notes as "the most natural extension of your existing data structures."
3. **Network intrusion detection / DPI flow-state table** — track millions of concurrent 5-tuple
   (src/dst IP, ports, protocol) TCP/IP flows; because lookup time is bounded and mathematically
   immune to cascading probe sequences, this is structurally immune to hash-flooding DDoS
   collision attacks, unlike a traditional table whose probe length explodes under adversarial key
   flooding.
4. **Local dedup / delta-sync** — an rsync alternative: a trigram index of the local filesystem
   finds duplicate *chunks* across entirely different files (not just whole-file hashes), so a
   renamed 500MB PDF with two edited pages gets recognized as ~99% identical to the original
   instantly, without reading both files sequentially.
5. **Real-time DLP / compliance auditing** — an in-memory trigram index of sensitive assets (core
   source code, trade secrets), hooked into OS file-system events (`inotify`/`FSEvents`), flags a
   staged git commit or USB copy with too much trigram overlap in microseconds, without the
   regex/heavy-string-matching cost of conventional DLP tools.
6. **Log anomaly detection via "structural templates"** — group streaming log lines into
   templates by trigram profile on the fly; a line with ~0% intersection to any known template is
   instantly flagged as a new error type — a much lighter-weight alternative to ELK/Loki-style
   full indexing.
7. **Code clone / dead-code detection** — cron-job trigram intersection across repositories to
   find copy-pasted blocks worth refactoring into shared libraries, and as an ultra-fast
   static-analysis pre-filter (narrow "50,000 files down to the 5 worth actually parsing" for a
   given unsafe-function pattern) rather than a full AST parse of everything.
8. **Offline/disconnected external-drive indexing** — the actual origin motivation (third note):
   crawl an external drive once while connected, keep the searchable reference catalog, and be
   able to search it (though not open files) even while the physical drive sits unplugged in a
   closet — then only reconnect the one specific drive you actually need. Motivating scenario
   given: a photographer with 300,000 photos spread across five external drives, no practical way
   to search all of them today. The notes floated, but didn't commit to, a portable/sellable
   product version given the apparent speed edge over existing tools (Spotlight/Everything/locate).

Two further second-wave ideas, built on `kmerstash`'s *containment score* rather than raw
trigram presence/absence (so likely to reuse `kmerstash`'s existing plumbing more than the raw
tiny-pointer table): **sliding-window stream classification** (feed a continuous stream — DNA
reads, logs, packets — through a rolling containment-score window; the score spikes to ~99% and
drops the instant a known signature passes through, giving a zero-latency streaming
trigger/router) and **"anti-screen" background/contaminant filtering** (index the *noise* — host
DNA in a viral sample, boilerplate headers in a corpus — and discard anything matching it, so
expensive downstream analysis only sees the concentrated, interesting residual). A third,
**structural-integrity/tamper detection without cryptographic hashing**, was also floated: unlike
SHA-256 (breaks completely on one bit flip), a containment score degrades gracefully in
proportion to real corruption — a fast, error-tolerant "is this massive distributed system still
essentially intact" check.

**A fresh competitive check (2026-07-29) found the benchmark target matters a lot**: the *static*
approximate-membership-query filter space (Bloom/Cuckoo/Quotient/XOR) is now dominated by Binary
Fuse filters and Ribbon/BuRR (Bumped Ribbon Retrieval), the latter reaching space overhead as low
as ~0.1% — a very recent (2026) JACM paper (Dietzfelbinger et al., "Ribbon: Fast Succinct Static
Retrieval and Approximate Membership") suggests this static side is already near information-
theoretic limits and heavily contested; **benchmarking a static filter there would not be a fair
or fresh fight.** The tiny-pointers mechanism's actual differentiator — stable handles under
concurrent insertion/growth, bounded zero-probing lookups, GPU warp-coalesced access — is a
**dynamic** hash-table property, not a static-filter one, so the fair, honest benchmark target is
dynamic GPU hash tables: WarpCore, SlabHash, DyCuckoo, and (found in this pass) a very recent
(Oct 2025) entrant, **Hive Hash Table** (arXiv 2510.15095), which reports sustaining 95% load
factor at 1.5-2x the throughput of SlabHash/DyCuckoo/WarpCore on an RTX 4090 (2-3.5 billion
ops/sec). **This is a real, strong, very recent competitor, not an empty field** — any claim that
the existing tiny-pointer implementation is under-optimized needs a real, controlled benchmark
against Hive specifically (same GPU, same workload shape) before concluding there's genuine room
to improve, not an assumption from 2022-era numbers on different hardware.

**Language, if either #2 (hash-join) or #1 (vector/ANN) is picked up — CUDA Fortran kernel + Rust
host, not a straight either/or (2026-07-29 discussion)**: the original 2022 notes frame both as
CUDA Fortran GPU projects, matching `tensor_core_engine_v5`/`MPDOK`'s own existing toolchain — but
the *productionized* `stash` family deliberately moved to pure Rust (`PLAN_PURE_RUST.md`,
libc-only, portable-package goals), and the competitive dynamic-hash-table field just surveyed
(WarpCore, SlabHash, Hive) is all CUDA C++, not CUDA Fortran, so there's no direct code to borrow
either way. The natural fit: keep the GPU kernel itself in CUDA Fortran (reuses existing expertise
and the `tensor_core_engine_v5`/`MPDOK` build system) and expose it to a **Rust host** via FFI —
the same shape as the existing Python+ctypes bridge to `mpdok.so`/`cuda_matlib.so`, just with Rust
standing in for Python, matching where `stash`/`mdstash`/`pdfstash` actually live and deploy today.

**Which to tackle first, if either — the hash-join engine, by a clear margin (2026-07-29
discussion)**: the hash-join engine is explicitly the smaller lift — the original notes call it
"the most natural extension of your existing data structures," since the GPU lab's insert/
dereference benchmark *is* the build/probe phase almost directly; it mainly needs a join
semantic wrapped around already-built, already-benchmarked machinery, plus a synthetic two-table
dataset. The vector/ANN index needs an entirely new graph-construction/ANN algorithm layer
(HNSW-style greedy neighbor selection or IVF clustering) built first — the tiny-pointer table only
replaces the adjacency-list storage *inside* that algorithm, so most of the actual hard work
(getting competitive recall/build-time) hasn't started — and it would enter a far more mature,
optimized competitive field (FAISS-GPU, RAPIDS RAFT, DiskANN, ScaNN) than dynamic hash tables are.
**Honest caveat**: unlike the dynamic-hash-table space above, the GPU hash-join competitive
landscape (cuDF/RAPIDS joins, academic GPU hash-join literature) was NOT checked in this pass —
worth a real survey before claiming the hash-join engine would win anything, only that it's the
easier one to *start*.

### Update — the hash-join engine already existed, got benchmarked for real, and got a real fix (2026-07-29 build session)
Turns out the hash-join idea above wasn't just a 2022 sketch — a full CUDA Fortran
implementation of the whole tiny-pointers mechanism, including `joindemo.cuf` (a GPU hash
join built on a reusable `tinymap` module), was already built and fidelity-checked on
2026-06-13 (RTX A1000, cc8.6) at
`.../tensor_core_engine_v5/tiny_pointers/` — genuinely easy to lose track of, exactly per
Fraser's own worry; see that directory's `README.md` for the full detail, kept there (not
duplicated here) since that's the source of truth.
- **Real, fresh numbers (2026-07-29)**, R=4.19M build keys / S=67.1M probe keys, B=16: at 99%
  load, linear probing manages 61.0 Mr/s build / 164.5 Mr/s probe; the existing single-level
  tiny-pointer table beats it (107.8 / 596.3 Mr/s) but with a real 9.5% build-failure rate.
- **A real fix, built this session (`joindemo_reliable.cuf`)**: wiring `tinymap`'s already-
  existing reliable 2-level path (`ttd_insert_r`/`ttd_find_r`, bucket + linear-probe overflow —
  no new mechanism) into the same benchmark gives **0.00% failure at every load factor AND
  higher throughput than the single-level table (128.9 / 680.9 Mr/s at 99% load)** — banked,
  not just a benchmark number: a real, working improvement to an existing engine.
- **A real, honest competitive-benchmarking attempt, and what it found**: cloned and built
  [BGHT](https://github.com/owensgroup/BGHT) ("Better GPU Hash Tables," the academic
  state-of-the-art static GPU hash table, 100% success to 0.991 load factor / ~1.43 avg
  probes in its own published results) on the same GPU — required `pip install cmake` and
  forcing `-allow-unsupported-compiler` (our CUDA 13.3/gcc 16 postdates this 2021-era code by
  several CUDA majors) — compiled clean, but its `bcht` table's constructor hangs, reproduced
  at both 4.19M- and 1,024-key scale, isolated to something beyond plain `thrust::fill`
  (tested separately, works fine). Most likely a genuine toolchain-version gap, not a flaw in
  BGHT's design — but a real same-GPU head-to-head was NOT achieved this session.
- **The sharper answer to "is there a more commonly-used reference," found after BGHT's
  build friction itself became the signal worth noting**: [**cuCollections**](https://github.com/NVIDIA/cuCollections)
  (`cuco`), NVIDIA's own actively-maintained header-only GPU data-structures library —
  genuinely more "commonly used" than BGHT in the sense that matters: it's the actual engine
  cuDF's own hash joins/groupbys run on top of in production, not an academic reference
  implementation, and being part of the RAPIDS ecosystem means it's continuously CI-tested
  against current CUDA toolkits — far likelier to build cleanly against CUDA 13.3 than BGHT
  was. Offers `cuco::static_map` (fixed-size, open addressing + linear probing — a direct
  analogue to our own linear-probe baseline) and `cuco::dynamic_map` (grows via linked
  static maps — the closer structural analogue to our reliable 2-level table). **Not yet
  attempted to build** — the natural next candidate if this thread is picked up again, ahead
  of further BGHT toolchain archaeology.
- **`HASH_JOIN_ENGINE.ipynb` DONE (2026-07-29)** — a real, human-readable worked example
  (`customers JOIN orders`, 8 customers/12 orders including 2 genuine no-match cases), built
  and probed independently through both the linear-probe reference and the reliable
  tiny-pointer engine, confirmed to agree on all 12 rows (a correctness demonstration, not just
  a speed one) — plus the full benchmark charts. **The hash-join work outgrew a single file and
  got its own subdirectory**: `.../tensor_core_engine_v5/tiny_pointers/hash_join_engine/`
  (own `Makefile`, own `README.md`, `joindemo.cuf`/`joindemo_reliable.cuf`/
  `join_example_small.cuf`/`build_join_notebook.py`/`HASH_JOIN_ENGINE.ipynb`), separated out
  from the parent `tiny_pointers/` directory's other one-off application demos
  (`hashbench`/`kvpage`/`succinctbst`/`stabledict`/`spacedict`), which still share the
  `../tinymap.cuf` reusable module. See that subdirectory's own `README.md` for the full
  detail, not duplicated here.
- **Packaged as a real, callable Python API (2026-07-29)** — `hash_join_api.cuf` wraps the
  reliable 2-level table in a small `bind(c)` surface (create/build/probe/destroy, up to 8
  independent tables at once), compiled into `libhashjoin.so` using the **exact same
  `bind(c)` + ctypes + CuPy bridge pattern `MPDOK/mpdok_ops.py` already uses** for its own
  solvers — not a new convention. `hash_join.py`'s `HashJoinTable` class wraps it into a
  three-line `build()`/`probe()` API. **Verified working, not just written**: matches the raw
  CUDA Fortran demo exactly on the toy `customers JOIN orders` data; a 200,000-random-key
  correctness + failure-detection check passes (zero build failures at 90% load, real matches
  correct, absent keys correctly return no-match); and an honest failure-mode demo (an
  undersized overflow region) makes `build()` raise `RuntimeError` rather than silently drop
  keys — confirmed empirically at a 4.9% failure rate, right in line with the documented ~6%
  worst-case random-key spill figure. `HOW_TO_USE.ipynb` (via `build_howto_notebook.py`) walks
  all three examples live against the real library, not pre-generated CSVs. **Language
  decision, recorded honestly**: CUDA Fortran kernel + Python host now (matches MPDOK,
  matches the immediate "notebook with more examples" ask); a Rust FFI wrapper (matching the
  `stash` family's ethos) deferred until a real daily-driver use case for this
  deliberately-niche, high-load-factor-only engine identifies itself, per this codebase's own
  established "don't build ahead of an obvious need" pattern.

### New lab — `vector_ann_lab` (idea #1 above, competitive survey + Phase 0, 2026-07-29)
Path: `.../tensor_core_engine_v5/tiny_pointers/vector_ann_lab/`. Picked up idea #1 (GPU
vector/ANN proximity-graph index) after the hash-join engine (idea #2) was wrapped up. Honest
reframing established before any building: not "beat FAISS-GPU/RAFT-cuVS/CAGRA at scale" (a
crowded, NVIDIA/Meta-funded space) but "fit a meaningfully larger index in scarce consumer VRAM
(4-8GB) than mainstream libraries' own memory layouts allow," reusing `kvpage.cuf`'s existing
HBM-directory + host-pinned-tier mechanism for whatever still overflows.
- **Competitive survey (`research/`, 5 sourced topic files + `RESEARCH.md`)**: no green field at
  the "GPU graph ANN + host-spill" level — cuVS/CAGRA already ships an out-of-core build (ACE), a
  CAGRA-to-HNSW CPU-search handoff, and generic UVM oversubscription; DiskANN already solves
  bigger-than-RAM via SSD (CPU-side); PQ/SQ/binary vector quantization is a mature, solved
  4-96x+ compression lever across FAISS/cuVS/Qdrant/Weaviate. **The one real, narrow gap**:
  nothing surveyed compresses the graph adjacency array itself — every library stores neighbor
  edges as flat 32-bit global indices. At full-precision vectors this is a rounding error
  (8-45% of total memory). But once vectors are PQ-compressed enough to fit a large corpus on a
  small card at all (the exact move any real VRAM-scarce deployment already makes), the graph
  inverts to become the dominant cost (~89% of total memory in the survey's worked example).
- **Phase 0 (`phase0_hnsw_baseline.py`, `RESULTS_PHASE0.md`) — real measurement, not just
  desk math**: built 4 real hnswlib indices (dim 384/768, M 16/32, n up to 500k), measured true
  graph-vs-vector byte split via `index_file_size()`. Confirmed graph bytes/node is scale- and
  dim-invariant (pure function of M), and re-deriving the PQ-regime split from these measured
  bytes (not the desk formula) lands at 82.3% (M=16) to **89.6% (M=32)** graph share — an
  independent confirmation of the survey's ~89% projection from a real built index. **Refined
  the compression-factor estimate too**: raw byte accounting of HNSW's actual per-node layout
  suggests tiny-pointer-compressed neighbor IDs could realistically hit ~4-5x on the link-list
  term (better than the survey's conservative 2x) — **but** this requires graph-adjacent nodes to
  share a local bucket, which needs a locality-aware node layout/renumbering pass that doesn't
  exist yet (unlike `tinymap.cuf`'s hash-bucket scheme, a graph's neighbors aren't independently
  hashable — they're whatever the ANN algorithm picked for proximity). Plan around 2x until that
  layout pass is designed and measured; treat 4-5x as upside, not the plan.
- **Phase 1 — done (2026-07-29): the locality-layout pass, designed and measured, and the 4-5x
  upside did not survive contact with real data.** Surveyed the graph-reordering-for-locality
  literature (Gorder/SIGMOD 2016, Rabbit Order/IPDPS 2016, METIS balanced partitioning —
  `research/06_graph_reordering_locality.md`), picked METIS (off-the-shelf, `pip install`-able,
  natively targets fixed-size parts) to prototype against a space-filling-curve baseline and
  arbitrary insertion order, measured on a real 30k-node kNN graph (`phase1_bucket_layout.py`,
  `RESULTS_PHASE1.md`). **METIS wins decisively** (40-60x locality over arbitrary order, 2-3x
  over the curve baseline) — but at the small bucket sizes (B=16-64) tiny pointers were
  originally pictured needing for a cheap few-bit local pointer, only 6-18% of edges are
  actually compressible. The real ~2x compression only shows up at **B≈512** (a `log₂512=9`-bit
  local pointer, not 4-6 bits) plus a genuine one-time METIS build-time partitioning pass — a
  materially more involved mechanism than a drop-in change to `tinymap.cuf`'s existing
  small-bucket scheme. A direct check against the corpus's true cluster labels explained why:
  locality saturates to 100% only once B reaches the corpus's natural semantic-neighborhood size
  (500 in this synthetic test), and degrades smoothly below it — a real structural ceiling, not
  a METIS weakness. **Two open questions flagged before any GPU kernel work**: whether a real
  embedding dataset's nested topic structure (vs. this phase's flat synthetic clusters) lets B
  come down, and whether METIS's own cost holds up at 1M-100M-node scale (only checked to 30k
  here). This is the same "measure before claiming" discipline that produced the SHM/hydro labs'
  own honest muted findings — Phase 0's optimistic desk-refined estimate was a real hypothesis
  worth testing, and testing it here is what caught it being too optimistic.
- **Phase 1b — done (2026-07-29): tested against a real embedding dataset, and the verdict came
  down again.** Re-ran Phase 1's identical measurement on real data (20 Newsgroups, 18,211 real
  posts, genuine 2-level topic hierarchy — e.g. `comp.graphics`/`comp.sys.mac.hardware` as real
  sub-topics of a broader `comp.*`) embedded with real `all-MiniLM-L6-v2` sentence embeddings —
  addressing Phase 1's own flagged caveat about its synthetic single-level clusters.
  `RESULTS_PHASE1B.md`. **The hierarchy does help a little at small B** (24.0% vs 18.0% locality
  at B=64, real vs. synthetic) — confirming the mechanism is real, not wishful thinking. **But
  real data never saturates to 100% locality the way clean synthetic blobs did — it plateaus
  around 45-58% across a 20x range of B (512-4096)**, because genuine semantic neighborhoods in
  real text overlap category boundaries (e.g. `sci.electronics` and
  `comp.sys.ibm.pc.hardware` legitimately share near-neighbors). Best achievable compression on
  real data: **~1.5x**, down from the synthetic graph's ~2x and Phase 0's original 4-5x desk
  estimate — three successive rounds of more-realistic testing each revised the number down,
  never up, which is itself a signal. **Final call: don't proceed to GPU kernel work on this
  mechanism.** A real, validated ~1.5x on a term that's ~89% of memory in the PQ regime is only
  a ~25-30% cut to total index memory — not enough to justify the engineering lift (a real
  METIS build-time pass, cost untested past 18k nodes, plus variable-width local/fallback
  pointers materially more complex than `tinymap.cuf`'s existing uniform-bucket scheme) at the
  scale this codebase put into `hash_join_engine` for a bigger per-effort payoff. If this thread
  is revisited, `RESULTS_PHASE1B.md` suggests chasing whether cuVS/CAGRA's own out-of-core build
  path has real gaps for this codebase's specific 4-8GB-VRAM/large-host-RAM hardware profile,
  rather than continuing to refine this specific compression number further.
- **Phase 2 — pivot assessment (2026-07-29): host-RAM reframing survives, two other pivot ideas
  ruled out as already-solved.** Three pivots proposed after Phase 1b's "skip the kernel" call;
  checked against the literature before writing code (`RESULTS_PHASE2_PIVOT.md`). A "two-tier
  coarse-routing + local-subgraph" engine turned out to already be **SPANN** (Microsoft, NeurIPS
  2021, open-sourced as `SPTAG`) — 2x faster than DiskANN at matched recall/memory at
  billion-scale, and its posting lists are flat vector lists with no internal graph, so the
  tiny-pointer-adjacency angle doesn't even apply to it. "Tiny-pointer PQ residuals" turned out
  to be standard residual quantization, already how IVF-PQ/DiskANN work. **The third pivot is
  real**: Phase 1b's "skip the kernel" call was specifically about branch divergence from
  per-thread real-time variable-width decode during graph traversal, not the compression being
  worthless — reframed as a host-RAM-resident structure inside an existing out-of-core pipeline,
  bulk-decoded once per batch before paging to VRAM (never per-thread on the hot path), that
  objection no longer applies. Mapping Phase 1b's real ~1.5x/~30% number onto host-RAM capacity
  instead of VRAM fit gives a concrete result using this codebase's actual target hardware:
  **~1.43x more vectors resident in RAM before hitting disk** (e.g. 166M→237M vectors on a 64GB
  host). **Next step, not yet started**: prototype and time the bulk-decode path itself to
  confirm decode cost doesn't eat the capacity win, before building this as a real add-on to an
  existing pipeline.
- **Phase 2b — done (2026-07-29): the bulk-decode prototype, timed for real on the RTX A1000 —
  currently a net loss, root cause diagnosed.** Built and timed the exact pipeline: host-RAM
  tiny-pointer-compressed adjacency (D=64, B=512, `f_local=0.405`, from `RESULTS_PHASE1B.md`'s
  real numbers) transferred + GPU-decoded, vs. just transferring an already-flat array
  (`RESULTS_PHASE2B_BULKDECODE.md`). **Real, reproducible at every scale tested (200K-4M nodes):
  the compressed pipeline is ~40% SLOWER end-to-end** — decode alone costs almost as much as the
  transfer it was meant to save. Root cause checked, not assumed: the decode kernel runs at only
  ~6-12% of the card's real memory bandwidth (measured ~13.7 GB/s vs. a 112-224 GB/s peak range),
  consistent across every scale — a latency-bound, one-thread-per-node kernel design, not a
  fundamental ceiling; a warp-cooperative redesign is the concrete, untried next step. **This is
  the fourth consecutive round of real measurement in this lab, and the fourth to come back
  weaker than the last** (4-5x desk estimate -> 2x synthetic graph -> 1.5x real embeddings -> net
  negative as currently implemented). Status at that point: stopping unless a concrete fix for
  the diagnosed root cause (latency-bound kernel) presented itself.
- **Phase 2c — done (2026-07-29): the definitive microbenchmark, and it REVERSES Phase 2b.**
  **Attribution: the fix — one warp per node, `__shfl_sync`-broadcast metadata, fixed-stride
  layout — was suggested by Gemini**, given Phase 2b's write-up as an external second opinion;
  it's the real unblock, credited to Gemini rather than this lab's own derivation. Tested against
  a hard bar (~50% of peak bandwidth, ~60-80 GB/s on
  the A1000) as the definitive yes/no on GPU decompression for this architecture
  (`RESULTS_PHASE2C_WARPDECODE.md`). **Result: ~187-191 GB/s, consistent across every scale
  tested (200K-4M nodes) — more than 3x past the bar**, implying ~83-85% of the card's real
  peak bandwidth. Decode cost dropped from ~50% of total pipeline time (Phase 2b's naive kernel)
  to ~6-7% (this kernel) — **net result: the compressed pipeline is 1.05-1.09x FASTER than a
  flat transfer at every scale, correctness-verified.** The data layout never changed (the
  existing CSR grouped-local/grouped-fallback structure already matched the "fixed-stride, no
  bit-parsing" layout the fix called for) — the entire difference was warp-cooperative memory
  access replacing one-thread-per-node. **Phase 2b's "stop here" is reversed for the
  kernel-feasibility question**: GPU decompression of this scheme is real and fast on this
  hardware, given a competent kernel — the earlier negative result was a naive-implementation
  artifact, not a property of the architecture or the scheme. **Next step, not yet attempted**:
  this phase's non-bit-packed int16 encoding only reaches 1.163x raw compression (vs. the
  ~1.43x/~30% ceiling measured earlier); with decode now cheap, there's real headroom to try
  true bit-packed 9-bit local pointers and see whether the fuller compression survives a
  comparably warp-cooperative kernel.
- **Phase 2d — done (2026-07-29): true 9-bit bit-packing, and it holds up — lab closed out.**
  9-bit local pointers (exact, `ceil(log₂512)`), packed LSB-first, decoded per-lane with a
  2-byte read+shift+mask (a 9-bit code never spans more than 2 bytes; no `__popc`/scan needed,
  since host-side CSR offsets already give each node's start position — that concern only
  applies to a flag-array design this codebase never used) (`RESULTS_PHASE2D_BITPACKED.md`).
  **Bandwidth barely dropped** (~187-191 -> ~181-184 GB/s, still ~3x past the bar) **while
  compression improved** (1.163x -> 1.296x), correctness holding at 100% every scale. **Net
  pipeline result improved to 1.17-1.22x faster than a flat transfer** (up from 1.05-1.09x).
  **Lab closed out**: four real fixes/checks in a row, each responding to the last one's honest
  finding (naive kernel, slow -> warp-cooperative + int16, fast, fix suggested by Gemini -> true
  bit-packing, fast and more compressed) — the host-RAM tiny-pointer bulk-decode pipeline is now
  validated end-to-end on real hardware with the full compression scheme: ~1.3x net speedup,
  ~23% less data moved, decode at ~6-7% of total time, 100% correctness-verified, combined with
  the earlier ~1.43x host-RAM capacity framing to close the loop this lab set out to test.
- **Packaged as `gpustash` (2026-07-29)** — `.../tensor_core_engine_v5/tiny_pointers/gpustash/`,
  a new sibling engine to `hash_join_engine`, same shape (CUDA Fortran kernel + `bind(c)` API +
  Python/CuPy wrapper + notebook). Name chosen by Fraser to echo the `stash`/`pdfstash`/
  `mdstash` family while flagging a GPU is involved. `gpustash.cuf`'s `gs_decode_kernel` is the
  real warp-cooperative, bit-packed decode kernel validated in Phase 2c/2d (stateless — a pure
  decode function, no persistent handle/table unlike `hash_join_engine`); `gpustash.py` provides
  `compress()` (host-side numpy: builds the CSR + bit-packed representation from a real
  fixed-degree adjacency array + bucket assignment) and `GPUStash.decode()` (the GPU side).
  **Verified against a real, non-synthetic-membership adjacency array**, not just re-running the
  lab's synthetic timing harness: 1.34x compression, 100% correctness (unordered neighbor-set
  match). Benchmarked at 200K/1M/4M nodes with proper warm-up + min-of-5 timing (first-call
  JIT/allocation overhead genuinely inflated an early naive re-benchmark by 3-4x before this was
  caught) — reproduces the lab's own numbers: **1.18-1.24x net pipeline speedup, ~1.3x
  compression, every scale**. `GPUSTASH.ipynb` (via `build_gpustash_notebook.py`) documents the
  mechanism, the Gemini-suggested fix, a worked correctness example, and the live benchmark —
  0 cell errors. `README.md` records the full lineage and honest scope (decoder, not an
  index-builder; depends on an upstream bucket assignment it doesn't itself choose).

### `embedding_index_lab` — recsys/DLRM-style ID→row index, small test (2026-08-10)
`.../tensor_core_engine_v5/tiny_pointers/embedding_index_lab/` (`test_embedding_index.py`,
`RESULTS.md`). New brainstorm idea (2026-08-10 session): DLRM/recsys embedding tables index
a dense row by an arbitrary sparse categorical ID — `hash_join_engine`'s existing
`HashJoinTable` is already that exact key→value shape (compact, GPU-resident, build-once/
probe-many, high load factor), untested until now against a real (non-synthetic) key
distribution. Reused `ragstash`'s real 216,865 embstore content-hashes as a proxy key set
(real distribution, not a real recsys ID pipeline — none available). **Result**: default
`ovf_frac=0.15` built fine at loadfactor=0.95 (real hashes behave like the paper's ~6%
random-key baseline), 2000/2000 correctness check passed, and after correcting for
first-call GPU/dict JIT warm-up (the same lesson `gpustash` benchmarking already caught):
HashJoinTable beat a plain Python dict by 3x at batch=1K queries widening to 157x at
batch=216K (dict is not slow — the gap is the batch-probe access pattern, which is exactly
DLRM's inference gather pattern), plus ~6x smaller memory footprint (3.8MB vs ~22.6MB).
**Honest caveat**: only tested at 216K keys, far short of a real large-scale recsys table's
billions of rows — validates the mechanism, not production scale. Not embedding-vector
compression (rows unchanged), only the ID→row index in front of them.
**Notebook shipped (2026-08-10)**: `EMBEDDING_INDEX.ipynb` (via
`build_embedding_index_notebook.py`, same convention as `GPUSTASH.ipynb`/
`HASH_JOIN_ENGINE.ipynb`) — executed live against real `ragstash` data and `HashJoinTable`,
0 cell errors, 100% correctness at every scale point (1M-64M synthetic + 216,865 real keys).
This is the idea picked to take further from the session's three (idea #1); idea #2 (JIT/
dispatch caching) stayed unresolved (`dispatch_cache_lab`/`variable_tp_lab`) pending further
planning.

### `id_index_lab` — wired `embedding_index_lab`'s ID→row index into `gpustash` (2026-08-18)
`.../tensor_core_engine_v5/tiny_pointers/id_index_lab/` (`id_index.py`, `test_id_index.py`,
`pipeline_demo.py`, `id_index-rs/`, `README.md`, `RESULTS.md`). Closed the gap
`embedding_index_lab` left open: `gpustash.compress()` assumes `bucket_id = row_index // B`
— neighbor ids ARE dense array positions — which only holds because a corpus like
`seedverify`'s embstore already hands out sequential row ids. A real recsys ID space is
sparse and arbitrary, so something has to sit in front of METIS mapping external id → dense
row, and map back after decode so callers see external ids, not row/rank numbers. `IdIndex`
is that layer: one `HashJoinTable` (forward) + a plain device array for the reverse
direction (row→id needs no hash lookup — rows are already dense, so it's array indexing, not
a second table — a design simplification found while building it, not planned upfront).
**Verified two ways on the real, now-217,759-key `ragstash` corpus** (grew ~900 keys since
`embedding_index_lab`'s 216,865): (1) `test_id_index.py` — `IdIndex` round trip 5000/5000,
full row→id coverage 217,759/217,759, plus a full external-id→row→METIS-rank→gpustash-decode
→row→external-id round trip at 2000/2000 on a small synthetic sub-graph; (2)
`pipeline_demo.py` — the real full pipeline (hnswlib kNN → METIS → relabel → gpustash) run
end to end on the **entire** real 217,759-vector corpus with `IdIndex` bookended on both
ends: 2000/2000 round-trip correctness, 48.4% locality and 1.287x compression (matches
`ragstash/full_scale_graph.py`'s own prior numbers — confirms the ID layer didn't change the
compression story, only added a sparse-ID front door). **Two real bugs caught while
building the test harness** (not in `IdIndex`/`partition.py`/`gpustash` themselves): a
non-symmetrized synthetic adjacency crashed pymetis with a real `Bus error`, exactly the
failure mode `partition.py`'s own docstring already warned about; and an early correctness
check compared decode's rank-space output against local-id space, reading as 0/2000 even
though decode itself was correct.
**Rust port (`id_index-rs/`)**: per `hash_join_engine`'s own README, which already flagged
"a Rust FFI wrapper... is the natural next step if a real daily-driver/production use case
identifies itself" — this is that use case. Built as an FFI wrapper around the *existing*
`libhashjoin.so` (not a from-scratch reimplementation like `variable_tp_lab`'s CPU port of
the algorithm itself). Real detail found reading `hje_api.cuf`: `hje_build`/`hje_probe`'s
pointers are DEVICE pointers, not host memory, so the crate isn't purely libc-only — it
links `libcudart` and does its own `cudaMalloc`/`cudaMemcpy`/`cudaFree`. `cargo run --release
--bin verify` reads the same real embstore via an independent Rust implementation of its
binary format and reproduces the Python reference's exact numbers (5000/5000, 217759/217759)
— a genuine FFI/ABI cross-check, same compiled `.so`, two languages, not a restated test.
**Honest scope, unchanged**: still the laptop GPU (217,759 keys well under
`embedding_index_lab`'s validated ~90M-key ceiling — the 8GB desktop is for a bigger real
corpus or re-running that ceiling, not this wiring); still real content hashes standing in
for sparse IDs, not a real recsys ID stream's popularity/skew distribution.

### `~/sparsebridge/` — `id_index_lab` promoted to a top-level project + real RAG bench (2026-08-18)
Same session as `id_index_lab` above. User's excitement after that result ("a sparse
external key space bridge to dense GPU-resident graph/vector representations... could
power a high-throughput RAG and vector search engine") prompted promoting it out of
`tiny_pointers/` into its own top-level dir, `~/sparsebridge/`, matching the `~/stash/`/
`~/pdfstash/`/`~/ragstash/` convention (`id_index_lab` itself left in place as the
historical lab record). `pipeline_demo.py`'s build steps were refactored into a reusable
`build_index.py`, and — the real gap this promotion surfaced — `gpustash` had **no
query-time search algorithm**, only a decoder for an existing corpus row's precomputed
neighbors. Built `ann_search.py`: real, new single-layer NSW-style greedy best-first
search over the decoded graph (`greedy_search`/`multi_start_search`), confirmed with the
user before building since it's genuinely new algorithm code, not glue.
**Real three-way RAG benchmark built** (`sparsebridge/rag_bench/`), per the user's own
steer toward a "correct"/standard comparison rather than VECSRCH (`mpdok_websearch`) —
ruled out as apples-to-oranges (TF-IDF + kernel ridge reranking, different corpus) and
left as a mentioned-only alternative: brute-force cosine (`ragstash/rag_query.py`'s real
`retrieve()`, unchanged — also ground truth) vs. hnswlib native search (the standard,
already-integrated reference library) vs. sparsebridge, on the real 217,759-vector
`ragstash` corpus and 18 real embedded questions.
**A real bug caught mid-benchmark**: a single fixed entry point made sparsebridge's
recall bimodal (0.00 on several technical queries, near-perfect on others) — a
mechanistic, not random, failure of single-layer greedy search losing its way with no
multi-layer hierarchy to route from. Fixed with `multi_start_search` (several random
restarts, keep the best) — mean recall@10 went from **0.533 → 0.744** on rerun, same
corpus/queries. **Real, honest final numbers**: recall@10 hnswlib 0.906 vs. sparsebridge
0.744; latency hnswlib 0.092ms vs. sparsebridge 3.320ms vs. brute-force 55.506ms —
sparsebridge's pure-Python search loses to hnswlib's compiled search (expected, stated up
front) but is **~17x faster than the brute-force path this codebase actually runs in
production today**. Memory: gpustash's compressed graph is ~33x smaller than hnswlib's
own index (~20MB vs. ~697MB), though both are small next to the ~669MB shared raw-vector
array every method still needs resident to score candidates — stated honestly as a graph-
compression story, not an overall-memory one.
**Live answer comparison** (`rag_answer_compare.py`, reusing `ragstash/rag_query.py`'s
real context-building/citation/`llm_provider.call_llm()` pipeline unchanged, retrieval
swapped): sparsebridge's graph rebuilt over the exact same resolvable SEC+PDF subset
`ragstash` restricts to (fairness — neither path wastes budget on unresolvable rows).
3 real questions, both paths produced grounded, correctly-cited, sensible answers (real
SEC EDGAR URLs, real kindle/web citations) despite low context overlap (1-2/6 passages
identical) — approximate retrieval genuinely surfaces different real sources without
visibly hurting answer quality in this small manual check. **A second real bug caught and
fixed**: building the graph-restriction set from real uint64 hashes via a plain `int()`
cast then `np.asarray(..., dtype=np.int64)` overflowed for hash values above `2**63-1` —
fixed by using `.view(np.int64)` (bit reinterpretation) consistently, matching
`id_index.py`'s own already-correct convention for this exact dtype.
**Follow-up: `ef`/entry-point tuning (`sweep_ef_entrypoints.py`, same session)**. The
first pass's `ef=80`/4-restart settings were an unscrutinized guess. Swept `ef ∈
{40,80,120,160,240} × entry_points ∈ {1,2,4,8,16}` (25 combos, same 18 queries, one
index build reused throughout). Two real, non-obvious findings: raising `ef` alone buys
more recall per ms than adding restarts does (`ef=240,entry_points=1` — recall 0.844,
2.75ms — dominates the original `ef=80,entry_points=4` — recall 0.744, 3.46ms — on both
axes); and going from 1→2 entry points changed recall by exactly zero at every `ef`
tested (only 4+ restarts helped), unexplained beyond what was actually measured. Picked
`ef=240,entry_points=4` as the new shipped default (Pareto-frontier point closest to
hnswlib's own recall without paying `entry_points=8`'s latency) — recall 0.894, matching
hnswlib's own 0.894 on a full rerun with these defaults. **Honest correction to the
first pass's headline**: the original "~17x faster than brute-force" was measured at
`ef=80`'s recall (0.744), well below hnswlib's — an unmatched comparison. At matched
recall (0.894 vs. 0.894), sparsebridge is **~6.4x faster than brute-force**, not 17x —
still real, just smaller once compared fairly; still ~92x slower than hnswlib's own
compiled search at that same matched recall, exactly as predicted before any of this
ran. `bench_recall_latency.py`/`rag_answer_compare.py` now ship with these tuned
defaults, not the original guess.

**Follow-up: variance check across 5 rebuilds before considering Rust (`variance_check.py`,
same session)**. User asked whether to refactor to Rust yet; answer was "test stability
first, don't port possibly-noisy numbers" — so ran 5 independent `hnswlib`+METIS+`gpustash`
rebuilds (2 at identical `hnsw_seed=42`, 3 at different seeds) at the tuned defaults. **Real
finding**: locality/compression vary meaningfully even at the SAME seed (63.5% vs 57.8%
locality between two `hnsw_seed=42` runs) — confirms `hnswlib`'s multi-threaded `add_items()`
insertion order isn't fully pinned by its seed alone, matching the unexplained locality swings
already noticed in `bench_recall_latency.py`'s own earlier repeated runs. **The reassuring
result**: despite that graph-level variance, recall@10 stayed tight — sparsebridge 0.882±0.011
(1.2%), hnswlib 0.898±0.006 (0.6%) — the tuned `ef=240`/4-restart operating point isn't a
lucky single draw, it reliably lands near hnswlib's own recall across independently-built
graphs. **An honest, left-open caveat**: absolute latencies in this 5-trial session ran higher
than the earlier single dedicated benchmark (brute-force ~104ms vs ~55.6ms; sparsebridge
~11.3ms vs ~8.6ms) — plausibly cumulative CPU load from 5 consecutive heavy builds back-to-back
in one process, but not confirmed with an isolated comparison; derived speedup-vs-brute-force
ranged 6.4x-9.27x and slowdown-vs-hnswlib 92x-127x across the two sessions, reported as a range
rather than a single pinned number until a cleaner isolated run is done.
**Follow-up: Rust port of `ann_search` (`ann_search-rs/`, same session)**. User: "go ahead
with the Rust refactor now; we can worry about speed later" — so ported `ann_search.py`'s
own algorithm (`greedy_search`/`multi_start_search`), NOT the hnswlib/METIS/gpustash build
pipeline (out of scope — a much larger separate undertaking; `export_graph.py` builds the
real graph once in Python via the unchanged `build_index.py` and dumps it — real 217,759×768
vectors, real decoded neighbors, Python's own real-query results — to a ~725MB binary file
for the Rust port to run against, same division of labor `id_index-rs` used for verifying
an FFI binding rather than the whole pipeline). **Verified, not just written**: 18/18 real
queries produced an EXACT result-set match against Python's output — byte-for-byte identical
row sets on real data, not just close. Bar was set at set-equality (gpustash's own
established "order carries no meaning" convention, specifically to tolerate Python
`heapq`/Rust `BinaryHeap` tie-breaking differences) but the real result came back exact
anyway. Zero external Rust dependencies (a small local `Sim(f32)` `Ord` wrapper instead of
pulling in `ordered-float`, matching this codebase's minimal-deps convention).
**Follow-up: speed optimization pass (same session)**. User: "now optimize sparsebridge
for speed; great progress!" Baseline (the correctness-first port): 4.70ms mean/query
(`bin/bench.rs`, added since no timing had been measured yet — only correctness). Two
concrete changes: `visited` `HashSet<usize>` (allocated + hashed per search) replaced with
a generation-stamped `Vec<u32>` bitset sized to the corpus (no clearing, no hashing); and
`multi_start_search`'s sequential entry-point restarts parallelized via
`std::thread::scope` (zero extra crate dep — each restart was already fully independent
state in the original Python semantics, nothing to synchronize). **Isolated each change's
real share, not assumed** (`bin/bench_sequential.rs` diagnostic): bitset alone 4.70→4.16ms
(~11%), +parallelism → **2.73ms** (~1.5x further) — parallelism did most of the work.
**Re-verified 18/18 exact result-set match after both changes** — a silent correctness
regression from "optimizing" would defeat the point of the port. New standing: Rust
sparsebridge is now ~3-4x faster than the Python reference (was already faster, now more
so) and ~20x faster than brute-force (up from Python's ~6.4x), while closing the gap to
hnswlib's native search from ~92-127x slower down to ~30x slower.
**Follow-up: closed out the "METIS graph partitioning" query anomaly (same session)**.
The one query that scored low for both hnswlib and sparsebridge at every setting, left
open as "genuinely hard, not asserted as solved." `investigate_metis_query.py` pulled the
real top-20 brute-force hits/scores instead of continuing to guess: the corpus has no
genuine content about the METIS library at all (top hits are topically-adjacent HPC/graph
material, MPI course slides, none naming METIS); the ground-truth top-10/11 boundary is an
**exact numerical tie** (0.4415 twice) — a coin flip, mechanically explaining why methods
disagree there; and a control query with real matching content ("customer concentration
risk," genuine SEC filings) scores meaningfully higher (0.5511 vs 0.4912 max), confirming
via VECSRCH's own established score bands (0.30-0.50 = "tangential," not strong) that this
query never had a real answer to find. Verdict: a real corpus-coverage gap, not a
retrieval bug — closed, not left open.
**Follow-up: wired into COBOLMM/rustmm's live menu as a new `SP` item (same session)**.
User: "should we test out our new sparsebridge engine with our RUSTMM menu item?" — with
two explicit scope decisions: build real disk-cache persistence first (not accept the
~50-60s HNSW+METIS rebuild every launch), and add a NEW menu item rather than touching the
live, daily-use `RQ` item. **Caching**: `build_index.py`'s expensive, picklable part
(`neighbors_row` + stats, NOT the unpicklable `hnsw` object, which also isn't needed to
serve a query) now goes through `ragstash/cache_utils.cached_build`/`file_fingerprint`
unchanged — no new caching code. Verified: 63.14s cold → 1.13s cache hit (55.7x), with
`neighbors_row` confirmed byte-identical between runs. **New `sb_query.py`**: mirrors
`ragstash/rag_query.py`'s own resident-loop shape, sourcing retrieval from sparsebridge
instead of brute-force, reusing `load_corpus()`/citation resolution/`llm_provider.call_llm()`
directly (same pattern `rag_answer_compare.py` already established). Verified end-to-end
on real questions: grounded, cited answers, real SEC EDGAR links, 8.5s total wall time
(vs. ~50-60s pre-caching). **`cfg/MENU.DAT` edit**: backed up first; new fixed-width
records built programmatically. **A real bug caught before shipping**: the first draft
used an em-dash in a label — a 3-byte UTF-8 character that silently passes Python's
character-count `len()` check but would have broken COBOL's byte-oriented `PIC X(50)`
fixed-width field; caught by checking `.encode('ascii')` byte length instead, fixed with a
plain hyphen. **Verified by actually running the real compiled COBOL binary**
(`mainmenu_bin`, driven headless via `pty.openpty()`, no gnome-terminal relaunch/NAS
dependency) and capturing its real rendered output — the new `SPARSEBRIDGE RAG` header/`SP`
item render correctly, positioned exactly between `RQ` and `DECISION ENGINE`, box borders
aligned. Honest limit: rendering/dispatch verified, not a full scripted interactive
session choosing `SP` end-to-end through the COBOL UI.
**Follow-up: real correctness bug caught live, fixed at the root (same session)**. User
ran `sb_query.py` live for "handwritten notes" and reported "results seem similar and
faster despite fewer hits" (30 brute-force hits vs. 8 sparsebridge, different `k`s so not
inherently alarming — but sparsebridge's `k=10` should have resolved all 10). Investigated
rather than assumed benign (`investigate_missing_hits.py`): every missing hit had a
*negative* `ext_id`, every resolved one positive. Root cause: `ragstash`'s `combined_idx`
is keyed by genuine **unsigned** FNV-1a-64 hashes, but `id_index.py`'s `IdIndex.to_id()`
was returning the **signed** int64 copy `HashJoinTable`'s GPU API internally requires —
negative for any hash `>= 2**63` — silently breaking its own documented contract ("same
dtype IdIndex was built with") and dropping roughly **half of all real resolvable
results** from citation lookups, on every query, since `IdIndex` was first built. Invisible
to every prior correctness check because `test_id_index.py`'s round-trips only ever
compared `IdIndex` against itself, never against an external unsigned source like
`combined_idx` — only live use surfaced it. **Fixed at the root** in `id_index.py`
(`_row_to_id` now preserves the caller's original uint64 values; a separate signed int64
copy is used only for the internal `HashJoinTable.build()` call) rather than patched at
each of the several call sites that would otherwise all need the same fix independently.
One follow-on fix: `pipeline_demo.py`'s round-trip check explicitly value-cast `to_id()`'s
output back to int64, which would now overflow given the corrected output dtype — fixed by
relying on `to_row()`'s own already-correct internal handling instead. **Verified, not just
asserted**: `test_id_index.py` re-run clean (unaffected, as expected — self-consistent
checks). `investigate_missing_hits.py` re-run: 10/10 resolved (was 8/10) on the exact query
that surfaced it, recovering the exact same real file ("Screenshot 2021-07-20 173204.jpg")
brute-force had already found independently — not a different result, the same real
content sparsebridge had been silently unable to show. **Honest retroactive note**: doesn't
affect `bench_recall_latency.py`'s recall/latency/tuning numbers (row-index comparisons,
never touch `IdIndex`); does mean the earlier "Live RAG answer comparison" 3-question
spot-check ran under this bug — never showed anything false, just a smaller true candidate
pool than reported. Not re-run; flagged honestly rather than left uncorrected.
**Follow-up: hit-count mismatch, same day**. With the hash bug fixed (10/10), user
immediately noticed sparsebridge still showed only 10 hits vs. `RQ`'s 30 for the same
query — not a bug, `sb_query.py`'s `K=10` had been borrowed from `rag_bench`'s own
`recall@10` evaluation metric, the right constant for *measuring* recall, the wrong one
for matching `ragstash`'s actual retrieval width. Fixed by importing `rag_query.py`'s real
`TOP_K` (30) directly instead of hardcoding a mismatched value — now reports 30 hits,
matching `RQ` exactly, latency unchanged (~22ms, confirms `ef` not `k` dominates cost).
**Live validation, same day**: user ran "CUDA Fortran" through both `sb_query.py` and
`RQ` side by side with both real bugs above already fixed — identical top-6 retrieval,
identical generated answer, sparsebridge 15ms vs. brute-force 137ms (~9.1x faster) — a
real, self-picked query, not a cherry-picked benchmark one. User: "I think this
sparsebridge is a great success!" Arc closed for this session: promoted from a lab
experiment, tuned, ported to Rust and verified byte-identical, optimized, wired into the
live production menu, two real correctness bugs found via actual use and fixed at the
root, now confirmed matching production quality at a real measured speedup in daily use.
**Follow-up: GPU-GNN-over-Obsidian idea scoped down to a cheap "related items" validation
first (same session)**. User's next idea: a GPU graph neural net over dynamic/sparse
graphs (Obsidian notes as an example), sparse 64-bit node IDs, GPU-side neighbor sampling
without host remapping. Checked the real target before scoping any new engine: the real
Obsidian vault (`~/SynologyDrive/Adversa 2022/`) has only **274 explicit `[[link]]` edges
across 2,534 notes** — too small to justify GPU/compression infra on its own (same lesson
`embedding_index_lab` already taught: dict wins until real scale forces the issue). The
"large adjacency matrix" the user described better fits a *semantic* similarity graph
(node features = embeddings), which sparsebridge's existing cached graph already provides
at real 217K-node scale. Rather than build new sparse-graph-sampling/GNN infrastructure
speculatively, built a **zero-new-infrastructure validation** first
(`related_items_demo.py`): shells out to the real `concept_search` binary (a separate Rust
codebase, not modified — uses its own `--report` flag), matches each real hit back to a
content hash via `ragstash`'s `combined_idx` (keyed by exact chunk text, more reliable
than label since many chunks share a label), then looks up "related items" in
sparsebridge's already-cached graph. **Real result**: 5/5 hits resolved for "CUDA
Fortran," with genuinely interesting cross-document connections — a Kindle excerpt of a
book surfaced the same book's real PDF pages; an Obsidian web clipping surfaced CUDA
Fortran interface PDFs it doesn't literally name. Validates the underlying idea has real
value at zero engine cost; any further sparse-ID neighbor-sampling/GNN scoping is real,
separate work, not started here. See `sparsebridge/RELATED_ITEMS_DEMO.md`.
**Follow-up: promoted to a real live trial on `concept_search` (`CP`) (same session)**.
User: "wire it into the menu as a real feature... add it to concept search menu item as
it is lower risk... test it out for a few weeks." Deliberate site choice: `CP` is
lower-traffic than `universal_search` (the actual daily driver) — promote further only if
this proves compelling. User also asked for the related-items section to use a distinct
heading color, and — mid-implementation — to fork `seedverify` rather than edit it in
place. **Why a fork**: `concept_search` is a resident, in-process interactive loop (loads
its corpus once, then answers many queries) — a wrapper that shells out to a *fresh*
`concept` process per query would lose that property (real measured cost: ~4.9s combined
`load_corpus()`+cached `build_index()`, paid once per session vs. once per query). Forked
the whole `seedverify` directory (`cp -r`, no git repo to branch from) to
`seedverify_related/`; the original is never touched. Added three new functions to the
fork's `search()` — hand-rolled JSON (no new crate dependency, matching the crate's
existing zero-deps `Cargo.toml`) and a helper-subprocess call with a background-thread +
channel 15s timeout (avoids both a pipe-buffer deadlock and a hung main thread) — one call
site, after the existing hit-printing loop, never touching the separate `--report`
file-writing block. **Wired into the LIVE menu with zero edits to any currently-running
script**: `concept_search.sh` already reads `${MM_CONCEPT_BIN:-...}`, built for exactly
this override (its own error message says so) — one new line in
`~/COBOLMM/config.thinkpad-p16.env` (backed up first) repoints it at a new
`~/concept_related/concept` wrapper. Rollback is deleting that one line. **Verified, not
assumed**: fork's real binary run directly — concept_search's own real output unchanged
byte-for-byte, plus a new light-purple (`\x1b[38;5;141m`, distinct from concept_search's
own orange) "Related items" section; `--report` confirmed clean (grepped, zero heading
leakage); graceful degradation actually tested (renamed the helper away mid-test —
concept_search's own real 19-match/48.7s search completed normally, no crash/hang/visible
error, just no related-items section); `MM_CONCEPT_BIN` override confirmed to actually
take effect via `concept_search.sh` directly.
**Follow-up: content snippets per related item (same day)**. User feedback: bare
`[source] label` lines weren't enough — wanted the same content preview (image
descriptions, PDF page text) the main search already shows per hit. `combined_idx`'s
stored text was already available at resolution time — `find_related_items()` now
returns `(rel_source, rel_label, rel_text)`, and the helper prints a truncated,
word-wrapped snippet (`textwrap`, stdlib, no new dependency) under each related item, dim
and matching the main search's own body-text style. No Rust rebuild needed — the fork
only ever prints the helper's stdout verbatim.
**Follow-up: real clickable links + a genuinely compelling first result (same day)**.
User: "handwritten notes" surfaced a Kindle highlight about "writing on paper" improving
conceptual grasp — real conceptual retrieval the main concept_search hybrid search itself
did NOT surface for that query, exactly the value case for this feature. Also asked for
clickable links, ctrl+click-openable like `universal_search`. `combined_idx` already
carries the real URL per item — `find_related_items()` now returns `(source, label, text,
url)`; the helper wraps each label via `ragstash/term_links.hyperlink()` (reused, not
reimplemented). Verified live: 34 real, correctly-formatted OSC8 links in one query.
See `sparsebridge/README.md` and `sparsebridge/rag_bench/RESULTS.md` for full detail.
**New lab started (2026-08-21): `sparsebridge/recsys_lab/`** — a second application of
`IdIndex`, this time DLRM/recsys-style sparse-ID embedding-table lookup (user/item/ad IDs
-> dense row -> GPU gather -> pooling), chosen over the GPU-GNN-on-dynamic-graphs idea
above as the more approachable first test (direct reuse of `IdIndex.to_row()` + a dense
gather, no new kernel needed for Phase 0 — the GNN idea needs graph storage + multi-step
sampling before any test is possible, and this session's own earlier finding above already
showed the one concrete GNN target on hand, the Obsidian vault, is too small in practice to
justify it). See `sparsebridge/recsys_lab/LAB_PLAN.md`.
**Phase 0 DONE (2026-08-21):** `datasets.py` (synthetic sparse-uint64-ID embedding table)
+ `bench_lookup.py` (correctness vs. ground truth + a throughput sweep separating remap
cost from gather+pool cost, host-dict baseline vs. `IdIndex`/GPU path). Correctness: exact
match at n_rows=20k incl. OOV handling. **Real finding, not assumed**: first CUDA/CuPy call
per process pays a one-time ~150ms JIT cost (measured directly — identical call, 2nd run
300x faster on gather+pool alone) — sweep numbers below are post-warmup. GPU path's
per-batch cost is nearly flat across 100k-1M rows and 1,024-65,536 batch size (remap
0.05-0.18ms, gather+pool 0.3-1.5ms); host-dict path scales ~linearly with batch. Net: **a
wash-to-loss at batch=1,024 (0.7-3.4x), 58-104x at batch=65,536** — the advantage is
batch-size-dependent, not universal, stated as such. Also found: `IdIndex` build is
1.5-2x *slower* than a plain dict build at every scale (one-time-per-table cost, not yet
amortized/break-even-measured). Honest limits: only tested to 1M rows (`HashJoinTable`
previously verified only to ~217k keys for the RAG use case); host baseline is a naive
Python dict, not a tuned CPU baseline (`EmbeddingBag`/vectorized numpy); single table,
fixed bag length only. See `sparsebridge/recsys_lab/RESULTS_PHASE0.md`.
**Follow-up (2026-08-21): break-even point + 10M-row scale test.** `bench_breakeven.py`:
break-even is batch-size-dominated, not table-size-dominated — at large batch (65,536)
the GPU path pays back its ~1.5-2x-slower setup cost within 1-3 batches (large per-batch
gap); at small batch (1,024) break-even stretches to 26-350 batches (tiny per-batch gap).
`bench_10m.py`: pushed `IdIndex`/`HashJoinTable` to **10M keys**, an order past this
lab's own §3 (1M) and the engine's prior-largest verified scale (~217k, RAG use case) —
memory-constrained to dim=16 on this 4GB laptop GPU (measured real footprint via
`nvidia-smi`, not CuPy's own pool, which undercounts — the CUDA Fortran backend
allocates outside it). **Flat per-batch-cost pattern holds at 10M rows**: correctness
exact, speedup up to **153.7x** at batch=65,536 (higher than 1M rows' 103.7x — host dict
remap cost itself grows with table size, ~47ms→64ms at same batch, GPU stays flat).
Build cost ratio (~1.5-2x slower than dict) holds too but now costs **seconds** (7.2-7.8s
vs ~4s), pushing 10M-row break-even to ~28-2,608 batches depending on batch size — 7-14x
higher than the 1M-row numbers. **Follow-up (2026-08-21): build-time scaling, 3rd point added.** `bench_build_scaling.py`
added a 3M-row point, re-measured 1M/3M/10M fresh in one process (fixes the earlier
2-points-from-separate-sessions gap). **Both paths are mildly superlinear** (full-range
exponent 1.23 host, 1.25 idx — not the p=1.0 linear assumption), but the **ratio between
them stays stable across the 10x range** (1.85x, 2.16x, 1.94x, no systematic drift) —
reads as a shared Python-dict/GPU-allocator effect, not something specific to
`HashJoinTable`. Practical implication: straight-line extrapolation of *absolute* build
time undershoots — 100M rows would cost ~17x the 10M-row build time at this exponent
(~130s), not 10x.
See `sparsebridge/recsys_lab/RESULTS_PHASE0.md` §5-7.
**Follow-up (2026-08-21): tuned CPU baseline, `bench_embeddingbag.py`.** §3's own
honest-limits note predicted `torch.nn.EmbeddingBag` (real DLRM's actual op) would
narrow the naive-baseline speedup — confirmed, with the mechanism now visible:
EmbeddingBag's vectorized C++ kernel crushes the naive Python loop on gather+pool
specifically (~27x at 1M rows/batch=65,536), but its **remap stage is unchanged** (still
the same Python dict — EmbeddingBag has no sentinel/sparse-id support, doesn't touch
remap at all), so remap now dominates its total time at large batch (58.97 of 60.97ms).
**Top-line GPU speedup narrows from 96x (vs. naive) to 56x (vs. EmbeddingBag) at the
largest scale/batch, but doesn't disappear** — `IdIndex` wins specifically because its
GPU hash probe attacks the stage EmbeddingBag leaves untouched. Correctness exact
(n=20,000, 10% OOV via `per_sample_weights`). Open, not run here: a vectorized/compiled
CPU remap (Numba, sorted-array binary search) paired with EmbeddingBag — the fairest
possible full CPU pipeline.
See `sparsebridge/recsys_lab/RESULTS_PHASE0.md` §8.
**Follow-up (2026-08-21): fully-vectorized CPU pipeline, `bench_searchsorted.py`.**
Closed §8's own open question — replaced EmbeddingBag's dict-based remap with
`np.searchsorted` on a once-sorted id array (no Python loop anywhere in the CPU path;
correctness needs an explicit exact-match verification step since searchsorted returns
an insertion point, not a guaranteed match — verified against dict ground truth, both
pooled output and row-level remap). **Genuine surprise**: sorting is the *cheapest*
build of all three structures (dict/sort/IdIndex), not the middle option — 41-45ms vs.
dict's 206-225ms and IdIndex's 382-486ms at 1M rows. **This moves the headline number**:
against this fully-tuned CPU pipeline, GPU speedup is **2-16x at DLRM-realistic large
batches — not 56-104x** (§3/§8 measured real but weaker baselines) — and **at the
smallest batch/table combo tested, the CPU pipeline actually wins outright** (0.37ms vs
0.50ms), the same small-batch fixed-overhead regime §5's break-even math already
flagged. The GPU path's real, defensible advantage across every comparison run in this
lab (§3/§8/§9) is that its per-batch cost stays flat with batch AND table size while
every CPU variant's remap cost scales with batch size — the win concentrates exactly
where DLRM production traffic lives (large batches, long-lived tables), and **2-16x, not
56-104x, is the number to trust** for "how good is the real CPU competition."
See `sparsebridge/recsys_lab/RESULTS_PHASE0.md` §9.
**Phase 0 closed out with a summary + roadmap in `LAB_PLAN.md` (2026-08-21).** Bottom
line: `IdIndex` earns its keep specifically in large-batch, long-lived-table serving
(real: 2-16x over the best CPU pipeline built, not the 56-154x weaker-baseline numbers)
— a real, narrow, defensible result, not a universal GPU win. **Phase 1 DONE (2026-08-21, `bench_gpu_searchsorted.py`):** does "sort beats hash"
(Phase 0 §9, CPU-side) transfer to the GPU? **Yes, more strongly.** `cupy.argsort`
builds 5-100x faster than `IdIndex` (3.79-6.13ms vs. 354-395ms at 1M rows), gap widening
with table size. `IdIndex` stays faster per-batch (1.2-3.2x, still sub-ms) but that's a
small edge against a sometimes-two-orders-of-magnitude-larger build cost. **Break-even:
31-58 batches at 100k rows, 1,079-1,946 batches at 1M rows** — growing with scale, the
opposite direction from what would favor a fixed default. Practical conclusion: neither
GPU structure is universally right — `IdIndex` for tables probed thousands of times per
build, `cupy.searchsorted` for tables rebuilt often or where build latency is on a
critical path (cold start). Adds a second axis to the Phase 2 router: not just device
(CPU/GPU) by batch size, but which GPU structure, by expected batches-per-build. See
`sparsebridge/recsys_lab/RESULTS_PHASE1.md`.
**Phase 2 DONE (2026-08-21, `hybrid_router.py`/`bench_router.py`):** a real router that
calibrates all four candidates (`cpu_naive`, `cpu_vectorized`, `gpu_searchsorted`,
`gpu_idindex`) LIVE against the real table (not hardcoded historical numbers), picks
whichever minimizes `build + expected_n_batches*per_batch`. **Picked correctly in all 3
test regimes** (long-lived large table→`gpu_idindex`; short-lived large table→
`gpu_searchsorted`; small batch/small table→`cpu_vectorized`), each verified for output
correctness and a live head-to-head, not just decision-match. **A real fairness bug
found and fixed first**: initial calibration reused a pre-uploaded query buffer,
understating GPU per-batch cost vs. real serving (fresh query upload every call) —
caught because the router's own live check disagreed with its calibration, not assumed
away; the bug actually flipped one scenario's decision before the fix. **Genuine
surprise**: at small scale, plain `cpu_naive` (dict+loop) beat `gpu_searchsorted` in the
ranking (cheap build wins over few batches) — concrete proof a fixed CPU/GPU rule would
misfire, only live calibration gets it right. Also found: build costs varied 20-60%
run-to-run this session (`IdIndex` build at 1M rows: 425-611ms) — would have silently
broken a router built on this lab's own earlier point estimates instead of live
recalibration. See `sparsebridge/recsys_lab/RESULTS_PHASE2.md`.
**Phase 3 (variable-length bags) DONE (2026-08-21, `bench_variable_bags.py`):** real
DLRM's actual `EmbeddingBag` calling convention (flat id array + offsets) instead of
Phase 0's fixed `bag_size` reshape-and-sum. CPU side is native (`EmbeddingBag` already
supports it); GPU side needed a new segmented-sum primitive since `cp.add.reduceat`
doesn't exist on CuPy. **Two approaches built and measured, not assumed**: a
cumsum-difference trick (no atomics, appealing on paper) was caught **initially wrong**
by the correctness check (float32 catastrophic cancellation on long cumulative sums,
fixed via float64 accumulation) **and** ~10x slower than `cupyx.scatter_add` (1.20ms vs.
12.3ms at 1M rows/8,000 bags) — kept both functions rather than deleting the slower one,
a real negative result worth keeping visible. Correctness exact across all four
candidates incl. genuinely zero-length bags. Same central finding holds with the more
realistic bag structure: GPU stays flat (0.4-1.8ms) while CPU remap scales with total id
count (up to 13.5x speedup at largest scale). **Not yet done**: `hybrid_router.py` still
only dispatches fixed-`bag_size` paths — wiring it to variable-length is the natural
next integration step. See `sparsebridge/recsys_lab/RESULTS_PHASE3.md`.
**Multi-table (Phase 3's other half) DONE (2026-08-21, `bench_multitable.py`):** real
DLRM has dozens of categorical features, each its own table. **Two hypotheses tested,
one confirmed one refuted**: `IdIndex.to_row()`'s FFI call has no stream parameter (read
directly from `hash_join_api.cuf`'s `hje_probe`) — hypothesized not to benefit from
concurrent CUDA streams, confirmed (0.99-1.06x, noise); `gpu_searchsorted` (pure CuPy,
genuinely stream-aware) was hypothesized TO benefit — refuted, no gain either, tables
this size are too small for cross-table overlap to matter regardless of stream support.
**A second constraint found the hard way**: 26-table run crashed on `hje_create`'s
`MAX_HANDLES=8` hard cap (fixed-size static array in the Fortran source, not
runtime-configurable) — `IdIndex` literally can't hold real DLRM's 26+ tables open at
once; `gpu_searchsorted` has no such limit. **Open question answered**: per-table cost
is flat for both (`gpu_idindex` ~0.075-0.079ms/table, `gpu_searchsorted`
~0.109-0.120ms/table, stable 2→8→26 tables) — no super-linear blowup, and GPU-vs-CPU
speedup GROWS with table count (up to 25.6x). The real scaling constraint is
`MAX_HANDLES`, an engine limit, not a performance one. See `sparsebridge/recsys_lab/
RESULTS_PHASE3.md` Part 2.
**Follow-up (2026-08-21, later session): `MAX_HANDLES` raised 8→64, router wired to
Phase 3.** Fraser asked directly whether `MAX_HANDLES=8` was arbitrary — confirmed yes
(`hash_join_api.cuf`'s own comment: sized for a different, narrower original use case,
never revisited for DLRM). Raised to 64 (essentially free — the capped array holds only
allocatable pointers + a few scalars per slot, no preallocated GPU memory), rebuilt
`libhashjoin.so`, verified no regression against `sparsebridge`'s own 218k-key RAG-path
test first. **26 and 40 concurrent `IdIndex` tables now both work**, per-table cost
still flat past the old ceiling (0.073-0.074ms/table) — confirms this lab's own
prediction made before the fix existed. `hybrid_router.py` gained `mode="variable"`
(Phase 3 Part 1's scatter_add/EmbeddingBag-offsets pooling, kept as a separate code path
from fixed mode rather than unified, since the two were never benchmarked head-to-head
at equal bag length) and `MultiTableRouter` (N independent `HybridRouter`s, sequential
dispatch — deliberately not concurrent, per Part 2's own finding that streams don't
help — plus a `MAX_HANDLES` budget guard that raises with an actionable message instead
of failing opaquely inside `hje_create`). All four new paths verified: variable-mode
correctness on out-of-calibration batches at two different decision-flipping
`expected_n_batches`, heterogeneous real per-table decisions in a mixed-table test,
guard fires with zero handle leak. `bench_router_extended.py`; original fixed-mode
`bench_router.py` scenarios re-confirmed unregressed. See `sparsebridge/recsys_lab/
RESULTS_PHASE2.md` and `RESULTS_PHASE3.md` Part 2, both follow-up sections.
**Update-path: capacity-growth cascade BUILT (2026-08-21, later session).** Scoping
found incremental insert within a table's *original* reserved capacity already worked
with zero new code (verified directly), and Fraser asked whether the codebase's
existing cascading-overflow mechanism (`tinyfull.cuf`'s Extension 1 — 0 failures in 6
levels at 8M keys) could solve growth PAST that capacity. **Built it**, isolated in a
new `tiny_pointers/hash_join_cascade/` directory — `tinymap.cuf`/`hash_join_api.cuf`
copied byte-identical and never edited; only a new `hash_join_cascade_api.cuf` (one new
kernel, five new entry points for lazily-allocated linear-probe "extra levels") plus a
Python orchestration layer (`GrowableHashJoinTable`). **Two real bugs caught before
trusting any measurement**: a classic CuPy+ctypes bug (an inline `.data.ptr` read with
no Python reference held let a temp array get freed before the kernel read it, silently
dropping a value update — fixed by binding to a local variable first), and a genuine
~430x performance cliff — 24.9 SECONDS for a 1M-key growth batch vs. 55ms to just
rebuild — in the *original, unmodified* `ttd_insert_r`: its overflow-region linear
probe has no early exit once the region is fully saturated, a latent behavior no prior
usage had ever triggered (normal single-shot builds never deliberately saturate a
region; this cascade design is the first thing that does, by construction). Fixed via
empirically-tuned chunked insertion (5,000 keys/chunk — tested 1,000/5,000/50,000, 5,000
won) on the CALLING side, without touching the original engine file.
**The material-difference question, answered honestly and unevenly**: when growth
fits within a table's reserved headroom (no new level needed), cascade-growth is a
real, large win — **3.3-12.2x faster than a full rebuild**, advantage growing with
table size, exactly matching `IdIndex`'s known weakness shape. When growth genuinely
exceeds that headroom, the picture flips: cascade is **5-13x slower** than rebuilding,
even after the chunking fix — a real, stated limitation, not solved yet.
**Superseded same day**: Fraser's call — never cascade past headroom, just rebuild —
proved both simpler and faster. `hash_join_rebuild.py`'s `RebuildingHashJoinTable`
(none of the cascade-level machinery, built on unmodified `HashJoinTable`) **beats the
cascade approach outright in every case tested (1.2-1.7x faster, both regimes)**.
Building it caught two MORE real bugs: a Python-dict bookkeeping loop that made the
first version *slower* than both alternatives at scale (fixed via an append-only log,
deduplicated only at rebuild time), and an unconditional CPU-side `np.unique` dedup
pass running even with zero duplicates, adding ~600ms-1s to a 57ms rebuild (fixed by
skipping it when provably unneeded). `RebuildingHashJoinTable` is now the recommended
strategy; the cascade version is kept as a measured comparison point, not deleted. See
`tiny_pointers/hash_join_cascade/RESULTS.md`'s follow-up section for the full
three-way comparison, and `hash_join_engine/CASCADE_GROWTH_DESIGN.md` (status header
updated).
**Wired into `IdIndex`/`hybrid_router.py` (2026-08-21, same day).** `IdIndex` gained
opt-in `growable=False`/`insert_rows()` (uses `RebuildingHashJoinTable` only when
asked — zero cost to existing callers like `sparsebridge`'s RAG path). `HybridRouter`
gained `support_growth=False`/`add_rows()` — each of the four candidates gets its OWN
real growth strategy (dict extension for `cpu_naive`; full re-sort, already known cheap
per Phase 1, for `cpu_vectorized`/`gpu_searchsorted`; `insert_rows()` for
`gpu_idindex`), no re-calibration after growth. Verified: all four candidates' growth
path tested by forcing each via the same scenarios Phase 2's own router tests already
validated, both old and brand-new rows correct afterward, `support_growth=False` guard
fires correctly, no regression anywhere (`bench_router_growth.py`; full details
`sparsebridge/recsys_lab/RESULTS_PHASE2.md`'s follow-up section).
**Root-caused (2026-08-21, same day).** Per-stage timing at the worst case (5M
base/1M growth) found the chunked "discover it doesn't fit" phase was ~90% of the
overhead (462.9ms of ~514ms) — 179 round trips of a fixed 5,000-key chunk, each paying
kernel-launch+sync overhead regardless of GPU work done — not the rebuild (46.1ms,
near a plain rebuild's own ~55ms) or the existence probe (4.6ms). Tried adaptive
chunk-size doubling; a 50,000 cap made the worst case **~7x WORSE** (5,133ms) — growing
the chunk size also grows whichever chunk finally hits saturation, re-triggering the
original `ttd_insert_r` pathology at a bigger scale, a real reported negative result.
Landed on a 10,000 cap: **1.6x better at 1M/200K growth** (90.82ms→55.86ms), ~10-25%
worse at 5M/1M growth (733ms→~822-960ms) — an honest, imperfect trade-off, not a clean
win. Side effect of simplifying the failure logic (any failure now triggers rebuild,
not a 10% threshold): closed a latent silent-data-loss risk found while investigating
(never actually observed at this lab's scale, but real depending on tuning). See
`tiny_pointers/hash_join_cascade/RESULTS.md`'s follow-up section.
**Closed (2026-08-22).** The deferred "smarter chunking" question was solved:
`RebuildingHashJoinTable` now estimates headroom directly (its own tracked exact
occupancy vs. `HashJoinTable`'s sized capacity) instead of discovering it by
trial-and-error chunking — one round trip if a batch should fit, zero wasted
attempts if it clearly won't. **1M/200K growth: 90.82ms→9.44ms (~9.6x better)**;
**5M/1M growth: 733ms→~53-55ms**, now matching plain rebuild's own ~57ms instead of
being 1.0-13x worse across every earlier version tried. See
`tiny_pointers/hash_join_cascade/RESULTS.md`'s newest follow-up section.

**Native Rust port (`~/sparsebridge/recsys_router-rs/`, same day)** — all four
candidates + the router decision logic + capacity-aware growth, callable from Python
as `import recsys_router` (PyO3, `maturin develop`), zero CuPy/PyTorch dependency.
`-core`/`-cli`/`-py` Cargo workspace matching `decision_engine`/`rbfx`'s own
conventions; builds on the pre-existing `id_index-rs` FFI wrapper (extended with
`RebuildingHashJoinTable`) plus a new CUDA Fortran module,
`tiny_pointers/gpu_pool_engine/`, hand-writing sorted binary search + bag pooling
(no existing C-ABI for CuPy's `searchsorted`/`scatter_add`) — a real design
improvement over the Python original, not just a port: the offsets convention means
pooling needs no atomics at all (one thread per (bag,dim) summing its own contiguous
range), unlike CuPy's `scatter_add` workaround. Verified in stages against the
Python reference at each layer (new kernels vs. CuPy directly, growth vs.
`hash_join_rebuild.py`'s own tests, black-box oracle correctness, then a real
cross-language check against the live `HybridRouter` on identical seeded data) —
output matched to float32 precision in every case, calibration **4-5x faster**
(5M rows: 7,222ms→1,680ms). Full writeup: `recsys_lab/LAB_PLAN.md` Phase 4.
**Next planned**: none currently scoped. Deferred: online re-calibration after
growth, eviction (still no primitive anywhere in `tinymap.cuf`), full dense-tower
DLRM forward pass, 100M+-row scale, closing `gpu_idindex`'s known host-round-trip
gap in the Rust port (measured, not a real cost problem at any scale tested, so
low priority) and giving `cpu_vectorized` its own dedicated oracle test.

**Small end-use application (same day)**: `recsys_lab/movielens_demo.ipynb` — real
movie embeddings (SVD on MovieLens `ml-latest-small`'s rating matrix, 610 users/9,742
movies/100,836 ratings) served through `recsys_router.Router(mode="variable")` to
recommend from a user's variable-length "liked movies" bag. Evaluated honestly:
leave-one-out Recall@10 **0.120 vs. a 0.090 popularity baseline** across 200 users —
a real, modest lift, not cherry-picked.

**Research + Phase 0 for the deferred GNN idea, new lab `~/sparsebridge/graph_lab/`
(2026-08-22)**: researched (web search, not assumed) a real public dataset + real
evidence that avoiding host-side ID remapping matters — found the classic Elliptic
Bitcoin transaction dataset (203,769 real transactions, 234,355 edges, genuinely sparse
anonymized `txId`s, a real 49-timestep growth history, Hugging-Face-mirrored, no auth)
and 2026 GNN-systems literature confirming host-side orchestration/data-management
overhead is a real, current bottleneck class (up to 5.28x speedup found by removing it
in one paper; 53-58% of training time in another) — though no single paper isolates ID
remapping specifically, stated honestly. Phase 0 (`bench_growth_replay.py`, zero new
engine code) replayed the graph's real 49-step growth history through three candidates
(host dict, host sorted-array, `IdIndex(growable=True)`). **Real, mixed result**: the
lookup/resolve step is a genuine ~6x win for the GPU-growable index (confirmed on real
data), but the insert/growth step is currently a loss against a plain Python dict at
this dataset's real per-step scale (real, unamortized per-call GPU overhead) — net 0.91x
vs. plain dict, 2.70x vs. the more realistic re-sort-every-step baseline (which visibly
degrades ~9x over the replay, a real confirmed cost of that common strategy). Not a
refutation: this Phase 0 tested a 1:1 insert:resolve ratio, while a real GNN training
loop resolves IDs far more often than it inserts — the measured lookup win likely
dominates a realistic workload, untested here. Full numbers: `graph_lab/RESULTS.md`.
**Follow-up, same day: the realistic resolve:insert ratio, measured not guessed.**
Rather than pick one "realistic" epoch count, measured the actual crossover directly
(`bench_resolve_insert_ratio.py`): build each candidate via the same real 49-step
insert replay, then time one full resolve pass over all 234,355 edges at once (5
repeats, confirmed stable ~5-20% spread, not assumed) as "one training epoch over the
finished graph." **Decisive result**: `gpu_growable` already beats the re-sort baseline
at zero epochs (69.0ms insert vs. 140.5ms); against a plain Python dict it starts 48.9ms
behind on insert but wins 60.0ms back per epoch — **crossover at E ≈ 0.82 epochs**. Any
real deployment reusing a graph more than once after building/updating it (every GNN
training loop; every serving system with more than one query per update) already comes
out ahead overall. This directly answers the "material, not just nice to have" question
from a real measured number. Full numbers: `graph_lab/RESULTS.md`. Next: scale up (full
Elliptic++ actor graph or the 252M-node Bitcoin graph, likely on the 8GB-GPU desktop).
Also recorded: Fraser's desktop has an RTX 4060 (8GB VRAM) + 4TB SSD, vs. this laptop's
4GB-VRAM A1000.

**Demo notebook (2026-08-22)**: `recsys_lab/demo_recsys_lab.ipynb` — runnable
walkthrough (every number measured live, not copied from `RESULTS.md`): build a
synthetic sparse-ID table, watch the router's decision flip between `cpu_vectorized`
and `gpu_idindex` purely from expected table lifetime, run/verify `lookup_and_pool`
against a plain-dict ground truth, grow the table live via `add_rows()`, reproduce
the capacity-aware growth-cost win directly, and a 3-table `MultiTableRouter` demo.

### `dispatch_cache_lab` — JIT/megamorphic dispatch cache, small test (2026-08-10)
`.../tensor_core_engine_v5/tiny_pointers/dispatch_cache_lab/` (`dispatch_bench.c`,
`RESULTS.md`). Second brainstorm idea from the same session, tested in C (the user asked
about Rust mid-session; the C numbers already answered the question so the port wasn't
needed). **Result: the GPU win does not transfer.** Faithful CPU port of `tinymap.cuf`'s
reliable 2-level bucket mechanism (B=16, same murmur3 hash) loses to a plain flat
linear-probe hash table (~8% slower) though it still beats the real pointer-chasing inline-
cache representation (~12% faster). Root cause, mechanical not incidental: the mechanism's
real edge is warp-parallel bucket scanning (32 SIMT lanes checking a bucket's 16 slots at
once) — a single CPU thread scans the same bucket serially, at ~7.2 average comparisons
(B=16, load 0.90) vs. plain linear probing's ~2.6. **Real bug caught and fixed**: single-
level bucket alone spilled the same ~6% random-key rate already on record; needed the
reliable 2-level overflow port to reach 100% correctness before trusting timings. **Real
finding from a bucket-size sweep**: shrinking B to 4 hung — fixed `ovf_frac=0.15` (tuned
for B=16) can't absorb B=4's higher spill rate without the overflow region itself
saturating; `ovf_frac` needs to scale with B, not stay fixed.
- **Paper connection, verified against the actual arXiv:2111.12800v1 PDF** (not memory —
  found via `pdfstash`, at `~/machine_learning/Machine Learning research/2111.12800v1.pdf`):
  the paper formally distinguishes **fixed-size tiny pointers** (§3 — one global bucket
  width for every key, Θ(log log log n + log k) bits, what `tinymap.cuf` actually
  implements) from **variable-size tiny pointers** (§4 — a recursive cascade of log₂(s)
  halving-size levels, each with its own overflow array; most keys succeed cheaply at
  level 0, rare keys pay more bits only when they cascade deeper; Θ(log k) *expected*
  bits, asymptotically better). `tinymap.cuf`'s "reliable" variant is a 2-level
  *approximation* of this (one flat overflow, not the paper's recursive structure) — the
  real variable-size construction has never been built here. It's the natural fix for
  the CPU dispatch-cache loss above (few comparisons for the common case, graceful
  fallback for rare ones) but is real implementation work, not a small test — flagged as
  the next candidate if this thread is picked up again, not attempted this session.

### `variable_tp_lab` — the paper's real variable-size construction, built in Rust (2026-08-10)
`.../tensor_core_engine_v5/tiny_pointers/variable_tp_lab/` (Rust crate: `src/lib.rs`
`VariableTable`, `src/bin/bench.rs`). Third step in the same brainstorm session — a real
port of arXiv:2111.12800v1 §4's variable-size cascade (not the 2-level approximation
`tinymap.cuf` has), reviewed against the `rust-error-harness` skill (typed errors, no
panics, 5/5 failure-mode + correctness-oracle tests passing). **Correctness confirmed**:
100% match against a HashMap oracle at 8M keys, 86.1% of keys resolve at level 0, avg tiny
pointer 4.46 bits — the paper's claimed behavior, reproduced for real.
**Performance: still loses** (185-220 ns/query vs. the C fixed-size port's 92.5ns and plain
linear-probe's 85.3ns) — but the root cause is now precisely identified, not mysterious:
this implementation runs one *global* cascade over all n keys rather than the paper's
n/log(n) independent per-container cascades, and that alone verifiably inflates memory 2-4x
(exact match between calculated and measured MB) — every level pre-allocates against the
*entire* key count even though 99.9% of keys resolve by level 1. The paper's container
partitioning isn't just for its probabilistic proof bounds, as flagged going in — it's what
keeps this multi-level redundancy cheap in practice; skipping it is the actual, verified
cause of the slowdown. Tried `ovf_frac` (1.0→0.3) and `b` (4→8) as quick fixes — neither
closes the gap, confirming this isn't a tuning problem. **Real next step, not attempted**:
implement the actual per-container partitioning (many small independent cascades instead of
one large one) — materially bigger build than this session's, flagged honestly as the only
remaining path if idea #2 (JIT/dispatch caching) is worth pursuing further.

### New project — `ragstash`, RAG glue (2026-07-29)
`~/ragstash/` (own top-level entry, matching `~/stash/`/`~/pdfstash/`/`~/mdstash/`/`~/concept/`'s
placement convention). Born from checking overlap between `gpustash`'s planned "embed documents"
step and existing work — found almost total overlap, not built fresh:
- **`seedverify` already has a real, populated, content-addressed embedding store**:
  `~/.cache/seedverify/embstore_embeddinggemma.bin`, confirmed on disk 2026-07-29 — 668.8 MB,
  768-dim, **216,865 real vectors** across SEC filings, PDFs (reusing `pdfstash`'s own extracted
  `pdf_search/PDF.DAT`, 1,685 PDFs/244,692 pages — no PDF re-parsing needed, ever), markdown,
  transcripts, images, Kindle highlights. Binary format confirmed directly against
  `seedverify/src/embed.rs`: `[u64 hash][u32 dim][dim x f32]` records, trivially readable from
  Python with no Rust dependency (`ragstash/embstore_reader.py`).
- **Honest scale caveat**: at ~217k vectors this fits comfortably in any GPU's VRAM as a flat
  array — `seedverify`'s existing brute-force cosine search is already fine here; `gpustash`'s
  compression story has no urgent need yet at this scale, but it's real, growing, available data
  worth validating the pipeline against regardless.
- **"Step 4" (the bucket-assignment gap flagged after `gpustash` shipped) built**:
  `ragstash/partition.py` packages `vector_ann_lab`'s validated METIS approach
  (`build_knn_graph`/`metis_bucket_assign`/`measure_locality`/`relabel_neighbors`) into reusable
  functions. **A real bug caught building it**: `relabel_neighbors` initially remapped only
  neighbor *values* to the new rank order, leaving *rows* in original order — silently breaking
  `gpustash.compress()`'s `bucket_id = row_index // B` assumption for almost every node. Caught
  because the first real end-to-end run measured a compression ratio *below 1.0* despite 59.6%
  real locality, which didn't match hand-computed expectations — fixed by reordering rows too.
- **First real end-to-end result** (`pipeline_demo.py`, real 20,000-vector random subsample of
  the actual store — full-corpus exact k-NN is O(n²), out of scope for a first demo): **59.6%
  real locality (k=32, B=256), 1.475x real compression, 1.22x real net GPU decode speedup, 100%
  correctness** — better locality than `vector_ann_lab`'s own 20-Newsgroups test, plausibly from
  this corpus's stronger multi-source separation (SEC vs. PDF vs. images vs. Kindle), not yet
  confirmed why.
- **Full RAG build-out, done (2026-07-29)** — `RESULTS.md`. Filled every gap the first pass
  flagged:
  - **`sec_reader.py`**: reconstructs seedverify's exact chunking (`chunk_prose`, 700-char
    default) + exact FNV-1a-64 content hash, ported directly from `seedverify/src/data.rs`/
    `embed.rs`. **Verified byte-exact, not assumed**: 43,587/43,634 (99.9%) independently
    reconstructed SEC passage hashes matched real store entries.
  - **`rag_query.py`**: the real, live, end-to-end loop — embeds a query via ollama
    (`embeddinggemma`), retrieves via exact brute-force GPU cosine over all 216,865 real
    vectors (213ms, honest choice at this scale over the compressed-graph path), resolves
    hits to real SEC text, generates a grounded, correctly-cited answer via `gemma4:e2b`
    (3.7s). Ran live on a real query ("customer concentration risk") — real tickers (HOOD,
    QS, PG), real filing language, real cited answer, not synthetic.
  - **`full_scale_graph.py`**: ran the graph+METIS+gpustash chain over the **entire** current
    corpus (216,865 vectors, not a subsample) via an hnswlib-built approximate k-NN graph
    (exact kNN is O(n²), doesn't reach this scale). Real result: **47.4% locality, 1.275x
    compression (27.8MB->21.8MB), 1.02x decode speedup (essentially even, not negative), 100%
    correctness**. Honest reading: at today's ~28MB adjacency size there's too little data
    moved for a dramatic decode-time edge (matches `vector_ann_lab`'s own finding that the win
    needs hundreds of MB to show clearly) — but ~22% less memory at zero decode cost is a real,
    banked win today, and the mechanism is proven ready for when corpus growth (the
    still-mostly-unembedded 244,692 PDF pages, growing transcripts/images/Kindle) makes
    capacity the actual bottleneck.
  - **Still open**: real text recovery only built for SEC (5 more sources share the pattern,
    not yet done); the compressed-graph path isn't wired into live query serving (brute force
    is still the right choice at this scale); HNSW build is a one-time full-rebuild cost, not
    yet incremental. **Language stays Python/CuPy** reading the Rust-produced store directly —
    no FFI, no Rust rewrite, per this codebase's "don't build a language port ahead of a
    concrete need" pattern.
- **PDF source recovery + COBOLMM/rustmm menu integration + menu-level LLM selection, done
  (2026-07-29)** — `RESULTS.md` §4-6.
  - **`pdf_reader.py`**: reconstructed seedverify's exact PDF chunking (`source.rs::load_pdf`,
    reusing `pdfstash`'s own `PDF.DAT`, no re-parsing) and FNV-1a hash. **100.0%** (149,314/
    149,314) of reconstructed hashes matched the store — even cleaner than SEC's 99.9%. This
    also corrected an earlier guess: **PDFs are the dominant source** (149,314 of 216,865
    vectors, 69%), not "mostly unembedded" as first assumed. Combined with SEC: **89.0%**
    (192,901/216,865) of the entire real corpus now resolves to real text. Re-ran the live RAG
    query with both sources wired in ("How do I configure a Zowe CLI profile?") — real,
    page-cited, technically correct answer pulling from `CLIReference_Zowe.pdf`/`REXX1.pdf`.
  - **Wired into the COBOLMM/rustmm menu**, per explicit guidance to touch only `MENU.DAT` (no
    full module directory like `alert_scan/`): backed up `cfg/MENU.DAT` first, added two
    records, verified by re-parsing field widths (135-byte records, all well-formed) and by
    rendering the live menu (`rustmm`) before/after — no corruption. New items: **`RQ`** ("Ask
    a question," real retrieval + local LLM, under a new "RAG QUERY" section, dispatching
    `ragstash/menu_run.sh`, following the established `/dev/tty`-read pattern for menu-launched
    scripts) and **`SL`** ("Select LLM," under a new "LLM SETTINGS" section). Honest note: full
    interactive pty-dispatch validation (the standard this codebase used for `AL`/`CP`) was not
    run, only rendering + parsing — worth doing before daily reliance.
  - **Menu-level LLM selection, no `.env` editing** (the user's one requested add-on): found
    `llm_provider.py` already provides unified Claude/Ollama dispatch with a 2-tier config load
    — added a 3rd tier, `~/COBOLMM/session_llm.env`, checked between the machine config and
    live env vars, so a session choice applies immediately to every tool built on
    `call_llm()`/`call_llm_small()` while a Makefile-level env override still wins unchanged.
    `select_llm.sh` (menu item `SL`) picks Claude or any live-listed installed Ollama model, or
    resets to the config default. **Verified working**: session file flips
    `llm_provider.provider_label()`'s output immediately; removing it reverts cleanly.
  - **Still open**: 3 more sources (markdown/transcripts, images, Kindle) need their own reader
    the same way; only one combined `RQ` item exists so far (the plan is per-source items as
    each reader is built, then a universal option at the end — this is item #2 of that plan,
    not the finale); full pty-based menu validation not yet done.
- **Real bugs found by using it, plus all six sources completed (2026-07-29)** — `RESULTS.md`
  §8-10.
  - **Model selection had zero effect**: `rag_query.py` called `ollama_client.generate` with a
    hardcoded model, never touching `llm_provider.py`/the session override — fixed by routing
    through `llm_provider.call_llm()`. Verified live switching to `gemma4:31b-cloud`.
  - **A real, colloquial query ("how has tesla performed recently?") returned 0 resolved hits**
    despite TSLA genuinely having 71 real filing records — casual phrasing scored higher
    against not-yet-recovered sources, crowding SEC/PDF out of the global top-30 entirely.
    Fixed by restricting retrieval to the resolvable subset.
  - **A deeper finding, not fully solved**: even after adding a ticker/company-name detector
    that boosts a named company's own SEC filings (`sec_reader.build_ticker_names()`,
    verified: real TSLA earnings passages with actual reported revenue figures exist in the
    store but the best one ranked 1,170th globally, similarity 0.386, vs ~0.49 for narrative
    news prose), the boosted top TSLA passages were STILL risk-factor boilerplate, not the
    figures — a genuine embeddinggemma weakness (tabular/numeric disclosure text underranked
    vs. narrative prose) that a company-level boost alone doesn't fix. Left as-is per explicit
    user decision, not silently declared solved.
  - **Two citation-linkifier gaps**, both found by running real generated answers, not code
    review: semicolon-separated multi-citations, then comma-separated ones (unsafe to split
    on comma — filenames contain commas) — fixed by substring-matching known labels directly
    instead of splitting.
  - **Resident loop + house styling**: reworked into load-once/ask-many/`q`-to-quit (fixing
    the slow-reload complaint as a side effect, not a separate pass), and restyled to match
    `universal_search`'s own convention exactly (orange section labels, dim secondary text,
    compact `SRC(N)` counts line) per direct feedback to trim the explanatory banner.
  - **All six sources now resolve real text**: built `md_reader.py` (transcripts + web
    clippings via the same `load_md_dir` mechanism, plus images and Kindle highlights),
    verified 96.7-100% match per source against the real store. **Combined with SEC+PDF:
    216,712/216,865 (99.9%) of the ENTIRE real store now resolves to real text** (only 153
    hashes left unresolved). Useful simplification confirmed directly from Rust source:
    `chunk_hash` depends only on `(model, text)`, never on label/link/metadata, so exact
    link-string reproduction was never necessary for hash matching. Verified live: a
    cross-source query ("inflation risk") pulled real Kindle highlights, a real transcript,
    and a real SEC filing into one coherent answer — the first genuinely multi-source result.
    Menu header updated to "6 sources" (was "SEC + PDF"), same backup+verify process.
  - **Model switch had no effect mid-session — a real bug in `llm_provider.py` itself, not
    ragstash-specific**: it cached its config chain in a module-level global loaded once at
    import time, so any long-running process (ragstash's own resident query loop, built
    earlier to avoid reloading the corpus per question — but this would hit *any* long-lived
    COBOLMM tool built on `call_llm()`) never saw a later `session_llm.env` change at all.
    Fixed by re-reading the config chain on every call instead of caching it once — a few KB
    of file I/O, negligible next to an actual LLM round-trip. Verified directly: in one
    long-lived process with no re-import, switching the session file now changes
    `provider_label()`'s answer immediately, every time.
  - **A second, separate bug in `select_llm.sh` itself**: even after a full `rustmm`
    restart, reselecting `gemma4:e2b` still silently had no effect. Reproduced directly:
    the script runs under `set -u`, and `read -rp ... CHOICE </dev/tty || true` leaves
    `CHOICE` completely unassigned if the read ever fails — referencing it under `set -u`
    is a hard crash regardless of `|| true`, killing the script before it reaches the
    `case` statement and silently leaving the prior model selection in place. Fixed by
    pre-initializing `CHOICE`/`MNUM` to empty strings before their reads. Verified twice:
    under a real pseudo-terminal (`pty.fork()`) driving the exact interactive sequence
    from a cold cloud-selected state — correctly wrote `gemma4:e2b`, no crash — then
    end-to-end through `rag_query.py`, which now correctly answers using it.
  - **Model now shown live in the header**: `LLM: <provider label>` prints fresh before
    every question in the resident loop (not just once at startup or in the answer
    footer), so a mid-session model switch is visible on the very next question.
  - **The REAL root cause, found after both prior fixes still didn't resolve it through
    the actual menu**: `~/COBOLMM/mainmenu` runs `set -a` before sourcing
    `config.<host>.env`, exporting every value in it (including `LLM_MODEL=gemma4:31b-
    cloud`) as a real environment variable for the whole process tree the menu launches.
    `llm_provider.py`'s own documented rule ("env vars always override") meant the
    session override could only ever beat the *config files*, never the *environment* —
    and the menu always injects the config file's own values into the environment first.
    Every standalone test passed because none had those env vars exported; the bug was
    invisible outside the actual menu launch path, exactly where it was being tested.
    Fixed by making `session_llm.env` the single highest-priority source, checked before
    `os.environ` itself. Verified by directly simulating the exact failure (exporting
    `LLM_MODEL=gemma4:31b-cloud` in the shell, precisely what `set -a` produces) and
    confirming the session override now wins regardless; also verified no regression —
    with no session override active, an ad-hoc env var still wins exactly as before.

### `~/machine_learning/sparsebridge/` — moved from `~/sparsebridge/` to the NAS-backed
`machine_learning/` tree to build/run on the desktop's 8GB GPU (2026-08-22)
Pre-move session (`sparsebridge/SESSION_2026-08-22.md`) had already flagged what would
break: 26 files with hardcoded `sys.path.insert(0, "/var/home/fraser/sparsebridge...")`,
plus the `maturin`-installed `recsys_router` Python package. Post-move, fixed all of it:
24 `.py` files + 2 notebooks (`demo_recsys_lab.ipynb`, `movielens_demo.ipynb`) switched to
`os.path.dirname(os.path.abspath(__file__))`-derived paths (notebooks use `os.getcwd()`
instead, since no `__file__`); one file outside `sparsebridge/` itself
(`tiny_pointers/gpu_pool_engine/verify_gpu_pool.py`) had its own hardcoded old path fixed
directly. **New, unpredicted problem found along the way**: three Rust `build.rs` files
(`id_index-rs`, `recsys_router-rs/{recsys_router-py,recsys_router-cli}`) link against the
NVIDIA HPC SDK Fortran runtime via a *hardcoded* path,
`/var/home/fraser/nvidia/Linux_x86_64/26.5/compilers/lib` — wrong on this desktop, where
the SDK actually lives one directory deeper at `.../nvidia/hpc_sdk/Linux_x86_64/26.5/...`
(confirmed by listing the dir and finding `libcudafor.so`/`libcudanvhpc.so` there, not at
the old path). Not caused by the sparsebridge move itself (a pre-existing machine-specific
absolute path, thinkpad vs. desktop), but it blocked the exact same task, so fixed in the
same pass. `maturin`/`patchelf` weren't installed anywhere on this desktop either —
installed into the `py314` conda env (user's choice over `rapids-25.04`/`py313`) and
`recsys_router` now builds and installs cleanly there. **Fully re-verified working**:
`id_index-rs`'s `verify`/`verify_growth`, `recsys_router-rs`'s `verify_router`,
`tiny_pointers/gpu_pool_engine/verify_gpu_pool.py` (real CUDA kernels vs. CuPy reference
on the desktop's RTX 4060), `verify_rust_router.py` (Python vs. Rust router, max abs error
1.4e-6, matches pre-move numbers), and `recsys_lab/bench_lookup.py` — all PASS with real
numbers. **Follow-up, same session**: `hnswlib`/`pymetis` installed into `py314`,
closing the last gap — `test_id_index.py`'s full METIS pipeline and `sb_query.py`/
`related_items_lib.py` now verified working too. Full dependency list (pip packages +
system-level CUDA/HPC-SDK/Rust requirements, incl. the `hpc_sdk/` path gotcha) written up
in `sparsebridge/README.md`'s new "Dependencies / environment setup" section for the next
machine move.

**`graph_lab/` Phase 1 (2026-08-22, later same day) — the scale-up the desktop move was
for.** Real Elliptic++ Actor Interaction graph (822,942 wallet addresses, 2,868,964
edges, ~4x/~12x Phase 0's transaction-graph scale), fetched straight from
`git-disl/EllipticPlusPlus`'s GitHub LFS media endpoint via `curl` (no `git-lfs`
install needed). New wrinkle: base58 string node IDs, hashed to uint64 via plain
FNV-1a-64 (checked zero-collision across all 822,942 real addresses, not assumed).
Reused Phase 0's `HostDict`/`HostSorted`/`GpuGrowable` candidates and harness verbatim
(`bench_growth_replay_actors.py`, only the data loader is new). **Real result: Phase
0's near-wash at small scale (0.91x vs `host_dict` on a strict 1:1 insert:resolve
replay) becomes an outright 6.68x win at this scale** (6.41x vs `host_sorted`), gap
visibly widening over the 49-step replay rather than flat. See `sparsebridge/graph_lab/
RESULTS.md`'s Phase 1 section and `LAB_PLAN.md`. Left open: the 252M-node Bitcoin graph
(Nature *Scientific Data* 2025) remains untested at that far larger scale.

**`graph_lab/` Phase 2 (2026-08-22, later same day) — the 252M-node Bitcoin graph, at
the scale the GPU actually supports.** Restored the real PostgreSQL dump (Schnoering &
Vazirgiannis, *Scientific Data* 2025, figshare 10.6084/m9.figshare.26305093.v3, 17.4GB
MD5-verified download) via an ephemeral `podman` postgres container — real row counts
matched the paper exactly (252,219,007 nodes, 785,954,737 edges). **Real correction
found along the way, the actual headline result**: a clean one-shot-build GPU-memory
estimate (~55 bytes/key, suggesting ~110-120M nodes would fit) turned out ~4x optimistic
against the dataset's real bursty per-step growth pattern — confirmed by a real CUDA OOM
crash at a 110M-node attempt, then root-caused with a live calibration run that hit the
practical ceiling at ~32M keys / ~215-220 bytes/key. Landed on a measured-safe
**25,000,000-node** real temporal prefix (96,408,977 real edges). **Result: 7.69x vs
`host_dict`, 4.73x vs `host_sorted`**, consistent with Phase 1, gap still widening at
~30x Phase 1's scale. Full writeup: `sparsebridge/graph_lab/RESULTS.md`'s Phase 2
section and `LAB_PLAN.md`. Left open: the remaining ~90% of the 252M-node graph exceeds
this desktop's real 8GB-card ceiling under the engine's current growth strategy — more
GPU memory or a `RebuildingHashJoinTable` capacity-management change would be needed to
go further, neither attempted here.

**`graph_lab/` Phase 2 engine fix (2026-08-22, later same day) — user asked to fix
`RebuildingHashJoinTable` rather than accept the reduced ceiling.** Found and fixed
THREE real, distinct bugs, each confirmed by an actual crash or direct measurement, not
guessed: (1) `hash_join_rebuild.py`'s rebuild bookkeeping log lived permanently on the
GPU, duplicating the table's own contents (`cp.concatenate`-grown) — the real mechanism
behind the earlier ~215-220 bytes/key figure — fixed by moving it to host memory; (2)
the benchmark's two sequential full-scale builds (correctness check, then timed replay)
hit real CuPy memory-pool fragmentation, not a genuine leak (confirmed: `nvidia-smi`
showed the device fully clean right after the crashed process exited) — fixed with
`cp.get_default_memory_pool().free_all_blocks()` at candidate-close boundaries; (3)
`id_index.py`'s `IdIndex.close()` never released `_row_to_id` (a second GPU-resident
duplicate array, same anti-pattern as bug 1, grown via `cp.concatenate` on every
`insert_rows()` call) — fixed by clearing it in `close()`. **Result: the real 110M-node
prefix (the full extent already exported from Postgres) now runs successfully —
5.38x vs `host_dict`, 4.40x vs `host_sorted`**, correctness PASS, gap still widening.
Honest tradeoff: bug 1's fix costs some insert-side speed from host round trips —
Phase 1's own number dropped from 6.68x/6.41x to 4.25x/3.97x, a real, accepted cost for
real memory correctness (was silently overcounting device memory before). Full writeup:
`sparsebridge/graph_lab/RESULTS.md`'s Phase 2 follow-up section and `LAB_PLAN.md`. Left
open: the remaining ~140M of the real 252M-node graph beyond what's already exported —
now blocked by Postgres export size, not a known engine/hardware limit.

**`graph_lab/` Phase 3 (2026-08-22, later same day) — graph storage + message-passing,
real Elliptic scale.** User: "let's build out 1) graph storage, 2) message-passing. We
can scale later as 110M gives us a good idea right now." Built at Elliptic's real
203,769-node scale (only dataset in this lab with real per-node features AND labels),
not Phase 1/2's larger graphs, per direct instruction. **Graph storage**
(`graph_storage.py`): real txIds → dense rows via the same `IdIndex` every earlier
phase used → a symmetric, binary `cupyx.scipy.sparse` GPU CSR adjacency (reused, not
hand-rolled) — verified against a plain-Python ground-truth adjacency, degree and
total-edge-count both exact matches. **Message-passing** (`message_passing.py`):
one-hop mean-neighbor aggregation as a single SpMM (`adj @ features`) via cupyx,
degree-normalized — verified against a manual per-node gather-and-average, exact match.
**Real application** (`elliptic_gnn_eval.py`): does this actually help illicit-
transaction classification, using the exact published protocol for this benchmark
(Pareja et al. temporal split, time_step≤34 train/>34 test, illicit F1 — confirmed via
web research)? `LogisticRegression`, baseline (165 raw features) vs. `+neighbors` (raw
++ 1-hop aggregated, 330-dim). **Real result: illicit F1 0.3053 → 0.3431 (+12%
relative)**, precision up substantially at a modest recall cost — real, positive signal
that graph structure helps. Honest limit: minimal viable pipeline (unlearned
aggregation + linear classifier), not a SOTA GNN reproduction (published EvolveGCN/GCN
results on this benchmark report ~0.55-0.7 F1) — proves the signal and plumbing are
real; a trained multi-layer GNN and scaling to larger graphs are separate future work.
Full writeup: `sparsebridge/graph_lab/RESULTS.md`'s Phase 3 section and `LAB_PLAN.md`.

**`graph_lab/` Phase 3 follow-up (2026-08-22, later same day) — Actor graph, a false
negative caught and fixed, then a bigger win than Elliptic's own.** User: "try it on
the Actor graph next." `actor_gnn_eval.py` reused `graph_storage`/`message_passing`
completely unchanged on the real Actor Interaction graph (822,942 wallet addresses,
2,868,964 edges). **Real mistake caught before being reported**: first run showed
1-hop aggregation making illicit F1 WORSE (0.2866 → 0.2306), plus a real `sklearn`
convergence warning — investigated rather than accepted, root cause was the Actor
dataset's unscaled raw features (unlike Elliptic's own pre-normalized ones). Fixed with
a shared `gnn_eval_common.py` adding proper `StandardScaler`, applied to BOTH datasets
for comparability (Elliptic's own number moved slightly too, updated above). **After
the fix, the result flipped entirely: illicit F1 0.1405 → 0.2699 (+92% relative)** —
more than 3x Elliptic's own absolute gain. The discipline that caught it: always
investigate a surprising result before writing it down as a finding, applied here to a
modeling result, not just a performance number. Full writeup: `sparsebridge/graph_lab/
RESULTS.md`'s Phase 3 follow-up section and `LAB_PLAN.md`.

**`graph_lab/` Phase 4 (2026-08-22, later same day) — a real trained multi-layer GNN,
on Elliptic.** User: "trained multi-layer GNN and scaling to the bigger graphs sound
like good next steps." `gnn_train.py`: real 2-layer GCN (Kipf & Welling's symmetric-
normalized propagation rule), trained via PyTorch autodiff (already installed with CUDA
support, confirmed before writing any code), reusing `graph_storage.build_csr_adjacency()`
unchanged. Real train/val/test discipline — val carved from Phase 3's own train range
for model selection, test set never touched until the end. **Result: illicit F1
0.4320** on the real held-out test set, beating Phase 3's unlearned aggregation+LR
pipeline (0.3431). Verified the "published GCN results" comparison this time instead
of approximating it: Weber et al. 2019's own GCN scores F1=0.628 (their simple LR
baseline: 0.53); EvolveGCN reaches 0.77. Our own absolute numbers sit below all of
these, including their LR baseline — stated plainly as a real, untuned floor (no
hyperparameter search done), not a claimed reproduction; the DIRECTION of every result
in this lab matches the literature. Full writeup: `sparsebridge/graph_lab/RESULTS.md`'s
Phase 4 section and `LAB_PLAN.md`. In progress: re-fetching the 110M-node Bitcoin
graph's real node-feature/label columns from Postgres (Phase 2's own export only had
alias+first_block) to scale this same trained-GCN pipeline up, per direct instruction.

**`graph_lab/` Phase 4 follow-up (2026-08-22, later same day) — scaled to the real
110M-node Bitcoin graph, DONE.** Re-fetched real `node_features` columns (degree/
transaction/satoshi/cluster stats + a real entity-type `label` taxonomy, different from
Elliptic/Actor's illicit/licit `class`) — only 32,027 of 110M nodes labeled at all,
8 classes kept (>=100 examples each). **Real "neighbor explosion" bug caught and
fixed**: exact 2-hop BFS reached 71,000,000 of 110,000,000 nodes (64.5% of the WHOLE
graph) because some labeled seeds are themselves real hubs (one had 725,204 neighbors;
a hop-2 frontier node had 11,526,657) — also flatly impossible regardless (full-batch
110M-node training needs ~28GB for one hidden layer alone). Fixed with GraphSAGE-style
capped neighbor sampling (15/node/hop, 600K-node ceiling) → a real 600,000-node
training subgraph, built and trained cleanly on the same 2-layer GCN architecture.
**Result: macro F1=0.4177**, a real, interpretable per-class pattern tracking class
support (`BET` F1=0.9515 at n=1345 down to `FAUCET` F1=0.0308 at n=20) — a real data
limit (too few labeled rare-class examples), not a pipeline failure. Full writeup:
`sparsebridge/graph_lab/RESULTS.md`'s Phase 4 follow-up section and `LAB_PLAN.md`.

**`graph_lab/` Phase 4 follow-up #2 (2026-08-25) — trained on the remaining ~142.2M
nodes too.** User: "try training on the remaining 140M nodes too." Real continuation of
Phase 2's own `OFFSET 110000000` methodology — 142,219,007 real nodes, 265,819,564 real
edges. This time kept the Postgres container/`pgdata` running afterward (third re-fetch
of the same 17.4GB file, clearly worth persisting). **Real finding**: only 2,071 total
labeled nodes here vs. 32,027 in the earlier 110M set — confirms real labeled entities
concentrate heavily early in the graph's history (94% of all labels in the first 110M
nodes). **Real silent-OOM bug caught and fixed**: the CSV loader died with no Python
traceback at all on this larger file — pandas' default dtype inference was memory-
hungry enough to trigger a kernel OOM kill; fixed with explicit dtypes (float32
features, `category` label), confirmed by peak RSS dropping ~4.4x at the same load
point. **Result: macro F1=0.5961** (`EXCHANGE`/`GAMBLING`/`INDIVIDUAL`/`RANSOMWARE`,
the only classes clearing the same >=100-example threshold) — higher than the 110M
run's 0.4177, but explicitly flagged as NOT apples-to-apples (that run averaged across
8 classes including several much harder ones that don't exist above-threshold here).
Full writeup: `sparsebridge/graph_lab/RESULTS.md`'s newest Phase 4 follow-up section
and `LAB_PLAN.md`.

**`graph_lab/` Phase 4 follow-up #3 (2026-08-25) — combined model, all 252.2M nodes
together.** User: "train a combined model on all 252M nodes together." Needed a
genuinely new export (the FULL 785,954,737-edge table, including edges crossing the
110M/142.2M boundary neither earlier run ever saw) — pulled from the already-running
Postgres container, no third 17.4GB re-download. **Two real memory bugs found and
fixed at this larger scale**: chunked CSV reading (the dtype-explicit fix alone wasn't
enough on the 18.1GB combined file), then fusing edge-loading with alias→row mapping
into one chunked pass (a second, different OOM — 46GB/46GB RAM, 91% swap — from ever
materializing the full raw edge-alias arrays). All 785,954,737 edges resolved (100%).
**Real finding**: the TRUE hub degree only becomes visible when combined (23.8M,
bigger than either half alone showed). **Result: macro F1=0.3898** across 9 classes
(test n=6,790, the largest/most complete evaluation run) — close to but below the
110M-only run's 0.4177, honestly explained (one more hard class, `MARKETPLACE`,
entering the average). Full writeup with a side-by-side comparison table of all three
runs: `sparsebridge/graph_lab/RESULTS.md`'s newest Phase 4 follow-up section and
`LAB_PLAN.md`.

**`graph_lab/` Phase 5 side-finding (2026-08-25) — the 110M/142.2M split was not a
perfect partition.** Found by accident while building Phase 5's temporal sampler (pandas
rejected a stitched `first_block` reconstruction with "cannot reindex on an axis with
duplicate labels"), then verified deliberately rather than worked around. The two prefix
CSVs came from two SEPARATE `ORDER BY first_block LIMIT/OFFSET` queries and `first_block`
has heavy ties, so the tied group straddling row 110,000,000 was ordered differently
between the two executions. **Measured**: each file internally all-unique, but **70 nodes
appear in BOTH exports and 70 in NEITHER** (union 252,218,937 of 252,219,007) — a
0.000028% tie-boundary defect. **Practical impact measured, not assumed: a Postgres query
over all 140 affected aliases returns `(unlabeled) | 140` — every one is unlabeled**, so
none was ever a train/val/test example and no reported metric changes. Does not affect
the combined 252.2M run (single unfiltered export, all 785,954,737 edges independently
confirmed resolving) or Phases 0-3. Fixed at the root: `first_block` now exported in one
query keyed on the `alias` primary key. Written up as a correction block in
`sparsebridge/graph_lab/RESULTS.md`.

**`graph_lab/` Phase 5, review point 1 (2026-08-25) — closing the gap to published
baselines.** External review suggested `pos_weight`, learning-rate tuning, or residual
connections would close the gap between this lab's untuned GCN (~0.43) and Weber et
al.'s 0.628. Rather than test the three suggestions in the order given, read the
diagnostic off the existing numbers first: our precision/recall (0.309/0.718) was the
near **mirror image** of Weber's (0.812/0.512) — an operating-point signature, not an
architecture one. `gnn_ablation.py` (staged ablation, every config selected on
VALIDATION only, test scored but never used to choose). **Real finding: the dominant
lever was none of the review's three suggestions** — `gnn_train.py` never standardized
its input features while `elliptic_gnn_eval.py` did (the Phase 3 Actor-graph lesson
never back-ported); **that fix alone moved test F1 0.3502 → 0.5163**, precision
0.2297 → 0.4707. LR tuning never beat the default on validation; **residual connections
actively hurt** (0.4727 vs 0.5298) — expected in a 2-layer net. The `pos_weight`
instinct was half right: strong class weighting is fine, but only once the decision
threshold is tuned instead of hardcoded at 0.5. **Final val-selected config, 5 seeds:
illicit F1 = 0.5120 ± 0.0233**, precision/recall now balanced (0.509/0.520). Fixed
`gnn_train.py` itself (`STANDARDIZE=True`, now reproduces 0.5151). Noted honestly: the
single best TEST config (0.5571) was deliberately NOT selected because its validation
score lost — selecting it would be test-set tuning. Gap narrowed, not closed (0.512 vs
0.628). Full writeup: `sparsebridge/graph_lab/RESULTS.md`'s Phase 5 section.

**`graph_lab/` Phase 5, review point 2 (2026-08-25) — neighbor-sampling strategies, a
real negative result.** Review suggested degree-weighted or temporal-weighted neighbor
selection might beat fixed-cap uniform sampling, "retaining stronger illicit signatures
around major hubs." `gnn_sampling_ablation.py` compared 5 strategies x 3 seeds on the
real combined 252.2M-node/785.9M-edge graph. Made cheap deliberately: a one-time `.npy`
cache (features/labels/first_block/CSR structure, 1.57B directed entries) so the 30-min
CSV parse and ~25-30GB CSR build happen once ever and each strategy costs ~45s; two
memory bugs caught by arithmetic before running (a 12.6GB-per-run `np.repeat` replaced
by a chunked gather; per-artifact resumable caching). **Result: no strategy beats
uniform by more than seed noise** — degree_low 0.3983+/-0.0046, uniform_k30
0.3963+/-0.0034, uniform 0.3918+/-0.0076, temporal 0.3875+/-0.0059, degree_high
0.3744+/-0.0027. **The one real effect contradicts the review's hypothesis**:
`degree_high` (preferring hubs, the suggested direction) is reliably WORSE, ranges
non-overlapping. Mechanism visible in the data — it was the only strategy that never
reached the 600K ceiling, plateauing at ~433,700 nodes, because preferring hubs
re-draws the same few high-degree nodes and collapses neighborhood diversity ~28% at
identical budget; hubs like exchanges connect to everything, so they are nearly
uninformative about any specific node's class. The `uniform_k30` control settles the
rest: doubling the budget also gained nothing, so **the sampling budget is not the
binding constraint** — consistent with review point 1's finding that the real limits
here are data-side (34,098 labeled nodes out of 252.2M), not sampling- or
architecture-side. Full writeup: `sparsebridge/graph_lab/RESULTS.md`'s Phase 5 sections.

**`graph_lab/` Phase 6 (2026-08-25) — the end use: AML alert triage.** User asked what
this GNN is actually for and how it would normally be applied. Real deployment pattern
in AML/blockchain forensics is a **ranked alert queue feeding human review**, never an
autonomous blocker — so the operational metric is precision@k at a fixed analyst budget,
not F1 (which assumes acting on every positive at a 0.5 threshold). `aml_triage_demo.py`:
trained on timesteps <=27, scoring genuinely unseen 35-49, base rate 6.50%. **Result:
precision@500 = 66.8%, a 10.3x lift** — reviewing 3% of volume surfaces 31% of all
illicit activity. **The finding that decides deployability**: per-timestep scoring shows
70.4% mean precision for timesteps 35-42 then **10.1% for 43-49**, with three timesteps
returning ZERO real cases from a full budget — the documented dark-market shutdown
destroys the model. Measured, not asserted; the most important operational fact in the
lab. **A ranking pathology diagnosed**: precision@k non-monotonic (32.0% at k=25 vs
66.8% at k=500) from real overconfidence on outliers, not tie-ordering. Noted precisely
that monotonic calibration (Platt/isotonic) provably cannot fix it — it preserves order,
and precision@k depends only on order. 5-seed ensembling tested instead
(`aml_triage_ensemble.py`): substantially improves precision at every realistic budget
(+24.2pp at k=100) and collapses single-model variance, but does NOT repair the
non-monotonicity — the top-of-queue problem is systematic, every seed wrong together.
Published as an analyst-facing artifact. Full writeup: `sparsebridge/graph_lab/RESULTS.md`'s
Phase 6 section.

**`graph_lab/` (2026-08-26) — Phase 6 demo as a live notebook + a CUDA-stack finding.**
`AML_TRIAGE_DEMO.ipynb`: the AML triage demo as a fully live notebook (trains 5 seeded
GCNs from scratch, alert queue, per-timestep collapse chart, ensemble test, alert
drill-down; every number computed in-notebook). One real bug caught on first execution
and fixed -- the drill-down cell mixed index spaces (`ensemble`/`y_true` are TEST-aligned
at 16,670; `tx_ids`/`labels`/`indptr` are FULL-graph-aligned at 203,769).
**CUDA stack**: Fraser installed `cuda-python`/`cuda-toolkit` 13.3.1 into `base` and then
`py314`, both printing pip dependency conflicts against torch's pinned
`cuda-bindings`/`nvidia-cuda-runtime`. **Verified end-to-end that nothing is broken** --
torch, cupy, `cupyx.scipy.sparse`, `libhashjoin.so` and the full `aml_triage_demo.py`
pipeline all run correctly in py314; PyTorch ships its own CUDA libs in `torch/lib/` and
loads those preferentially, so the pip metadata pin is stricter than the real runtime
requirement. **On the stated goal of bypassing CuPy**: checked rather than assumed --
`cuda-python` provides NO sparse support (no `cuda.cusparse`, no
`cuda.bindings.cusparse`, nothing sparse in `cuda.core`), so it cannot replace
`cupyx.scipy.sparse`'s CSR+SpMM that `graph_storage.py`/`message_passing.py` depend on.
**PyTorch can**, and is already a hard dependency: `torch.sparse_csr_tensor` +
`torch.sparse.mm`, `.cuda()`/`.cpu()`, and `.data_ptr()` for the ctypes FFI into
`libhashjoin.so`. Verified numerically on the real Elliptic graph -- the torch path
reproduces `mean_aggregate`'s CuPy output to max abs diff 1.5e-5. Offered as a scoped
follow-up (touches graph_storage/message_passing/id_index, real regression risk), not
done unasked. Written up in `sparsebridge/README.md`.

### sparsebridge — streaming index phase SCOPED AND SHELVED (2026-08-29)

Reviewed whether a fourth sparsebridge lab is warranted after `graph_lab` closed. Judgement:
the primitive is validated and has stopped generating new questions — all three labs ended
with the interesting result somewhere other than the ID bridge (`rag_bench` lost to hnswlib
on static recall/latency; `recsys_lab` returned a calibrated "it depends"; `graph_lab`'s arc
concluded the deliverable is a **monitoring design, not a model**).

The one axis where sparsebridge structurally *wins* rather than loses is a **mutating**
index, and there is a real, measured gap there: `build_index.py`'s cache is keyed on the
embstore's **mtime+size** (`build_index.py:189`) while the seedverify embstore is
**append-only**, so **one new embedded chunk invalidates the whole cache and forces a full
~50-60s rebuild of all 217,759 vectors**, paid interactively by `sb_query.py`/the `SP` menu item.

Per Fraser's direction — *"directing our efforts to fixing real gaps and problems rather than
necessarily covering every base"* — this was **scoped but deliberately not started**:
`sparsebridge/STREAMING_INDEX_PLAN.md`. S0 measures whether the rebuild is actually felt
(with an explicit kill condition: if misses are rare, stop and record "rebuild is fine");
S1 is the cheap fix (persist/extend the hnswlib index instead of rebuilding it, killing most
of the ~45-50s, with a recall guard against `variance_check.py`'s ±0.011 band); S2, the real
LSM-style compressed-base + hot-delta index, only if S1 is insufficient. **Eviction/deletion
is an explicit non-goal** — the store is content-addressed and append-only, nothing is ever
deleted, so there is no driver for the missing eviction primitive `recsys_lab` noted.
Unshelve triggers are written down in the plan.

### sparsebridge — S0 RUN, verdict STOP (2026-08-29)

Ran phase S0 of `STREAMING_INDEX_PLAN.md` the same day it was scoped, rather than shelving
it unmeasured. Full writeup: `sparsebridge/RESULTS_S0.md`.

**The premise didn't hold.** The seedverify embstore has been unchanged since **2026-07-22
(38 days)** — zero appends, therefore zero append-driven cache invalidations. There was no
recurring cost to remove.

**But S0 found a real live problem it wasn't looking for.** There were **zero**
`sparsebridge_graph_*.pkl` cache files on this desktop — the graph cache never survived the
thinkpad→desktop move — so every `sb_query.py`/`SP` launch here was paying a **full ~75s
rebuild**, with no appends involved at all. Verified (miss 75.53s → hit 2.17s) and **fixed
by populating it**. The actual cost in live use was a cold cache from a machine move, not
index churn.

**Rebuild breakdown, all 9 stages, `n=216,865`, RTX 4060** (`s0_rebuild_breakdown.py`):
`hnsw_build` 36.87s (49.4%), `knn_query` 25.49s (34.2%), `symmetrize_py` 4.99s, `metis`
3.79s, `load_embstore` 2.65s, everything else <0.5s — **74.6s total, so the README's
"~50-60s" was stale/understated**. S1-addressable (persist+extend the hnswlib index)
**62.4s = 83.6%**; residual **12.3s**. That residual is payable interactively, which
**closes the S2 gate permanently on measurement** — the LSM-style delta index is
unnecessary even if growth resumes. Notable: gpustash compress+decode are **0.48s of 74.6s
(0.6%)** — the GPU part isn't the cost; two hnswlib CPU calls are 83.6% of it.

**Corpus discrepancy flagged**: this machine's store holds **216,865** vectors, not the
**217,759** every published `rag_bench/RESULTS.md` number cites (exact, not estimated:
3084 B/record × 216,865 = the file size, remainder 0). The desktop's corpus is not the one
the benchmarks were measured on — reruns here will differ slightly; don't compare as equal.

**Instrumentation left in place**: `build_index.py` appends one `hit`/`miss`/`forced_build`/
`uncached` event per call to `~/.cache/sparsebridge/build_index_events.jsonl` (hit/miss
decided by whether the builder callable runs — exact, and `ragstash/cache_utils.py` was left
untouched). `python s0_cache_log_report.py` is now the one-command unshelve check.

### escalation_lab — the uncertainty machinery meets a real recurring decision (2026-08-29)

- **Path:** `~/machine_learning/escalation_lab/` (`LAB_PLAN.md`, Phase E0 not yet run)
- **What:** repairs the `GF`/`QX` escalation decision in RUSTMM's SEC fast-scoring path —
  cheap ridge probe vs. expensive GL/QL LLM scoring — using `gp_engine`'s posterior,
  `decision.py`/`voi.py`'s VoI machinery, and `ARC.md`'s monitoring lessons. **The first real
  domain for `decision_harness_lab`'s never-started Phase 3**: a decision that is real,
  recurring, and costly, which is exactly what `vol_regime_lab` failed to satisfy.
- **A correction made while scoping**, recorded because JOURNEY.md §9 asserted it wrongly:
  the fast scorer is **already per-dimension gated** (`phase1_probes.py` computes LOO-by-ticker
  Spearman + residual band per dim; `phase1_score.py` gates at `DISTILL_THRESHOLD=0.60`) —
  grouping LOO by ticker is correct practice. JOURNEY.md §9 has been corrected in place.
- **The five real defects (D1-D5)**: uncertainty is per-dimension but never per-filing; the
  within-filing chunk spread is computed by `per_chunk_probe()` and then **discarded by a
  `.mean()`**; `novelty()` is a covariate-shift detector (ARC §4.6: structurally cannot see
  concept drift) and is chunk-averaged so a few novel sections wash out; the `0.60` threshold
  has no cost model; and the two gates don't compose at the real *(filing, dimension)* unit.
- **Phases:** E0 harness + kill condition (`decision.has_probe_niche()` on real elicited costs
  — if no probe niche exists, the lab stops there); E1 per-filing posterior, cheapest first
  (chunk spread → **Bayesian-ridge closed form**, since `_ridge()` is already a Bayesian linear
  posterior mean → GP only on measured shortfall); E2 the VoI policy replacing the threshold;
  E3 turn the existing `--anchor` label budget into ARC §4.8 lift monitoring. **E4 (generalise
  into a reusable decision-under-uncertainty workflow) deliberately deferred** — this codebase
  extracts patterns after the second instance, not before the first.
- **Small-n objection deliberately overruled and documented**: n≈31/270 fails the four-point
  litmus test on size, so **no claim is made that this exercises the engines** — the value is
  the statistics and decision layer. A chunk-level GP (10^5-10^6 embeddings) would clear the
  bar and is a legitimate later escalation.
- **Terminology trap pinned in the plan**: in `voi.py` "probe" = the cheap *information action*;
  in RUSTMM "probe" = the cheap *scorer*. Here the **LLM call is voi.py's probe**. Invert this
  and every payoff matrix is backwards.
