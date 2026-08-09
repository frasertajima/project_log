// project log

### OS/2 VX-REXX

1995: OS/2 VX-REXX Real Estate conveyancing document production application

A full real estate conveyancing document production application for production use during real estate conveyances in British Columbia within my own law firm. Produced all documents required for the transaction with all calculations. Each document and calculation was verified manually for each client file but in production the system was without error.


## GitHub repositories

GitHub profile: https://github.com/frasertajima

The work below clusters into a few themes: two flagship **GPU compute innovations** (the tensor core engine and MPDOK), a **tiny-pointers research line** of labs, a body of **consumer-GPU scientific computing** (training huge scientific datasets on an 8GB card), **security/YubiKey tooling**, and **personal data/productivity utilities** that feed the COBOLMM daily driver.

### Timeline (2022 → 2026)

A rough arc: from **personal data utilities** (transcription, Kindle, YubiKey), through a Fortran/CUDA self-education on **consumer-GPU scientific computing**, into the **GPU compute innovations** (tensor core engine, MPDOK) and the **tiny-pointers** research line — all now converging into the COBOLMM daily driver.

| Year | Focus | Key repos (created) |
|---|---|---|
| **2022** | First data utilities | `whisper_bulk_transcription` (bulk audio→text); `zoose-gitpod-public` |
| **2023** | Security + personal corpora | `random_number_yubikey` (YubiKey RNG); `kindle_clippings`; `tutorials` *(fork)* |
| **2024** | GPG/Linux tooling + Fortran begins | `kleopatra-Fedora-preparation`; **`fortran`** (examples repo created — later becomes the tensor-core engine home); `zed_unable_to_launch_doc_fix` *(fork)* |
| **2025** | Consumer-GPU scientific computing | `WeatherBench2` (72GB ERA5, 8GB GPU); `Cryo-EM_Denoising` (matches Topaz on 117GB); `tailscale_shortcuts` |
| **2026 (H1)** | COBOL, cryo-EM scale-up, then the innovations | COBOL: `cobol-programming-course` *(fork)*, `cobol-check-automation` · cryo-EM/physics: `cryo_em_denoising2` (200GB+), `cryo-em-3d-viewer`, `turbulence-spectral-lab`, `emergency-response-simulation` · security: `OpenPGP_Nautilus_menu` · transcription: `cohere-transcribe-script` · **innovations:** `tensor-core-quant-sim-demo` (engine v5), **`MPDOK`** · **tiny-pointers labs:** `tiny_pointers_lab`, `tiny_pointers_digitstash`, `tiny_pointers_dna_lab` |


### Innovations

#### Tensor core engine

The recurring throughline of the recent work: a custom CUDA Fortran kernel family that exploits NVIDIA tensor cores for general matrix operations while *controlling* precision loss, instead of accepting the low precision tensor cores normally impose.

- **fortran** — https://github.com/frasertajima/fortran — the home of the engine and its progressive optimisation history. `matrix_dot` reached **35.5 TFLOPS vs ~300 GFLOPS for CuPy** (~100×); matmul, batched, and 4D-tensor variants show comparable wins. Also collects the supporting CUDA Fortran work: Mandelbrot, Monte Carlo option pricing, radix sort, 3D Laplace, GEMM kernels climbing 300 GFLOPS → 3 TFLOPS, plus *Numerical Recipes* ports and the YubiKey RNG.
- **tensor-core-quant-sim-demo** — https://github.com/frasertajima/tensor-core-quant-sim-demo — engine **v5** applied to quantum dynamics (1D transverse-field Ising chain quench). Key trick: a "real-doubling" map turns complex unitary evolution `exp(-iHt)` into a real *orthogonal* matrix problem whose entries are bounded by 1, so **TF32 never overflows**. Result: **~60× over CuPy** at the tensor-core baseline, and an "improved32" mode giving ~6× FP64 speed at 1e-5 unitarity error. *Learning: choosing a representation where values are bounded is what makes low-precision hardware safe to use.*

#### MPDOK — Mixed-Precision Dense-Operator Krylov

- **MPDOK** — https://github.com/frasertajima/MPDOK — a GPU iterative solver (CG / GMRES / BiCGSTAB) that delivers **double-precision accuracy at near-half-precision speed**. Three tiers: dense-operator Krylov (BEM, GP regression), GMRES-IR mixed-precision refinement (FP16/FP32/FP64), and mixed-precision preconditioning for sparse systems. Uses cuBLAS-Lt epilogue fusion and the **Ozaki 5×FP32** technique (~1e-7 accuracy). Benchmarks as *"faster than SciPy but with more precision, not less,"* crossing into superiority around 128³. Fills the gap PETSc/AMGX leave for dense operators. *Learning: the usual speed-vs-accuracy trade-off in iterative solvers is not fundamental — precision can be controlled per-stage rather than accepted globally.* Now the semantic-search kernel behind COBOLMM's RSS and web-clip ranking.

---

### Tiny-pointers research line
A series of labs exploring Bender et al.'s tiny-pointers paper (O(log log n)-bit "hints" instead of full pointers) and pushing the idea into new domains — the same engine that powers the millisecond `stash` search.

- **tiny_pointers_lab** — https://github.com/frasertajima/tiny_pointers_lab — the core GPU benchmark. Tiny-pointer lookups hold **~2,500–2,700 Mops regardless of load**, while linear probing collapses 2,306 → 213 Mops at 99% load; external references shrink **5.5× at 99% load** (22 bits → 4 bits). Crossover ~95% load. Demonstrates all five paper applications (optimal stashes, key-less value stores, stable/space-efficient dictionaries, succinct BSTs). *Learning: tiny pointers win specifically at high load factor — a boundary win, not a universal one.*
- **tiny_pointers_digitstash** — https://github.com/frasertajima/tiny_pointers_digitstash — applies the trigram/k-mer index to MNIST pixel space (approx k-NN). A tiny verified shortlist **beats exhaustive k-NN: 97.82% vs 96.88%, and ~5× faster** — the shortlist acts as a regulariser on the vote. *Learning: a good retrieval boundary can beat brute force on both accuracy and speed.*
- **tiny_pointers_dna_lab** — https://github.com/frasertajima/tiny_pointers_dna_lab — the same engine as a laptop-portable DNA k-mer presence screener (pathogen / AMR / contaminant alert). *Learning: a trigram engine generalises cleanly from text → pixels → genomic k-mers.*

---

### Consumer-GPU scientific computing (8GB-card, huge datasets)
A theme of training/serving large scientific workloads on a single consumer RTX 4060 (8GB) via Fortran + CUDA/cuDNN streaming data loaders — "you don't need a cluster."

- **Cryo-EM_Denoising** — https://github.com/frasertajima/Cryo-EM_Denoising — a **3-layer CNN, 5 epochs, 117GB dataset, 8GB GPU, ~1.75h** that **matches Topaz-Denoise SSIM (0.86) and exceeds its PSNR (21.57 dB)**. *Learning: with a large, naturally diverse dataset, architectural simplicity beats deep models + augmentation.*
- **cryo_em_denoising2** — https://github.com/frasertajima/cryo_em_denoising2 — Fortran/CUDA streaming loader scaling the above to **200GB+ datasets on an 8GB GPU**.
- **cryo-em-3d-viewer** — https://github.com/frasertajima/cryo-em-3d-viewer — interactive 3D viewer for the denoised micrographs.
- **WeatherBench2** — https://github.com/frasertajima/WeatherBench2 — a Fortran cuDNN climate U-Net trained on the **72GB WeatherBench2 ERA5** dataset on the same 8GB RTX 4060.
- **turbulence-spectral-lab** — https://github.com/frasertajima/turbulence-spectral-lab — spectral turbulence experiments.
- **emergency-response-simulation** — https://github.com/frasertajima/emergency-response-simulation — interactive 3D scenario builder + GPU-accelerated pollutant dispersion simulator.

---

### Security / YubiKey / GPG
- **random_number_yubikey** — https://github.com/frasertajima/random_number_yubikey — cryptographically strong RNG from the YubiKey secure element (`scd random`), with Linux / WSL2 / Fortran variants and a password generator.
- **OpenPGP_Nautilus_menu** — https://github.com/frasertajima/OpenPGP_Nautilus_menu — right-click GPG encrypt/decrypt in Nautilus backed by a YubiKey.
- **kleopatra-Fedora-preparation** — https://github.com/frasertajima/kleopatra-Fedora-preparation — makes Kleopatra work on Fedora Silverblue and similar immutable Linux systems.

---

### Personal data & productivity tooling (COBOLMM feeders)
- **kindle_clippings** — https://github.com/frasertajima/kindle_clippings — notebooks to turn the Kindle `My Clippings.txt` dump into something useful; the canonical reference for the COBOLMM Kindle module.
- **whisper_bulk_transcription** — https://github.com/frasertajima/whisper_bulk_transcription — bulk audio→text with OpenAI Whisper (the original transcription pipeline).
- **cohere-transcribe-script** — https://github.com/frasertajima/cohere-transcribe-script — batch transcription via Cohere Transcribe.
- **tailscale_shortcuts** — https://github.com/frasertajima/tailscale_shortcuts — securely trigger Python scripts on a Linux desktop from iOS/Android Shortcuts over Tailscale via FastAPI.

---

### COBOL
- **cobol-check-automation** — https://github.com/frasertajima/cobol-check-automation — automated unit testing for COBOL (groundwork for the COBOLMM stack).
- **cobol-programming-course** *(fork)* — https://github.com/frasertajima/cobol-programming-course — reference training materials.

---

Under construction (not on github):

### COBOLMM:
A COBOL IBM terminal-styled CLI application that gathers various projects and makes them accessible. Vocabulary, books and Kindle clippings gathered decades-old collections and enabled easy loading of data, browsing, searching, editing and other maintenance as well as export. Missing vocabulary definitions can be filled in with local AI to reduce workload. 

The unifying goal of the project is to enable easy search and access to all information that I am creating or using (without irrelevant material or hits as is common with standard search and other tools). Original 1-2s search times under COBOL upgraded to millisecond search using trigram search and Rust search engines. COBOL limitations uncovered and required the switch to Rust.

Charts supports equity trading by downloading daily or weekly charts along with local or cloud LLM analysis of the charts. Custom charts can also be collected and analysed. SEC 10-K and 10-Q filings conveniently gather and analyse SEC filings using local or cloud LLM and enable browsing and searching of these reports. CVaR portfolio runs a portfolio optimisation run over S&P500 stocks using Nvidia’s cuFolio portfolio optimiser and produces a report concerning the portfolio composition and changes from the past.

Transcription batch transcribes any audio files using a local LLM and enables browsing and search of the transcription, with a default to transcribe any new podcasts downloaded to the machine.

Web Clippings integrates with Obsidian’s web clipper to enable browsing and searching.

RSS feeds take the user’s OPML-saved RSS feeds and download new feeds. Millisecond search of RSS feeds. MPDOK-powered semantic search enables ranking of the most relevant search results without the computational burden and slowness of local LLM processing while still providing human-like ranking.

PDF search enables both yazi and stash selection of individual PDFs to index content. Bulk indexing is also easy with any image-only pdf automatically OCRed. Once PDFs are indexed their content can be searched with a millisecond response that is constant time (irrespective of dataset size).

Markdown search indexes all Markdown files on the NAS and local drive relevant to the user. Most users will use Synology Drive to sync frequently used NAS folders to the local SSD and our search will read the Synology Drive sync folder list to only index local SSD copies and ignore the slower NAS folders. This has resulted in 80% or more of Markdown files being indexed locally rather than traversing the slower NAS.

OCR images enable a manual OCR operation using a LLM in a set folder.

Image search enables compression of images, description of images with a local LLM for a human-like summary of the image and indexing and search of images. Any descriptions which are not quite correct can be manually edited.

COBOLMM editorial reviews run a cloud or local LLM over the day’s RSS feeds to curate a summary of the day’s feed, useful for quickly summarising the day’s events. Catalyst search looks for catalysts to stocks from the day’s RSS feeds using cloud or local LLM and produces a report. Any stock tickers missing from our SEC filings can also be batch processed for analysis.

Universal and quick search tie all the datasets together to provide a convenient way to search all relevant datasets. Quick search and universal search were originally separated due to the slower search times with pdfs when using linear search but after the upgrade to trigram search for all search, search times are identical. Quick search is still retained as the results would be most relevant for search of recent items (pdfs and other sources tend to be older).

Semantic search is somewhat experimental and runs MPDOK vector kernel search over web clippings to provide the most relevant search results without the slowness of local or cloud LLM processing. Relevance is colour coded with the most relevant search results and their score at the top.

Stash file search replicates the command line utility `stash` to enable search of a file in milliseconds without the complications or irrelevant search results common in most command line utilities.


### stash:

Stash (file search), mdstash (markdown content search) and pdfstash (pdf content search) employ tiny pointers and trigram search for millisecond search results using a compact index for constant time results (irrespective of growth in dataset size) in Rust. This is particularly useful when data is on a lower-end/slow NAS, slow external HD or other devices that are not necessarily connected to the machine at the time of search, and other messy collections of data. There are 0 I/O operations and no file operations on the NAS until the user ctrl+clicks the file link to open the file. The user can tailor the data sources (internal and external) so that the search results are always filled with user-relevant data (not junk files created by the OS or other programs).

### concept search engine:

https://youtu.be/CLoJR8PJD3c?si=1kFbPsyXgyBLqEtX

### RANSAC Android scanner

[RANSAC scanner on android
](https://felixquinihildebet.wordpress.com/2026/07/05/ransac-scanner-on-android/)

### Upgraded Portfolio Studio and MPDOK signal board

- integrated into COBOLMM menu structure for ease of access
- experiment with RANSAC inspired alert board
<img width="3640" height="592" alt="Screenshot From 2026-07-07 15-12-14" src="https://github.com/user-attachments/assets/cfdc1369-8694-45b5-ae18-117a170902df" />

### GP Engine
[gp_engine](https://github.com/frasertajima/gp_engine)
- 3rd engine after tensor core, MPDOK: gp engine handles Gaussian process pipeline
- OOC component running (in development) and tested to n>300k
- [gp lab](https://github.com/frasertajima/gp_engine/tree/main/gb_lab)
- place_gpc_lab

### Updated Place Android Scanner
- Laplace gpc and svm
- on phone video processing and easy route acquistion for testing
- higher lock ratio achieved on complex route

### GP mining lab (extension of MPDOK mining lab)
- economic benefit to Laplace GPC vs SVM in low confidence environments
- compare vs MPDOK as same dataset used
- [gp mining](https://github.com/frasertajima/gp_engine/tree/main/mining_gpc_lab)

### bayesian decision lab
- exploring payoff outcomes with SVM, Laplace GPC mean only and Laplace GPC full posterior to sketch out decision making under conditions of uncertainty
- [MNIST lab](https://github.com/frasertajima/gp_engine/tree/main/mnist_gpc_lab)

### porphyry_cu_gpc_lab
- same bayesian decision engine as with gp mining lab (gold) but this time with copper
- similar holdings
- [porphyry cu gpc lab](https://github.com/frasertajima/gp_engine/tree/main/porphyry_cu_gpc_lab)

### rbfx
- A compact Rust library wrapping MPDOK/gp_engine's proven mixed-precision dense
solver (`gp_solver.so`: FP32 Cholesky factor + FP64 iterative refinement,
CUDA Fortran/cuBLAS/cuSOLVER) so it's usable as a generic dense
scattered-data-interpolation kernel — from a plain Rust program, or from
Python/Jupyter via a PyO3 wheel — rather than only through this codebase's
existing app-specific ctypes wrappers (`gp_engine`, `MPDOK/kriging`,
`rbf_pointcloud`, `rbf_spatial`).
- [rbfx](https://github.com/frasertajima/rbfx)

### cvar_gp_lab
- Bayesian CVaR
- integrated into Portfolio Studio
- integrated into COBOLMM as mirror to cuFolio portfolio (including Smart1 and Smart2 portfolios)
- [gb lab](https://github.com/frasertajima/gp_engine/tree/main/gb_lab)

### climate cat_lab
- [climate cat lab](https://github.com/frasertajima/gp_engine/tree/main/gp_cat_lab)
- same shortcoming as portfolio optimisation, this time in insurance
- simulated dollar amounts are considerable; industry knows about the issue but has not fixed it, like in finance

### gaussian process soft-EM portfolio studio addition
- cuFolio, MPDOK and now gp soft-EM
- best performance, lowest drawdown
- [Portfolio studio](https://github.com/frasertajima/MPDOK/tree/main/portfolio_studio)

### grid reserve lab
- [grid reserve lab](https://github.com/frasertajima/gp_engine/tree/main/grid_reserve_lab)
- only option that provides reliable results and lowest cost (gp soft-EM)

### smh lab (bridge structure health/preventative maintenance)
- key finding: gp soft-em does not have a decisive advantage (over pooling, which is simpler)
- soft-EM only shows a decisive advantage when there are rare, recurring regimes to iteratively model
- [shm_lab](https://github.com/frasertajima/gp_engine/blob/main/shm_lab/SHM_LAB.ipynb)

### hydro reserve lab
- [hydro reserve lab](https://github.com/frasertajima/gp_engine/tree/main/hydro_reserve_lab)
- regime change breaks statistical assumptions of past environment, e.g., climate change

### [tiny pointers hash join](https://github.com/frasertajima/tiny_pointers_hash_join/tree/main)
- very niche use case where load > 95% but it may become useful when that is a constant condition

### gp soft-em decision engine
- refactored gp soft-em bayesian decision engine into rust and integrated into RUSTMM/COBOLMM for ease of access
- added diagnostic function to enable selection of options:
  Validated across 5 random-holdout splits at target reliability 0.99:
  empirical:      reserve=1696.5725  won 60% of splits, mean gap from target -0.0120
[-0.0460,+0.0090]
  one_component:  reserve=111867.8001  won 20% of splits, mean gap from target +0.0100
[+0.0100,+0.0100]
  soft_em:        reserve=1814.1437  won 20% of splits, mean gap from target +0.0000
[-0.0180,+0.0100]

- gp soft-em tends to be extremely niche so it has been somewhat difficult to surface good lab examples but with this CLI framework we can surface the function easily on the command line, in Jupyter notebooks and with a local LLM workflow in the future
- also ran a standard decision lab
---

Later projects such as COBOLMM and stash use Claude extensively. 
