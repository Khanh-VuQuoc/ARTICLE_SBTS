# ARTICLE\_SBTS — Deep Hedging of Rainbow Options via Schrödinger Bridge

> **Paper:** *Multivariate Schrödinger Bridge for Deep Hedging of Rainbow Options under Transaction Costs*
> **Status:** Under review

---

## Overview

This repository contains the **complete reproducible pipeline** for the paper.
We compare three path generators — **GBM**, **Heston**, and **SBTS** (Schrödinger Bridge Time Series) — as training environments for a deep hedging agent on **multi-asset rainbow options** (Basket Asian Call and Asian Worst-of Put) with proportional transaction costs `c = 0.001`.

```
yfinance 2005–2026
       │
       ├── 2005–2019 (calibration) ──► GBM    20,000 paths
       │                           ──► Heston 20,000 paths
       │                           ──► SBTS   20,000 paths
       │                                    │
       │                                    ▼  split 16k/2k/2k (seed=42)
       │                                360 checkpoints (training)
       │                                    │
       └── 2019–2025 (OOS)  ──► 3 historical periods
                                            │
                                            ▼
                              t-tests (216) · MCS (18)
```

---

## Repository Structure

```
ARTICLE_SBTS/
│
├── notebooks/
│   ├── SBTS_CANONICAL_A100.ipynb    # ★ canonical pipeline — data → training → tests → tables
│   ├── SBTS_CANONICAL_A100.py       #   generated audit export of the notebook
│   ├── article_data.ipynb           # legacy: data download, calibration, path generation
│   ├── article_gbm.ipynb            # legacy: GBM training
│   ├── article_heston_sbts.ipynb    # legacy: Heston + SBTS training
│   └── article_4_3period.ipynb      # legacy: OOS evaluation, tests, tables, figure
│
├── tools/
│   └── export_notebook_py.py        # regenerates the .py audit snapshot
│
├── THESIS_RESULT_MAPPING.md         # thesis object → artifact index
├── requirements-canonical.txt       # locked environment for the canonical run
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

Auto-generated outputs (not tracked in git): the canonical run directory
`ARTICLE_SBTS/canonical_runs/<run_id>/` on Drive, and, for the legacy pipeline,
`article_results_3p_covid_ext/` and `checkpoints_article/` (360 `.pt` files — see *Checkpoints* below).

---

## The canonical notebook

`notebooks/SBTS_CANONICAL_A100.ipynb` is the **single execution source** for the
paper. It runs from a clean Colab A100 kernel and carries the whole pipeline:
frozen data snapshot → generator calibration → path generation → deep-hedging
training → uniform evaluation → statistical tests → support diagnostics →
tables/figures → signed manifest. The legacy four-notebook pipeline is kept for
comparison; its outputs are **not** treated as final evidence.

What the canonical notebook fixes relative to the legacy pipeline:

| Legacy issue | Canonical behaviour |
|---|---|
| `seed_all` missing in the GBM session | defined in the environment bootstrap; a unit test covers it |
| Seed 3 replaced by seed 10 under the old label | seed replacement is impossible — a failed run is recorded `status="failed"` and the tests report the real common-seed `n` |
| Ad-hoc MCS called Hansen–Lunde–Nason | `arch.bootstrap.MCS` at a locked version, with four synthetic unit tests |
| COVID window described as strictly out of sample | locked two-tier disclosure: 2019 overlaps the calibration window, 2020+ is out of sample |
| Regimes reported by start date only | per-regime start/end ranges, crash share and data cutoff are stored and printed |
| `M // batch_size` dropped 3,712 of 16,000 paths per epoch | `range(0, M, batch_size)`; every epoch asserts it consumed all `M` samples |
| Best epoch restored weights only | the best bundle restores model, optimizer, scheduler and the CVaR `ν` |
| Phase-2 `ν` initialised at 0 | `ν` starts at the empirical VaR₀.₉₅ of the Phase-1 validation residuals |
| `grad_norm_post_clip` logged the pre-clip norm | the post-clip norm is measured after clipping, with the invariant asserted |
| Xavier applied to the output layer only | every `Linear` layer initialised, verified by a unit test |
| Bandwidth selected on a random split of overlapping windows | chronological split with a purge/embargo of one horizon, plus the full held-out objective (`paper_full`) |
| Support counts derived from clamped weights | exact boolean compact-support masks; NNZ/ESS/entropy/concentration regenerated from code |
| Artifacts mixed across runs | every cache checks `schema_version` + `config_hash` + data hashes; mismatches are quarantined, not merged |

### Running it

```python
RUN_MODE = "SMOKE"   # validate the pipeline and size the machine
# RUN_MODE = "FULL"          # the canonical 180-configuration experiment
# RUN_MODE = "ANALYSIS_ONLY" # reuse existing checkpoints, no training
# RUN_MODE = "DIAGNOSTICS"   # support diagnostics only
```

Cell 1 installs `arch` and `yfinance` (required — it stops with the pip output
if either is unavailable) and only *verifies* numpy/pandas/scipy/torch/matplotlib
against the lock, recording any deviation, because force-downgrading the Colab
CUDA build would need a runtime restart. Set `DEPENDENCY_POLICY = "install_locked"`
in Cell 1 to force the exact pins anyway.

Run `SMOKE` first: it executes every stage on a reduced grid, runs the unit and
integration tests (Gate 1) and prints a runtime estimate measured on the machine
you are actually using. Only then switch to `FULL`. Outputs land in
`MyDrive/ARTICLE_SBTS/canonical_runs/<run_id>/` with `config/`, `environment/`,
`data/`, `generators/`, `checkpoints/`, `evaluations/`, `statistics/`,
`diagnostics/`, `tables/`, `figures/`, `logs/` and `manifest.json`.

### Resuming and running shards in parallel

A `FULL` run is tens of hours and will be interrupted. Leaving `RESUME_RUN_ID`
as `None` resumes the newest unfinished run with the same `config_hash` and
says so. Resuming works at three levels:

* completed configurations are skipped once both checkpoints verify;
* a finished Phase 1 is reused rather than re-trained, together with the
  random state it ended in, so an interruption during Phase 2 does not repeat
  the longer phase and still gives exactly the uninterrupted result (without
  that state, Phase 1 is re-trained instead);
* a phase persists model, loss, optimizer, scheduler, history, patience and
  RNG state every `CHECKPOINT_EVERY_EPOCHS` epochs (default 25) and continues
  from the epoch it reached.

At most the epochs since the last periodic save are repeated.
`FORCE_NEW_RUN = True` starts a separate run instead.

To split the work across two T4 sessions, `notebooks/shards/` holds two
ready-to-run notebooks, one per Colab session. Open each and run them at the
same time; nothing to edit:

| File | GPU | Trains |
|---|---|---|
| `run_T4_1.ipynb` | T4 | seeds 0, 2, 4, 6, 8 of GBM, Heston and SBTS |
| `run_T4_2.ipynb` | T4 | seeds 1, 3, 5, 7, 9 of GBM, Heston and SBTS |

Together they cover all 180 configurations with no overlap; completed
configurations are skipped. Both join the existing run automatically — the one
with the same `config_hash` and the most completed configurations — and must
print the same `run_id` in Cell 3.

The whole experiment is trained on T4: these notebooks set
`REQUIRE_TRAINED_ON_GPU = "T4"`. Every checkpoint records the GPU it was
trained on, and a configuration whose checkpoint came from another GPU (the
earlier A100 session) is trained again from scratch; its old checkpoints are
quarantined, and the notebook lists them before the queue starts.
**Whichever finishes last continues straight into the audit, evaluation,
statistics, diagnostics and tables**; the other stops after Cell 19 with
`ShardTrainingComplete`, the expected end and not an error. After a Colab
disconnect, run the same notebook again. If both stop without the analysis —
possible because Google Drive syncs between machines with a delay — run either
one again: it finds every configuration complete and goes straight to the
analysis.

Stop any session still running an older copy of the notebook first, or two
processes will train the same configurations.

All shards run on a T4. The A100_FAST numerical mode needs bf16/TF32
(compute capability 8.0+), so on a T4 Gate 3 benchmarks only REFERENCE_FP32.
Each session records its own GPU and environment
(`environment/environment__<shard>.json` and `gpu_info__<shard>.txt`), and every checkpoint
records the GPU it was trained on, which is what `REQUIRE_TRAINED_ON_GPU` checks.

These files are generated — edit `SBTS_CANONICAL_A100.ipynb` and re-run
`python3 tools/make_shard_notebooks.py`.

A run is only quotable in the thesis when its manifest reports
`publishable: true` — that requires `RUN_MODE="FULL"`, the real hashed Yahoo
snapshot, the `paper_full` selection engine, no recorded failures and every
audit check passing. See `THESIS_RESULT_MAPPING.md` for the object-to-artifact
index.

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/Khanh-VuQuoc/ARTICLE_SBTS.git
cd ARTICLE_SBTS
pip install -r requirements.txt
```

### 2. Run on Google Colab (recommended — GPU)

Open each notebook in Colab and mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

The notebooks expect a folder `MyDrive/ARTICLE_SBTS/` in your own Google Drive. Download checkpoints from Zenodo (see *Checkpoints* below) into `MyDrive/ARTICLE_SBTS/checkpoints_article/`.

### 3. Pipeline order

**Canonical (recommended):** open `notebooks/SBTS_CANONICAL_A100.ipynb` on an
A100 runtime, `pip install -r requirements-canonical.txt` (the notebook does
this itself in Cell 1), run `RUN_MODE="SMOKE"` end to end, then `RUN_MODE="FULL"`.
No runtime figure is quoted here on purpose — Cell 18 measures it on your machine.

**Legacy (for comparison only):**

```
article_data.ipynb          — Data & path generation         (~15 min, CPU)
article_gbm.ipynb           — GBM training                   (~10 h on T4 GPU)
article_heston_sbts.ipynb   — Heston + SBTS training         (~20 h on T4 GPU)
article_4_3period.ipynb     — OOS evaluation + tests         (~5 min on T4 GPU, given checkpoints)
```

To reproduce only the legacy statistical analysis (skip training), download the 360 checkpoints from Zenodo and run `article_4_3period.ipynb` directly.

---

## Checkpoints

The 360 model checkpoints (`.pt` files) are archived on Zenodo with a permanent DOI:

> 📦 **Zenodo:** https://doi.org/10.5281/zenodo.20176723

Each checkpoint follows a **5-layer provenance scheme**:

1. **Model state** — `net_state_dict`, `V0`, `optimizer_state_dict`, `scheduler_state_dict`, CVaR auxiliary `nu`
2. **Architecture + hyperparameters** — complete spec to rebuild the network and reproduce training
3. **Training history** — per-epoch train/val loss, learning rate, gradient norms (pre/post clip), `V0`, `nu`
4. **Evaluation artifacts** — full per-path arrays (residuals, PnL, cost, payoff) on synthetic test + historical periods, with SHA-256 hashes
5. **Provenance** — timestamps, environment snapshot (torch / CUDA / GPU), data-file SHA-256, RNG state, git commit, integrity hash

---

## Experimental Design

| Component | Details |
|---|---|
| **Assets** | AAPL (Tech), JPM (Financials), XOM (Energy) |
| **Calibration period** | 2005-01-01 → 2019-12-31 |
| **OOS periods** | COVID 2019–2020 · PostCOVID 2021–2022 · Recent 2023–2025 |
| **Options** | Basket Asian Call · Asian Worst-of Put |
| **Strike levels κ** | {0.95, 1.00, 1.05} |
| **Transaction cost** | c = 0.001 (proportional) |
| **Seeds** | 10 per configuration |
| **Total configurations** | 180 (3 generators × 2 options × 3 strikes × 10 seeds) |
| **Total checkpoints** | 360 (180 × 2 phases: MSE warmup + CVaR fine-tune) |

### Network architecture

- **Type:** Feedforward MLP, semi-recurrent in time (shared weights across 252 rebalancing dates, $\delta_{t-1}$ fed back as input)
- **Layers:** `[10, 64, 64, 3]` with ReLU, ~5,060 parameters
- **Input features:** `[spots (3), running_avg (3), prev_delta (3), time_left (1)]`
- **Output:** hedge positions $\delta \in \mathbb{R}^3$
- **Initialization:** Xavier uniform, last-layer gain 0.1
- **Initial wealth $V_0$:** scalar parameter

### Two-phase curriculum

| Phase | Loss | Epochs | LR | Patience | $V_0$ |
|---|---|---|---|---|---|
| **Phase 1** | MSE | 500 | $10^{-3}$ | 20 | trainable |
| **Phase 2** | CVaR$_{0.95}$ | 200 | $10^{-4}$ | 20 | frozen |

Optimizer: Adam $(\beta_1, \beta_2) = (0.9, 0.999)$. Scheduler: `ReduceLROnPlateau` (factor 0.5, patience 10, min lr $10^{-7}$). Gradient clipping: max-norm 1.0. Batch size: 4,096.

### Three-way fairness principle

All three generators observe **only** pre-2020 data for calibration. OOS evaluation uses the **same** 3 historical periods, the **same** network architecture, and the **same** hyperparameters — ensuring apples-to-apples comparison.

---

## Statistical Evaluation

| Test | Details |
|---|---|
| **Paired t-test** | 216 pairwise tests (3 pairs × 2 options × 3 strikes × 3 periods × 2 phases × 2 metrics), BH-FDR adjusted at α = 0.05 |
| **Model Confidence Set** | 18 MCS computations (3 periods × 2 options × 3 strikes), Hansen–Lunde–Nason (2011), α = 0.10, seed-level bootstrap B = 5,000 |
| **Scoreboard** | Win/Tie/Loss across 6 cells per period × pair |

---

## Key Results

> **Status:** the figures in this section come from the legacy pipeline. They are
> being regenerated by `SBTS_CANONICAL_A100.ipynb` and must not be quoted as
> locked thesis results until a `FULL` run reports `publishable: true`. The
> claims flagged **pending canonical rerun** in `THESIS_RESULT_MAPPING.md` —
> the selected bandwidth and Markov order, the MCS membership counts, and the
> `313 → 7` / `45×` support-contraction numbers — are the ones most likely to
> change, because the procedures that produced them were replaced.


- **Representative regime (Recent 2023–2025):** SBTS is the **singleton MCS** at all 6 (option, strike) cells; paired t-tests reject equality vs both baselines at all 6 cells. SBTS reduces hedging-error standard deviation by **13–42%** relative to GBM and **8–39%** relative to Heston.
- **COVID 2019–2020 stress regime:** Ranking **reverses** — SBTS hedging-error standard deviation exceeds the parametric baselines by an average of **≈101%** (range 60–156% across cells). The MCS at α = 0.10 retains all three generators in 5/6 cells (elevated cross-seed variance under stress); paired t-tests still detect an SBTS deficit in 5/6 cells at BH-adjusted 5%.
- **Aggregated:** SBTS belongs to MCS in 15/18 cells, GBM in 9/18, Heston in 9/18.
- **Mechanism:** SBTS underperformance under stress is interpreted as a **bias–variance trade-off under distribution shift** combined with the **support constraint of the non-parametric bridge construction** (the calibration-window kernel support collapses sharply, from 313 to 7 effective neighbours), which limits extrapolation beyond the historical calibration window. This is *not* a positive endorsement of the parametric baselines as stress models — GBM and Heston also degrade by several-fold relative to the representative regime.

---

## Requirements

```
python >= 3.9
torch >= 2.0
numpy
pandas
scipy
matplotlib
yfinance
jupyter
```

Full list: see `requirements.txt`. The canonical notebook pins exact versions in
`requirements-canonical.txt`; any deviation is recorded in the run's
`environment/environment.json` and surfaces as a manifest warning.
Reproducibility is claimed as *reproducible from the frozen snapshot and locked
environment within declared numerical tolerances* — not bit-for-bit across
different GPUs or software stacks.

---

## Seed policy

**Legacy pipeline (what was done).** One training run (SBTS,
asian\_worst\_of\_put, $\kappa = 0.95$, seed 3) diverged during Phase 2: $\nu$
collapsed to a pathological fixed point and $\sigma(R)$ exceeded $10^{10}$ on
stress paths. Three additional seeds (10, 11, 12) were re-run and seed 3 was
replaced by seed 10 while keeping the seed-3 label, so that $n = 10$ was
preserved. The log is in `seed_replacement_log.json` (archived on Zenodo).

**Canonical pipeline (what happens now).** Seed replacement is not available.
`SEED_ALIAS_MAP` is empty and asserted empty, the checkpoint audit fails if any
checkpoint records a seed different from the run key pointing at it, and a run
that still fails after its retry budget — retries always reuse the same seed and
the same configuration — is recorded `status="failed"`. Statistical tests then
use the seed-set intersection and report the real `n`, never a padded one. The
divergence itself is also addressed directly: Phase 2 now initialises $\nu$ at
the empirical VaR₀.₉₅ of the Phase-1 validation residuals rather than at zero,
and the best-epoch bundle restores $\nu$ together with the network.

---

## Citation

```bibtex
@article{VuNguyen_SBTS_deephedging_2026,
  title   = {Multivariate Schr\"odinger Bridge for Deep Hedging
             of Rainbow Options under Transaction Costs},
  author  = {Vu, Quoc Khanh and Nguyen, Thai},
  journal = {[JOURNAL — under review]},
  year    = {2026}
}
```

---

## License

This code is released under the **MIT License**. See `LICENSE` for details.

Data sourced from Yahoo Finance via `yfinance` — subject to Yahoo Finance terms of use.
