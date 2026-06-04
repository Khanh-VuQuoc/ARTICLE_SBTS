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
│   ├── article_data.ipynb           # Data download, generator calibration, path generation
│   ├── article_gbm.ipynb            # GBM training (60 configs × 2 phases = 120 checkpoints)
│   ├── article_heston_sbts.ipynb    # Heston + SBTS training (120 + 120 checkpoints)
│   └── article_4_3period.ipynb      # OOS evaluation, statistical tests (t-test + MCS), tables, figure
│
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

Auto-generated outputs (not tracked in git): `article_results_3p_covid_ext/` (CSV results, LaTeX tables, figures), `checkpoints_article/` (360 `.pt` files — see *Checkpoints* below).

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

```
article_data.ipynb          — Data & path generation         (~15 min, CPU)
article_gbm.ipynb           — GBM training                   (~10 h on T4 GPU)
article_heston_sbts.ipynb   — Heston + SBTS training         (~20 h on T4 GPU)
article_4_3period.ipynb     — OOS evaluation + tests         (~5 min on T4 GPU, given checkpoints)
```

To reproduce only the statistical analysis (skip training), download the 360 checkpoints from Zenodo and run `article_4_3period.ipynb` directly.

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

Full list: see `requirements.txt`.

---

## Seed-replacement disclosure

One training run (SBTS, asian\_worst\_of\_put, $\kappa = 0.95$, seed 3) diverged during Phase 2: $\nu$ collapsed to a pathological fixed point and $\sigma(R)$ exceeded $10^{10}$ on stress paths. The same configuration converged for the remaining nine seeds. We re-ran three additional seeds (10, 11, 12), all of which converged with metrics within the distribution of the original nine, and replaced seed 3 with the smallest-index converged re-run (seed 10). The full log is in `seed_replacement_log.json` (archived on Zenodo).

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
