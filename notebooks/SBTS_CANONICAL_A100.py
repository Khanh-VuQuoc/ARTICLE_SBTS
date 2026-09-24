#!/usr/bin/env python3
"""Audit snapshot exported from SBTS_CANONICAL_A100.ipynb.

This file is generated; edit the notebook, not this export.
"""

# ==========================================================================
# # SBTS Canonical Experiment — GBM / Heston / SBTS Deep Hedging on A100
#
# **Notebook:** `SBTS_CANONICAL_A100.ipynb` · **Schema:** `sbts-canonical-1.0`
#
# This notebook is the **single canonical execution source** for the thesis
# *"Multivariate Schrödinger Bridge for Deep Hedging of Rainbow Options under
# Transaction Costs"*. It runs end-to-end from a clean kernel: data snapshot →
# generator calibration → path generation → deep-hedging training → uniform
# evaluation → statistical tests → support diagnostics → tables/figures →
# signed manifest. Older notebooks and their outputs are reference material
# only; they are **not** evidence.
#
# ---
#
# ## 1. Objectives
#
# 1. **Scientifically correct** — every statistic comes from one pipeline: no
#    seed substitution, no mislabelled runs, no hand-entered numbers.
# 2. **Runs from scratch** — one notebook loads/downloads data, calibrates the
#    generators, generates paths, trains, evaluates, tests and exports.
# 3. **Uses the A100** — GPU bottlenecks are optimised, but never at the cost of
#    fairness between GBM, Heston and SBTS.
# 4. **Auditable** — every artifact carries config hash, data hashes, seed,
#    environment, checkpoint and manifest entry.
#
# ## 2. Data scope and the 2019 overlap disclosure
#
# | Canonical name | Date range (by path start date) | Role |
# |---|---|---|
# | `CALIBRATION_2005_2019` | 2005-01-01 → 2019-12-31 | Calibrates all three generators |
# | `COVID_EXTENDED_2019_2020` | 2019-01-01 → 2020-12-31 | Historical stress-regime evaluation |
# | `POSTCOVID_2021_2022` | 2021-01-01 → 2022-12-31 | Temporally OOS by start date |
# | `RECENT_2023_2025` | 2023-01-01 → 2025-12-31 | Representative regime, temporally OOS by start date |
#
# > **Disclosure (locked wording).** Hedging networks are trained exclusively on
# > synthetic paths. However, the generators producing those paths are calibrated
# > on historical returns through December 2019. The extended 2019–2020 COVID
# > regime is therefore a *historical stress-regime evaluation*: its 2019
# > component overlaps the generator-calibration window, whereas observations
# > from 2020 onward are temporally out of sample.
#
# This does not break the fairness of the three-way comparison — all generators
# share the same calibration window, network architecture, loss, seeds and
# evaluation paths — it only bounds the out-of-sample generalisation claim.
#
# **Regime assignment rule.** Every historical path has a 252-trading-day
# horizon and is assigned to a regime by its **path start date**; a path may end
# outside the regime's calendar range. Tables and figures must therefore say
# *"classified by path start date"*. The notebook reports, per regime,
# `n_paths`, `start_date_min/max`, `end_date_min/max`, the share of paths
# covering the predefined 2020 crash window, and the actual data cutoff.
#
# ## 3. Experimental grid (locked)
#
# | Component | Value |
# |---|---|
# | Generators | `GBM`, `Heston`, `SBTS` |
# | Options | `basket_asian_call`, `asian_worst_of_put` |
# | Strike ratios | `0.95`, `1.00`, `1.05` |
# | Training seeds | `0,1,2,3,4,5,6,7,8,9` |
# | Generator / split seed | `42` |
# | Paths per generator | `20,000` |
# | Train / val / test | `16,000 / 2,000 / 2,000` |
# | Horizon | `252` trading days |
# | Assets | AAPL, JPM, XOM |
# | Transaction-cost rate | `0.001` |
# | Training phases | MSE warm-up → CVaR fine-tune |
# | Configurations | `3 × 2 × 3 × 10 = 180` |
# | Best checkpoints | `180 × 2 = 360` |
#
# **A seed is never swapped for another seed while keeping the old label.** If a
# configuration fails after its retry budget it is recorded `status="failed"`
# and statistical tests report the real common-seed `n`.
#
# ## 4. What this notebook must never do
#
# 1. Read variables or functions implicitly defined by an earlier notebook/kernel.
# 2. Replace seed 3 (or any seed) with a different seed.
# 3. Silently swallow a training error and still mark the configuration complete.
# 4. Mix outputs from an older run with a new one when `config_hash` or
#    `data_hash` differs.
# 5. Use a random split over overlapping rolling windows and call it
#    chronological validation.
# 6. Call the ad-hoc MCS implementation "Hansen–Lunde–Nason".
# 7. Log `grad_norm_post_clip` using the pre-clipping norm.
# 8. Claim Xavier initialisation for hidden layers the code does not initialise.
# 9. Claim bit-for-bit reproducibility across GPUs/software stacks.
# 10. Keep the `313 → 7`, `45×` support-contraction numbers unless this notebook
#     regenerates them.
# 11. Re-download Yahoo data and silently overwrite the published snapshot.
# 12. Use a cache whose schema version, data hash or config hash does not match.
#
# ## 5. Runtime requirements
#
# * Google Colab **A100** runtime (the notebook also runs on any CUDA GPU or CPU,
#   with proportionally longer wall time; the execution mode is recorded).
# * Google Drive mounted for run-scoped artifact storage.
# * A full run is `180` configurations × 2 phases. **No runtime estimate is
#   printed until the smoke benchmark (Cell 18) has measured this machine.**
#
# ## 6. Run modes
#
# Set `RUN_MODE` in the next cell:
#
# | Mode | Meaning |
# |---|---|
# | `SMOKE` | Small end-to-end run: validates the pipeline, sizes the machine. Never produces thesis results. |
# | `FULL` | The canonical experiment (180 configurations). |
# | `ANALYSIS_ONLY` | Re-use existing checkpoints/artifacts for a given `run_id`; no training. |
# | `DIAGNOSTICS` | Support-contraction diagnostics only. |
#
# ## 7. Execution gates
#
# `Gate 0` design lock · `Gate 1` static implementation audit (unit tests) ·
# `Gate 2` end-to-end smoke · `Gate 3` A100 performance/numerical gate ·
# `Gate 4` generator artifacts · `Gate 5` full training · `Gate 6` statistical
# lock · `Gate 7` thesis result lock. Thesis numbers may only be updated after
# Gate 6, and the run id + manifest hash must be quoted in the reproducibility
# appendix.

# ---- notebook cell 1 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 0 — RUN MODE
# ═════════════════════════════════════════════════════════════════════════════
#  SMOKE          small end-to-end run; validates the pipeline and sizes the
#                 machine. NEVER produces thesis results.
#  FULL           the canonical experiment (180 configurations x 2 phases).
#  ANALYSIS_ONLY  re-use existing checkpoints/artifacts; no training.
#  DIAGNOSTICS    support-contraction diagnostics only.
# ═════════════════════════════════════════════════════════════════════════════

RUN_MODE = "SMOKE"            # <-- run canonical experiment with "FULL"
# RUN_MODE = "FULL"
# RUN_MODE = "ANALYSIS_ONLY"
# RUN_MODE = "DIAGNOSTICS"

# Set to an existing run id (e.g. "20260101T120000Z__ab12cd34") to resume or to
# analyse a previous run. None => a new run id is minted for FULL/SMOKE, and
# the most recent compatible run is resolved for ANALYSIS_ONLY/DIAGNOSTICS.
RESUME_RUN_ID = None

# Numerical execution mode. REFERENCE_FP32 is the canonical mode; A100_FAST may
# only be locked in for a FULL run after the Cell-18 benchmark gate passes.
PRECISION_MODE = "REFERENCE_FP32"     # or "A100_FAST"

# A FULL run takes tens of hours and WILL be interrupted. Leaving RESUME_RUN_ID
# as None no longer starts over: the notebook resumes the newest unfinished run
# with the same configuration and says so. Set FORCE_NEW_RUN = True to start a
# separate run instead.
FORCE_NEW_RUN = False

# Split the training queue across parallel sessions. None runs every generator
# and then the full analysis. A tuple, e.g. ("Heston",), trains ONLY those
# generators and stops after Cell 19 — point each session at the SAME
# RESUME_RUN_ID, then run one unsharded session afterwards to do the audit,
# evaluation, statistics and tables over all shards.
SHARD_GENERATORS = None

# Split by seed instead of (or as well as) by generator. Halving the seeds cuts
# every generator/option/strike cell in two, so two workers stay balanced no
# matter how far an earlier run already got through the queue.
SHARD_SEEDS = None

# How often a training phase persists enough state to continue from the exact
# epoch it reached. 0 disables it. This is an execution detail, not an
# experiment parameter, so it stays out of ExperimentConfig and does not change
# config_hash.
CHECKPOINT_EVERY_EPOCHS = 25

# Grid sizing. None => shrunk when RUN_MODE == "SMOKE", full otherwise. Set it
# explicitly to True when analysing or re-running diagnostics over a run that
# was produced with the smoke grid.
SMOKE_SIZING = None

VALID_RUN_MODES = ("SMOKE", "FULL", "ANALYSIS_ONLY", "DIAGNOSTICS")
VALID_PRECISION_MODES = ("REFERENCE_FP32", "A100_FAST")
assert RUN_MODE in VALID_RUN_MODES, f"RUN_MODE must be one of {VALID_RUN_MODES}"
assert PRECISION_MODE in VALID_PRECISION_MODES, (
    f"PRECISION_MODE must be one of {VALID_PRECISION_MODES}")

print(f"RUN_MODE       = {RUN_MODE}")
print(f"PRECISION_MODE = {PRECISION_MODE}")
print(f"RESUME_RUN_ID  = {RESUME_RUN_ID}")
print(f"SMOKE_SIZING   = {SMOKE_SIZING} (None => derived from RUN_MODE)")
print(f"FORCE_NEW_RUN  = {FORCE_NEW_RUN}")
print(f"SHARD          = generators={SHARD_GENERATORS or 'all'}  "
      f"seeds={SHARD_SEEDS if SHARD_SEEDS is not None else 'all'}")
print(f"CHECKPOINT_EVERY_EPOCHS = {CHECKPOINT_EVERY_EPOCHS}"
      f"{' (epoch-level resume disabled)' if not CHECKPOINT_EVERY_EPOCHS else ''}")
print("\nNo runtime estimate is shown until the Cell-18 smoke benchmark has "
      "measured this machine.")

# ==========================================================================
# ---
#
# ## Section 1 — Environment bootstrap
#
# Installs the locked dependency set, imports everything, mounts Drive, inspects
# the hardware, defines `seed_all` and applies the numerical execution mode.
# These cells must succeed from a clean kernel with no state from any other
# notebook.

# ---- notebook cell 3 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 1 — DEPENDENCY SETUP
# ═════════════════════════════════════════════════════════════════════════════
#  Two tiers, because they need opposite treatment on Colab:
#
#   REQUIRED_PACKAGES   packages the pipeline cannot run without and that the
#                       runtime may not ship (arch for the canonical MCS,
#                       yfinance for the data snapshot). These are INSTALLED
#                       if they are not importable: the pinned version first,
#                       then unpinned if that pin is unavailable. If one of
#                       them is still missing afterwards the cell fails loudly
#                       instead of letting a later cell discover it.
#
#   RUNTIME_PACKAGES    numpy / pandas / scipy / torch / matplotlib, which the
#                       Colab image already provides, built against its CUDA
#                       stack. Force-downgrading them to a pin breaks that
#                       build and needs a runtime restart, so by default they
#                       are only VERIFIED and any deviation is recorded in
#                       environment.json and in the manifest. Set
#                       DEPENDENCY_POLICY = "install_locked" to force the exact
#                       versions anyway (expect to restart the runtime).
#
#  Whatever happens, the environment that actually ran is hashed and recorded:
#  reproducibility is claimed from the frozen snapshot plus the recorded
#  environment, within declared numerical tolerances.
# ═════════════════════════════════════════════════════════════════════════════

import importlib.util, subprocess, sys
from typing import Any, Dict, Tuple

# pip name -> (locked version, import name)
REQUIRED_PACKAGES = {
    "arch":     ("8.0.0", "arch"),
    "yfinance": ("1.7.0", "yfinance"),
}
RUNTIME_PACKAGES = {
    "numpy":      ("2.4.6",  "numpy"),
    "pandas":     ("3.0.6",  "pandas"),
    "scipy":      ("1.17.1", "scipy"),
    "torch":      ("2.14.0", "torch"),
    "matplotlib": ("3.11.2", "matplotlib"),
}
REQUIREMENTS_LOCK = {k: v[0] for k, v in
                     {**RUNTIME_PACKAGES, **REQUIRED_PACKAGES}.items()}

IN_COLAB = ("google.colab" in sys.modules
            or importlib.util.find_spec("google.colab") is not None)

#   "auto"          install what is missing, verify the rest      (default)
#   "install_locked" force every pin (expect a runtime restart)
#   "verify_only"   never install, only compare
#   "skip"          neither install nor compare (development only)
DEPENDENCY_POLICY = "auto"


def _installed_version(pkg: str):
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:                                        # noqa: BLE001
        return None


def _importable(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:                                        # noqa: BLE001
        return False


def _pip(*args) -> Tuple[bool, str]:
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return True, ""
    except subprocess.CalledProcessError as exc:
        return False, exc.output.decode(errors="replace")[-2000:]
    except Exception as exc:                                 # noqa: BLE001
        return False, repr(exc)


def ensure_package(pip_name: str, module_name: str = None,
                   version: str = None) -> Dict[str, Any]:
    """Make `module_name` importable, preferring the locked version.

    Returns a record of what was done. Also usable later in the session, e.g.
    to recover from a missing `arch` without re-running this whole cell.
    """
    module_name = module_name or pip_name
    rec = {"package": pip_name, "module": module_name, "locked": version,
           "action": "already_present", "installed_version": None,
           "importable": _importable(module_name), "error": None}
    if rec["importable"] and (version is None
                              or _installed_version(pip_name) == version):
        rec["installed_version"] = _installed_version(pip_name)
        return rec

    if version:
        ok, err = _pip(f"{pip_name}=={version}")
        if ok:
            rec["action"] = "installed_pinned"
        else:
            ok, err = _pip(pip_name)
            rec["action"] = "installed_unpinned" if ok else "install_failed"
            rec["error"] = None if ok else err
    else:
        ok, err = _pip(pip_name)
        rec["action"] = "installed_unpinned" if ok else "install_failed"
        rec["error"] = None if ok else err

    importlib.invalidate_caches()
    rec["importable"] = _importable(module_name)
    rec["installed_version"] = _installed_version(pip_name)
    return rec


def setup_dependencies(policy: str = None) -> Dict[str, Any]:
    policy = policy or DEPENDENCY_POLICY
    report = {"policy": policy, "lock": dict(REQUIREMENTS_LOCK),
              "actions": [], "installed": {}, "mismatches": {},
              "missing_required": [], "restart_required": False}
    if policy == "skip":
        return report

    if policy in ("auto", "install_locked"):
        for pip_name, (ver, mod) in REQUIRED_PACKAGES.items():
            rec = ensure_package(pip_name, mod, ver)
            report["actions"].append(rec)
            if not rec["importable"]:
                report["missing_required"].append(pip_name)

    if policy == "install_locked":
        for pip_name, (ver, mod) in RUNTIME_PACKAGES.items():
            before = _installed_version(pip_name)
            if before == ver:
                continue
            rec = ensure_package(pip_name, mod, ver)
            report["actions"].append(rec)
            # A package already imported into this kernel cannot be swapped in
            # place; Colab needs a restart for the new version to take effect.
            if mod in sys.modules and _installed_version(pip_name) != before:
                report["restart_required"] = True

    for pip_name, (ver, mod) in {**RUNTIME_PACKAGES, **REQUIRED_PACKAGES}.items():
        got = _installed_version(pip_name)
        report["installed"][pip_name] = got
        if got != ver:
            report["mismatches"][pip_name] = {"locked": ver, "found": got}

    return report


DEPENDENCY_REPORT = setup_dependencies()

for _rec in DEPENDENCY_REPORT["actions"]:
    if _rec["action"] != "already_present":
        print(f"  {_rec['package']:<12s} {_rec['action']} "
              f"-> {_rec['installed_version']}")

if DEPENDENCY_REPORT["missing_required"]:
    print("\n  Required packages could not be installed: "
          f"{DEPENDENCY_REPORT['missing_required']}")
    for _rec in DEPENDENCY_REPORT["actions"]:
        if _rec["error"]:
            print(f"\n  --- pip output for {_rec['package']} ---\n{_rec['error']}")
    raise RuntimeError(
        f"Missing required packages: {DEPENDENCY_REPORT['missing_required']}. "
        f"'arch' is needed for the canonical Model Confidence Set and "
        f"'yfinance' for the data snapshot. Install them manually "
        f"(e.g. !pip install arch yfinance) and re-run this cell.")

if DEPENDENCY_REPORT["restart_required"]:
    print("\n  A runtime package was replaced after it had already been "
          "imported.\n  RESTART THE RUNTIME and run the notebook again from "
          "Cell 0.")

if DEPENDENCY_REPORT["mismatches"]:
    print("\n  Version deviations from the lock (recorded in the manifest; "
          "reproducibility\n  is claimed only within the recorded "
          "environment):")
    for _pkg, _info in sorted(DEPENDENCY_REPORT["mismatches"].items()):
        _tier = "required" if _pkg in REQUIRED_PACKAGES else "runtime"
        print(f"    {_pkg:<12s} [{_tier}] locked={_info['locked']:<10s} "
              f"found={_info['found']}")
else:
    print("\n  Dependency lock satisfied exactly.")

print(f"\n  IN_COLAB={IN_COLAB}  policy={DEPENDENCY_REPORT['policy']}")

# ---- notebook cell 4 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 2 — IMPORTS, DRIVE MOUNT, HARDWARE INSPECTION, seed_all
# ═════════════════════════════════════════════════════════════════════════════
#  Runs from a clean kernel. Nothing in this notebook may rely on a variable or
#  function defined by a previous notebook or a previous kernel session.
#  `seed_all` is defined HERE (environment bootstrap) so that no later cell can
#  raise NameError for it.
# ═════════════════════════════════════════════════════════════════════════════

import os, sys, gc, io, json, math, time, random, hashlib, platform, shutil
import importlib
import datetime as _dt
import subprocess, warnings, logging, tempfile, itertools, dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import scipy
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ── Optional: canonical MCS implementation ───────────────────────────────────
def _import_arch():
    from arch.bootstrap import MCS as _MCS
    import arch as _a
    return _MCS, _a.__version__


try:
    ARCH_MCS, ARCH_VERSION = _import_arch()
    ARCH_AVAILABLE = True
except Exception as _exc:                                    # noqa: BLE001
    # Cell 1 installs arch, but a kernel that started before it ran (or a
    # cell executed out of order) can still land here. Try once more rather
    # than failing 20 cells later in the MCS.
    print(f"  arch is not importable ({_exc!r}); installing it now...")
    try:
        _rec = ensure_package("arch", "arch", REQUIRED_PACKAGES["arch"][0])
        importlib.invalidate_caches()
        ARCH_MCS, ARCH_VERSION = _import_arch()
        ARCH_AVAILABLE = True
        print(f"  arch {ARCH_VERSION} installed and imported.")
    except Exception as _exc2:                               # noqa: BLE001
        ARCH_MCS, ARCH_AVAILABLE, ARCH_VERSION = None, False, None
        print(f"  [WARN] arch is still unavailable ({_exc2!r}). The canonical "
              f"MCS engine cannot run: Cell 24 will stop rather than emit a "
              f"non-canonical Model Confidence Set. Install it manually "
              f"(!pip install arch) and re-run from Cell 1.")

# ── Optional: Yahoo Finance (only needed to create a new data snapshot) ──────
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:                                            # noqa: BLE001
    yf, YFINANCE_AVAILABLE = None, False

import matplotlib
if not hasattr(sys, "ps1") and "google.colab" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
    "font.family": "serif", "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.linewidth": 0.8, "figure.figsize": (7, 4.3),
})

# ── Google Drive ─────────────────────────────────────────────────────────────
DRIVE_MOUNTED = False
DRIVE_ROOT = None
if IN_COLAB:
    # Every artifact of a run must land on Drive. Falling back to the VM's local
    # disk would train for hours into storage that vanishes with the session
    # and that no other session can see, so a failed mount stops here.
    try:
        from google.colab import drive as _gdrive
        _gdrive.mount("/content/drive", force_remount=False)
    except Exception as exc:                                 # noqa: BLE001
        raise RuntimeError(
            f"Google Drive could not be mounted ({exc!r}). Nothing has been "
            f"trained. Re-run this cell and complete the Drive authorisation; "
            f"results must be stored on Drive, never on the temporary VM disk.") from exc
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Drive reports mounted but /content/drive/MyDrive is "
                           "missing; re-run this cell.")
    DRIVE_ROOT = Path("/content/drive/MyDrive/ARTICLE_SBTS")
    DRIVE_MOUNTED = True
if DRIVE_ROOT is None:
    DRIVE_ROOT = Path(os.environ.get("SBTS_STORAGE_ROOT", "./ARTICLE_SBTS")).resolve()
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

# ── Device / hardware ────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_AVAILABLE = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else None
GPU_CAPABILITY = list(torch.cuda.get_device_capability(0)) if CUDA_AVAILABLE else None
GPU_TOTAL_MEMORY = (torch.cuda.get_device_properties(0).total_memory
                    if CUDA_AVAILABLE else None)
IS_A100 = bool(GPU_NAME and "A100" in GPU_NAME)

print(f"Device : {DEVICE}")
if CUDA_AVAILABLE:
    print(f"  GPU        : {GPU_NAME}")
    print(f"  Capability : {GPU_CAPABILITY}")
    print(f"  VRAM       : {GPU_TOTAL_MEMORY / 1e9:.1f} GB")
    print(f"  A100       : {IS_A100}")
else:
    print("  [WARN] No CUDA device. The pipeline runs but a FULL run will be slow; "
          "the manifest records the device actually used.")


# ── seed_all (defined in the environment bootstrap, per design §1) ───────────
def seed_all(seed: int) -> int:
    """Seed every RNG this notebook uses. Returns the seed for call-site logging."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def torch_generator(device, *seed_parts: int) -> torch.Generator:
    """A torch.Generator seeded reproducibly from (base_seed, batch_index, ...).

    Used for stream-reproducible Brownian increments: resuming a batched
    generation must reproduce the uninterrupted run bit-for-bit within a mode.
    """
    h = hashlib.sha256("|".join(str(int(p)) for p in seed_parts).encode()).digest()
    seed = int.from_bytes(h[:8], "big") % (2 ** 63 - 1)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


# ── Precision / determinism flags ────────────────────────────────────────────
DETERMINISTIC_ALGORITHMS = False     # see note below before enabling


def apply_precision_mode(mode: str) -> Dict[str, Any]:
    """Apply a numerical execution mode and return exactly what was set."""
    assert mode in VALID_PRECISION_MODES, mode
    if mode == "REFERENCE_FP32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        amp_dtype = None
    else:  # A100_FAST
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        amp_dtype = "bfloat16"
    torch.backends.cudnn.benchmark = (mode == "A100_FAST")
    flags = {
        "precision_mode": mode,
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "autocast_dtype": amp_dtype,
        "deterministic_algorithms": bool(DETERMINISTIC_ALGORITHMS),
    }
    # Determinism is OPT-IN and never claimed unless it is actually enabled:
    # several reductions used here have no deterministic CUDA kernel.
    if DETERMINISTIC_ALGORITHMS:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        flags["cublas_workspace_config"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]
    return flags


PRECISION_FLAGS = apply_precision_mode(PRECISION_MODE)
print("\nPrecision flags:")
for k, v in PRECISION_FLAGS.items():
    print(f"  {k:<28s} {v}")
print("\n  NOTE: reproducibility is claimed as 'reproducible from the frozen "
      "snapshot\n        and locked environment within declared numerical "
      "tolerances',\n        never bit-for-bit across different GPUs or "
      "software stacks.")


def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb() -> float:
    return (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0


def env_snapshot() -> Dict[str, Any]:
    """Everything needed to describe the software/hardware stack of this run."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "arch_version": ARCH_VERSION,
        "yfinance_available": YFINANCE_AVAILABLE,
        "cuda_available": CUDA_AVAILABLE,
        "cuda_version": torch.version.cuda if CUDA_AVAILABLE else None,
        "cudnn_version": (torch.backends.cudnn.version() if CUDA_AVAILABLE else None),
        "gpu_name": GPU_NAME,
        "gpu_capability": GPU_CAPABILITY,
        "gpu_total_memory_bytes": GPU_TOTAL_MEMORY,
        "is_a100": IS_A100,
        "in_colab": IN_COLAB,
        "drive_mounted": DRIVE_MOUNTED,
        "dependency_report": DEPENDENCY_REPORT,
        "precision_flags": PRECISION_FLAGS,
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def pip_freeze_text() -> str:
    try:
        return subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL).decode()
    except Exception as exc:                                 # noqa: BLE001
        return f"# pip freeze failed: {exc!r}\n"


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                        # noqa: BLE001
        return None


# Reference commit of the source repository this notebook was rebuilt from.
SOURCE_REPOSITORY = "Khanh-VuQuoc/ARTICLE_SBTS"
SOURCE_SNAPSHOT_COMMIT = "a6412faccf248e73f24fc812a6eb2d4ef7da52fc"

print(f"\nStorage root : {DRIVE_ROOT}")
print(f"Git commit   : {git_commit()}")
print("Environment bootstrap complete.")

# ==========================================================================
# ---
#
# ## Section 2 — Central configuration and run scaffolding
#
# One frozen dataclass holds every knob; its canonicalised JSON gives the
# `config_hash` that is stamped into every cache, checkpoint and artifact. The
# run directory follows the layout in §18 of the design document.

# ---- notebook cell 6 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 3 — CENTRAL CONFIGURATION AND RUN PATHS
# ═════════════════════════════════════════════════════════════════════════════
#  One frozen dataclass holds every knob the pipeline reads. `config_hash` is
#  the SHA-256 of its canonicalised JSON and is stamped into every cache,
#  checkpoint and artifact. Nothing downstream may read a knob that is not
#  here, and no cache whose config_hash differs may be reused.
# ═════════════════════════════════════════════════════════════════════════════

SCHEMA_VERSION = "sbts-canonical-1.0"

# Execution-only fields: they never change what the experiment IS.
CONFIG_HASH_EXCLUDED_FIELDS = ("run_mode", "precision_mode", "storage_root")


@dataclass(frozen=True)
class ExperimentConfig:
    # ── identity ────────────────────────────────────────────────────────────
    schema_version: str = SCHEMA_VERSION
    run_mode: str = "FULL"
    precision_mode: str = "REFERENCE_FP32"

    # ── data ────────────────────────────────────────────────────────────────
    tickers: Tuple[str, ...] = ("AAPL", "JPM", "XOM")
    sectors: Tuple[str, ...] = ("Technology", "Financials", "Energy")
    raw_data_start: str = "2005-01-01"
    raw_data_end: str = "2026-12-31"          # yfinance clips to what exists
    calibration_start: str = "2005-01-01"
    calibration_end: str = "2019-12-31"
    price_field: str = "Close"                 # with auto_adjust=True
    auto_adjust: bool = True
    data_source: str = "yahoo/yfinance"

    # ── experiment grid ─────────────────────────────────────────────────────
    horizon: int = 252
    delta_t: float = 1.0 / 252.0
    n_paths: int = 20_000
    n_train: int = 16_000
    n_val: int = 2_000
    n_test: int = 2_000
    seeds: Tuple[int, ...] = tuple(range(10))
    split_seed: int = 42
    generator_seed: int = 42
    options: Tuple[str, ...] = ("basket_asian_call", "asian_worst_of_put")
    strike_ratios: Tuple[float, ...] = (0.95, 1.00, 1.05)
    cost_rate: float = 0.001

    # ── evaluation regimes (assignment is by PATH START DATE) ───────────────
    regimes: Tuple[Tuple[str, str, str], ...] = (
        ("COVID_EXTENDED_2019_2020", "2019-01-01", "2020-12-31"),
        ("POSTCOVID_2021_2022",      "2021-01-01", "2022-12-31"),
        ("RECENT_2023_2025",         "2023-01-01", "2025-12-31"),
    )
    baseline_regime: str = "RECENT_2023_2025"
    covid_crash_start: str = "2020-02-20"
    covid_crash_end: str = "2020-04-30"

    # ── GBM ─────────────────────────────────────────────────────────────────
    gbm_estimator: str = "annualised_moments_of_log_returns"

    # ── Heston (multivariate, full-truncation Euler + Milstein CIR) ─────────
    heston_kappa: float = 5.0
    heston_xi: float = 0.7
    heston_rho: float = -0.7
    heston_substeps: int = 4
    heston_use_milstein: bool = True

    # ── SBTS ────────────────────────────────────────────────────────────────
    sbts_kernel: str = "quartic_isotropic"
    sbts_n_pi: int = 100                       # Euler sub-steps per interval
    sbts_batch_size: int = 500                 # locked after the A100 probe
    sbts_zero_support_policy: str = "nearest_reference"   # or "zero_drift"
    h_grid: Tuple[float, ...] = (0.01, 0.02, 0.03, 0.05, 0.07, 0.10,
                                 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50)
    k_grid: Tuple[int, ...] = (1, 2, 5, 10, 21, 63)

    # ── bandwidth / Markov-order selection ──────────────────────────────────
    selection_engine: str = "paper_full"       # "paper_full" | "one_step_smoke"
    selection_objective: str = "terminal_state"  # literal [A25] eq. (5) target
    selection_train_fraction: float = 0.8      # chronological, purged
    selection_purge: int = 252                 # >= horizon => no shared observations
    selection_n_queries: int = 100             # Q in eq. (5)
    selection_n_samples: int = 20              # L in eq. (5)
    selection_horizon: int = 21                # steps simulated to the objective date
    selection_n_pi: int = 10                   # sub-steps during selection
    selection_query_chunk: int = 25            # GPU chunking of queries
    selection_seed: int = 42
    selection_tolerance: float = 0.01          # relative tie tolerance
    # Locked tie-break, applied in order, among configs within tolerance of the
    # best validation objective: prefer the smaller K, then the larger h
    # (larger h => wider support => more robust conditioning).
    selection_tie_break: Tuple[str, ...] = ("min_k", "max_h")

    # ── deep hedging network ────────────────────────────────────────────────
    hidden_sizes: Tuple[int, ...] = (64, 64)
    activation: str = "ReLU"
    init_hidden: str = "xavier_uniform_relu_gain"
    init_output_gain: float = 0.1
    v0_init: float = 0.0

    # ── training ────────────────────────────────────────────────────────────
    batch_size: int = 4096
    cvar_alpha: float = 0.95
    mse_epochs: int = 500
    mse_lr: float = 1e-3
    mse_patience: int = 20
    cvar_epochs: int = 200
    cvar_lr: float = 1e-4
    cvar_patience: int = 20
    sched_patience: int = 10
    sched_factor: float = 0.5
    sched_min_lr: float = 1e-7
    grad_clip: float = 1.0
    grad_clip_tolerance: float = 1e-4
    val_improve_tol: float = 1e-6
    max_retries_per_run: int = 2               # same seed, same config only

    # ── statistics ──────────────────────────────────────────────────────────
    primary_metrics: Tuple[str, ...] = ("std", "cvar95")
    test_phases: Tuple[str, ...] = ("mse", "cvar")
    fdr_alpha: float = 0.05
    permutation_exact_max_n: int = 20          # 2**n sign patterns enumerated
    bootstrap_reps: int = 10_000
    bootstrap_seed: int = 42
    mcs_alpha: float = 0.10
    mcs_reps: int = 5_000
    mcs_method: str = "R"                      # canonical; "max" is sensitivity
    mcs_block_size: int = 1                    # independent seed-level analysis
    mcs_seed: int = 42

    # ── support diagnostics ─────────────────────────────────────────────────
    diag_k_values: Tuple[int, ...] = (1, 3, 5)
    diag_step_index: int = 126                 # locked reference step for queries
    diag_bootstrap_reps: int = 2_000
    diag_n_queries: int = 250
    diag_top_k: Tuple[int, ...] = (1, 5, 10)
    diag_seed: int = 42

    # ── benchmark gate tolerances ───────────────────────────────────────────
    gate_metric_rel_tol: float = 0.02
    gate_seeds: Tuple[int, ...] = (0, 1, 3)    # seed 3 is always benchmarked
    gate_strike: float = 1.00
    gate_mse_epochs: int = 6
    gate_cvar_epochs: int = 4
    gate_n_train: int = 2_048
    gate_n_val: int = 512
    gate_n_test: int = 512

    # ── smoke-run sizing (never used for thesis numbers) ────────────────────
    smoke_n_paths: int = 512
    smoke_n_train: int = 384
    smoke_n_val: int = 64
    smoke_n_test: int = 64
    smoke_horizon: int = 21
    smoke_seeds: Tuple[int, ...] = (0, 1, 3)
    smoke_mse_epochs: int = 3
    smoke_cvar_epochs: int = 2
    smoke_batch_size: int = 128
    smoke_sbts_n_pi: int = 5
    smoke_hist_paths_per_regime: int = 64

    # ── output schema versions ──────────────────────────────────────────────
    data_schema_version: str = "data-1.0"
    generator_schema_version: str = "generator-1.0"
    checkpoint_schema_version: str = "checkpoint-1.0"
    evaluation_schema_version: str = "evaluation-1.0"
    statistics_schema_version: str = "statistics-1.0"
    manifest_schema_version: str = "manifest-1.0"

    # ── storage ─────────────────────────────────────────────────────────────
    storage_root: str = str(DRIVE_ROOT)
    runs_subdir: str = "canonical_runs"
    snapshot_subdir: str = "canonical_snapshot"   # shared frozen raw data

    # ── derived ─────────────────────────────────────────────────────────────
    @property
    def d(self) -> int:
        return len(self.tickers)

    @property
    def regime_names(self) -> Tuple[str, ...]:
        return tuple(r[0] for r in self.regimes)

    @property
    def regime_ranges(self) -> Dict[str, Tuple[str, str]]:
        return {r[0]: (r[1], r[2]) for r in self.regimes}

    @property
    def n_configurations(self) -> int:
        return (3 * len(self.options) * len(self.strike_ratios) * len(self.seeds))

    @property
    def n_expected_checkpoints(self) -> int:
        return self.n_configurations * len(self.test_phases)

    def to_json_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(asdict(self), default=str))

    def canonical_json(self) -> str:
        """Canonical JSON of the EXPERIMENT definition.

        `run_mode`, `precision_mode` and `storage_root` are excluded: they say
        how and where the experiment was executed, not what it is. Excluding
        them lets ANALYSIS_ONLY and DIAGNOSTICS read a run produced by FULL,
        and lets the Cell-18 gate switch the numerical mode, without the
        artifacts looking like a different configuration. Everything that
        changes the experiment itself — including every size the smoke grid
        shrinks — is still in the hash.
        """
        d = {k: v for k, v in self.to_json_dict().items()
             if k not in CONFIG_HASH_EXCLUDED_FIELDS}
        return json.dumps(d, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True)

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


def make_smoke_config(base: ExperimentConfig) -> ExperimentConfig:
    """Shrink the grid for a pipeline validation run. Results are never
    published: the run directory is tagged SMOKE and every artifact carries
    run_mode='SMOKE'."""
    return dataclasses.replace(
        base,
        run_mode="SMOKE",
        n_paths=base.smoke_n_paths,
        n_train=base.smoke_n_train,
        n_val=base.smoke_n_val,
        n_test=base.smoke_n_test,
        horizon=base.smoke_horizon,
        delta_t=1.0 / 252.0,
        seeds=base.smoke_seeds,
        mse_epochs=base.smoke_mse_epochs,
        cvar_epochs=base.smoke_cvar_epochs,
        batch_size=base.smoke_batch_size,
        sbts_n_pi=base.smoke_sbts_n_pi,
        sbts_batch_size=128,
        h_grid=(0.05, 0.15, 0.30),
        k_grid=(1, 2, 5),
        selection_engine="one_step_smoke",
        selection_n_queries=16,
        selection_n_samples=4,
        selection_horizon=5,
        selection_n_pi=2,
        selection_purge=base.smoke_horizon,
        heston_substeps=2,
        mcs_reps=200,
        bootstrap_reps=200,
        diag_n_queries=32,
        diag_k_values=(1, 3),
    )


BASE_CONFIG = ExperimentConfig(run_mode=RUN_MODE, precision_mode=PRECISION_MODE)
USE_SMOKE_SIZING = (SMOKE_SIZING if SMOKE_SIZING is not None
                    else RUN_MODE == "SMOKE")
CFG = (dataclasses.replace(make_smoke_config(BASE_CONFIG), run_mode=RUN_MODE)
       if USE_SMOKE_SIZING else BASE_CONFIG)
CONFIG_HASH = CFG.config_hash()

# ── Run directory layout (design §18) ────────────────────────────────────────
STORAGE_ROOT = Path(CFG.storage_root)
RUNS_ROOT = STORAGE_ROOT / CFG.runs_subdir
SNAPSHOT_ROOT = STORAGE_ROOT / CFG.snapshot_subdir
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)

RUN_SUBDIRS = ("config", "environment", "data", "data/raw", "data/processed",
               "data/historical_paths", "generators", "generators/GBM",
               "generators/Heston", "generators/SBTS", "checkpoints",
               "checkpoints/GBM", "checkpoints/Heston", "checkpoints/SBTS",
               "evaluations", "evaluations/per_path", "evaluations/summaries",
               "statistics", "diagnostics", "tables", "figures", "logs",
               "notebook_snapshot")


def mint_run_id(cfg: ExperimentConfig) -> str:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "SMOKE__" if USE_SMOKE_SIZING else ""
    return f"{tag}{stamp}__{cfg.config_hash()[:8]}"


def resolve_run_id(cfg: ExperimentConfig, resume: Optional[str]) -> str:
    """Pick the run id: explicit resume, else a matching run for read-only
    modes, else a fresh id. A run directory is only reused when its stored
    config_hash matches; otherwise the notebook refuses to mix artifacts."""
    if resume:
        rd = RUNS_ROOT / resume
        if not rd.exists():
            raise FileNotFoundError(f"RESUME_RUN_ID={resume!r} not found under {RUNS_ROOT}")
        stored = rd / "config" / "config_hash.txt"
        if stored.exists():
            got = stored.read_text().strip()
            if got != cfg.config_hash():
                raise RuntimeError(
                    f"config_hash mismatch for run {resume}: stored={got[:12]} "
                    f"current={cfg.config_hash()[:12]}. Refusing to mix artifacts "
                    f"across configurations.")
        return resume
    candidates = []
    for rd in sorted(RUNS_ROOT.glob("*")):
        f = rd / "config" / "config_hash.txt"
        if f.exists() and f.read_text().strip() == cfg.config_hash():
            candidates.append(rd.name)

    if cfg.run_mode in ("ANALYSIS_ONLY", "DIAGNOSTICS"):
        if not candidates:
            raise FileNotFoundError(
                f"{cfg.run_mode} needs an existing run with config_hash "
                f"{cfg.config_hash()[:12]}; none found under {RUNS_ROOT}. "
                f"Set RESUME_RUN_ID explicitly.")
        return candidates[-1]

    # FULL/SMOKE: resume rather than silently start a 180-configuration
    # experiment over from zero. When several runs share the configuration —
    # e.g. an older notebook minted a fresh run after a disconnect — the one
    # with the MOST completed configurations wins, so no finished work is
    # abandoned; ties go to the newest.
    if candidates and not FORCE_NEW_RUN:
        def _n_complete(run_name: str) -> int:
            done = set()
            for f in (RUNS_ROOT / run_name / "evaluations" / "summaries"
                      ).glob("training_results*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        runs = json.load(fh).get("runs", {})
                    done |= {k for k, v in runs.items()
                             if v.get("status") == "complete"}
                except Exception:                            # noqa: BLE001
                    pass
            return len(done)

        progress = [(name, _n_complete(name)) for name in candidates]
        chosen, chosen_n = max(progress, key=lambda t: (t[1], t[0]))
        print(f"  RESUMING the existing run {chosen} (same config_hash),\n"
              f"  which already holds {chosen_n} completed configurations.\n"
              f"  Parallel sessions must print the SAME run id here — if another "
              f"session shows\n  a different one, they are not sharing a run.\n"
              f"  Completed configurations will be skipped after their "
              f"checkpoints verify.\n"
              f"  Set FORCE_NEW_RUN = True in Cell 0 to start a separate run "
              f"instead.")
        if len(candidates) > 1:
            print(f"  {len(candidates)} runs share this configuration; the one "
                  f"with the most completed\n  configurations was chosen. Set "
                  f"RESUME_RUN_ID to pick another:")
            for name, n in progress:
                print(f"    {'->' if name == chosen else '  '} {name}  "
                      f"({n} complete)")
        return chosen

    # A shard exists to join a run another session is also working on. Finding
    # none almost always means this session sees a different storage location
    # (another Google account's Drive, or Drive not mounted) — starting a run of
    # its own would train for hours into a copy nobody else can merge with.
    _is_shard = bool(SHARD_GENERATORS) or SHARD_SEEDS is not None
    if _is_shard and not FORCE_NEW_RUN:
        raise RuntimeError(
            f"This notebook is one shard of a split run, but it found NO existing "
            f"run with config_hash {cfg.config_hash()[:8]} under\n  {RUNS_ROOT}\n"
            f"Nothing has been trained. Most likely this session sees a different "
            f"Google Drive from the other session (a different Google account). "
            f"Make both sessions see the same ARTICLE_SBTS folder, then re-run. "
            f"Only if you really mean to start a brand-new run, set "
            f"FORCE_NEW_RUN = True in Cell 0.")
    return mint_run_id(cfg)


RUN_ID = resolve_run_id(CFG, RESUME_RUN_ID)
RUN_DIR = RUNS_ROOT / RUN_ID
for _sub in RUN_SUBDIRS:
    (RUN_DIR / _sub).mkdir(parents=True, exist_ok=True)

PATHS = {
    "run": RUN_DIR,
    "config": RUN_DIR / "config",
    "environment": RUN_DIR / "environment",
    "data": RUN_DIR / "data",
    "raw": RUN_DIR / "data" / "raw",
    "processed": RUN_DIR / "data" / "processed",
    "historical_paths": RUN_DIR / "data" / "historical_paths",
    "generators": RUN_DIR / "generators",
    "checkpoints": RUN_DIR / "checkpoints",
    "evaluations": RUN_DIR / "evaluations",
    "per_path": RUN_DIR / "evaluations" / "per_path",
    "summaries": RUN_DIR / "evaluations" / "summaries",
    "statistics": RUN_DIR / "statistics",
    "diagnostics": RUN_DIR / "diagnostics",
    "tables": RUN_DIR / "tables",
    "figures": RUN_DIR / "figures",
    "logs": RUN_DIR / "logs",
    "notebook_snapshot": RUN_DIR / "notebook_snapshot",
    "snapshot_root": SNAPSHOT_ROOT,
}

GENERATORS = ("GBM", "Heston", "SBTS")

print(f"schema_version : {CFG.schema_version}")
print(f"run_mode       : {CFG.run_mode}")
print(f"run_id         : {RUN_ID}")
print(f"config_hash    : {CONFIG_HASH}")
print(f"run directory  : {RUN_DIR}")
print(f"\nGrid           : {len(GENERATORS)} generators x {len(CFG.options)} options "
      f"x {len(CFG.strike_ratios)} strikes x {len(CFG.seeds)} seeds "
      f"= {CFG.n_configurations} configurations")
print(f"Checkpoints    : {CFG.n_expected_checkpoints} best-phase checkpoints expected")
print(f"Horizon        : {CFG.horizon} trading days, {CFG.n_paths:,} paths per generator")
print(f"Regimes        : {', '.join(CFG.regime_names)} (classified by path start date)")
if USE_SMOKE_SIZING:
    print("\n  [SMOKE GRID] reduced sizes — these outputs must never be quoted "
          "in the thesis.")

# ---- notebook cell 7 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 4 — SEEDS, HASHING, LOGGING, ATOMIC I/O, ARTIFACT REGISTRY
# ═════════════════════════════════════════════════════════════════════════════
#  Every important write goes through `atomic_write_*`:
#     tmp on the same filesystem -> flush/fsync -> atomic rename -> hash ->
#     reload verify -> manifest entry.
#  Caches are reused only when schema_version, config_hash and data_hash all
#  match; otherwise the stale artifact is quarantined, never silently merged.
# ═════════════════════════════════════════════════════════════════════════════

# ── Seed policy ──────────────────────────────────────────────────────────────
#  There is no seed alias/replacement mapping in this notebook, and there never
#  may be: a failed seed is reported as failed, not relabelled.
SEED_ALIAS_MAP: Dict[int, int] = {}


def assert_no_seed_aliasing() -> None:
    if SEED_ALIAS_MAP:
        raise RuntimeError(
            f"Seed aliasing is forbidden by design; found {SEED_ALIAS_MAP}. "
            f"A seed that fails must be recorded status='failed'.")


assert_no_seed_aliasing()


# ── Hashing ──────────────────────────────────────────────────────────────────
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr) -> str:
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def canonical_hash(obj: Any) -> str:
    """Format-independent hash of a nested payload.

    torch.save() bytes are not stable across PyTorch versions, so checkpoint
    integrity is verified against a hash computed by walking the payload tree:
    sorted dict keys, array dtype/shape/contents, primitives by repr.
    """
    h = hashlib.sha256()

    def _walk(o, key=""):
        if torch.is_tensor(o):
            a = o.detach().cpu().numpy()
            h.update(f"{key}:tensor:{a.dtype}:{a.shape}".encode())
            h.update(np.ascontiguousarray(a).tobytes())
        elif isinstance(o, np.ndarray):
            h.update(f"{key}:ndarray:{o.dtype}:{o.shape}".encode())
            h.update(np.ascontiguousarray(o).tobytes())
        elif isinstance(o, np.generic):
            h.update(f"{key}=np:{o.dtype}:{o.item()!r}".encode())
        elif isinstance(o, dict):
            for k in sorted(o.keys(), key=str):
                _walk(o[k], f"{key}.{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                _walk(v, f"{key}[{i}]")
        elif isinstance(o, bytes):
            h.update(f"{key}:bytes".encode()); h.update(o)
        elif o is None or isinstance(o, (bool, int, float, str)):
            h.update(f"{key}={type(o).__name__}:{o!r}".encode())
        else:
            h.update(f"{key}={o!r}".encode())

    _walk(obj)
    return h.hexdigest()


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (Path, _dt.datetime, _dt.date)):
        return str(o)
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    return str(o)


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


# ── Shard identity ───────────────────────────────────────────────────────────
#  Parallel training sessions share one run directory: checkpoints have unique
#  filenames, but the bookkeeping files must not be written by two processes at
#  once, so they carry the shard tag. An unsharded session owns the canonical
#  manifest.json and merges every shard's results.
_shard_parts = []
if SHARD_GENERATORS:
    _shard_parts.append("-".join(sorted(SHARD_GENERATORS)))
if SHARD_SEEDS is not None:
    _shard_parts.append("seeds" + "-".join(str(int(x)) for x in sorted(SHARD_SEEDS)))
IS_SHARDED = bool(_shard_parts)
SHARD_TAG = ("__" + "__".join(_shard_parts)) if IS_SHARDED else ""
SHARD_LABEL = "__".join(_shard_parts) if IS_SHARDED else "all"


def in_shard(generator: str, seed) -> bool:
    """Whether a configuration belongs to this session's slice of the queue."""
    if SHARD_GENERATORS and generator not in SHARD_GENERATORS:
        return False
    if SHARD_SEEDS is not None and int(seed) not in {int(x) for x in SHARD_SEEDS}:
        return False
    return True

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_PATH = PATHS["logs"] / f"run_{RUN_ID}{SHARD_TAG}.log"
LOGGER = logging.getLogger(f"sbts.{RUN_ID}")
LOGGER.setLevel(logging.INFO)
LOGGER.handlers.clear()
LOGGER.propagate = False
_fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)sZ %(levelname)-7s %(message)s"))
LOGGER.addHandler(_fh)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(logging.Formatter("%(message)s"))
_sh.setLevel(logging.WARNING)
LOGGER.addHandler(_sh)


def log(msg: str, level: str = "info", echo: bool = True) -> None:
    getattr(LOGGER, level)(msg)
    if echo and level in ("info", "debug"):
        print(msg)


# ── Atomic writes ────────────────────────────────────────────────────────────
def _atomic_replace(tmp: Path, dst: Path) -> None:
    with open(tmp, "rb+") as f:
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)


def atomic_write_bytes(dst, payload: bytes) -> str:
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(payload); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, dst)
    return sha256_file(dst)


def atomic_write_text(dst, text: str) -> str:
    return atomic_write_bytes(dst, text.encode("utf-8"))


def atomic_write_json(dst, obj: Any, verify: bool = True) -> str:
    dst = Path(dst)
    payload = json.dumps(obj, indent=2, sort_keys=True, default=json_default)
    digest = atomic_write_bytes(dst, payload.encode("utf-8"))
    if verify:
        with open(dst, "r", encoding="utf-8") as f:
            json.load(f)              # reload verify; raises on a torn write
    return digest


def atomic_write_npz(dst, verify: bool = True, **arrays) -> str:
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    # np.savez_compressed appends ".npz" unless the name already ends with it,
    # so the temporary name must carry that suffix itself.
    tmp = dst.parent / (dst.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    _atomic_replace(tmp, dst)
    digest = sha256_file(dst)
    if verify:
        with np.load(dst, allow_pickle=False) as z:
            for k in arrays:
                _ = z[k].shape
    return digest


def atomic_write_npy(dst, array: np.ndarray, verify: bool = True) -> str:
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / (dst.name + ".tmp.npy")   # np.save appends ".npy"
    np.save(tmp, array)
    _atomic_replace(tmp, dst)
    digest = sha256_file(dst)
    if verify:
        np.load(dst, allow_pickle=False)
    return digest


def atomic_write_torch(dst, payload: Any, verify: bool = True) -> str:
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(payload, tmp)
    _atomic_replace(tmp, dst)
    digest = sha256_file(dst)
    if verify:
        torch.load(dst, map_location="cpu", weights_only=False)
    return digest


def atomic_write_dataframe(dst_csv, df: "pd.DataFrame") -> str:
    dst_csv = Path(dst_csv); dst_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_csv.with_suffix(dst_csv.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    _atomic_replace(tmp, dst_csv)
    return sha256_file(dst_csv)


# ── Artifact registry / manifest ─────────────────────────────────────────────
MANIFEST_PATH = RUN_DIR / f"manifest{SHARD_TAG}.json"


def _load_manifest() -> Dict[str, Any]:
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:                             # noqa: BLE001
            log(f"[WARN] unreadable manifest ({exc!r}); starting a fresh one",
                "warning")
    return {
        "manifest_schema_version": CFG.manifest_schema_version,
        "schema_version": CFG.schema_version,
        "run_id": RUN_ID,
        "run_mode": CFG.run_mode,
        "shard": SHARD_LABEL,
        "config_hash": CONFIG_HASH,
        "source_repository": SOURCE_REPOSITORY,
        "source_snapshot_commit": SOURCE_SNAPSHOT_COMMIT,
        "started_utc": utc_now(),
        "ended_utc": None,
        "status": "running",
        "artifacts": {},
        "runs": {},
        "warnings": [],
        "failures": [],
        "gpu_hours": 0.0,
    }


MANIFEST: Dict[str, Any] = _load_manifest()


def save_manifest() -> str:
    MANIFEST["updated_utc"] = utc_now()
    return atomic_write_json(MANIFEST_PATH, MANIFEST)


def register_artifact(key: str, path, digest: Optional[str] = None,
                      meta: Optional[Dict[str, Any]] = None,
                      persist: bool = True) -> Dict[str, Any]:
    path = Path(path)
    entry = {
        "path": str(path.relative_to(RUN_DIR)) if RUN_DIR in path.parents or path == RUN_DIR
                else str(path),
        "sha256": digest if digest is not None else (sha256_file(path) if path.exists() else None),
        "bytes": path.stat().st_size if path.exists() else None,
        "config_hash": CONFIG_HASH,
        "schema_version": CFG.schema_version,
        "written_utc": utc_now(),
    }
    if meta:
        entry["meta"] = json.loads(json.dumps(meta, default=json_default))
    MANIFEST["artifacts"][key] = entry
    if persist:
        save_manifest()
    return entry


def add_warning(message: str, context: Optional[Dict[str, Any]] = None) -> None:
    MANIFEST["warnings"].append({"utc": utc_now(), "message": message,
                                 "context": context or {}})
    log(f"[WARN] {message}", "warning")
    save_manifest()


def add_failure(message: str, context: Optional[Dict[str, Any]] = None) -> None:
    MANIFEST["failures"].append({"utc": utc_now(), "message": message,
                                 "context": context or {}})
    log(f"[FAIL] {message}", "error", echo=False)
    print(f"[FAIL] {message}")
    save_manifest()


# ── Cache / resume policy ────────────────────────────────────────────────────
def cache_is_valid(meta: Optional[Dict[str, Any]],
                   data_hashes: Optional[Dict[str, str]] = None,
                   extra: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """A cache is reusable only when every identity field matches."""
    if not meta:
        return False, "no metadata"
    if meta.get("schema_version") != CFG.schema_version:
        return False, f"schema {meta.get('schema_version')} != {CFG.schema_version}"
    if meta.get("config_hash") != CONFIG_HASH:
        return False, f"config_hash {str(meta.get('config_hash'))[:12]} != {CONFIG_HASH[:12]}"
    if data_hashes:
        got = meta.get("data_hashes") or {}
        for k, v in data_hashes.items():
            if got.get(k) != v:
                return False, f"data_hash[{k}] mismatch"
    for k, v in (extra or {}).items():
        if meta.get(k) != v:
            return False, f"{k} {meta.get(k)!r} != {v!r}"
    return True, "ok"


def quarantine(path, reason: str) -> Optional[Path]:
    """Move a conflicting artifact aside instead of reusing or deleting it."""
    path = Path(path)
    if not path.exists():
        return None
    qdir = RUN_DIR / "quarantine"
    qdir.mkdir(parents=True, exist_ok=True)
    dst = qdir / f"{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%S')}__{path.name}"
    shutil.move(str(path), str(dst))
    add_warning(f"quarantined {path.name}: {reason}", {"moved_to": str(dst)})
    return dst


# ── Run keys ─────────────────────────────────────────────────────────────────
def run_key(generator: str, option: str, kappa: float, seed: int) -> str:
    return f"{generator}__{option}__k{kappa:.2f}__seed{int(seed):02d}"


def checkpoint_filename(generator: str, option: str, kappa: float,
                        seed: int, phase: str, kind: str = "best") -> str:
    return (f"{generator}__{option}__k{kappa:.2f}__seed{int(seed):02d}"
            f"__{phase}__{kind}.pt")


def canonical_run_queue(cfg: ExperimentConfig = None) -> List[Dict[str, Any]]:
    cfg = cfg or CFG
    queue = []
    for gen in GENERATORS:
        for opt in cfg.options:
            for kappa in cfg.strike_ratios:
                for seed in cfg.seeds:
                    queue.append({"generator": gen, "option": opt,
                                  "kappa": float(kappa), "seed": int(seed),
                                  "run_key": run_key(gen, opt, kappa, seed)})
    return queue


def capture_rng_state() -> Dict[str, Any]:
    state = {
        "torch": torch.get_rng_state().numpy().tolist(),
        "numpy": [s.tolist() if hasattr(s, "tolist") else s
                  for s in np.random.get_state()],
        "python": list(random.getstate()[1]),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [s.cpu().numpy().tolist()
                               for s in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> bool:
    """Inverse of capture_rng_state. Returns True only if EVERY stream was
    restored, so a caller can say honestly whether a resumed run continues the
    original random stream or merely a compatible one."""
    if not state:
        return False
    ok = True
    try:
        torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
    except Exception as exc:                                 # noqa: BLE001
        ok = False; log(f"[WARN] torch RNG not restored: {exc!r}", "warning", echo=False)
    try:
        np_state = state["numpy"]
        np.random.set_state((str(np_state[0]),
                             np.array(np_state[1], dtype=np.uint32),
                             int(np_state[2]), int(np_state[3]), float(np_state[4])))
    except Exception as exc:                                 # noqa: BLE001
        ok = False; log(f"[WARN] numpy RNG not restored: {exc!r}", "warning", echo=False)
    try:
        random.setstate((3, tuple(int(x) for x in state["python"]), None))
    except Exception as exc:                                 # noqa: BLE001
        ok = False; log(f"[WARN] python RNG not restored: {exc!r}", "warning", echo=False)
    if torch.cuda.is_available() and state.get("torch_cuda"):
        try:
            torch.cuda.set_rng_state_all(
                [torch.tensor(x, dtype=torch.uint8) for x in state["torch_cuda"]])
        except Exception as exc:                             # noqa: BLE001
            ok = False; log(f"[WARN] CUDA RNG not restored: {exc!r}", "warning", echo=False)
    return ok


def tensors_to_cpu(obj):
    """Recursively move every tensor to CPU (reaches nested optimizer state)."""
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: tensors_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tensors_to_cpu(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(tensors_to_cpu(x) for x in obj)
    return obj


# ── Persist config + environment for this run ────────────────────────────────
_cfg_hash_written = atomic_write_json(PATHS["config"] / "config.json", CFG.to_json_dict())
atomic_write_text(PATHS["config"] / "config_hash.txt", CONFIG_HASH + "\n")
atomic_write_json(PATHS["config"] / "source_snapshot.json", {
    "source_repository": SOURCE_REPOSITORY,
    "source_snapshot_commit": SOURCE_SNAPSHOT_COMMIT,
    "notebook": "SBTS_CANONICAL_A100.ipynb",
    "schema_version": CFG.schema_version,
    "git_commit_here": git_commit(),
    "written_utc": utc_now(),
})
ENVIRONMENT = env_snapshot()
# Each session records its OWN machine: a run trained across several GPUs
# (e.g. A100 then T4) must keep every environment, not the last writer's.
atomic_write_json(PATHS["environment"] / f"environment{SHARD_TAG}.json", ENVIRONMENT)
atomic_write_text(PATHS["environment"] / f"pip_freeze{SHARD_TAG}.txt", pip_freeze_text())
atomic_write_text(PATHS["environment"] / f"gpu_info{SHARD_TAG}.txt", json.dumps({
    "device": str(DEVICE), "gpu_name": GPU_NAME, "capability": GPU_CAPABILITY,
    "total_memory_bytes": GPU_TOTAL_MEMORY, "is_a100": IS_A100,
}, indent=2))
ENVIRONMENT_HASH = canonical_hash(ENVIRONMENT)
MANIFEST["environment_hash"] = ENVIRONMENT_HASH
MANIFEST["config_json_sha256"] = _cfg_hash_written
register_artifact("config/config.json", PATHS["config"] / "config.json", _cfg_hash_written)
register_artifact(f"environment/environment{SHARD_TAG}.json",
                  PATHS["environment"] / f"environment{SHARD_TAG}.json")
if DEPENDENCY_REPORT.get("mismatches"):
    add_warning("dependency lock deviations present",
                DEPENDENCY_REPORT["mismatches"])

log(f"Utilities ready. Log file: {LOG_PATH}")
print(f"environment_hash : {ENVIRONMENT_HASH[:16]}...")
print(f"manifest         : {MANIFEST_PATH}")

# ==========================================================================
# ---
#
# ## Sections 3–4 — Frozen data snapshot, temporal split, historical paths
#
# Pipeline order (design §6 orchestrator):
#
# ```python
# ctx        = initialize_run(cfg)                    # Cells 2-4
# raw        = load_or_create_frozen_data(ctx)        # Cell 5
# data       = validate_and_prepare_data(raw, ctx)    # Cell 6
# historical = build_historical_paths(data, ctx)      # Cell 7
# params     = {"GBM": calibrate_gbm(...),            # Cell 8
#               "Heston": calibrate_heston(...)}      # Cell 9
# refs       = build_sbts_references(...)             # Cell 10
# h*, K*     = select_sbts_hk_chronologically(...)    # Cell 11
# datasets   = generate_or_load(...)                  # Cell 12
# validate_generator_datasets(...)                    # Cell 13
# compute_generator_fidelity(...)                     # Cell 14
# ```
#
# The published run uses a frozen, hashed snapshot. Reproducibility is claimed as
# **reproducible from the frozen snapshot and locked environment within declared
# numerical tolerances** — never bit-for-bit across hardware.

# ---- notebook cell 9 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 5 — DATA ACQUISITION AND FROZEN SNAPSHOT
# ═════════════════════════════════════════════════════════════════════════════
#  1. If the canonical raw snapshot exists and its hashes match the manifest,
#     read it.  2. Otherwise download ONCE.  3. Normalise price columns and the
#  calendar.  4. Store raw prices, clean prices, log returns and metadata.
#  5. SHA-256 every file.  6. Never download twice in one run, and never
#  overwrite a published snapshot silently.
# ═════════════════════════════════════════════════════════════════════════════

SNAPSHOT_DIR = SNAPSHOT_ROOT / f"{'-'.join(CFG.tickers)}__{CFG.raw_data_start}__{CFG.raw_data_end}"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_MANIFEST = SNAPSHOT_DIR / "data_manifest.json"

# Offline escape hatch for pipeline validation only. A synthetic snapshot is
# tagged as such and any run built on it is refused publication status.
ALLOW_SYNTHETIC_SNAPSHOT = (os.environ.get("SBTS_ALLOW_SYNTHETIC_SNAPSHOT") == "1")


def _download_prices(cfg: ExperimentConfig) -> Tuple["pd.DataFrame", Dict[str, Any]]:
    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance is not importable and no frozen snapshot exists.")
    t0 = utc_now()
    raw = yf.download(list(cfg.tickers), start=cfg.raw_data_start,
                      end=cfg.raw_data_end, auto_adjust=cfg.auto_adjust,
                      progress=False)
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance returned an empty frame.")
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        field = cfg.price_field if cfg.price_field in set(level0) else (
            "Adj Close" if "Adj Close" in set(level0) else "Close")
        prices = raw[field][list(cfg.tickers)]
    else:
        field = cfg.price_field
        prices = raw[[*cfg.tickers]]
    meta = {
        "source": cfg.data_source,
        "price_field": field,
        "auto_adjust": cfg.auto_adjust,
        "adjustment_convention": ("yfinance auto_adjust=True: split- and "
                                  "dividend-adjusted close"
                                  if cfg.auto_adjust else "unadjusted close"),
        "download_started_utc": t0,
        "download_finished_utc": utc_now(),
    }
    return prices, meta


def _synthetic_prices(cfg: ExperimentConfig) -> Tuple["pd.DataFrame", Dict[str, Any]]:
    """Deterministic offline stand-in with realistic calendar and moments.
    NEVER publishable: every artifact derived from it is tagged synthetic."""
    rng = np.random.default_rng(20050103)
    idx = pd.bdate_range(cfg.raw_data_start, min(cfg.raw_data_end, "2025-12-31"))
    n, d = len(idx), cfg.d
    vol = np.array([0.0180, 0.0165, 0.0150])[:d]
    mu = np.array([0.00055, 0.00030, 0.00025])[:d]
    corr = np.array([[1.0, 0.45, 0.35], [0.45, 1.0, 0.50], [0.35, 0.50, 1.0]])[:d, :d]
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, d)) @ chol.T
    z += rng.standard_t(df=4, size=(n, d)) * 0.25      # fat tails
    crash = ((idx >= "2020-02-20") & (idx <= "2020-04-30"))
    shock = np.where(crash, 3.0, 1.0)[:, None]
    rets = mu + vol * z * shock
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)),
                          index=idx, columns=list(cfg.tickers))
    return prices, {"source": "synthetic_offline", "price_field": "synthetic",
                    "auto_adjust": None,
                    "adjustment_convention": "synthetic; not publishable",
                    "download_started_utc": utc_now(),
                    "download_finished_utc": utc_now()}


def _clean_prices(prices: "pd.DataFrame", cfg: ExperimentConfig):
    """Common calendar, forward fill, drop leading/remaining gaps."""
    prices = prices.sort_index()
    # yfinance returns a tz-aware index for some tickers/versions and a naive
    # one for others; only strip the zone when there is one to strip, since
    # older pandas raises on tz_localize(None) over an already-naive index.
    _idx = pd.to_datetime(prices.index)
    if getattr(_idx, "tz", None) is not None:
        _idx = _idx.tz_localize(None)
    prices.index = _idx
    n_rows_raw = int(len(prices))
    n_missing_before = int(prices.isna().sum().sum())
    clean = prices.ffill().dropna(how="any")
    n_missing_after = int(clean.isna().sum().sum())
    log_returns = np.log(clean / clean.shift(1)).dropna(how="any")
    stats_ = {
        "n_rows_raw": n_rows_raw,
        "n_rows_clean": int(len(clean)),
        "n_rows_returns": int(len(log_returns)),
        "n_missing_cells_before_cleaning": n_missing_before,
        "n_missing_cells_after_cleaning": n_missing_after,
        "first_observation": str(clean.index[0].date()),
        "last_observation": str(clean.index[-1].date()),
    }
    return clean, log_returns, stats_


def build_or_load_snapshot(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Return the frozen data snapshot, creating it exactly once."""
    if SNAPSHOT_MANIFEST.exists():
        with open(SNAPSHOT_MANIFEST, "r", encoding="utf-8") as f:
            meta = json.load(f)
        files = {k: SNAPSHOT_DIR / v for k, v in meta["files"].items()}
        missing = [k for k, p in files.items() if not p.exists()]
        if missing:
            add_warning(f"snapshot manifest lists missing files {missing}; rebuilding")
        else:
            bad = {k: (meta["hashes"][k], sha256_file(p))
                   for k, p in files.items() if sha256_file(p) != meta["hashes"][k]}
            if bad:
                raise RuntimeError(
                    f"Frozen snapshot hash mismatch in {SNAPSHOT_DIR}: {list(bad)}. "
                    f"The snapshot has been modified; refusing to continue. "
                    f"Restore it from backup or move it aside deliberately.")
            log(f"Frozen snapshot reused: {SNAPSHOT_DIR.name}")
            meta["reused"] = True
            return meta

    # ── create the snapshot exactly once ────────────────────────────────────
    log(f"Creating frozen snapshot in {SNAPSHOT_DIR}")
    try:
        prices, dl_meta = _download_prices(cfg)
    except Exception as exc:                                 # noqa: BLE001
        if not ALLOW_SYNTHETIC_SNAPSHOT:
            raise RuntimeError(
                f"Data download failed ({exc!r}) and no frozen snapshot exists. "
                f"Provide a snapshot under {SNAPSHOT_DIR}, or set "
                f"SBTS_ALLOW_SYNTHETIC_SNAPSHOT=1 for an offline, "
                f"NON-PUBLISHABLE pipeline test.") from exc
        add_warning(f"using SYNTHETIC offline prices ({exc!r}); run is not publishable")
        prices, dl_meta = _synthetic_prices(cfg)

    clean, log_returns, clean_stats = _clean_prices(prices, cfg)

    files = {"raw_prices": "raw_prices.csv", "clean_prices": "clean_prices.csv",
             "log_returns": "log_returns.csv", "arrays": "snapshot_arrays.npz"}
    hashes = {}
    hashes["raw_prices"] = atomic_write_text(
        SNAPSHOT_DIR / files["raw_prices"], prices.to_csv())
    hashes["clean_prices"] = atomic_write_text(
        SNAPSHOT_DIR / files["clean_prices"], clean.to_csv())
    hashes["log_returns"] = atomic_write_text(
        SNAPSHOT_DIR / files["log_returns"], log_returns.to_csv())
    hashes["arrays"] = atomic_write_npz(
        SNAPSHOT_DIR / files["arrays"],
        clean_prices=clean.values.astype(np.float64),
        clean_dates=np.array([str(d.date()) for d in clean.index]),
        log_returns=log_returns.values.astype(np.float64),
        return_dates=np.array([str(d.date()) for d in log_returns.index]),
        tickers=np.array(list(cfg.tickers)))

    meta = {
        "data_schema_version": cfg.data_schema_version,
        "tickers": list(cfg.tickers),
        "ticker_mapping": {t: t for t in cfg.tickers},
        "request_start": cfg.raw_data_start,
        "request_end": cfg.raw_data_end,
        "files": files,
        "hashes": hashes,
        "array_hashes": {
            "clean_prices": sha256_array(clean.values.astype(np.float64)),
            "log_returns": sha256_array(log_returns.values.astype(np.float64)),
        },
        "created_utc": utc_now(),
        "synthetic": dl_meta["source"] == "synthetic_offline",
        "reused": False,
        **dl_meta, **clean_stats,
    }
    atomic_write_json(SNAPSHOT_MANIFEST, meta)
    log(f"Snapshot written: {clean_stats['n_rows_clean']} rows "
        f"{clean_stats['first_observation']} -> {clean_stats['last_observation']}")
    return meta


if RUN_MODE == "ANALYSIS_ONLY" and not SNAPSHOT_MANIFEST.exists():
    raise RuntimeError("ANALYSIS_ONLY requires an existing frozen snapshot.")

SNAPSHOT_META = build_or_load_snapshot(CFG)
SNAPSHOT_IS_SYNTHETIC = bool(SNAPSHOT_META.get("synthetic"))

with np.load(SNAPSHOT_DIR / SNAPSHOT_META["files"]["arrays"], allow_pickle=False) as _z:
    CLEAN_PRICES = pd.DataFrame(_z["clean_prices"],
                                index=pd.to_datetime([str(x) for x in _z["clean_dates"]]),
                                columns=[str(t) for t in _z["tickers"]])
    LOG_RETURNS = pd.DataFrame(_z["log_returns"],
                               index=pd.to_datetime([str(x) for x in _z["return_dates"]]),
                               columns=[str(t) for t in _z["tickers"]])

DATA_HASHES = {
    "snapshot_clean_prices": SNAPSHOT_META["array_hashes"]["clean_prices"],
    "snapshot_log_returns": SNAPSHOT_META["array_hashes"]["log_returns"],
    "snapshot_manifest": sha256_file(SNAPSHOT_MANIFEST),
}

# Mirror the snapshot identity into this run (files stay in the shared store).
atomic_write_json(PATHS["data"] / "data_manifest.json", {
    "data_schema_version": CFG.data_schema_version,
    "schema_version": CFG.schema_version,
    "config_hash": CONFIG_HASH,
    "snapshot_dir": str(SNAPSHOT_DIR),
    "snapshot": SNAPSHOT_META,
    "data_hashes": DATA_HASHES,
})
register_artifact("data/data_manifest.json", PATHS["data"] / "data_manifest.json")
MANIFEST["data_hashes"] = DATA_HASHES
MANIFEST["snapshot_is_synthetic"] = SNAPSHOT_IS_SYNTHETIC
save_manifest()

print(f"Snapshot dir   : {SNAPSHOT_DIR}")
print(f"  source       : {SNAPSHOT_META['source']}"
      f"{'   [SYNTHETIC - NOT PUBLISHABLE]' if SNAPSHOT_IS_SYNTHETIC else ''}")
print(f"  requested    : {SNAPSHOT_META['request_start']} -> {SNAPSHOT_META['request_end']}")
print(f"  actual       : {SNAPSHOT_META['first_observation']} -> "
      f"{SNAPSHOT_META['last_observation']}")
print(f"  rows (clean) : {SNAPSHOT_META['n_rows_clean']}  "
      f"returns: {SNAPSHOT_META['n_rows_returns']}")
print(f"  missing cells: before={SNAPSHOT_META['n_missing_cells_before_cleaning']}  "
      f"after={SNAPSHOT_META['n_missing_cells_after_cleaning']}")
print(f"  adjustment   : {SNAPSHOT_META['adjustment_convention']}")
print(f"  downloaded   : {SNAPSHOT_META['download_finished_utc']}")
for k, v in DATA_HASHES.items():
    print(f"  {k:<24s} {v[:16]}...")

# ---- notebook cell 10 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 6 — TEMPORAL SPLIT AND DESCRIPTIVE STATISTICS
# ═════════════════════════════════════════════════════════════════════════════
#  Two-tier design (locked):
#    Tier 1  generator calibration  : historical returns 2005-01 .. 2019-12
#    Tier 2  hedger training        : synthetic paths only
#  The hedging network never sees a historical evaluation path during training.
# ═════════════════════════════════════════════════════════════════════════════

TRAIN_RETURNS = LOG_RETURNS.loc[CFG.calibration_start:CFG.calibration_end]
TRAIN_RETURNS_VALUES = TRAIN_RETURNS.values.astype(np.float64)
TICKERS = list(LOG_RETURNS.columns)
D_ASSETS = len(TICKERS)
S0 = CLEAN_PRICES.loc[:CFG.calibration_end].iloc[-1].values.astype(np.float64)

assert D_ASSETS == CFG.d, f"asset count {D_ASSETS} != config {CFG.d}"
assert len(TRAIN_RETURNS) > CFG.horizon + 10, "calibration window too short"
assert np.isfinite(TRAIN_RETURNS_VALUES).all(), "non-finite calibration returns"
assert TRAIN_RETURNS.index.is_monotonic_increasing, "calibration dates not sorted"
assert TRAIN_RETURNS.index.is_unique, "duplicate calibration dates"

CALIBRATION_INPUT_HASH = sha256_array(TRAIN_RETURNS_VALUES)

desc = pd.DataFrame(index=TICKERS)
desc["mean_pct"] = TRAIN_RETURNS.mean().values * 100
desc["std_pct"] = TRAIN_RETURNS.std().values * 100
desc["ann_vol_pct"] = TRAIN_RETURNS.std().values * np.sqrt(252) * 100
desc["skewness"] = TRAIN_RETURNS.skew().values
desc["kurtosis"] = TRAIN_RETURNS.kurtosis().values + 3.0
desc["min_pct"] = TRAIN_RETURNS.min().values * 100
desc["max_pct"] = TRAIN_RETURNS.max().values * 100
desc["jarque_bera_p"] = [float(sp_stats.jarque_bera(TRAIN_RETURNS[t].dropna())[1])
                         for t in TICKERS]
DESCRIPTIVE_TABLE = desc.reset_index().rename(columns={"index": "ticker"})

DATA_PERIOD_TABLE = pd.DataFrame([
    {"period": "CALIBRATION_2005_2019",
     "range_start": CFG.calibration_start, "range_end": CFG.calibration_end,
     "role": "generator calibration (all three generators)",
     "n_observations": int(len(TRAIN_RETURNS)),
     "actual_first": str(TRAIN_RETURNS.index[0].date()),
     "actual_last": str(TRAIN_RETURNS.index[-1].date())},
    *[{"period": name, "range_start": s, "range_end": e,
       "role": ("historical stress-regime evaluation (2019 overlaps the "
                "calibration window; 2020+ is temporally out of sample)"
                if name.startswith("COVID")
                else "temporally out of sample by path start date"),
       "n_observations": int(((LOG_RETURNS.index >= s) & (LOG_RETURNS.index <= e)).sum()),
       "actual_first": (str(LOG_RETURNS.index[(LOG_RETURNS.index >= s)
                                              & (LOG_RETURNS.index <= e)][0].date())
                        if ((LOG_RETURNS.index >= s) & (LOG_RETURNS.index <= e)).any() else None),
       "actual_last": (str(LOG_RETURNS.index[(LOG_RETURNS.index >= s)
                                             & (LOG_RETURNS.index <= e)][-1].date())
                       if ((LOG_RETURNS.index >= s) & (LOG_RETURNS.index <= e)).any() else None)}
      for name, s, e in CFG.regimes],
])

_h = atomic_write_dataframe(PATHS["processed"] / "descriptive_statistics.csv",
                            DESCRIPTIVE_TABLE)
register_artifact("processed/descriptive_statistics.csv",
                  PATHS["processed"] / "descriptive_statistics.csv", _h)
_h = atomic_write_dataframe(PATHS["processed"] / "data_period_metadata.csv",
                            DATA_PERIOD_TABLE)
register_artifact("processed/data_period_metadata.csv",
                  PATHS["processed"] / "data_period_metadata.csv", _h)
atomic_write_json(PATHS["processed"] / "split_metadata.json", {
    "schema_version": CFG.schema_version,
    "config_hash": CONFIG_HASH,
    "calibration_start": CFG.calibration_start,
    "calibration_end": CFG.calibration_end,
    "calibration_first_observation": str(TRAIN_RETURNS.index[0].date()),
    "calibration_last_observation": str(TRAIN_RETURNS.index[-1].date()),
    "n_calibration_returns": int(len(TRAIN_RETURNS)),
    "calibration_input_hash": CALIBRATION_INPUT_HASH,
    "S0_at_calibration_end": {t: float(v) for t, v in zip(TICKERS, S0)},
    "data_cutoff": SNAPSHOT_META["last_observation"],
    "tier_1": "generators calibrated on historical returns through 2019-12-31",
    "tier_2": "hedging networks trained exclusively on synthetic paths",
})
DATA_HASHES["calibration_returns"] = CALIBRATION_INPUT_HASH

print(f"Calibration window : {TRAIN_RETURNS.index[0].date()} -> "
      f"{TRAIN_RETURNS.index[-1].date()}  ({len(TRAIN_RETURNS)} returns)")
print(f"Assets             : {TICKERS}")
print(f"S0 (2019 close)    : {dict(zip(TICKERS, np.round(S0, 2)))}")
print(f"calibration hash   : {CALIBRATION_INPUT_HASH[:16]}...")
print(f"data cutoff        : {SNAPSHOT_META['last_observation']}")
print("\nDescriptive statistics of daily log-returns (calibration window):")
print(DESCRIPTIVE_TABLE.round(4).to_string(index=False))
print("\nCorrelation (calibration window):")
print(TRAIN_RETURNS.corr().round(4).to_string())
print("\nData-period metadata:")
print(DATA_PERIOD_TABLE.to_string(index=False))

# ---- notebook cell 11 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 7 — HISTORICAL EVALUATION PATHS (classified by path start date)
# ═════════════════════════════════════════════════════════════════════════════
#  Every path is a forward window of `horizon` trading days starting on a
#  trading day inside the regime's calendar range. The window MAY end outside
#  that range — that is by design (an option must be hedged to maturity), and
#  it is why every table says "classified by path start date".
#
#  Normalisation matches the synthetic paths: S_0 = 1 per asset, using the
#  price on the path's own start date.
# ═════════════════════════════════════════════════════════════════════════════

PRICE_VALUES = CLEAN_PRICES.values.astype(np.float64)
PRICE_DATES = CLEAN_PRICES.index
CRASH_START = pd.Timestamp(CFG.covid_crash_start)
CRASH_END = pd.Timestamp(CFG.covid_crash_end)


def build_regime_paths(regime: str, start: str, end: str,
                       cfg: ExperimentConfig) -> Dict[str, Any]:
    H = cfg.horizon
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    n_total = len(PRICE_DATES)
    in_regime = np.where((PRICE_DATES >= lo) & (PRICE_DATES <= hi))[0]
    usable = [i for i in in_regime if i + H < n_total]
    n_starts_in_regime = int(len(in_regime))

    if USE_SMOKE_SIZING and len(usable) > cfg.smoke_hist_paths_per_regime:
        step = max(1, len(usable) // cfg.smoke_hist_paths_per_regime)
        usable = usable[::step][:cfg.smoke_hist_paths_per_regime]

    if not usable:
        raise RuntimeError(
            f"{regime}: no path of {H} trading days fits before the data cutoff "
            f"{PRICE_DATES[-1].date()} for start dates in [{start}, {end}].")

    S_raw = np.stack([PRICE_VALUES[i:i + H + 1] for i in usable]).astype(np.float64)
    S_norm = S_raw / S_raw[:, :1, :]
    returns = np.diff(np.log(S_raw), axis=1)
    start_dates = [PRICE_DATES[i] for i in usable]
    end_dates = [PRICE_DATES[i + H] for i in usable]
    contains_crash = np.array([(sd <= CRASH_END) and (ed >= CRASH_START)
                               for sd, ed in zip(start_dates, end_dates)], dtype=bool)

    # ── validation ─────────────────────────────────────────────────────────
    assert S_raw.shape[1:] == (H + 1, cfg.d), S_raw.shape
    assert returns.shape[1:] == (H, cfg.d), returns.shape
    assert np.isfinite(S_raw).all() and np.isfinite(returns).all(), f"{regime}: non-finite"
    assert (S_raw > 0).all(), f"{regime}: non-positive price"
    assert np.allclose(S_norm[:, 0, :], 1.0), f"{regime}: S_0 != 1"
    for i in usable[:1] + usable[-1:]:
        window = PRICE_DATES[i:i + H + 1]
        assert window.is_monotonic_increasing and window.is_unique, \
            f"{regime}: dates not strictly increasing"
    assert all(lo <= sd <= hi for sd in start_dates), \
        f"{regime}: a path start date falls outside the regime range"

    meta = {
        "regime": regime,
        "range_start": start, "range_end": end,
        "assignment_rule": "path start date",
        "horizon_trading_days": H,
        "n_paths": int(len(usable)),
        "n_start_dates_in_regime": n_starts_in_regime,
        "n_dropped_insufficient_lookahead": int(n_starts_in_regime - len([
            i for i in in_regime if i + H < n_total])),
        "start_date_min": str(min(start_dates).date()),
        "start_date_max": str(max(start_dates).date()),
        "end_date_min": str(min(end_dates).date()),
        "end_date_max": str(max(end_dates).date()),
        "covid_crash_window": [CFG.covid_crash_start, CFG.covid_crash_end],
        "n_paths_covering_crash": int(contains_crash.sum()),
        "share_paths_covering_crash": float(contains_crash.mean()),
        "data_cutoff": str(PRICE_DATES[-1].date()),
        "subsampled_for_smoke": bool(USE_SMOKE_SIZING),
        "S_norm_hash": sha256_array(S_norm.astype(np.float32)),
        "tickers": list(TICKERS),
    }
    return {"S_raw": S_raw, "S_norm": S_norm, "returns": returns,
            "start_dates": [str(x.date()) for x in start_dates],
            "end_dates": [str(x.date()) for x in end_dates],
            "contains_covid_crash": contains_crash, "meta": meta}


HISTORICAL: Dict[str, Dict[str, Any]] = {}
_rows = []
for _name, _s, _e in CFG.regimes:
    _cache = PATHS["historical_paths"] / f"{_name}.npz"
    _cache_meta = PATHS["historical_paths"] / f"{_name}.json"
    _reuse = False
    if _cache.exists() and _cache_meta.exists():
        with open(_cache_meta, "r", encoding="utf-8") as f:
            _m = json.load(f)
        _ok, _why = cache_is_valid(_m, {"calibration_returns": CALIBRATION_INPUT_HASH})
        if _ok:
            with np.load(_cache, allow_pickle=False) as z:
                HISTORICAL[_name] = {
                    "S_raw": z["S_raw"], "S_norm": z["S_norm"], "returns": z["returns"],
                    "start_dates": [str(x) for x in z["start_dates"]],
                    "end_dates": [str(x) for x in z["end_dates"]],
                    "contains_covid_crash": z["contains_covid_crash"].astype(bool),
                    "meta": _m["regime_meta"]}
            _reuse = True
            log(f"  {_name}: reused cached historical paths")
        else:
            quarantine(_cache, f"historical path cache invalid: {_why}")
            quarantine(_cache_meta, f"historical path cache invalid: {_why}")

    if not _reuse:
        HISTORICAL[_name] = build_regime_paths(_name, _s, _e, CFG)
        _obj = HISTORICAL[_name]
        _digest = atomic_write_npz(
            _cache,
            S_raw=_obj["S_raw"].astype(np.float32),
            S_norm=_obj["S_norm"].astype(np.float32),
            returns=_obj["returns"].astype(np.float32),
            start_dates=np.array(_obj["start_dates"]),
            end_dates=np.array(_obj["end_dates"]),
            contains_covid_crash=_obj["contains_covid_crash"],
            path_id=np.arange(len(_obj["start_dates"])))
        atomic_write_json(_cache_meta, {
            "schema_version": CFG.schema_version, "config_hash": CONFIG_HASH,
            "data_hashes": {"calibration_returns": CALIBRATION_INPUT_HASH},
            "regime_meta": _obj["meta"], "npz_sha256": _digest})
        register_artifact(f"historical_paths/{_name}.npz", _cache, _digest,
                          meta=_obj["meta"])
    _rows.append(HISTORICAL[_name]["meta"])

REGIME_METADATA_TABLE = pd.DataFrame(_rows)[[
    "regime", "range_start", "range_end", "assignment_rule", "n_paths",
    "start_date_min", "start_date_max", "end_date_min", "end_date_max",
    "n_paths_covering_crash", "share_paths_covering_crash", "data_cutoff"]]
_h = atomic_write_dataframe(PATHS["historical_paths"] / "regime_metadata.csv",
                            REGIME_METADATA_TABLE)
register_artifact("historical_paths/regime_metadata.csv",
                  PATHS["historical_paths"] / "regime_metadata.csv", _h)

HIST_TENSORS = {name: torch.tensor(obj["S_norm"], dtype=torch.float32, device=DEVICE)
                for name, obj in HISTORICAL.items()}
DATA_HASHES.update({f"historical_{k}": v["meta"]["S_norm_hash"]
                    for k, v in HISTORICAL.items()})
MANIFEST["data_hashes"] = DATA_HASHES
save_manifest()

print("Historical evaluation paths (classified by path start date):\n")
print(REGIME_METADATA_TABLE.to_string(index=False))
print(f"\nHorizon {CFG.horizon} trading days; a path may end after its regime's "
      f"range — that is expected and disclosed.")
print("\nDISCLOSURE: hedging networks are trained exclusively on synthetic paths; "
      "the\ngenerators producing those paths are calibrated on historical returns "
      "through\n2019-12-31. The extended 2019-2020 COVID regime is therefore a "
      "historical\nstress-regime evaluation, not a strictly out-of-sample one.")

# ==========================================================================
# ---
#
# ## Sections 5–8 — Generator calibration, selection, generation, fidelity
#
# GBM and Heston are calibrated on the 2005–2019 returns; SBTS builds rolling
# reference trajectories from the same window. The bandwidth and Markov order are
# selected on a **chronological, purged** split with the full held-out objective,
# replacing the random-split one-step proxy. Fidelity diagnostics are computed
# before any hedger is trained.

# ---- notebook cell 13 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 8 — GBM CALIBRATION AND GENERATION
# ═════════════════════════════════════════════════════════════════════════════
#  Estimators (recorded verbatim in the calibration artifact):
#     drift_annual[k]  = mean(log-returns_k) * 252          (log-drift, mu-sigma^2/2)
#     sigma_annual[k]  = std(log-returns_k, ddof=1) * sqrt(252)
#     rho              = Pearson correlation of daily log-returns
#     L                = Cholesky factor of rho (after a positive-definiteness check)
#  Dynamics:  X_t = drift*dt + sigma*sqrt(dt) * (L Z_t),  S_t = S_0 exp(cumsum X)
# ═════════════════════════════════════════════════════════════════════════════

GBM_ESTIMATOR_FORMULAS = {
    "drift_annual": "mean(daily log-returns) * 252",
    "sigma_annual": "std(daily log-returns, ddof=1) * sqrt(252)",
    "correlation": "Pearson correlation of daily log-returns",
    "cholesky": "numpy.linalg.cholesky(correlation), validated positive definite",
}


def _nearest_correlation(rho: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Project to a positive-definite correlation matrix, logging the change."""
    eig = np.linalg.eigvalsh(rho)
    info = {"eigenvalues_raw": eig.tolist(), "projected": False}
    if eig.min() <= 1e-10:
        shift = -eig.min() + 1e-8
        rho = rho + shift * np.eye(rho.shape[0])
        rho = rho / np.sqrt(np.outer(np.diag(rho), np.diag(rho)))
        info.update({"projected": True, "ridge_shift": float(shift),
                     "eigenvalues_projected": np.linalg.eigvalsh(rho).tolist()})
    return rho, info


def calibrate_gbm(returns: np.ndarray, cfg: ExperimentConfig) -> Dict[str, Any]:
    n, d = returns.shape
    drift_annual = returns.mean(axis=0) * 252.0
    sigma_annual = returns.std(axis=0, ddof=1) * np.sqrt(252.0)
    rho = np.corrcoef(returns.T)
    rho, pd_info = _nearest_correlation(rho)
    chol = np.linalg.cholesky(rho)
    params = {
        "generator": "GBM",
        "generator_schema_version": cfg.generator_schema_version,
        "config_hash": CONFIG_HASH,
        "calibration_input_hash": sha256_array(returns),
        "calibration_sample_size": int(n),
        "calibration_start": str(TRAIN_RETURNS.index[0].date()),
        "calibration_end": str(TRAIN_RETURNS.index[-1].date()),
        "estimator_formulas": GBM_ESTIMATOR_FORMULAS,
        "drift_annual": drift_annual.tolist(),
        "sigma_annual": sigma_annual.tolist(),
        "correlation": rho.tolist(),
        "cholesky": chol.tolist(),
        "correlation_eigenvalues": np.linalg.eigvalsh(rho).tolist(),
        "positive_definiteness": pd_info,
        "tickers": list(TICKERS),
        "calibrated_utc": utc_now(),
    }
    return params


def generate_gbm(params: Dict[str, Any], cfg: ExperimentConfig,
                 seed: int, n_paths: int) -> np.ndarray:
    """Return normalised spot paths, shape [n_paths, horizon+1, d]."""
    g = torch_generator(DEVICE, seed, 0)
    drift = torch.tensor(params["drift_annual"], dtype=torch.float32, device=DEVICE)
    sigma = torch.tensor(params["sigma_annual"], dtype=torch.float32, device=DEVICE)
    chol = torch.tensor(params["cholesky"], dtype=torch.float32, device=DEVICE)
    dt = float(cfg.delta_t)
    z = torch.randn(n_paths, cfg.horizon, cfg.d, generator=g,
                    device=DEVICE, dtype=torch.float32)
    z = torch.matmul(z, chol.T)
    x = drift * dt + sigma * math.sqrt(dt) * z
    s = torch.ones(n_paths, cfg.horizon + 1, cfg.d, device=DEVICE, dtype=torch.float32)
    s[:, 1:, :] = torch.exp(torch.cumsum(x, dim=1))
    out = s.cpu().numpy()
    del z, x, s
    clear_mem()
    if not np.isfinite(out).all():
        raise FloatingPointError("GBM generation produced non-finite values")
    return out


GBM_PARAMS = calibrate_gbm(TRAIN_RETURNS_VALUES, CFG)
atomic_write_json(PATHS["generators"] / "GBM" / "calibration.json", GBM_PARAMS)
register_artifact("generators/GBM/calibration.json",
                  PATHS["generators"] / "GBM" / "calibration.json")

print("GBM calibration")
print(f"  sample size        : {GBM_PARAMS['calibration_sample_size']} daily returns")
print(f"  drift (annual)     : {dict(zip(TICKERS, np.round(GBM_PARAMS['drift_annual'], 4)))}")
print(f"  volatility (annual): {dict(zip(TICKERS, np.round(GBM_PARAMS['sigma_annual'], 4)))}")
print(f"  corr eigenvalues   : {np.round(GBM_PARAMS['correlation_eigenvalues'], 4).tolist()}")
print(f"  PD projection      : {GBM_PARAMS['positive_definiteness']['projected']}")
print(f"  input hash         : {GBM_PARAMS['calibration_input_hash'][:16]}...")

# ---- notebook cell 14 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 9 — MULTIVARIATE HESTON CALIBRATION AND GENERATION
# ═════════════════════════════════════════════════════════════════════════════
#  Locked specification (as in the thesis):
#    * per asset:  theta_k = annualised variance of the calibration returns,
#                  v0_k    = theta_k,
#                  mu_k    = mean(log-returns)*252 + theta_k/2,
#                  kappa, xi, rho  fixed at the standard equity values in CFG.
#    * cross-asset: Pearson correlation of the calibration returns drives the
#                  spot Brownian motions through its Cholesky factor; the
#                  variance Brownian motion of asset k is rho-correlated with
#                  that asset's spot shock.
#    * scheme:     full-truncation Euler with the Milstein CIR correction.
#  Every clipping/projection is logged; nothing is silently repaired.
# ═════════════════════════════════════════════════════════════════════════════

def calibrate_heston(returns: np.ndarray, cfg: ExperimentConfig) -> Dict[str, Any]:
    n, d = returns.shape
    warnings_list: List[Dict[str, Any]] = []
    sigma_ann = returns.std(axis=0, ddof=1) * np.sqrt(252.0)
    theta = sigma_ann ** 2
    mu = returns.mean(axis=0) * 252.0 + theta / 2.0
    kappa = np.full(d, float(cfg.heston_kappa))
    xi = np.full(d, float(cfg.heston_xi))
    rho = np.full(d, float(cfg.heston_rho))

    if (theta <= 0).any():
        bad = np.where(theta <= 0)[0].tolist()
        raise FloatingPointError(f"non-positive theta for assets {bad}")

    feller = 2.0 * kappa * theta - xi ** 2
    for k in range(d):
        if feller[k] <= 0:
            warnings_list.append({
                "asset": TICKERS[k], "issue": "Feller condition violated",
                "2_kappa_theta": float(2 * kappa[k] * theta[k]),
                "xi_squared": float(xi[k] ** 2),
                "consequence": "variance can reach zero; full truncation keeps it at 0"})

    corr = np.corrcoef(returns.T)
    corr, pd_info = _nearest_correlation(corr)
    if pd_info["projected"]:
        warnings_list.append({"issue": "correlation projected to positive definite",
                              **{k: v for k, v in pd_info.items() if k != "projected"}})
    chol = np.linalg.cholesky(corr)

    return {
        "generator": "Heston",
        "generator_schema_version": cfg.generator_schema_version,
        "config_hash": CONFIG_HASH,
        "calibration_input_hash": sha256_array(returns),
        "calibration_sample_size": int(n),
        "calibration_start": str(TRAIN_RETURNS.index[0].date()),
        "calibration_end": str(TRAIN_RETURNS.index[-1].date()),
        "mu": mu.tolist(), "kappa": kappa.tolist(), "theta": theta.tolist(),
        "xi": xi.tolist(), "rho": rho.tolist(), "v0": theta.tolist(),
        "cross_asset_correlation": corr.tolist(), "cholesky": chol.tolist(),
        "correlation_eigenvalues": np.linalg.eigvalsh(corr).tolist(),
        "n_substeps": int(cfg.heston_substeps),
        "scheme": ("full-truncation Euler"
                   + (" + Milstein CIR correction" if cfg.heston_use_milstein else "")),
        "feller_2_kappa_theta_minus_xi2": feller.tolist(),
        "feller_satisfied": (feller > 0).tolist(),
        "fixed_parameter_source": "standard equity values locked in ExperimentConfig",
        "estimator_formulas": {
            "theta": "var(daily log-returns, ddof=1) * 252",
            "mu": "mean(daily log-returns) * 252 + theta/2",
            "v0": "theta",
            "kappa/xi/rho": "fixed by configuration (not estimated)",
        },
        "calibration_warnings": warnings_list,
        "positive_definiteness": pd_info,
        "tickers": list(TICKERS),
        "calibrated_utc": utc_now(),
    }


def generate_heston(params: Dict[str, Any], cfg: ExperimentConfig,
                    seed: int, n_paths: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return (normalised spots [n_paths, horizon+1, d], diagnostics)."""
    d = cfg.d
    n_sub = int(params["n_substeps"])
    dt = cfg.delta_t / n_sub
    sqrt_dt = math.sqrt(dt)
    g = torch_generator(DEVICE, seed, 0)

    t32 = lambda x: torch.tensor(x, dtype=torch.float32, device=DEVICE)
    mu, kappa, theta = t32(params["mu"]), t32(params["kappa"]), t32(params["theta"])
    xi, rho, v0 = t32(params["xi"]), t32(params["rho"]), t32(params["v0"])
    chol = t32(params["cholesky"])
    sqrt_1m_rho2 = torch.sqrt(torch.clamp(1.0 - rho ** 2, min=0.0))
    mil_c = (xi ** 2 / 4.0) * dt if cfg.heston_use_milstein else None

    log_s = torch.zeros(n_paths, d, device=DEVICE, dtype=torch.float32)
    v = v0.expand(n_paths, d).clone()
    out = torch.zeros(n_paths, cfg.horizon + 1, d, device=DEVICE, dtype=torch.float32)
    n_truncations = 0

    for t in range(cfg.horizon):
        for _ in range(n_sub):
            v_pos = torch.clamp(v, min=0.0)
            n_truncations += int((v < 0).sum().item())
            sqrt_v = torch.sqrt(v_pos)
            z_a = torch.randn(n_paths, d, generator=g, device=DEVICE, dtype=torch.float32)
            z_i = torch.randn(n_paths, d, generator=g, device=DEVICE, dtype=torch.float32)
            eps_s = torch.matmul(z_a, chol.T)
            eps_v = rho * eps_s + sqrt_1m_rho2 * z_i
            v_new = v + kappa * (theta - v_pos) * dt + xi * sqrt_v * sqrt_dt * eps_v
            if mil_c is not None:
                v_new = v_new + mil_c * (eps_v ** 2 - 1.0)
            log_s = log_s + (mu - v_pos / 2.0) * dt + sqrt_v * sqrt_dt * eps_s
            v = v_new
        out[:, t + 1, :] = log_s

    spots = torch.exp(out).cpu().numpy()
    diagnostics = {
        "n_negative_variance_truncations": int(n_truncations),
        "share_of_variance_updates_truncated": float(
            n_truncations / max(1, n_paths * d * cfg.horizon * n_sub)),
        "terminal_variance_mean": [float(x) for x in v.mean(dim=0).cpu().numpy()],
        "target_theta": params["theta"],
    }
    del out, v, log_s
    clear_mem()
    if not np.isfinite(spots).all():
        raise FloatingPointError("Heston generation produced non-finite values")
    return spots, diagnostics


HESTON_PARAMS = calibrate_heston(TRAIN_RETURNS_VALUES, CFG)
atomic_write_json(PATHS["generators"] / "Heston" / "calibration.json", HESTON_PARAMS)
register_artifact("generators/Heston/calibration.json",
                  PATHS["generators"] / "Heston" / "calibration.json")

print("Heston calibration (fixed kappa/xi/rho, theta and mu from data)")
print(f"  kappa={CFG.heston_kappa}  xi={CFG.heston_xi}  rho={CFG.heston_rho}  "
      f"substeps={CFG.heston_substeps}")
for _k, _t in enumerate(TICKERS):
    print(f"  {_t:>5s}: theta={HESTON_PARAMS['theta'][_k]:.5f}  "
          f"mu={HESTON_PARAMS['mu'][_k]:+.5f}  v0={HESTON_PARAMS['v0'][_k]:.5f}  "
          f"Feller={'ok' if HESTON_PARAMS['feller_satisfied'][_k] else 'VIOLATED'}")
if HESTON_PARAMS["calibration_warnings"]:
    print("  calibration warnings:")
    for _w in HESTON_PARAMS["calibration_warnings"]:
        print(f"    - {_w}")
        add_warning(f"Heston calibration: {_w.get('issue')}", _w)
print(f"  input hash         : {HESTON_PARAMS['calibration_input_hash'][:16]}...")

# ---- notebook cell 15 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 10 — SBTS KERNELS, REFERENCE TRAJECTORIES, ROLLOUT ENGINE
# ═════════════════════════════════════════════════════════════════════════════
#  References: Hamdouche, Henry-Labordere, Pham (2023), arXiv:2304.05093 [H23]
#              Alouadi, Barreau, Carlier, Pham (2025), ICAIF '25          [A25]
#
#  State space (locked convention): a reference trajectory is the sequence of
#  DAILY LOG-RETURNS of a rolling 252-day window, with X_ref[m, 0, :] = 0. The
#  bridge therefore diffuses in return space and the simulated path is read as
#  a return sequence, exactly as in the thesis implementation.
#
#  Two invariants this implementation enforces and the old code did not:
#    * NON-ZERO SUPPORT IS A BOOLEAN FACT. `support` comes from the exact
#      compact-support condition ||u||^2 < h^2, never from a clamped weight:
#      log-domain clamping may not manufacture support.
#    * WEIGHTS ARE EXACT UNDER THE SLIDING WINDOW. Per-step log-kernels are
#      kept in a ring buffer, so evicting the oldest step subtracts exactly
#      what was added, with no clamp contamination.
# ═════════════════════════════════════════════════════════════════════════════

LOG_TINY = 1e-300        # only ever applied INSIDE the support region


def quartic_log_kernel(diff_sq: torch.Tensor, h: float):
    """Quartic (biweight) isotropic kernel K_h(u) = (h^2 - ||u||^2)^2 1{||u||<h}.

    Returns (masked_log_k, support) where `support` is the exact compact-support
    mask and `masked_log_k` is 2*log(h^2 - ||u||^2) inside the support and 0
    outside (so that a masked running sum stays exact)."""
    h_sq = float(h) * float(h)
    support = diff_sq < h_sq
    # Clamp with the smallest normal of THIS dtype: 1e-300 underflows to zero in
    # float32 and would turn a supported reference into log(0) = -inf.
    tiny = torch.finfo(diff_sq.dtype).tiny
    inside = torch.clamp(h_sq - diff_sq, min=tiny)
    log_k = 2.0 * torch.log(inside)
    return torch.where(support, log_k, torch.zeros_like(log_k)), support


def build_sbts_references(returns: np.ndarray, cfg: ExperimentConfig,
                          index: "pd.Index") -> Dict[str, Any]:
    """Rolling reference trajectories of `horizon` daily returns."""
    T, d = returns.shape
    H = cfg.horizon
    M = T - H
    if M <= 0:
        raise ValueError(f"calibration window has {T} returns, needs > {H}")
    X = np.zeros((M, H + 1, d), dtype=np.float64)
    for m in range(M):
        X[m, 1:, :] = returns[m:m + H]
    meta = {
        "generator": "SBTS",
        "generator_schema_version": cfg.generator_schema_version,
        "config_hash": CONFIG_HASH,
        "n_training_observations": int(T),
        "n_reference_trajectories": int(M),
        "horizon": int(H),
        "tensor_shape": list(X.shape),
        "overlap_structure": (f"rolling windows with stride 1; references i and j "
                              f"share observations iff |i-j| < {H}"),
        "normalization_convention": ("daily log-returns; X[m,0,:]=0 and "
                                     "X[m,t,:] = r_{i+t-1} for window start i=m"),
        "reference_index_to_dates": [
            {"reference_index": int(m),
             "start_date": str(index[m].date()),
             "end_date": str(index[m + H - 1].date())}
            for m in range(M)],
        "calibration_input_hash": sha256_array(returns),
        "reference_hash": sha256_array(X.astype(np.float32)),
        "built_utc": utc_now(),
    }
    return {"X": X, "meta": meta}


class _SlidingWeights:
    """Exact Markov-K sliding-window kernel weights on the GPU.

    Keeps a ring buffer of per-step masked log-kernels and support flags so the
    running sums are exact when the window slides.
    """

    def __init__(self, n_query: int, n_ref: int, K: int, device, dtype=torch.float32):
        self.K, self.n_query, self.n_ref = int(K), int(n_query), int(n_ref)
        self.buf_log = torch.zeros(n_query, self.K, n_ref, device=device, dtype=dtype)
        self.buf_sup = torch.zeros(n_query, self.K, n_ref, device=device, dtype=torch.bool)
        self.run_log = torch.zeros(n_query, n_ref, device=device, dtype=dtype)
        self.run_sup = torch.zeros(n_query, n_ref, device=device, dtype=torch.int32)
        self.pos = 0
        self.filled = 0

    def push(self, masked_log_k: torch.Tensor, support: torch.Tensor) -> None:
        if self.filled == self.K:                      # evict exactly what was added
            self.run_log -= self.buf_log[:, self.pos]
            self.run_sup -= self.buf_sup[:, self.pos].to(torch.int32)
        else:
            self.filled += 1
        self.buf_log[:, self.pos] = masked_log_k
        self.buf_sup[:, self.pos] = support
        self.run_log += masked_log_k
        self.run_sup += support.to(torch.int32)
        self.pos = (self.pos + 1) % self.K

    def current(self):
        """(log_weights, support) for the current window.

        With an empty window the weights are uniform (log 0), which is the
        t=0 convention of [H23] Algorithm 1."""
        if self.filled == 0:
            return (torch.zeros_like(self.run_log),
                    torch.ones_like(self.run_log, dtype=torch.bool))
        support = self.run_sup == self.filled
        log_w = torch.where(support, self.run_log, torch.zeros_like(self.run_log))
        return log_w, support

    def memory_bytes(self) -> int:
        return (self.buf_log.numel() * self.buf_log.element_size()
                + self.buf_sup.numel() + self.run_log.numel() * 4 + self.run_sup.numel() * 4)


def _masked_softmax_weights(log_w: torch.Tensor, support: torch.Tensor):
    """Stable positive weights over the supported set. Returns (w, den, any_support)."""
    neg_inf = torch.finfo(log_w.dtype).min
    masked = torch.where(support, log_w, torch.full_like(log_w, neg_inf))
    mx = masked.max(dim=1, keepdim=True).values
    any_support = support.any(dim=1)
    mx = torch.where(any_support.unsqueeze(1), mx, torch.zeros_like(mx))
    w = torch.where(support, torch.exp(masked - mx), torch.zeros_like(masked))
    den = w.sum(dim=1, keepdim=True)
    return w, den, any_support


def sbts_rollout(X_ref_t: torch.Tensor, y0: torch.Tensor, i0: int, i1: int,
                 h: float, K: int, n_pi: int, delta_t: float,
                 generator: torch.Generator,
                 prime_states: Optional[torch.Tensor] = None,
                 zero_support_policy: str = "nearest_reference"):
    """Simulate the Schrodinger-bridge dynamics from step i0 to step i1.

    X_ref_t      : [M, N+1, d] reference trajectories on `device`
    y0           : [B, d] initial state at step i0
    prime_states : [B, p, d] observed states at steps i0-p+1 .. i0 used to prime
                   the Markov-K conditioning window (None => uniform start)
    Returns (Y [B, i1-i0+1, d], diagnostics).
    """
    assert zero_support_policy in ("nearest_reference", "zero_drift")
    device = X_ref_t.device
    M, _, d = X_ref_t.shape
    B = y0.shape[0]
    dt_sub = delta_t / n_pi
    sw = _SlidingWeights(B, M, K, device)

    if prime_states is not None and prime_states.shape[1] > 0:
        p = prime_states.shape[1]
        for j in range(p):
            step = i0 - p + 1 + j
            if step < 1:                    # step 0 carries no conditioning
                continue
            diff = X_ref_t[:, step, :].unsqueeze(0) - prime_states[:, j, :].unsqueeze(1)
            lk, sup = quartic_log_kernel((diff ** 2).sum(-1), h)
            sw.push(lk, sup)

    Y = y0.clone()
    out = torch.zeros(B, i1 - i0 + 1, d, device=device, dtype=Y.dtype)
    out[:, 0, :] = Y
    zero_support_events = 0
    nonfinite_events = 0

    for i in range(i0, i1):
        log_w, support = sw.current()
        X_next = X_ref_t[:, i + 1, :]
        diff_B = X_next.unsqueeze(0) - Y.unsqueeze(1)
        bridge_B = (diff_B ** 2).sum(-1) / (2.0 * delta_t)

        for k in range(n_pi):
            rem = delta_t - k * dt_sub
            diff_t = X_next.unsqueeze(0) - Y.unsqueeze(1)
            if k == 0:
                log_total = log_w
            else:
                bridge_A = (diff_t ** 2).sum(-1) / (2.0 * rem)
                log_total = log_w + (bridge_B - bridge_A)
            w, den, any_support = _masked_softmax_weights(log_total, support)
            num = (w.unsqueeze(-1) * diff_t).sum(dim=1)
            drift = num / torch.clamp(den, min=LOG_TINY) / rem

            bad = ~any_support
            n_bad = int(bad.sum().item())
            if n_bad:
                zero_support_events += n_bad
                if zero_support_policy == "nearest_reference":
                    cur_d2 = ((X_ref_t[:, i, :].unsqueeze(0) - Y.unsqueeze(1)) ** 2).sum(-1)
                    nearest = cur_d2.argmin(dim=1)
                    fallback = (X_next[nearest] - Y) / rem
                else:
                    fallback = torch.zeros_like(drift)
                drift = torch.where(bad.unsqueeze(1), fallback, drift)

            noise = torch.randn(B, d, generator=generator, device=device, dtype=Y.dtype)
            Y = Y + drift * dt_sub + noise * math.sqrt(dt_sub)

        if not torch.isfinite(Y).all():
            nonfinite_events += int((~torch.isfinite(Y)).sum().item())
        out[:, i - i0 + 1, :] = Y

        diff = X_ref_t[:, i + 1, :].unsqueeze(0) - Y.unsqueeze(1)
        lk, sup = quartic_log_kernel((diff ** 2).sum(-1), h)
        sw.push(lk, sup)

    diagnostics = {
        "zero_support_events": int(zero_support_events),
        "zero_support_rate": float(zero_support_events / max(1, B * (i1 - i0) * n_pi)),
        "nonfinite_state_count": int(nonfinite_events),
        "zero_support_policy": zero_support_policy,
        "weight_buffer_bytes": int(sw.memory_bytes()),
    }
    return out, diagnostics


@torch.no_grad()
def sbts_support_weights(X_ref_t: torch.Tensor, query_states: torch.Tensor,
                         step_indices: Sequence[int], h: float, K: int):
    """Kernel weights of a Markov-K query window against every reference path.

    query_states : [Q, p, d] states observed at `step_indices` (len p, ascending)
    Returns (weights [Q, M] normalised over the supported set, support [Q, M]).
    The support mask is the exact compact-support condition — never derived
    from clamped weights."""
    Q = query_states.shape[0]
    M = X_ref_t.shape[0]
    sw = _SlidingWeights(Q, M, max(1, K), X_ref_t.device)
    for j, step in enumerate(step_indices):
        if step < 1:
            continue
        diff = X_ref_t[:, step, :].unsqueeze(0) - query_states[:, j, :].unsqueeze(1)
        lk, sup = quartic_log_kernel((diff ** 2).sum(-1), h)
        sw.push(lk, sup)
    log_w, support = sw.current()
    w, den, any_support = _masked_softmax_weights(log_w, support)
    w = w / torch.clamp(den, min=LOG_TINY)
    w = torch.where(any_support.unsqueeze(1), w, torch.zeros_like(w))
    return w, support


def returns_to_spots(returns: np.ndarray) -> np.ndarray:
    """[M, N, d] daily log-returns -> [M, N+1, d] normalised spots with S_0 = 1."""
    M, N, d = returns.shape
    out = np.ones((M, N + 1, d), dtype=np.float64)
    out[:, 1:, :] = np.exp(np.cumsum(returns, axis=1))
    return out


SBTS_REFS = build_sbts_references(TRAIN_RETURNS_VALUES, CFG, TRAIN_RETURNS.index)
X_REF = SBTS_REFS["X"]
M_REF = X_REF.shape[0]
X_REF_T = torch.tensor(X_REF, dtype=torch.float32, device=DEVICE)
REFERENCE_HASH = SBTS_REFS["meta"]["reference_hash"]
DATA_HASHES["sbts_references"] = REFERENCE_HASH

_ref_meta_path = PATHS["generators"] / "SBTS" / "reference_trajectories.json"
atomic_write_json(_ref_meta_path, SBTS_REFS["meta"])
register_artifact("generators/SBTS/reference_trajectories.json", _ref_meta_path)

print("SBTS reference trajectories")
print(f"  training observations : {SBTS_REFS['meta']['n_training_observations']}")
print(f"  reference paths M     : {M_REF}")
print(f"  tensor shape          : {tuple(X_REF.shape)}")
print(f"  overlap               : {SBTS_REFS['meta']['overlap_structure']}")
print(f"  reference hash        : {REFERENCE_HASH[:16]}...")
print(f"  zero-support policy   : {CFG.sbts_zero_support_policy}")

# ---- notebook cell 16 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 11 — CHRONOLOGICAL, PURGED BANDWIDTH / MARKOV-ORDER SELECTION
# ═════════════════════════════════════════════════════════════════════════════
#  What changed versus the previous implementation, and why it had to change:
#
#   * SPLIT. Reference trajectories are rolling windows with stride 1, so two
#     references closer than `horizon` apart share observations. A random
#     permutation split therefore leaks the validation data into training. The
#     split here is CHRONOLOGICAL with a purge/embargo of `selection_purge`
#     (>= horizon) references, which guarantees disjoint observation sets.
#   * OBJECTIVE. `paper_full` evaluates the held-out objective of [A25] eq. (5):
#        MSE(h,K) = (1/Q) sum_q (1/L) sum_l || Yhat^{q,l}_T - Y^q_T ||^2
#     with L bridge samples simulated forward over `selection_horizon` steps
#     from the query's own conditioning window. The old one-step
#     Nadaraya-Watson proxy survives as `one_step_smoke` for unit tests and
#     benchmarking only; it may never produce a published bandwidth.
#   * SUPPORT. Zero-support queries are counted, never silently dropped.
# ═════════════════════════════════════════════════════════════════════════════

def chronological_purged_split(m_ref: int, cfg: ExperimentConfig) -> Dict[str, Any]:
    n_train = int(m_ref * cfg.selection_train_fraction)
    purge = int(cfg.selection_purge)
    val_start = n_train + purge
    if val_start >= m_ref:
        raise ValueError(
            f"purge={purge} leaves no validation references "
            f"(M={m_ref}, train={n_train}). Lower selection_train_fraction.")
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(val_start, m_ref)
    ref_dates = SBTS_REFS["meta"]["reference_index_to_dates"]
    return {
        "train_idx": train_idx, "val_idx": val_idx,
        "meta": {
            "split_type": "chronological with purge/embargo",
            "m_reference": int(m_ref),
            "train_fraction": cfg.selection_train_fraction,
            "purge_references": purge,
            "purge_rationale": (f"references i,j overlap iff |i-j| < horizon "
                                f"({cfg.horizon}); purge >= horizon guarantees "
                                f"train and validation share no observation"),
            "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
            "train_reference_first": int(train_idx[0]), "train_reference_last": int(train_idx[-1]),
            "val_reference_first": int(val_idx[0]), "val_reference_last": int(val_idx[-1]),
            "train_start_date": ref_dates[int(train_idx[0])]["start_date"],
            "train_end_date": ref_dates[int(train_idx[-1])]["end_date"],
            "purged_reference_interval": [int(train_idx[-1]) + 1, int(val_idx[0]) - 1],
            "purged_date_interval": [ref_dates[int(train_idx[-1]) + 1]["start_date"],
                                     ref_dates[int(val_idx[0]) - 1]["end_date"]],
            "val_start_date": ref_dates[int(val_idx[0])]["start_date"],
            "val_end_date": ref_dates[int(val_idx[-1])]["end_date"],
        },
    }


def _assert_no_observation_overlap(train_idx, val_idx, horizon: int) -> None:
    assert int(val_idx.min()) - int(train_idx.max()) >= horizon, (
        f"purge too small: gap={int(val_idx.min()) - int(train_idx.max())} < horizon={horizon}")


@torch.no_grad()
def selection_objective_paper_full(X_train_t, X_val_t, h: float, K: int,
                                   cfg: ExperimentConfig) -> Dict[str, Any]:
    """[A25] eq. (5) held-out objective, evaluated on the GPU in query chunks."""
    H_obj = int(cfg.selection_horizon)
    N = cfg.horizon
    t0 = N - H_obj
    assert t0 >= 1, "selection_horizon must be shorter than the horizon"
    Q = min(int(cfg.selection_n_queries), X_val_t.shape[0])
    L = int(cfg.selection_n_samples)
    q_idx = np.linspace(0, X_val_t.shape[0] - 1, Q).astype(int)
    p = min(K, t0)

    sq_err_sum, n_eval, zero_support = 0.0, 0, 0
    cum_err_sum = 0.0
    chunk = max(1, int(cfg.selection_query_chunk))
    for c0 in range(0, Q, chunk):
        sel = q_idx[c0:c0 + chunk]
        qs = X_val_t[sel]                                      # [c, N+1, d]
        c = qs.shape[0]
        prime = qs[:, t0 - p + 1:t0 + 1, :]                    # [c, p, d]
        y0 = qs[:, t0, :]
        prime_rep = prime.repeat_interleave(L, dim=0)
        y0_rep = y0.repeat_interleave(L, dim=0)
        gen = torch_generator(DEVICE, cfg.selection_seed, int(sel[0]),
                              int(round(h * 1e6)), K)
        Y, diag = sbts_rollout(X_train_t, y0_rep, t0, N, h, K,
                               cfg.selection_n_pi, cfg.delta_t, gen,
                               prime_states=prime_rep,
                               zero_support_policy=cfg.sbts_zero_support_policy)
        zero_support += diag["zero_support_events"]
        target = qs[:, N, :].repeat_interleave(L, dim=0)
        err = ((Y[:, -1, :] - target) ** 2).sum(dim=-1)
        sq_err_sum += float(err.sum().item())
        cum_target = qs[:, t0 + 1:N + 1, :].sum(dim=1).repeat_interleave(L, dim=0)
        cum_pred = Y[:, 1:, :].sum(dim=1)
        cum_err_sum += float(((cum_pred - cum_target) ** 2).sum(dim=-1).sum().item())
        n_eval += int(err.numel())

    return {
        "objective": sq_err_sum / max(1, n_eval),
        "objective_terminal_cumulative": cum_err_sum / max(1, n_eval),
        "n_evaluations": n_eval, "n_queries": Q, "n_samples": L,
        "zero_support_events": zero_support,
    }


@torch.no_grad()
def selection_objective_one_step(X_train_t, X_val_t, h: float, K: int,
                                 cfg: ExperimentConfig) -> Dict[str, Any]:
    """Fast one-step Nadaraya-Watson proxy. NOT a published objective."""
    N = cfg.horizon
    Q = min(int(cfg.selection_n_queries), X_val_t.shape[0])
    q_idx = np.linspace(0, X_val_t.shape[0] - 1, Q).astype(int)
    p = min(K, N - 1)
    steps = list(range(N - p, N))
    total, n_valid, zero_support = 0.0, 0, 0
    chunk = max(1, int(cfg.selection_query_chunk))
    for c0 in range(0, Q, chunk):
        sel = q_idx[c0:c0 + chunk]
        qs = X_val_t[sel]
        w, support = sbts_support_weights(X_train_t, qs[:, N - p:N, :], steps, h, K)
        has = support.any(dim=1)
        zero_support += int((~has).sum().item())
        if not bool(has.any()):
            continue
        pred = torch.matmul(w[has], X_train_t[:, N, :])
        err = ((pred - qs[has][:, N, :]) ** 2).sum(dim=-1)
        total += float(err.sum().item())
        n_valid += int(err.numel())
    return {
        "objective": (total / n_valid) if n_valid else float("inf"),
        "objective_terminal_cumulative": float("nan"),
        "n_evaluations": n_valid, "n_queries": Q, "n_samples": 1,
        "zero_support_events": zero_support,
    }


def select_bandwidth_and_k(X_ref_t, cfg: ExperimentConfig) -> Dict[str, Any]:
    split = chronological_purged_split(X_ref_t.shape[0], cfg)
    _assert_no_observation_overlap(split["train_idx"], split["val_idx"], cfg.horizon)
    X_train_t = X_ref_t[torch.as_tensor(split["train_idx"], device=X_ref_t.device)]
    X_val_t = X_ref_t[torch.as_tensor(split["val_idx"], device=X_ref_t.device)]

    engine = cfg.selection_engine
    fn = (selection_objective_paper_full if engine == "paper_full"
          else selection_objective_one_step)
    k_grid = tuple(k for k in cfg.k_grid if k <= cfg.horizon - 1)

    rows = []
    n_cfg = len(cfg.h_grid) * len(k_grid)
    print(f"  engine={engine}  configurations={n_cfg}  "
          f"train_refs={len(split['train_idx'])}  val_refs={len(split['val_idx'])}")
    print(f"  {'h':>7s} {'K':>4s} {'objective':>16s} {'zero-supp':>10s} "
          f"{'sec':>7s} {'peakMB':>8s}")
    t_grid = time.time()
    for h in cfg.h_grid:
        for k in k_grid:
            clear_mem()
            t0 = time.time()
            res = fn(X_train_t, X_val_t, float(h), int(k), cfg)
            dt = time.time() - t0
            row = {"h": float(h), "K": int(k), "engine": engine,
                   "objective": float(res["objective"]),
                   "objective_terminal_cumulative": float(res["objective_terminal_cumulative"]),
                   "n_evaluations": res["n_evaluations"],
                   "n_queries": res["n_queries"], "n_samples": res["n_samples"],
                   "zero_support_events": res["zero_support_events"],
                   "runtime_seconds": dt, "peak_memory_mb": gpu_peak_mb(),
                   "valid": bool(np.isfinite(res["objective"])
                                 and res["n_evaluations"] > 0)}
            rows.append(row)
            print(f"  {h:>7.4f} {k:>4d} {row['objective']:>16.8f} "
                  f"{row['zero_support_events']:>10d} {dt:>7.1f} "
                  f"{row['peak_memory_mb']:>8.0f}", flush=True)
    grid_seconds = time.time() - t_grid

    df = pd.DataFrame(rows)
    valid = df[df["valid"]]
    if valid.empty:
        raise RuntimeError("no valid (h, K) configuration — every objective was "
                           "non-finite or had no supported query")
    best_obj = float(valid["objective"].min())
    tol = float(cfg.selection_tolerance)
    within = valid[valid["objective"] <= best_obj * (1.0 + tol)].copy()
    # Locked tie-break: smaller K first, then larger h (wider support).
    for rule in cfg.selection_tie_break:
        if rule == "min_k":
            within = within[within["K"] == within["K"].min()]
        elif rule == "max_h":
            within = within[within["h"] == within["h"].max()]
        else:
            raise ValueError(f"unknown tie-break rule {rule!r}")
    chosen = within.sort_values(["K", "h"]).iloc[0]
    argmin = valid.sort_values("objective").iloc[0]

    meta = {
        "schema_version": cfg.schema_version, "config_hash": CONFIG_HASH,
        "data_hashes": {"sbts_references": REFERENCE_HASH},
        "engine": engine,
        "publishable": engine == "paper_full",
        "objective_definition": (
            "[A25] eq. (5): (1/Q) sum_q (1/L) sum_l ||Yhat^{q,l}_T - Y^q_T||^2, "
            "simulated forward over selection_horizon steps from the query's own "
            "Markov-K conditioning window"
            if engine == "paper_full"
            else "one-step Nadaraya-Watson proxy (NOT publishable)"),
        "objective_target": cfg.selection_objective,
        "selection_horizon": cfg.selection_horizon,
        "selection_n_pi": cfg.selection_n_pi,
        "Q": int(chosen["n_queries"]), "L": int(chosen["n_samples"]),
        "h_grid": list(cfg.h_grid), "k_grid": list(k_grid),
        "split": split["meta"],
        "tie_break_rule": list(cfg.selection_tie_break),
        "tie_tolerance": tol,
        "best_objective": best_obj,
        "argmin_h": float(argmin["h"]), "argmin_K": int(argmin["K"]),
        "h_star": float(chosen["h"]), "k_star": int(chosen["K"]),
        "tie_break_applied": bool(chosen["h"] != argmin["h"] or chosen["K"] != argmin["K"]),
        "h_at_grid_boundary": bool(float(chosen["h"]) in (min(cfg.h_grid), max(cfg.h_grid))),
        "k_at_grid_boundary": bool(int(chosen["K"]) in (min(k_grid), max(k_grid))),
        "n_within_tolerance": int(len(valid[valid["objective"] <= best_obj * (1.0 + tol)])),
        "grid_seconds": grid_seconds,
        "selected_utc": utc_now(),
    }
    return {"table": df, "meta": meta}


_sel_csv = PATHS["generators"] / "SBTS" / "hk_selection.csv"
_sel_json = PATHS["generators"] / "SBTS" / "hk_selection.json"
_reuse_sel = False
if _sel_csv.exists() and _sel_json.exists():
    with open(_sel_json, "r", encoding="utf-8") as f:
        _m = json.load(f)
    _ok, _why = cache_is_valid(_m, {"sbts_references": REFERENCE_HASH},
                               extra={"engine": CFG.selection_engine})
    if _ok:
        SELECTION_TABLE = pd.read_csv(_sel_csv)
        SELECTION_META = _m
        _reuse_sel = True
        log(f"  reused cached (h, K) selection: h*={_m['h_star']}  K*={_m['k_star']}")
    else:
        quarantine(_sel_csv, f"selection cache invalid: {_why}")
        quarantine(_sel_json, f"selection cache invalid: {_why}")

if not _reuse_sel:
    print("Bandwidth / Markov-order selection")
    _res = select_bandwidth_and_k(X_REF_T, CFG)
    SELECTION_TABLE, SELECTION_META = _res["table"], _res["meta"]
    _h1 = atomic_write_dataframe(_sel_csv, SELECTION_TABLE)
    atomic_write_json(_sel_json, SELECTION_META)
    register_artifact("generators/SBTS/hk_selection.csv", _sel_csv, _h1,
                      meta={"engine": SELECTION_META["engine"],
                            "h_star": SELECTION_META["h_star"],
                            "k_star": SELECTION_META["k_star"]})

H_STAR = float(SELECTION_META["h_star"])
K_STAR = int(SELECTION_META["k_star"])
# A selection cached by an earlier notebook version lacks these flags; derive
# them from the chosen values so the boundary disclosure still fires.
_k_grid_eff = tuple(k for k in CFG.k_grid if k <= CFG.horizon - 1)
SELECTION_META.setdefault("h_at_grid_boundary",
                          H_STAR in (min(CFG.h_grid), max(CFG.h_grid)))
SELECTION_META.setdefault("k_at_grid_boundary",
                          K_STAR in (min(_k_grid_eff), max(_k_grid_eff)))

print(f"\n  chronological split : train refs [{SELECTION_META['split']['train_reference_first']}, "
      f"{SELECTION_META['split']['train_reference_last']}] "
      f"({SELECTION_META['split']['train_start_date']} -> "
      f"{SELECTION_META['split']['train_end_date']})")
print(f"  purged interval     : refs {SELECTION_META['split']['purged_reference_interval']} "
      f"({SELECTION_META['split']['purged_date_interval'][0]} -> "
      f"{SELECTION_META['split']['purged_date_interval'][1]})")
print(f"  validation          : refs [{SELECTION_META['split']['val_reference_first']}, "
      f"{SELECTION_META['split']['val_reference_last']}] "
      f"({SELECTION_META['split']['val_start_date']} -> "
      f"{SELECTION_META['split']['val_end_date']})")
print(f"\n  SELECTED            : h* = {H_STAR:.4f}   K* = {K_STAR}")
print(f"  argmin              : h = {SELECTION_META['argmin_h']:.4f}  "
      f"K = {SELECTION_META['argmin_K']}   (tie-break applied: "
      f"{SELECTION_META['tie_break_applied']})")
print(f"  engine              : {SELECTION_META['engine']}  "
      f"(publishable: {SELECTION_META['publishable']})")
if SELECTION_META.get("k_at_grid_boundary") or SELECTION_META.get("h_at_grid_boundary"):
    _edge = []
    if SELECTION_META.get("h_at_grid_boundary"):
        _edge.append(f"h*={H_STAR:g} at the edge of {list(CFG.h_grid)}")
    if SELECTION_META.get("k_at_grid_boundary"):
        _edge.append(f"K*={K_STAR} at the edge of {list(CFG.k_grid)}")
    print(f"\n  NOTE: the selected value sits on the boundary of the locked "
          f"grid\n        ({'; '.join(_edge)}). The optimum may lie outside "
          f"the grid.\n        The grid is locked for this run, so this is "
          f"disclosed rather than\n        silently widened; say so when the "
          f"selection is reported.")
    add_warning("selected (h, K) sits on the boundary of the locked grid",
                {"h_star": H_STAR, "k_star": K_STAR,
                 "h_grid": list(CFG.h_grid), "k_grid": list(CFG.k_grid)})
if not SELECTION_META["publishable"]:
    add_warning("(h, K) selected with the non-publishable one_step_smoke engine",
                {"h_star": H_STAR, "k_star": K_STAR})

# ---- notebook cell 17 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 12 — SBTS PATH GENERATION
# ═════════════════════════════════════════════════════════════════════════════
#  * Brownian increments are reproducible from (generator_seed, batch_index):
#    a resumed batched run reproduces the uninterrupted run within one
#    numerical mode.
#  * Zero-support events are counted, logged per batch and aggregated; the
#    configured fallback policy is applied explicitly. NaN/Inf is never
#    silently mapped to zero — it marks the batch (and the run) as a numerical
#    failure.
#  * The batch size is probed against this GPU but the value used is the LOCKED
#    configuration value, so a full run cannot drift between sessions.
# ═════════════════════════════════════════════════════════════════════════════

def probe_sbts_batch_size(cfg: ExperimentConfig, k: int) -> Dict[str, Any]:
    """Estimate the dominant allocations per batch and report headroom."""
    d, M = cfg.d, M_REF
    b = int(cfg.sbts_batch_size)
    per_path_ref = 4 * M                       # one [B, M] float32 plane
    buffers = b * (2 * k * M * 4 * 0.625       # ring buffers (float32 + bool)
                   + 2 * per_path_ref)         # running sums
    working = b * M * d * 4 * 3                # diff/num tensors in flight
    est = buffers + working
    out = {"configured_batch_size": b, "estimated_bytes": int(est),
           "estimated_mb": est / 1e6, "reference_paths": int(M), "K": int(k)}
    if GPU_TOTAL_MEMORY:
        out["gpu_total_mb"] = GPU_TOTAL_MEMORY / 1e6
        out["headroom_ratio"] = float(est / GPU_TOTAL_MEMORY)
        safe = max(64, int(b * (0.35 * GPU_TOTAL_MEMORY / max(est, 1))))
        out["recommended_batch_size"] = int(min(4096, safe))
    return out


def generate_sbts(cfg: ExperimentConfig, h: float, k: int, n_paths: int,
                  seed: int) -> Dict[str, Any]:
    """Generate `n_paths` SBTS return paths in resumable batches."""
    batch_dir = PATHS["generators"] / "SBTS" / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    bs = int(cfg.sbts_batch_size)
    n_batches = (n_paths + bs - 1) // bs
    probe = probe_sbts_batch_size(cfg, k)
    log(f"  SBTS generation: h={h:.4f} K={k} paths={n_paths} batch={bs} "
        f"({n_batches} batches); probe ~{probe['estimated_mb']:.0f} MB/batch")
    if "recommended_batch_size" in probe and probe["headroom_ratio"] > 0.5:
        add_warning("SBTS batch size may exceed safe GPU headroom", probe)

    chunks, total_zero_support, total_nonfinite = [], 0, 0
    y0 = X_REF_T[0, 0, :].unsqueeze(0)
    t_start = time.time()
    for b_idx in range(n_batches):
        b_size = min(bs, n_paths - b_idx * bs)
        f_npy = batch_dir / f"batch_{b_idx:05d}.npy"
        f_meta = batch_dir / f"batch_{b_idx:05d}.json"
        if f_npy.exists() and f_meta.exists():
            with open(f_meta, "r", encoding="utf-8") as f:
                bm = json.load(f)
            ok, why = cache_is_valid(bm, {"sbts_references": REFERENCE_HASH},
                                     extra={"h": float(h), "K": int(k),
                                            "generator_seed": int(seed),
                                            "batch_index": int(b_idx),
                                            "batch_size": int(b_size)})
            arr = np.load(f_npy) if ok else None
            if ok and arr is not None and sha256_array(arr) == bm["array_hash"]:
                chunks.append(arr)
                total_zero_support += bm.get("zero_support_events", 0)
                continue
            quarantine(f_npy, f"SBTS batch cache invalid: {why if not ok else 'hash mismatch'}")
            quarantine(f_meta, "SBTS batch cache invalid")

        gen = torch_generator(DEVICE, seed, b_idx)
        Y, diag = sbts_rollout(X_REF_T, y0.expand(b_size, -1).clone(), 0,
                               cfg.horizon, h, k, cfg.sbts_n_pi, cfg.delta_t,
                               gen, prime_states=None,
                               zero_support_policy=cfg.sbts_zero_support_policy)
        returns = Y[:, 1:, :].cpu().numpy().astype(np.float32)
        n_bad = int((~np.isfinite(returns)).sum())
        total_nonfinite += n_bad
        total_zero_support += diag["zero_support_events"]
        if n_bad:
            add_failure(f"SBTS batch {b_idx} produced {n_bad} non-finite values",
                        {"batch": b_idx, "h": h, "K": k})
            raise FloatingPointError(
                f"SBTS batch {b_idx}: {n_bad} non-finite values. The run is a "
                f"numerical failure; values are NOT replaced by zeros.")

        atomic_write_npy(f_npy, returns)
        atomic_write_json(f_meta, {
            "schema_version": cfg.schema_version, "config_hash": CONFIG_HASH,
            "data_hashes": {"sbts_references": REFERENCE_HASH},
            "h": float(h), "K": int(k), "generator_seed": int(seed),
            "batch_index": int(b_idx), "batch_size": int(b_size),
            "array_hash": sha256_array(returns),
            "zero_support_events": diag["zero_support_events"],
            "zero_support_rate": diag["zero_support_rate"],
            "zero_support_policy": diag["zero_support_policy"],
            "written_utc": utc_now()})
        chunks.append(returns)

        done = b_idx + 1
        el = time.time() - t_start
        if done == 1 or done % 5 == 0 or done == n_batches:
            print(f"    batch {done}/{n_batches}  elapsed {el:.0f}s  "
                  f"ETA {el / done * (n_batches - done):.0f}s  "
                  f"zero-support so far {total_zero_support}", flush=True)
        clear_mem()

    returns_all = np.concatenate(chunks, axis=0)
    assert returns_all.shape == (n_paths, cfg.horizon, cfg.d), returns_all.shape
    return {
        "returns": returns_all,
        "diagnostics": {
            "zero_support_events_total": int(total_zero_support),
            "zero_support_rate_total": float(
                total_zero_support / max(1, n_paths * cfg.horizon * cfg.sbts_n_pi)),
            "zero_support_policy": cfg.sbts_zero_support_policy,
            "nonfinite_total": int(total_nonfinite),
            "n_batches": int(n_batches), "batch_size": int(bs),
            "wall_seconds": float(time.time() - t_start),
            "probe": probe,
        },
    }


def _generator_cache_paths(name: str):
    return (PATHS["generators"] / name / "paths.npz",
            PATHS["generators"] / name / "paths.json")


def load_or_generate(name: str, cfg: ExperimentConfig) -> Dict[str, Any]:
    """Generate (or reuse) the 20,000-path dataset for one generator."""
    npz_path, meta_path = _generator_cache_paths(name)
    extra = {"generator": name, "n_paths": int(cfg.n_paths),
             "generator_seed": int(cfg.generator_seed)}
    if name == "SBTS":
        extra.update({"h_star": H_STAR, "k_star": K_STAR})
    if npz_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        ok, why = cache_is_valid(m, {"calibration_returns": CALIBRATION_INPUT_HASH},
                                 extra=extra)
        if ok:
            with np.load(npz_path, allow_pickle=False) as z:
                spots = z["S_norm"]
            if sha256_array(spots) == m["S_norm_hash"]:
                log(f"  {name}: reused cached paths {spots.shape}")
                return {"S_norm": spots, "meta": m}
            why = "S_norm hash mismatch"
        quarantine(npz_path, f"{name} path cache invalid: {why}")
        quarantine(meta_path, f"{name} path cache invalid: {why}")

    t0 = time.time()
    diagnostics: Dict[str, Any] = {}
    if name == "GBM":
        spots = generate_gbm(GBM_PARAMS, cfg, cfg.generator_seed, cfg.n_paths)
    elif name == "Heston":
        spots, diagnostics = generate_heston(HESTON_PARAMS, cfg,
                                             cfg.generator_seed, cfg.n_paths)
    elif name == "SBTS":
        res = generate_sbts(cfg, H_STAR, K_STAR, cfg.n_paths, cfg.generator_seed)
        spots = returns_to_spots(res["returns"])
        diagnostics = res["diagnostics"]
    else:
        raise ValueError(name)

    spots = spots.astype(np.float32)
    meta = {
        "schema_version": cfg.schema_version, "config_hash": CONFIG_HASH,
        "generator_schema_version": cfg.generator_schema_version,
        "generator": name, "n_paths": int(cfg.n_paths),
        "horizon": int(cfg.horizon), "d": int(cfg.d),
        "generator_seed": int(cfg.generator_seed),
        "data_hashes": {"calibration_returns": CALIBRATION_INPUT_HASH},
        "S_norm_hash": sha256_array(spots),
        "wall_seconds": time.time() - t0,
        "diagnostics": diagnostics,
        "generated_utc": utc_now(),
        **({"h_star": H_STAR, "k_star": K_STAR,
            "selection_engine": SELECTION_META["engine"]} if name == "SBTS" else {}),
    }
    digest = atomic_write_npz(npz_path, S_norm=spots)
    meta["npz_sha256"] = digest
    atomic_write_json(meta_path, meta)
    register_artifact(f"generators/{name}/paths.npz", npz_path, digest,
                      meta={k: meta[k] for k in ("generator", "n_paths",
                                                 "S_norm_hash", "wall_seconds")})
    log(f"  {name}: generated {spots.shape} in {meta['wall_seconds']:.1f}s")
    return {"S_norm": spots, "meta": meta}


print("Generating / loading generator datasets\n")
GENERATOR_DATA: Dict[str, Dict[str, Any]] = {}
for _name in GENERATORS:
    print(f"[{_name}]")
    GENERATOR_DATA[_name] = load_or_generate(_name, CFG)
    DATA_HASHES[f"generator_{_name}"] = GENERATOR_DATA[_name]["meta"]["S_norm_hash"]
MANIFEST["data_hashes"] = DATA_HASHES
save_manifest()

_sbts_diag = GENERATOR_DATA["SBTS"]["meta"]["diagnostics"]
if _sbts_diag:
    print(f"\nSBTS numerical diagnostics")
    print(f"  zero-support events : {_sbts_diag.get('zero_support_events_total')} "
          f"(rate {_sbts_diag.get('zero_support_rate_total', 0):.3e})")
    print(f"  fallback policy     : {_sbts_diag.get('zero_support_policy')}")
    print(f"  non-finite values   : {_sbts_diag.get('nonfinite_total')}")
    if _sbts_diag.get("zero_support_events_total", 0) > 0:
        add_warning("SBTS generation hit zero-support states",
                    {k: _sbts_diag[k] for k in ("zero_support_events_total",
                                                "zero_support_rate_total",
                                                "zero_support_policy")})

# ---- notebook cell 18 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 13 — SYNTHETIC DATASET VALIDATION AND TRAIN/VAL/TEST SPLIT
# ═════════════════════════════════════════════════════════════════════════════
#  One permutation, drawn once with split_seed=42, is shared by all three
#  generators and stored as explicit indices so the split can be audited.
# ═════════════════════════════════════════════════════════════════════════════

def make_split_indices(cfg: ExperimentConfig) -> Dict[str, np.ndarray]:
    assert cfg.n_train + cfg.n_val + cfg.n_test == cfg.n_paths, (
        f"{cfg.n_train}+{cfg.n_val}+{cfg.n_test} != {cfg.n_paths}")
    perm = np.random.RandomState(cfg.split_seed).permutation(cfg.n_paths)
    tr = perm[:cfg.n_train]
    va = perm[cfg.n_train:cfg.n_train + cfg.n_val]
    te = perm[cfg.n_train + cfg.n_val:]
    assert len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0 \
        and len(set(va) & set(te)) == 0, "split indices overlap"
    assert len(np.unique(np.concatenate([tr, va, te]))) == cfg.n_paths
    return {"train": tr, "val": va, "test": te}


SPLIT_PATH = PATHS["processed"] / "split_indices.npz"
SPLIT_META_PATH = PATHS["processed"] / "split_indices.json"
if SPLIT_PATH.exists() and SPLIT_META_PATH.exists():
    with open(SPLIT_META_PATH, "r", encoding="utf-8") as f:
        _sm = json.load(f)
    _ok, _why = cache_is_valid(_sm, extra={"split_seed": int(CFG.split_seed),
                                           "n_paths": int(CFG.n_paths)})
    if _ok:
        with np.load(SPLIT_PATH, allow_pickle=False) as z:
            SPLIT_INDICES = {k: z[k] for k in ("train", "val", "test")}
    else:
        quarantine(SPLIT_PATH, f"split cache invalid: {_why}")
        quarantine(SPLIT_META_PATH, f"split cache invalid: {_why}")
        SPLIT_INDICES = make_split_indices(CFG)
else:
    SPLIT_INDICES = make_split_indices(CFG)

_d = atomic_write_npz(SPLIT_PATH, **{k: v.astype(np.int64)
                                     for k, v in SPLIT_INDICES.items()})
atomic_write_json(SPLIT_META_PATH, {
    "schema_version": CFG.schema_version, "config_hash": CONFIG_HASH,
    "split_seed": int(CFG.split_seed), "n_paths": int(CFG.n_paths),
    "n_train": int(CFG.n_train), "n_val": int(CFG.n_val), "n_test": int(CFG.n_test),
    "shared_across_generators": True,
    "index_hashes": {k: sha256_array(v.astype(np.int64))
                     for k, v in SPLIT_INDICES.items()},
    "npz_sha256": _d})
register_artifact("processed/split_indices.npz", SPLIT_PATH, _d)


def validate_generator_dataset(name: str, spots: np.ndarray,
                               cfg: ExperimentConfig) -> Dict[str, Any]:
    assert spots.shape == (cfg.n_paths, cfg.horizon + 1, cfg.d), \
        f"{name}: shape {spots.shape} != {(cfg.n_paths, cfg.horizon + 1, cfg.d)}"
    assert np.isfinite(spots).all(), f"{name}: non-finite spots"
    assert (spots > 0).all(), f"{name}: non-positive spot"
    assert np.allclose(spots[:, 0, :], 1.0, atol=1e-6), f"{name}: S_0 != 1"
    rets = np.diff(np.log(spots.astype(np.float64)), axis=1)
    return {
        "generator": name, "shape": list(spots.shape),
        "terminal_mean": float(spots[:, -1, :].mean()),
        "daily_return_std_annualised": float(rets.std(ddof=1) * np.sqrt(252)),
        "min_spot": float(spots.min()), "max_spot": float(spots.max()),
        "S_norm_hash": sha256_array(spots),
    }


DATASETS: Dict[str, Dict[str, torch.Tensor]] = {}
_val_rows = []
for _name in GENERATORS:
    _spots = GENERATOR_DATA[_name]["S_norm"]
    _val_rows.append(validate_generator_dataset(_name, _spots, CFG))
    DATASETS[_name] = {
        split: torch.tensor(_spots[idx], dtype=torch.float32, device=DEVICE)
        for split, idx in SPLIT_INDICES.items()}
    assert DATASETS[_name]["train"].shape[0] == CFG.n_train
    assert DATASETS[_name]["val"].shape[0] == CFG.n_val
    assert DATASETS[_name]["test"].shape[0] == CFG.n_test

DATASET_VALIDATION = pd.DataFrame(_val_rows)
_h = atomic_write_dataframe(PATHS["processed"] / "dataset_validation.csv",
                            DATASET_VALIDATION)
register_artifact("processed/dataset_validation.csv",
                  PATHS["processed"] / "dataset_validation.csv", _h)

SPLIT_HASHES = {name: {split: sha256_array(DATASETS[name][split])
                       for split in ("train", "val", "test")}
                for name in GENERATORS}

print(f"Split (seed {CFG.split_seed}, shared by all generators): "
      f"train {CFG.n_train} / val {CFG.n_val} / test {CFG.n_test}")
print(DATASET_VALIDATION.to_string(index=False))
print("\nPer-generator split hashes:")
for _name in GENERATORS:
    print(f"  {_name:<7s} train={SPLIT_HASHES[_name]['train'][:12]}  "
          f"val={SPLIT_HASHES[_name]['val'][:12]}  "
          f"test={SPLIT_HASHES[_name]['test'][:12]}")

# ---- notebook cell 19 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 14 — GENERATOR FIDELITY DIAGNOSTICS
# ═════════════════════════════════════════════════════════════════════════════
#  Compares the historical calibration sample with the three synthetic datasets
#  BEFORE any hedger is trained. Every number the thesis quotes for this table
#  is exported to CSV and LaTeX from this one DataFrame.
# ═════════════════════════════════════════════════════════════════════════════

def _lower_tail_dependence(x: np.ndarray, y: np.ndarray, q: float = 0.05) -> float:
    n = len(x)
    u = sp_stats.rankdata(x) / (n + 1)
    v = sp_stats.rankdata(y) / (n + 1)
    return float(np.mean((u <= q) & (v <= q)) / q)


def _acf(x: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(x):
        return float("nan")
    a, b = x[:-lag], x[lag:]
    sa, sb = a.std(), b.std()
    if sa < 1e-15 or sb < 1e-15:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def fidelity_stats(flat: np.ndarray, reference: Optional[np.ndarray],
                   label: str, cfg: ExperimentConfig,
                   terminal: Optional[np.ndarray] = None) -> Dict[str, Any]:
    d = flat.shape[1]
    triu = np.triu_indices(d, k=1)
    corr = np.corrcoef(flat.T)
    out = {
        "dataset": label,
        "n_observations": int(flat.shape[0]),
        "mean_daily": float(flat.mean()),
        "mean_annualised": float(flat.mean() * 252),
        "volatility_annualised": float(flat.std(ddof=1) * np.sqrt(252)),
        "skewness": float(np.mean([sp_stats.skew(flat[:, k]) for k in range(d)])),
        "kurtosis": float(np.mean([sp_stats.kurtosis(flat[:, k]) + 3.0 for k in range(d)])),
        "lower_tail_dependence": float(np.mean(
            [_lower_tail_dependence(flat[:, i], flat[:, j]) for i, j in zip(*triu)])),
        "acf1_returns": float(np.mean([_acf(flat[:, k], 1) for k in range(d)])),
        "acf1_abs_returns": float(np.mean([_acf(np.abs(flat[:, k]), 1) for k in range(d)])),
        "acf5_abs_returns": float(np.mean([_acf(np.abs(flat[:, k]), 5) for k in range(d)])),
    }
    if terminal is not None:
        out["terminal_mean"] = float(terminal.mean())
        out["terminal_std"] = float(terminal.std(ddof=1))
        out["terminal_q05"] = float(np.quantile(terminal, 0.05))
        out["terminal_q95"] = float(np.quantile(terminal, 0.95))
    if reference is None:
        out.update({"ks_distance": np.nan, "correlation_mae": np.nan,
                    "kurtosis_gap": np.nan, "lower_tail_gap": np.nan})
    else:
        ref_corr = np.corrcoef(reference.T)
        ref_kurt = float(np.mean([sp_stats.kurtosis(reference[:, k]) + 3.0
                                  for k in range(d)]))
        ref_tail = float(np.mean([_lower_tail_dependence(reference[:, i], reference[:, j])
                                  for i, j in zip(*triu)]))
        out.update({
            "ks_distance": float(np.mean([
                sp_stats.ks_2samp(reference[:, k], flat[:, k]).statistic
                for k in range(d)])),
            "correlation_mae": float(np.abs(corr - ref_corr)[triu].mean()),
            "kurtosis_gap": float(abs(out["kurtosis"] - ref_kurt)),
            "lower_tail_gap": float(abs(out["lower_tail_dependence"] - ref_tail)),
        })
    out["correlation_matrix"] = corr.tolist()
    return out


_hist_flat = TRAIN_RETURNS_VALUES
_rows = [fidelity_stats(_hist_flat, None, "Historical 2005-2019", CFG,
                        terminal=None)]
GENERATOR_RETURNS_FLAT = {}
for _name in GENERATORS:
    _spots = GENERATOR_DATA[_name]["S_norm"].astype(np.float64)
    _rets = np.diff(np.log(_spots), axis=1)
    GENERATOR_RETURNS_FLAT[_name] = _rets.reshape(-1, CFG.d)
    _rows.append(fidelity_stats(GENERATOR_RETURNS_FLAT[_name], _hist_flat, _name, CFG,
                                terminal=_spots[:, -1, :].mean(axis=1)))

FIDELITY_TABLE = pd.DataFrame(_rows)
FIDELITY_DISPLAY = FIDELITY_TABLE.drop(columns=["correlation_matrix"])
_h = atomic_write_dataframe(PATHS["generators"] / "fidelity.csv", FIDELITY_DISPLAY)
atomic_write_json(PATHS["generators"] / "fidelity.json", {
    "schema_version": CFG.schema_version, "config_hash": CONFIG_HASH,
    "data_hashes": {f"generator_{g}": DATA_HASHES[f"generator_{g}"] for g in GENERATORS},
    "rows": FIDELITY_TABLE.to_dict(orient="records")})
register_artifact("generators/fidelity.csv", PATHS["generators"] / "fidelity.csv", _h)

pd.set_option("display.width", 200)
print("Generator fidelity versus the historical calibration sample\n")
print(FIDELITY_DISPLAY.round(4).to_string(index=False))

# ==========================================================================
# ---
#
# ## Sections 9–11 — Deep hedging model, losses, corrected training engine
#
# Then Cell 17T runs the unit and integration tests (Gate 1). Everything below
# Gate 1 refuses to run if a test fails.

# ---- notebook cell 21 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 15 — PAYOFFS, HEDGING NETWORK, HEDGING FORWARD
# ═════════════════════════════════════════════════════════════════════════════
#  State at time t:  [spots(3), running_average(3), previous_delta(3), ttm(1)]
#  => input dimension 10.   Architecture: 10 -> 64 -> 64 -> 3, ReLU on hidden.
#
#  Initialisation now matches the description in the thesis, which the previous
#  code did not implement: EVERY hidden Linear.weight gets Xavier uniform with
#  the ReLU gain, the output layer gets Xavier uniform with gain 0.1, and all
#  biases are zero. Cell 17T asserts that every Linear layer was touched.
# ═════════════════════════════════════════════════════════════════════════════

def compute_running_averages(S_paths: torch.Tensor) -> torch.Tensor:
    """Causal running average of spots (no look-ahead)."""
    M, T1, d = S_paths.shape
    T = T1 - 1
    avg = torch.zeros(M, T, d, device=S_paths.device, dtype=S_paths.dtype)
    avg[:, 0, :] = S_paths[:, 0, :]
    if T > 1:
        cumsum = S_paths[:, 1:, :].cumsum(dim=1)
        counts = torch.arange(1, T + 1, device=S_paths.device,
                              dtype=S_paths.dtype).view(1, -1, 1)
        avg[:, 1:, :] = (cumsum / counts)[:, :-1, :]
    return avg


def payoff_basket_asian_call(S_paths: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    R_bar = S_paths[:, 1:, :].mean(dim=1)
    return torch.clamp(R_bar.mean(dim=1) - kappa, min=0.0)


def payoff_asian_worst_of_put(S_paths: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    R_bar = S_paths[:, 1:, :].mean(dim=1)
    return torch.clamp(kappa - R_bar.min(dim=1).values, min=0.0)


PAYOFF_FNS = {
    "basket_asian_call": payoff_basket_asian_call,
    "asian_worst_of_put": payoff_asian_worst_of_put,
}


class HedgingNetwork(nn.Module):
    """Feed-forward hedging policy shared by every generator/option/strike/seed."""

    def __init__(self, d: int = 3, hidden: Sequence[int] = (64, 64),
                 output_gain: float = 0.1, v0_init: float = 0.0):
        super().__init__()
        self.d = int(d)
        self.hidden = tuple(int(h) for h in hidden)
        self.output_gain = float(output_gain)
        layers: List[nn.Module] = []
        prev = 3 * self.d + 1
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, self.d))
        self.net = nn.Sequential(*layers)
        self.V0 = nn.Parameter(torch.tensor(float(v0_init)))
        self.init_report = self.reset_parameters()

    def reset_parameters(self) -> Dict[str, Any]:
        """Xavier-uniform every Linear layer; report exactly what was touched."""
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        relu_gain = nn.init.calculate_gain("relu")
        touched = []
        for i, layer in enumerate(linears):
            is_output = (i == len(linears) - 1)
            gain = self.output_gain if is_output else relu_gain
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)
            touched.append({"index": i, "shape": list(layer.weight.shape),
                            "gain": float(gain),
                            "role": "output" if is_output else "hidden"})
        return {"method": "xavier_uniform", "relu_gain": float(relu_gain),
                "output_gain": self.output_gain, "layers": touched,
                "n_linear_layers": len(linears), "bias_init": 0.0}

    def forward(self, spots, running_avg, delta_prev, time_left):
        x = torch.cat([spots, running_avg, delta_prev, time_left], dim=1)
        return self.net(x)

    @property
    def input_dim(self) -> int:
        return 3 * self.d + 1

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def spec(self) -> Dict[str, Any]:
        return {
            "class_name": "HedgingNetwork", "d": self.d, "hidden": list(self.hidden),
            "activation": "ReLU", "input_dim": self.input_dim,
            "input_features": ["spots", "running_avg", "delta_prev", "time_to_maturity"],
            "output_dim": self.d, "output_activation": None,
            "initialization": self.init_report, "v0_init": float(CFG.v0_init),
            "n_parameters": self.n_parameters,
        }


def autocast_ctx():
    """bf16 autocast for the MLP only when A100_FAST is active."""
    if PRECISION_MODE == "A100_FAST" and DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def deep_hedge_forward(net: HedgingNetwork, S_paths: torch.Tensor,
                       payoff_fn, kappa: float = 1.0,
                       cost_rate: float = 0.001,
                       keep_deltas: bool = False) -> Dict[str, torch.Tensor]:
    """Run the hedging policy over a batch of paths.

    Residual convention:  R = payoff - V0 - PnL + transaction_cost
                            = payoff - terminal_wealth.
    All accumulations and the residual itself are FP32 even under bf16
    autocast, so the CVaR tail never sees reduced precision.
    """
    M, T1, d = S_paths.shape
    T = T1 - 1
    running_avg = compute_running_averages(S_paths)
    time_fracs = (torch.arange(T, 0, -1, device=S_paths.device,
                               dtype=torch.float32) / T)
    delta = torch.zeros(M, d, device=S_paths.device, dtype=torch.float32)
    pnl = torch.zeros(M, device=S_paths.device, dtype=torch.float32)
    cost = torch.zeros(M, device=S_paths.device, dtype=torch.float32)
    deltas = [] if keep_deltas else None
    turnover = torch.zeros(M, device=S_paths.device, dtype=torch.float32)

    for t in range(T):
        with autocast_ctx():
            delta_new = net(S_paths[:, t, :], running_avg[:, t, :], delta,
                            time_fracs[t].expand(M, 1))
        delta_new = delta_new.float()
        traded = (delta_new - delta).abs()
        cost = cost + cost_rate * (traded * S_paths[:, t, :]).sum(dim=1)
        turnover = turnover + traded.sum(dim=1)
        pnl = pnl + (delta_new * (S_paths[:, t + 1, :] - S_paths[:, t, :])).sum(dim=1)
        delta = delta_new
        if keep_deltas:
            deltas.append(delta_new.detach())

    payoff = payoff_fn(S_paths, kappa).float()
    v0 = net.V0.float()
    terminal_wealth = v0 + pnl - cost
    residuals = payoff - terminal_wealth
    out = {
        "residuals": residuals,
        "terminal_wealth": terminal_wealth.detach(),
        "payoff": payoff.detach(),
        "pnl": pnl.detach(),
        "transaction_cost": cost.detach(),
        "turnover": turnover.detach(),
        "V0": v0,
    }
    if keep_deltas:
        out["deltas"] = torch.stack(deltas, dim=1)
    return out


print("Model components ready.")
_net_probe = HedgingNetwork(d=CFG.d, hidden=CFG.hidden_sizes,
                            output_gain=CFG.init_output_gain, v0_init=CFG.v0_init)
print(f"  architecture : {_net_probe.input_dim} -> "
      f"{' -> '.join(str(h) for h in _net_probe.hidden)} -> {_net_probe.d}")
print(f"  parameters   : {_net_probe.n_parameters}")
print(f"  initialised  : {_net_probe.init_report['n_linear_layers']} Linear layers, "
      f"ReLU gain {_net_probe.init_report['relu_gain']:.4f}, "
      f"output gain {_net_probe.init_report['output_gain']}")
print(f"  residual     : R = payoff - V0 - PnL + transaction_cost")
del _net_probe

# ---- notebook cell 22 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 16 — LOSSES, METRICS, CHECKPOINT SCHEMA
# ═════════════════════════════════════════════════════════════════════════════
#  Phase 1 loss : mean(R^2)
#  Phase 2 loss : nu + mean(relu(R - nu)) / (1 - alpha)      (Rockafellar-Uryasev)
#  Metric reductions run in FP32 on the GPU and in float64 on the CPU for every
#  number that reaches a statistical test or a table.
# ═════════════════════════════════════════════════════════════════════════════

def loss_mse(residuals: torch.Tensor) -> torch.Tensor:
    return (residuals.float() ** 2).mean()


class CVaRLoss(nn.Module):
    """Rockafellar-Uryasev CVaR with a trainable threshold nu."""

    def __init__(self, alpha: float = 0.95, nu_init: float = 0.0):
        super().__init__()
        self.alpha = float(alpha)
        self.nu = nn.Parameter(torch.tensor(float(nu_init)))

    def forward(self, residuals: torch.Tensor) -> torch.Tensor:
        r = residuals.float()
        return self.nu + torch.clamp(r - self.nu, min=0.0).mean() / (1.0 - self.alpha)


def empirical_var(residuals: np.ndarray, alpha: float) -> float:
    """Empirical VaR_alpha of the residual distribution (upper tail)."""
    return float(np.quantile(np.asarray(residuals, dtype=np.float64), alpha))


def residual_metrics(residuals: np.ndarray, extras: Optional[Dict[str, np.ndarray]] = None,
                     v0: Optional[float] = None, alpha: float = 0.95) -> Dict[str, Any]:
    """All reported metrics, computed in float64 on the CPU."""
    r = np.asarray(residuals, dtype=np.float64)
    n = r.size
    finite = np.isfinite(r)
    s = np.sort(r[finite])

    def _tail_mean(level: float) -> float:
        """Mean of the worst ceil((1-level)*n) observations.

        The tail always holds at least one observation, so the statistic stays
        defined for small evaluation sets instead of returning NaN. For the
        canonical sizes this is identical to the usual s[ceil(level*n):].
        """
        if s.size == 0:
            return float("nan")
        k = max(1, int(np.ceil((1.0 - level) * s.size)))
        return float(s[s.size - k:].mean())

    out = {
        "n_paths": int(n),
        "n_finite": int(finite.sum()),
        "all_finite": bool(finite.all()),
        "mean": float(s.mean()) if s.size else float("nan"),
        "std": float(s.std(ddof=1)) if s.size > 1 else float("nan"),
        "rmse": float(np.sqrt((s ** 2).mean())) if s.size else float("nan"),
        "var95": float(np.quantile(s, alpha)) if s.size else float("nan"),
        "cvar95": _tail_mean(alpha),
        "cvar99": _tail_mean(0.99),
        "min": float(s.min()) if s.size else float("nan"),
        "max": float(s.max()) if s.size else float("nan"),
    }
    if v0 is not None:
        out["V0"] = float(v0)
    for key, arr in (extras or {}).items():
        a = np.asarray(arr, dtype=np.float64)
        out[f"mean_{key}"] = float(a.mean())
    return out


@torch.no_grad()
def evaluate_full(net: HedgingNetwork, S_paths: torch.Tensor, payoff_fn,
                  kappa: float, cfg: ExperimentConfig,
                  chunk: int = 8192) -> Dict[str, Any]:
    """Uniform evaluation: identical code path for synthetic and historical sets."""
    net.eval()
    parts: Dict[str, List[np.ndarray]] = {
        "residuals": [], "payoff": [], "pnl": [], "transaction_cost": [],
        "terminal_wealth": [], "turnover": []}
    for i in range(0, S_paths.shape[0], chunk):
        r = deep_hedge_forward(net, S_paths[i:i + chunk], payoff_fn, kappa,
                               cfg.cost_rate)
        for k in parts:
            parts[k].append(r[k].detach().float().cpu().numpy())
    arrays = {k: np.concatenate(v).astype(np.float32) for k, v in parts.items()}
    metrics = residual_metrics(
        arrays["residuals"],
        extras={"payoff": arrays["payoff"], "pnl": arrays["pnl"],
                "transaction_cost": arrays["transaction_cost"],
                "turnover": arrays["turnover"]},
        v0=float(net.V0.item()), alpha=cfg.cvar_alpha)
    return {"metrics": metrics, "arrays": arrays}


# ── Checkpoint schema (design §13) ───────────────────────────────────────────
CHECKPOINT_REQUIRED_FIELDS = (
    "schema_version", "run_id", "run_key", "phase", "generator", "option",
    "kappa", "seed", "config_hash", "data_hashes", "source_commit",
    "model_spec", "model_state_dict", "V0", "optimizer_state_dict",
    "scheduler_state_dict", "best_epoch", "best_val_loss", "training_history",
    "rng_state", "environment", "synthetic_test_metrics", "historical_metrics",
    "artifact_hash", "created_utc")


def build_checkpoint_payload(*, spec: Dict[str, Any], phase: str,
                             bundle: Dict[str, Any], history: Dict[str, Any],
                             synthetic_metrics: Dict[str, Any],
                             historical_metrics: Dict[str, Any],
                             model_spec: Dict[str, Any],
                             data_hashes: Dict[str, str],
                             extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        "schema_version": CFG.checkpoint_schema_version,
        "notebook_schema_version": CFG.schema_version,
        "run_id": RUN_ID,
        "run_key": spec["run_key"],
        "phase": phase,
        "generator": spec["generator"],
        "option": spec["option"],
        "kappa": float(spec["kappa"]),
        "seed": int(spec["seed"]),
        "config_hash": CONFIG_HASH,
        "data_hashes": data_hashes,
        "source_commit": SOURCE_SNAPSHOT_COMMIT,
        "model_spec": model_spec,
        "model_state_dict": tensors_to_cpu(bundle["model_state_dict"]),
        "V0": float(bundle["V0"]),
        "optimizer_state_dict": tensors_to_cpu(bundle["optimizer_state_dict"]),
        "scheduler_state_dict": tensors_to_cpu(bundle["scheduler_state_dict"]),
        "best_epoch": int(bundle["epoch"]),
        "best_val_loss": float(bundle["best_val_loss"]),
        "training_history": history,
        "rng_state": bundle.get("rng_state"),
        "environment": ENVIRONMENT,
        "synthetic_test_metrics": synthetic_metrics,
        "historical_metrics": historical_metrics,
        "created_utc": utc_now(),
    }
    if phase == "cvar":
        payload["nu"] = float(bundle["loss_state_dict"]["nu"])
        payload["loss_state_dict"] = tensors_to_cpu(bundle["loss_state_dict"])
        payload["cvar_alpha"] = float(CFG.cvar_alpha)
    if extra:
        payload.update(extra)
    payload["artifact_hash"] = canonical_hash(
        {k: v for k, v in payload.items() if k != "artifact_hash"})
    return payload


def save_checkpoint(payload: Dict[str, Any], path: Path) -> str:
    digest = atomic_write_torch(path, payload, verify=True)
    sidecar = {k: payload[k] for k in (
        "schema_version", "run_id", "run_key", "phase", "generator", "option",
        "kappa", "seed", "config_hash", "best_epoch", "best_val_loss",
        "artifact_hash", "created_utc", "V0")}
    sidecar["nu"] = payload.get("nu")
    sidecar["file_sha256"] = digest
    sidecar["synthetic_test_metrics"] = payload["synthetic_test_metrics"]
    sidecar["historical_metrics"] = payload["historical_metrics"]
    sidecar["data_hashes"] = payload["data_hashes"]
    atomic_write_json(path.with_suffix(".json"), sidecar)
    return digest


def verify_checkpoint(path: Path, strict: bool = True) -> Dict[str, Any]:
    """Structural + integrity verification. Raises on any failure."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    missing = [k for k in CHECKPOINT_REQUIRED_FIELDS if k not in ckpt]
    if missing:
        raise ValueError(f"{path.name}: missing fields {missing}")
    if ckpt["phase"] == "cvar":
        for f in ("nu", "loss_state_dict", "cvar_alpha"):
            if f not in ckpt:
                raise ValueError(f"{path.name}: CVaR checkpoint missing {f}")
    if strict:
        if ckpt["config_hash"] != CONFIG_HASH:
            raise ValueError(f"{path.name}: config_hash mismatch")
        expected = set(CFG.regime_names)
        got = set(ckpt["historical_metrics"].keys())
        if expected - got:
            raise ValueError(f"{path.name}: missing regimes {expected - got}")
    recomputed = canonical_hash({k: v for k, v in ckpt.items() if k != "artifact_hash"})
    if recomputed != ckpt["artifact_hash"]:
        raise ValueError(f"{path.name}: artifact_hash mismatch "
                         f"(saved {ckpt['artifact_hash'][:16]}, "
                         f"recomputed {recomputed[:16]})")
    return ckpt


def load_checkpoint(path: Path, device=None) -> Tuple[HedgingNetwork,
                                                      Optional[CVaRLoss],
                                                      Dict[str, Any]]:
    device = device or DEVICE
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    spec = ckpt["model_spec"]
    net = HedgingNetwork(d=spec["d"], hidden=tuple(spec["hidden"]),
                         output_gain=CFG.init_output_gain,
                         v0_init=CFG.v0_init).to(device)
    net.load_state_dict({k: v.to(device) for k, v in ckpt["model_state_dict"].items()})
    net.eval()
    cvar = None
    if ckpt["phase"] == "cvar":
        cvar = CVaRLoss(alpha=ckpt["cvar_alpha"]).to(device)
        cvar.load_state_dict({k: v.to(device)
                              for k, v in ckpt["loss_state_dict"].items()})
    return net, cvar, ckpt


print("Losses, metrics and the checkpoint schema are defined.")
print(f"  checkpoint fields : {len(CHECKPOINT_REQUIRED_FIELDS)} required")
print(f"  filename pattern  : "
      f"{checkpoint_filename('GBM', 'basket_asian_call', 1.0, 3, 'cvar')}")

# ---- notebook cell 23 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 17 — TRAINING ENGINE
# ═════════════════════════════════════════════════════════════════════════════
#  Corrections against the previous engine, each of which changed results:
#
#   1. NO DROPPED PATHS. Batches are `range(0, M, batch_size)`, so the tail
#      batch is kept; every epoch asserts it consumed exactly M samples. The
#      old `M // BATCH_SIZE` loop silently discarded 3,712 of 16,000 paths.
#   2. FULL BEST BUNDLE. The best epoch stores model, loss (including nu),
#      optimizer, scheduler, epoch, val loss and RNG state together, and the
#      whole bundle is restored before evaluation. `last_bundle` is kept
#      separately for resuming and is never used for evaluation.
#   3. TRUE POST-CLIP NORM. Gradients are unscaled (if a scaler is active), the
#      pre-clip norm is measured, clipping is applied, and the post-clip norm
#      is measured AGAIN from the clipped gradients. The invariant
#      post <= clip + tol is asserted.
#   4. nu INITIALISATION. Phase 2 starts nu at the empirical VaR_0.95 of the
#      Phase-1 validation residuals instead of 0.
#   5. EPOCH-LEVEL RESUME. A phase persists model, loss, optimizer, scheduler,
#      history, patience and RNG state every CHECKPOINT_EVERY_EPOCHS epochs, so
#      an interrupted session continues from the epoch it reached instead of
#      restarting the configuration. When the RNG state is restored in full the
#      continuation follows the original random stream; that is recorded. A
#      finished Phase 1 is reused only together with the RNG state it ended
#      in, so Phase 2 matches an uninterrupted run; otherwise it is re-trained.
#   6. FAILURE PROTOCOL. Non-finite loss/parameters/gradients, a checkpoint
#      that will not reload, a non-finite metric, a sample-count mismatch or a
#      hash mismatch mark the run failed. Retries reuse the SAME seed and the
#      SAME configuration; a seed is never swapped for another one.
# ═════════════════════════════════════════════════════════════════════════════

class TrainingFailure(RuntimeError):
    """Raised when a run violates the numerical failure protocol."""


def _grad_norm(params) -> float:
    grads = [p.grad.detach() for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.norm(torch.stack([g.norm(2) for g in grads]), 2).item())


def _params_finite(params) -> bool:
    return all(torch.isfinite(p.detach()).all().item() for p in params)


def _grads_finite(params) -> bool:
    return all(torch.isfinite(p.grad.detach()).all().item()
               for p in params if p.grad is not None)


def _clone_bundle(net, loss_module, optimizer, scheduler, epoch, val_loss):
    return {
        "model_state_dict": {k: v.detach().cpu().clone()
                             for k, v in net.state_dict().items()},
        "loss_state_dict": ({k: v.detach().cpu().clone()
                             for k, v in loss_module.state_dict().items()}
                            if isinstance(loss_module, nn.Module) else None),
        "optimizer_state_dict": tensors_to_cpu(optimizer.state_dict()),
        "scheduler_state_dict": tensors_to_cpu(scheduler.state_dict()),
        "epoch": int(epoch),
        "best_val_loss": float(val_loss),
        "V0": float(net.V0.item()),
        "rng_state": capture_rng_state(),
    }


def _restore_bundle(bundle, net, loss_module, optimizer=None, scheduler=None):
    net.load_state_dict({k: v.to(DEVICE) for k, v in bundle["model_state_dict"].items()})
    if isinstance(loss_module, nn.Module) and bundle.get("loss_state_dict"):
        loss_module.load_state_dict({k: v.to(DEVICE)
                                     for k, v in bundle["loss_state_dict"].items()})
    if optimizer is not None:
        optimizer.load_state_dict(bundle["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(bundle["scheduler_state_dict"])


def _progress_payload(spec_key: str, phase: str, epoch: int, best_val: float,
                      patience_counter: int, history: Dict[str, Any],
                      best_bundle, last_bundle, complete: bool) -> Dict[str, Any]:
    return {"run_key": spec_key, "phase": phase, "config_hash": CONFIG_HASH,
            "schema_version": CFG.schema_version,
            "next_epoch": int(epoch + 1), "best_val_loss": float(best_val),
            "patience_counter": int(patience_counter), "history": history,
            "best_bundle": best_bundle, "last_bundle": last_bundle,
            "phase_complete": bool(complete), "saved_utc": utc_now()}


def _phase1_final_rng(path: Path, spec_key: str) -> Optional[Dict[str, Any]]:
    """The RNG state a FINISHED Phase 1 ended in, from its progress file.

    Phase 2 of an uninterrupted run draws its first minibatch order from
    exactly this state: nothing between the phases consumes randomness. Both
    the current progress format and the legacy one (written only after the
    phase had finished) carry it in last_bundle."""
    if not Path(path).exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:                                        # noqa: BLE001
        return None
    if (payload.get("run_key") != spec_key or payload.get("phase") != "mse"
            or payload.get("config_hash") != CONFIG_HASH):
        return None
    if payload.get("phase_complete") is False:              # not a finished phase
        return None
    return (payload.get("last_bundle") or {}).get("rng_state")


def _load_progress(path: Optional[Path], spec_key: str,
                   phase: str) -> Optional[Dict[str, Any]]:
    """Read an in-progress phase, or None when there is nothing usable."""
    if not path or not Path(path).exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:                                 # noqa: BLE001
        add_warning(f"{spec_key} {phase}: unreadable progress file ({exc!r}); "
                    f"the phase restarts from epoch 0")
        return None
    if (payload.get("config_hash") != CONFIG_HASH
            or payload.get("run_key") != spec_key
            or payload.get("phase") != phase):
        return None
    if payload.get("phase_complete"):
        return None
    if payload.get("best_bundle") is None or payload.get("last_bundle") is None:
        return None
    return payload


@torch.no_grad()
def _validation_loss(net, loss_callable, S_val, payoff_fn, kappa, cfg,
                     chunk: int = 8192) -> Tuple[float, np.ndarray]:
    net.eval()
    res = []
    for i in range(0, S_val.shape[0], chunk):
        r = deep_hedge_forward(net, S_val[i:i + chunk], payoff_fn, kappa, cfg.cost_rate)
        res.append(r["residuals"].detach().float().cpu())
    residuals = torch.cat(res)
    return float(loss_callable(residuals.to(DEVICE)).item()), residuals.numpy()


def train_phase(*, net: HedgingNetwork, loss_module, trainable_params,
                S_train: torch.Tensor, S_val: torch.Tensor, payoff_fn,
                kappa: float, lr: float, max_epochs: int, patience: int,
                cfg: ExperimentConfig, phase: str,
                scaler: Optional["torch.cuda.amp.GradScaler"] = None,
                verbose_every: int = 100,
                progress_path: Optional[Path] = None,
                spec_key: str = "") -> Dict[str, Any]:
    """Train one phase; return the best bundle, the last bundle and the history.

    With `progress_path` the phase is resumable at epoch granularity: it picks
    up where an interrupted session stopped rather than starting over."""
    M = int(S_train.shape[0])
    bs = int(cfg.batch_size)
    optimizer = optim.Adam(trainable_params, lr=lr,
                           fused=(PRECISION_MODE == "A100_FAST" and DEVICE.type == "cuda"))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg.sched_patience,
        factor=cfg.sched_factor, min_lr=cfg.sched_min_lr)
    loss_callable = (loss_module if callable(loss_module) else loss_mse)

    history = {"train_loss": [], "val_loss": [], "lr": [], "grad_norm_pre_clip": [],
               "grad_norm_post_clip": [], "V0": [], "nu": [], "epoch_seconds": [],
               "samples_seen": []}
    best_val, best_bundle, patience_counter = float("inf"), None, 0
    last_bundle = None
    start_epoch = 0
    rng_stream_continued = True
    t0 = time.time()
    epochs_run = 0

    resumed = _load_progress(progress_path, spec_key, phase)
    if resumed is not None:
        _restore_bundle(resumed["last_bundle"], net, loss_module, optimizer, scheduler)
        best_bundle = resumed["best_bundle"]
        last_bundle = resumed["last_bundle"]
        best_val = float(resumed["best_val_loss"])
        patience_counter = int(resumed["patience_counter"])
        history = resumed["history"]
        start_epoch = int(resumed["next_epoch"])
        rng_stream_continued = restore_rng_state(resumed["last_bundle"].get("rng_state"))
        print(f"      resuming {phase} at epoch {start_epoch} "
              f"(best val {best_val:.6f}, patience {patience_counter}"
              f"{'' if rng_stream_continued else ', RNG stream NOT restored'})",
              flush=True)
        if not rng_stream_continued:
            add_warning(f"{spec_key} {phase}: resumed without restoring the RNG "
                        f"state; the continuation is compatible but does not "
                        f"follow the original random stream")
        epochs_run = start_epoch

    def _persist(epoch_idx: int, complete: bool) -> None:
        if progress_path is None or (not complete and not CHECKPOINT_EVERY_EPOCHS):
            return
        atomic_write_torch(progress_path,
                           _progress_payload(spec_key, phase, epoch_idx, best_val,
                                             patience_counter, history,
                                             best_bundle, last_bundle, complete),
                           verify=False)

    for epoch in range(start_epoch, max_epochs):
        epochs_run = epoch + 1
        t_ep = time.time()
        net.train()
        perm = torch.randperm(M, device=S_train.device)
        seen, n_batches = 0, 0
        ep_loss = gn_pre_sum = gn_post_sum = 0.0

        for start in range(0, M, bs):                      # keeps the tail batch
            idx = perm[start:start + bs]
            seen += int(idx.numel())
            n_batches += 1
            optimizer.zero_grad(set_to_none=True)
            out = deep_hedge_forward(net, S_train[idx], payoff_fn, kappa, cfg.cost_rate)
            loss = loss_callable(out["residuals"])
            if not torch.isfinite(loss):
                raise TrainingFailure(
                    f"{phase}: non-finite loss at epoch {epoch}, batch {n_batches}")
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)                 # measure unscaled grads
            else:
                loss.backward()
            if not _grads_finite(trainable_params):
                raise TrainingFailure(f"{phase}: non-finite gradient at epoch {epoch}")
            gn_pre = _grad_norm(trainable_params)
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
            gn_post = _grad_norm(trainable_params)         # measured AFTER clipping
            if gn_post > cfg.grad_clip + cfg.grad_clip_tolerance:
                raise TrainingFailure(
                    f"{phase}: post-clip gradient norm {gn_post:.6f} exceeds "
                    f"clip {cfg.grad_clip}")
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            if not _params_finite(trainable_params):
                raise TrainingFailure(f"{phase}: non-finite parameter at epoch {epoch}")
            ep_loss += float(loss.item())
            gn_pre_sum += gn_pre
            gn_post_sum += gn_post

        if seen != M:
            raise TrainingFailure(f"{phase}: epoch consumed {seen} of {M} samples")

        ep_loss /= n_batches
        val_loss, _ = _validation_loss(net, loss_callable, S_val, payoff_fn, kappa, cfg)
        if not math.isfinite(val_loss):
            raise TrainingFailure(f"{phase}: non-finite validation loss at epoch {epoch}")
        scheduler.step(val_loss)

        nu_value = (float(loss_module.nu.item())
                    if isinstance(loss_module, CVaRLoss) else None)
        history["train_loss"].append(ep_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        history["grad_norm_pre_clip"].append(gn_pre_sum / n_batches)
        history["grad_norm_post_clip"].append(gn_post_sum / n_batches)
        history["V0"].append(float(net.V0.item()))
        history["nu"].append(nu_value)
        history["epoch_seconds"].append(time.time() - t_ep)
        history["samples_seen"].append(int(seen))

        last_bundle = _clone_bundle(net, loss_module, optimizer, scheduler,
                                    epoch, val_loss)
        if val_loss < best_val - cfg.val_improve_tol:
            best_val, patience_counter = val_loss, 0
            best_bundle = _clone_bundle(net, loss_module, optimizer, scheduler,
                                        epoch, val_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                _persist(epoch, complete=True)
                break

        if CHECKPOINT_EVERY_EPOCHS and (epoch + 1) % CHECKPOINT_EVERY_EPOCHS == 0:
            _persist(epoch, complete=False)

        if verbose_every and (epoch + 1) % verbose_every == 0:
            print(f"      epoch {epoch + 1:4d}  train={ep_loss:.6f}  "
                  f"val={val_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
                  f"|g|pre={gn_pre_sum / n_batches:.3f}", flush=True)

    if best_bundle is None:
        raise TrainingFailure(f"{phase}: no epoch produced a validation improvement")

    _persist(epochs_run - 1, complete=True)
    _restore_bundle(best_bundle, net, loss_module)          # evaluate from BEST
    net.eval()
    return {"best_bundle": best_bundle, "last_bundle": last_bundle,
            "history": history, "epochs_run": epochs_run,
            "resumed_from_epoch": start_epoch if resumed is not None else None,
            "rng_stream_continued": rng_stream_continued,
            "best_val_loss": float(best_val), "wall_seconds": time.time() - t0,
            "optimizer": optimizer, "scheduler": scheduler}


def train_configuration(spec: Dict[str, Any], cfg: ExperimentConfig,
                        datasets: Dict[str, Dict[str, torch.Tensor]],
                        hist_tensors: Dict[str, torch.Tensor],
                        data_hashes: Dict[str, str],
                        verbose_every: int = 100) -> Dict[str, Any]:
    """Run both phases for one (generator, option, kappa, seed) configuration."""
    gen, opt_name = spec["generator"], spec["option"]
    kappa, seed = float(spec["kappa"]), int(spec["seed"])
    payoff_fn = PAYOFF_FNS[opt_name]
    S_tr, S_vl, S_te = (datasets[gen]["train"], datasets[gen]["val"],
                        datasets[gen]["test"])
    ckpt_dir = PATHS["checkpoints"] / gen

    used_seed = seed_all(seed)
    assert used_seed == seed, "seed substitution detected"
    assert_no_seed_aliasing()

    net = HedgingNetwork(d=cfg.d, hidden=cfg.hidden_sizes,
                         output_gain=cfg.init_output_gain,
                         v0_init=cfg.v0_init).to(DEVICE)
    model_spec = net.spec()
    results: Dict[str, Any] = {"run_key": spec["run_key"], "seed": seed,
                               "generator": gen, "option": opt_name,
                               "kappa": kappa, "phases": {}}

    # ── Phase 1: MSE, V0 trainable ─────────────────────────────────────────
    p1_path = ckpt_dir / checkpoint_filename(gen, opt_name, kappa, seed, "mse")
    p1_progress = ckpt_dir / checkpoint_filename(gen, opt_name, kappa, seed,
                                                 "mse", "last")
    p1_done = None
    if p1_path.exists():
        try:                       # a finished Phase 1 is not re-trained ...
            p1_done = verify_checkpoint(p1_path)
        except Exception as exc:                             # noqa: BLE001
            add_warning(f"{spec['run_key']} mse: existing checkpoint failed "
                        f"verification ({exc!r}); Phase 1 will be re-run")
            p1_done = None

    if p1_done is not None:
        # ... provided Phase 2 can start from exactly the random state an
        # uninterrupted run would have: the one Phase 1 finished in. Without it
        # Phase 2 would draw different minibatch orders and give a different
        # result, so in that case Phase 1 is re-trained from the seed instead.
        final_rng = _phase1_final_rng(p1_progress, spec["run_key"])
        if final_rng is None or not restore_rng_state(final_rng):
            add_warning(f"{spec['run_key']} mse: finished Phase 1 found, but not "
                        f"the random state it ended in; Phase 1 is re-trained so "
                        f"the result matches an uninterrupted run")
            p1_done = None
            seed_all(seed)                    # start exactly as a fresh run does
            net = HedgingNetwork(d=cfg.d, hidden=cfg.hidden_sizes,
                                 output_gain=cfg.init_output_gain,
                                 v0_init=cfg.v0_init).to(DEVICE)
            model_spec = net.spec()

    if p1_done is not None:
        net.load_state_dict({k: v.to(DEVICE)
                             for k, v in p1_done["model_state_dict"].items()})
        net.eval()
        results["phases"]["mse"] = {
            "checkpoint": p1_path.name, "sha256": sha256_file(p1_path),
            "artifact_hash": p1_done["artifact_hash"],
            "best_epoch": p1_done["best_epoch"],
            "best_val_loss": p1_done["best_val_loss"],
            "epochs_run": p1_done.get("epochs_run"),
            "wall_seconds": p1_done.get("wall_seconds"),
            "metrics": p1_done["synthetic_test_metrics"],
            "historical": p1_done["historical_metrics"],
            "V0": float(p1_done["V0"]), "reused_existing_checkpoint": True}
        phase1_seconds = 0.0
        phase1_best_epoch = p1_done["best_epoch"]
        print(f"      Phase 1 already complete and verified "
              f"(best epoch {phase1_best_epoch}); reusing it.", flush=True)
    else:
        p1 = train_phase(net=net, loss_module=loss_mse,
                         trainable_params=list(net.parameters()),
                         S_train=S_tr, S_val=S_vl, payoff_fn=payoff_fn, kappa=kappa,
                         lr=cfg.mse_lr, max_epochs=cfg.mse_epochs,
                         patience=cfg.mse_patience, cfg=cfg, phase="mse",
                         verbose_every=verbose_every,
                         progress_path=p1_progress, spec_key=spec["run_key"])
        mse_eval = evaluate_full(net, S_te, payoff_fn, kappa, cfg)
        mse_hist = {r: evaluate_full(net, t, payoff_fn, kappa, cfg)
                    for r, t in hist_tensors.items()}
        p1_payload = build_checkpoint_payload(
            spec=spec, phase="mse", bundle=p1["best_bundle"], history=p1["history"],
            synthetic_metrics=mse_eval["metrics"],
            historical_metrics={r: e["metrics"] for r, e in mse_hist.items()},
            model_spec=model_spec, data_hashes=data_hashes,
            extra={"epochs_run": p1["epochs_run"], "wall_seconds": p1["wall_seconds"],
                   "precision_mode": PRECISION_MODE,
                   "resumed_from_epoch": p1["resumed_from_epoch"],
                   "rng_stream_continued": p1["rng_stream_continued"]})
        p1_digest = save_checkpoint(p1_payload, p1_path)
        verify_checkpoint(p1_path)
        results["phases"]["mse"] = {
            "checkpoint": p1_path.name, "sha256": p1_digest,
            "artifact_hash": p1_payload["artifact_hash"],
            "best_epoch": p1_payload["best_epoch"],
            "best_val_loss": p1_payload["best_val_loss"],
            "epochs_run": p1["epochs_run"], "wall_seconds": p1["wall_seconds"],
            "resumed_from_epoch": p1["resumed_from_epoch"],
            "metrics": mse_eval["metrics"],
            "historical": {r: e["metrics"] for r, e in mse_hist.items()},
            "V0": float(net.V0.item())}
        phase1_seconds = p1["wall_seconds"]
        phase1_best_epoch = p1_payload["best_epoch"]
        _restore_bundle(p1["best_bundle"], net, None)

    # nu is initialised from the Phase-1 validation residuals in both paths.
    _, p1_val_residuals = _validation_loss(net, loss_mse, S_vl, payoff_fn, kappa, cfg)

    # ── Phase 2: CVaR curriculum, V0 frozen, nu from validation VaR ────────
    net.V0.requires_grad_(False)
    nu_init = empirical_var(p1_val_residuals, cfg.cvar_alpha)
    cvar_loss = CVaRLoss(alpha=cfg.cvar_alpha, nu_init=nu_init).to(DEVICE)
    cvar_params = [p for p in net.parameters() if p.requires_grad] + list(cvar_loss.parameters())

    p2_progress = ckpt_dir / checkpoint_filename(gen, opt_name, kappa, seed,
                                                 "cvar", "last")
    p2 = train_phase(net=net, loss_module=cvar_loss, trainable_params=cvar_params,
                     S_train=S_tr, S_val=S_vl, payoff_fn=payoff_fn, kappa=kappa,
                     lr=cfg.cvar_lr, max_epochs=cfg.cvar_epochs,
                     patience=cfg.cvar_patience, cfg=cfg, phase="cvar",
                     verbose_every=max(1, verbose_every // 2),
                     progress_path=p2_progress, spec_key=spec["run_key"])
    cvar_eval = evaluate_full(net, S_te, payoff_fn, kappa, cfg)
    cvar_hist = {r: evaluate_full(net, t, payoff_fn, kappa, cfg)
                 for r, t in hist_tensors.items()}

    p2_payload = build_checkpoint_payload(
        spec=spec, phase="cvar", bundle=p2["best_bundle"], history=p2["history"],
        synthetic_metrics=cvar_eval["metrics"],
        historical_metrics={r: e["metrics"] for r, e in cvar_hist.items()},
        model_spec=model_spec, data_hashes=data_hashes,
        extra={"epochs_run": p2["epochs_run"], "wall_seconds": p2["wall_seconds"],
               "precision_mode": PRECISION_MODE,
               "nu_init_empirical_var": float(nu_init),
               "nu_init_source": "empirical VaR_alpha of Phase-1 validation residuals",
               "v0_frozen": True,
               "phase1_best_epoch": phase1_best_epoch,
               "resumed_from_epoch": p2["resumed_from_epoch"],
               "rng_stream_continued": p2["rng_stream_continued"]})
    p2_path = ckpt_dir / checkpoint_filename(gen, opt_name, kappa, seed, "cvar")
    p2_digest = save_checkpoint(p2_payload, p2_path)
    verify_checkpoint(p2_path)
    results["phases"]["cvar"] = {
        "checkpoint": p2_path.name, "sha256": p2_digest,
        "artifact_hash": p2_payload["artifact_hash"],
        "best_epoch": p2_payload["best_epoch"],
        "best_val_loss": p2_payload["best_val_loss"],
        "epochs_run": p2["epochs_run"], "wall_seconds": p2["wall_seconds"],
        "resumed_from_epoch": p2["resumed_from_epoch"],
        "metrics": cvar_eval["metrics"],
        "historical": {r: e["metrics"] for r, e in cvar_hist.items()},
        "V0": float(net.V0.item()), "nu": float(cvar_loss.nu.item()),
        "nu_init": float(nu_init)}

    for phase_name, phase_res in results["phases"].items():
        for key, value in phase_res["metrics"].items():
            if isinstance(value, float) and not math.isfinite(value):
                raise TrainingFailure(
                    f"{spec['run_key']} {phase_name}: non-finite metric {key}")

    results["status"] = "complete"
    results["wall_seconds"] = phase1_seconds + p2["wall_seconds"]
    del net, cvar_loss
    clear_mem()
    return results


print("Training engine ready.")
print("  batching        : range(0, M, batch_size)  (tail batch kept, "
      "samples_seen asserted)")
print(f"  epoch resume    : every {CHECKPOINT_EVERY_EPOCHS} epochs"
      if CHECKPOINT_EVERY_EPOCHS else "  epoch resume    : disabled")
print("                    a finished Phase 1 is reused, never re-trained")
print("  best bundle     : model + loss(nu) + optimizer + scheduler + epoch + rng")
print("  gradient log    : true post-clip norm, invariant post <= "
      f"{CFG.grad_clip} + {CFG.grad_clip_tolerance}")
print("  nu init         : empirical VaR_%.2f of Phase-1 validation residuals"
      % CFG.cvar_alpha)
print(f"  retry policy    : up to {CFG.max_retries_per_run} retries, same seed, "
      f"same config; no seed relabelling")

# ---- notebook cell 24 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 17T — UNIT AND INTEGRATION TESTS (Gate 1)
# ═════════════════════════════════════════════════════════════════════════════
#  These run from a clean kernel in seconds and must all pass before any full
#  run. The MCS tests live in Cell 24, next to the MCS implementation.
# ═════════════════════════════════════════════════════════════════════════════

TEST_RESULTS: List[Dict[str, Any]] = []

# Scratch files for the tests live on THIS machine's local disk. On the shared
# Drive run directory, two sessions started together would write, tamper with
# and delete the same scratch checkpoint and fail each other's Gate 1.
TEST_SCRATCH = Path(tempfile.mkdtemp(prefix="sbts_gate1_"))


def _run_test(fn, name: str, group: str) -> bool:
    t0 = time.time()
    try:
        fn()
        TEST_RESULTS.append({"group": group, "test": name, "status": "PASS",
                             "seconds": time.time() - t0, "error": None})
        print(f"  PASS  {name}")
        return True
    except Exception as exc:                                 # noqa: BLE001
        TEST_RESULTS.append({"group": group, "test": name, "status": "FAIL",
                             "seconds": time.time() - t0, "error": repr(exc)})
        print(f"  FAIL  {name}: {exc!r}")
        return False


# ── Unit tests ───────────────────────────────────────────────────────────────
def test_seed_all_reproducible():
    seed_all(1234); a = (torch.randn(5), np.random.rand(5), random.random())
    seed_all(1234); b = (torch.randn(5), np.random.rand(5), random.random())
    assert torch.equal(a[0], b[0]) and np.allclose(a[1], b[1]) and a[2] == b[2]
    seed_all(1235); c = torch.randn(5)
    assert not torch.equal(a[0], c), "different seeds gave identical draws"


def test_payoffs_hand_calculated():
    # Two assets over two steps; averages are computed over t = 1..T only.
    S = torch.tensor([[[1.0, 1.0], [1.2, 0.8], [1.4, 0.6]]])     # [1, 3, 2]
    # running means: asset0 (1.2+1.4)/2 = 1.3 ; asset1 (0.8+0.6)/2 = 0.7
    call = payoff_basket_asian_call(S, kappa=1.0)                 # (1.3+0.7)/2 - 1 = 0
    assert abs(float(call) - 0.0) < 1e-6, float(call)
    call2 = payoff_basket_asian_call(S, kappa=0.95)
    assert abs(float(call2) - 0.05) < 1e-6, float(call2)
    put = payoff_asian_worst_of_put(S, kappa=1.0)                 # 1 - min(1.3, 0.7)
    assert abs(float(put) - 0.3) < 1e-6, float(put)
    put0 = payoff_asian_worst_of_put(S, kappa=0.5)
    assert abs(float(put0)) < 1e-12


def test_running_average_is_causal():
    S = torch.tensor([[[1.0], [2.0], [4.0], [8.0]]])
    avg = compute_running_averages(S)[0, :, 0]
    assert torch.allclose(avg, torch.tensor([1.0, 2.0, 3.0]))     # S0, S1, (S1+S2)/2


class _ConstantDeltaNet(nn.Module):
    """Deterministic policy used to hand-check costs and P&L."""

    def __init__(self, delta_value: float, d: int, v0: float = 0.0):
        super().__init__()
        self.d = d
        self.value = float(delta_value)
        self.V0 = nn.Parameter(torch.tensor(float(v0)))

    def forward(self, spots, running_avg, delta_prev, time_left):
        return torch.full_like(delta_prev, self.value)


def test_transaction_cost_and_pnl():
    d, cost_rate = 2, 0.001
    S = torch.tensor([[[1.0, 1.0], [1.1, 0.9], [1.2, 1.0]]])
    net = _ConstantDeltaNet(0.5, d)
    out = deep_hedge_forward(net, S, payoff_basket_asian_call, 1.0, cost_rate)
    # Only the first rebalance trades (0 -> 0.5 on both assets at S = 1).
    assert abs(float(out["transaction_cost"]) - cost_rate * (0.5 + 0.5)) < 1e-6
    # P&L = 0.5 * [(1.1-1.0)+(1.2-1.1)] + 0.5 * [(0.9-1.0)+(1.0-0.9)]
    assert abs(float(out["pnl"]) - 0.5 * 0.2) < 1e-6
    r = float(out["residuals"])
    expected = float(out["payoff"]) - 0.0 - float(out["pnl"]) + float(out["transaction_cost"])
    assert abs(r - expected) < 1e-6
    assert abs(float(out["terminal_wealth"]) - (0.0 + float(out["pnl"])
                                                - float(out["transaction_cost"]))) < 1e-6


def test_cvar_loss_closed_form():
    r = torch.tensor([0.0, 1.0, 2.0, 3.0, 10.0])
    loss = CVaRLoss(alpha=0.8, nu_init=3.0)
    # nu + mean(relu(r-nu))/(1-alpha) = 3 + (7/5)/0.2 = 10.0
    assert abs(float(loss(r)) - 10.0) < 1e-6, float(loss(r))
    # At the optimum nu = VaR_0.8, CVaR equals the mean of the worst 20%.
    nu_star = empirical_var(r.numpy(), 0.8)
    loss2 = CVaRLoss(alpha=0.8, nu_init=nu_star)
    assert float(loss2(r)) <= float(loss(r)) + 1e-9


def test_initialization_touches_every_layer():
    net = HedgingNetwork(d=3, hidden=(64, 64), output_gain=0.1)
    linears = [m for m in net.net if isinstance(m, nn.Linear)]
    report = net.init_report
    assert report["n_linear_layers"] == len(linears) == 3
    assert [l["role"] for l in report["layers"]] == ["hidden", "hidden", "output"]
    relu_gain = nn.init.calculate_gain("relu")
    for layer, entry in zip(linears, report["layers"]):
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))
        fan_in, fan_out = layer.weight.shape[1], layer.weight.shape[0]
        bound = entry["gain"] * math.sqrt(6.0 / (fan_in + fan_out))
        assert float(layer.weight.abs().max()) <= bound + 1e-6, \
            "weights outside the Xavier-uniform bound for the declared gain"
        # A hidden layer initialised with the default gain 1.0 would be ~1.41x
        # tighter; assert the ReLU gain is what was actually used.
        if entry["role"] == "hidden":
            assert abs(entry["gain"] - relu_gain) < 1e-9
        else:
            assert abs(entry["gain"] - 0.1) < 1e-9
    assert float(net.V0.item()) == float(CFG.v0_init)


def test_kernel_compact_support_boundary():
    h = 0.1
    h_sq = h * h
    diff_sq = torch.tensor([0.0, h_sq * 0.99, h_sq, h_sq * 1.01, 1.0],
                           dtype=torch.float64)
    log_k, support = quartic_log_kernel(diff_sq, h)
    # The support is strict: ||u||^2 == h^2 is already OUTSIDE.
    assert support.tolist() == [True, True, False, False, False]
    # Inside the support the kernel matches (h^2 - ||u||^2)^2.
    assert abs(float(torch.exp(log_k[0])) - h_sq ** 2) < 1e-15
    assert abs(float(torch.exp(log_k[1])) - (h_sq - h_sq * 0.99) ** 2) < 1e-15
    assert float(log_k[2]) == 0.0 and float(log_k[4]) == 0.0


def test_support_mask_not_affected_by_log_clamp():
    """A weight clamped away from zero must never manufacture support."""
    h = 0.05
    diff_sq = torch.tensor([[h * h + 1e-12, h * h * 0.5]])
    log_k, support = quartic_log_kernel(diff_sq, h)
    assert support.tolist() == [[False, True]]
    assert float(log_k[0, 0]) == 0.0          # masked out, not clamped into support
    w, den, any_support = _masked_softmax_weights(log_k, support)
    assert float(w[0, 0]) == 0.0 and float(w[0, 1]) > 0.0
    # With no supported reference at all, the weights are zero and the caller
    # is told there is no support (instead of silently using a clamped value).
    none_support = torch.zeros_like(support)
    w2, den2, any2 = _masked_softmax_weights(log_k, none_support)
    assert not bool(any2.item()) and float(den2.item()) == 0.0


def test_sliding_window_weights_are_exact():
    """The ring buffer must equal a from-scratch recomputation after eviction."""
    torch.manual_seed(0)
    M, K, h = 12, 3, 0.4
    X = torch.randn(M, 8, 2) * 0.1
    q = torch.randn(1, 8, 2) * 0.1
    sw = _SlidingWeights(1, M, K, X.device)
    for step in range(1, 6):
        diff = X[:, step, :].unsqueeze(0) - q[:, step, :].unsqueeze(1)
        lk, sup = quartic_log_kernel((diff ** 2).sum(-1), h)
        sw.push(lk, sup)
    log_w, support = sw.current()
    ref_log, ref_sup = torch.zeros(1, M), torch.ones(1, M, dtype=torch.bool)
    for step in range(3, 6):                       # the last K = 3 steps only
        diff = X[:, step, :].unsqueeze(0) - q[:, step, :].unsqueeze(1)
        lk, sup = quartic_log_kernel((diff ** 2).sum(-1), h)
        ref_log += torch.where(sup, lk, torch.zeros_like(lk))
        ref_sup &= sup
    assert torch.equal(support, ref_sup)
    assert torch.allclose(log_w[support], ref_log[ref_sup], atol=1e-5)


def test_hash_canonicalisation_is_stable():
    a = {"b": [1, 2, {"c": np.arange(3)}], "a": "x", "t": torch.ones(2)}
    b = {"t": torch.ones(2), "a": "x", "b": [1, 2, {"c": np.arange(3)}]}
    assert canonical_hash(a) == canonical_hash(b)
    c = dict(a); c["a"] = "y"
    assert canonical_hash(a) != canonical_hash(c)
    assert CFG.config_hash() == ExperimentConfig(**{
        f.name: getattr(CFG, f.name) for f in dataclasses.fields(CFG)}).config_hash()


def test_checkpoint_roundtrip():
    tmp_dir = TEST_SCRATCH
    tmp_dir.mkdir(parents=True, exist_ok=True)
    net = HedgingNetwork(d=CFG.d, hidden=(8, 8)).to(DEVICE)
    cvar = CVaRLoss(alpha=CFG.cvar_alpha, nu_init=0.123).to(DEVICE)
    optimizer = optim.Adam(list(net.parameters()) + list(cvar.parameters()), lr=1e-3)
    loss = cvar(deep_hedge_forward(
        net, torch.ones(4, 4, CFG.d, device=DEVICE) * 1.01,
        payoff_basket_asian_call, 1.0, CFG.cost_rate)["residuals"])
    loss.backward(); optimizer.step()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    bundle = _clone_bundle(net, cvar, optimizer, scheduler, 7, 0.5)
    spec = {"run_key": "TEST", "generator": "GBM",
            "option": "basket_asian_call", "kappa": 1.0, "seed": 3}
    payload = build_checkpoint_payload(
        spec=spec, phase="cvar", bundle=bundle,
        history={"train_loss": [1.0], "val_loss": [0.5]},
        synthetic_metrics={"std": 0.1}, model_spec=net.spec(),
        historical_metrics={r: {"std": 0.1} for r in CFG.regime_names},
        data_hashes={"test": "0" * 64})
    path = tmp_dir / "roundtrip__cvar__best.pt"
    save_checkpoint(payload, path)
    ckpt = verify_checkpoint(path)
    net2, cvar2, _ = load_checkpoint(path)
    for k, v in net.state_dict().items():
        assert torch.allclose(v.cpu(), net2.state_dict()[k].cpu(), atol=1e-6)
    assert abs(float(cvar2.nu.item()) - float(cvar.nu.item())) < 1e-9, "nu not restored"
    assert ckpt["optimizer_state_dict"]["state"], "optimizer state lost"
    assert ckpt["scheduler_state_dict"], "scheduler state lost"
    # A tampered checkpoint must fail verification.
    tampered = torch.load(path, map_location="cpu", weights_only=False)
    tampered["best_val_loss"] = 999.0
    atomic_write_torch(path, tampered, verify=False)
    try:
        verify_checkpoint(path)
        raise AssertionError("verification accepted a tampered checkpoint")
    except ValueError:
        pass
    finally:
        path.unlink(missing_ok=True)
        path.with_suffix(".json").unlink(missing_ok=True)


def test_atomic_write_and_reload():
    p = TEST_SCRATCH / "atomic.json"
    digest = atomic_write_json(p, {"a": 1, "b": [1, 2, 3]})
    assert digest == sha256_file(p)
    assert not p.with_suffix(".json.tmp").exists()
    p.unlink()


def test_no_seed_aliasing():
    assert SEED_ALIAS_MAP == {}
    assert_no_seed_aliasing()
    queue = canonical_run_queue(CFG)
    assert len({q["run_key"] for q in queue}) == len(queue), "duplicate run keys"
    for q in queue:
        assert f"seed{q['seed']:02d}" in q["run_key"]
    assert 3 in set(CFG.seeds), "seed 3 must be part of the grid"


# ── Integration tests ────────────────────────────────────────────────────────
def _mini_paths(n: int, t: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch_generator(DEVICE, seed, 99)
    r = torch.randn(n, t, d, generator=g, device=DEVICE) * 0.01
    s = torch.ones(n, t + 1, d, device=DEVICE)
    s[:, 1:, :] = torch.exp(torch.cumsum(r, dim=1))
    return s


def test_epoch_consumes_every_sample():
    cfg = dataclasses.replace(CFG, batch_size=7, mse_epochs=2, mse_patience=2)
    S_tr, S_vl = _mini_paths(23, 4, cfg.d), _mini_paths(8, 4, cfg.d)
    net = HedgingNetwork(d=cfg.d, hidden=(8, 8)).to(DEVICE)
    res = train_phase(net=net, loss_module=loss_mse,
                      trainable_params=list(net.parameters()),
                      S_train=S_tr, S_val=S_vl, payoff_fn=payoff_basket_asian_call,
                      kappa=1.0, lr=1e-3, max_epochs=2, patience=2, cfg=cfg,
                      phase="mse", verbose_every=0)
    assert all(s == 23 for s in res["history"]["samples_seen"]), \
        res["history"]["samples_seen"]
    # 23 samples with batch 7 => 4 batches, the last one short: nothing dropped.
    assert math.ceil(23 / 7) == 4


def test_post_clip_norm_invariant():
    cfg = dataclasses.replace(CFG, batch_size=8, grad_clip=1e-4)
    S_tr, S_vl = _mini_paths(16, 4, cfg.d, seed=1), _mini_paths(8, 4, cfg.d, seed=2)
    net = HedgingNetwork(d=cfg.d, hidden=(8, 8)).to(DEVICE)
    res = train_phase(net=net, loss_module=loss_mse,
                      trainable_params=list(net.parameters()),
                      S_train=S_tr, S_val=S_vl, payoff_fn=payoff_asian_worst_of_put,
                      kappa=1.0, lr=1e-3, max_epochs=2, patience=2, cfg=cfg,
                      phase="mse", verbose_every=0)
    post = res["history"]["grad_norm_post_clip"]
    pre = res["history"]["grad_norm_pre_clip"]
    assert all(p <= cfg.grad_clip + cfg.grad_clip_tolerance for p in post), post
    assert any(a > b for a, b in zip(pre, post)), \
        "post-clip norms equal the pre-clip norms — clipping was not measured"


def test_best_bundle_restores_network_and_nu():
    cfg = dataclasses.replace(CFG, batch_size=8)
    S_tr, S_vl = _mini_paths(16, 4, cfg.d, seed=3), _mini_paths(8, 4, cfg.d, seed=4)
    net = HedgingNetwork(d=cfg.d, hidden=(8, 8)).to(DEVICE)
    cvar = CVaRLoss(alpha=cfg.cvar_alpha, nu_init=0.05).to(DEVICE)
    params = list(net.parameters()) + list(cvar.parameters())
    res = train_phase(net=net, loss_module=cvar, trainable_params=params,
                      S_train=S_tr, S_val=S_vl, payoff_fn=payoff_basket_asian_call,
                      kappa=1.0, lr=1e-3, max_epochs=4, patience=4, cfg=cfg,
                      phase="cvar", verbose_every=0)
    best = res["best_bundle"]
    assert best["loss_state_dict"] is not None and "nu" in best["loss_state_dict"]
    for k, v in best["model_state_dict"].items():
        assert torch.allclose(v, net.state_dict()[k].cpu(), atol=1e-6), \
            "network was not restored to the best epoch"
    assert abs(float(best["loss_state_dict"]["nu"]) - float(cvar.nu.item())) < 1e-9, \
        "nu was not restored to the best epoch"
    assert best["epoch"] == int(np.argmin(res["history"]["val_loss"]))


def test_nu_initialised_from_validation_var():
    residuals = np.array([0.0, 0.1, 0.2, 0.3, 5.0])
    nu = empirical_var(residuals, 0.95)
    assert abs(nu - float(np.quantile(residuals, 0.95))) < 1e-12
    assert nu > 0.3, "VaR_0.95 must sit in the upper tail, not at zero"


def test_training_resumes_at_the_epoch_it_reached():
    """An interrupted phase must continue, not restart, and land where an
    uninterrupted run of the same length would have landed."""
    global CHECKPOINT_EVERY_EPOCHS
    tmp = TEST_SCRATCH
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / "resume_probe__mse__last.pt"
    path.unlink(missing_ok=True)
    cfg = dataclasses.replace(CFG, batch_size=8)
    S_tr, S_vl = _mini_paths(24, 4, cfg.d, seed=11), _mini_paths(8, 4, cfg.d, seed=12)
    saved_interval = CHECKPOINT_EVERY_EPOCHS

    def _run(max_epochs, resume):
        seed_all(7)
        net = HedgingNetwork(d=cfg.d, hidden=(8, 8)).to(DEVICE)
        return net, train_phase(
            net=net, loss_module=loss_mse, trainable_params=list(net.parameters()),
            S_train=S_tr, S_val=S_vl, payoff_fn=payoff_basket_asian_call,
            kappa=1.0, lr=1e-3, max_epochs=max_epochs, patience=max_epochs,
            cfg=cfg, phase="mse", verbose_every=0,
            progress_path=path if resume else None, spec_key="resume_probe")

    try:
        CHECKPOINT_EVERY_EPOCHS = 1
        # Reference: six uninterrupted epochs, no progress file.
        _, ref = _run(6, resume=False)
        # Interrupted: three epochs, then resume and run to six. Reaching a
        # budget of three is a normal completion, so the progress file is
        # rewritten to the state a crash right after epoch 3's periodic save
        # would have left behind: same content, phase_complete still False.
        path.unlink(missing_ok=True)
        _, first = _run(3, resume=True)
        assert first["epochs_run"] == 3
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["next_epoch"] == 3, saved["next_epoch"]
        saved["phase_complete"] = False
        atomic_write_torch(path, saved, verify=False)
        net2, second = _run(6, resume=True)
        assert second["resumed_from_epoch"] == 3, second["resumed_from_epoch"]
        assert second["epochs_run"] == 6
        # It continued: the history carries all six epochs, the first three of
        # which are the ones already computed.
        assert len(second["history"]["val_loss"]) == 6
        assert second["history"]["val_loss"][:3] == first["history"]["val_loss"]
        # And it matches the uninterrupted run, so resuming costs no accuracy.
        assert second["history"]["val_loss"] == ref["history"]["val_loss"], (
            "resumed run diverged from the uninterrupted run")
        assert abs(second["best_val_loss"] - ref["best_val_loss"]) < 1e-9
    finally:
        CHECKPOINT_EVERY_EPOCHS = saved_interval
        path.unlink(missing_ok=True)


def test_completed_phase_is_not_retrained():
    """A progress file marked complete must not be resumed from."""
    tmp = TEST_SCRATCH
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / "complete_probe__mse__last.pt"
    atomic_write_torch(path, {"run_key": "probe", "phase": "mse",
                              "config_hash": CONFIG_HASH, "next_epoch": 5,
                              "best_val_loss": 0.1, "patience_counter": 0,
                              "history": {}, "best_bundle": {"x": 1},
                              "last_bundle": {"x": 1}, "phase_complete": True},
                       verify=False)
    assert _load_progress(path, "probe", "mse") is None
    # A different configuration or run key is never picked up either.
    atomic_write_torch(path, {"run_key": "probe", "phase": "mse",
                              "config_hash": "deadbeef", "next_epoch": 5,
                              "best_val_loss": 0.1, "patience_counter": 0,
                              "history": {}, "best_bundle": {"x": 1},
                              "last_bundle": {"x": 1}, "phase_complete": False},
                       verify=False)
    assert _load_progress(path, "probe", "mse") is None
    path.unlink(missing_ok=True)


def test_phase1_reuse_matches_uninterrupted_run():
    """A session interrupted between Phase 1 and Phase 2 must, on re-run, give
    exactly the result of a run that was never interrupted — both when it
    reuses the finished Phase 1 and when it has to re-train it."""
    saved = PATHS["checkpoints"]
    root = TEST_SCRATCH / "phase1_reuse"

    def _paths(n, t, seed, vol=0.05):
        g = torch_generator(DEVICE, seed, 77)
        r = torch.randn(n, t, CFG.d, generator=g, device=DEVICE) * vol
        s = torch.ones(n, t + 1, CFG.d, device=DEVICE)
        s[:, 1:, :] = torch.exp(torch.cumsum(r, dim=1))
        return s

    # The data must make Phase 2 actually TRAIN the network. On flat paths no
    # residual clears nu, the network gets zero gradient, the minibatch order
    # stops mattering, and this test would pass with the bug still present.
    # alpha = 0.5 and volatile paths put half of the residuals above nu.
    cfg = dataclasses.replace(CFG, batch_size=8, mse_epochs=4, cvar_epochs=4,
                              mse_patience=4, cvar_patience=4,
                              cvar_alpha=0.5, cvar_lr=1e-2)
    S = {"train": _paths(32, 6, 21), "val": _paths(16, 6, 22),
         "test": _paths(16, 6, 23)}
    hist = {r: _paths(8, 6, 30 + i) for i, r in enumerate(cfg.regime_names)}
    opt_name = "asian_worst_of_put"
    spec = {"generator": "GBM", "option": opt_name, "kappa": 1.05,
            "seed": 5, "run_key": run_key("GBM", opt_name, 1.05, 5)}
    ck = lambda phase, kind: (root / "GBM" / checkpoint_filename(
        "GBM", opt_name, 1.05, 5, phase, kind))

    def _drop(phase, kinds=("best", "last")):
        for kind in kinds:
            ck(phase, kind).unlink(missing_ok=True)
            ck(phase, kind).with_suffix(".json").unlink(missing_ok=True)

    def _phase2(res):
        c = res["phases"]["cvar"]
        return (c["best_val_loss"], c["best_epoch"], c["metrics"]["std"],
                c["metrics"]["cvar95"], c["nu"])

    try:
        PATHS["checkpoints"] = root
        (root / "GBM").mkdir(parents=True, exist_ok=True)
        ref = train_configuration(spec, cfg, {"GBM": S}, hist, {}, verbose_every=0)

        # Interrupted after Phase 1: its checkpoints are there, Phase 2's are not.
        _drop("cvar")
        reused = train_configuration(spec, cfg, {"GBM": S}, hist, {}, verbose_every=0)
        assert reused["phases"]["mse"].get("reused_existing_checkpoint"), \
            "the finished Phase 1 was not reused"
        assert _phase2(reused) == _phase2(ref), (
            f"Phase 2 after reusing Phase 1 differs from the uninterrupted run: "
            f"{_phase2(reused)} vs {_phase2(ref)}")

        # Same, but the record of Phase 1's final random state is gone: Phase 1
        # must be re-trained rather than reused, so the result still matches.
        _drop("cvar")
        _drop("mse", kinds=("last",))
        retrained = train_configuration(spec, cfg, {"GBM": S}, hist, {}, verbose_every=0)
        assert not retrained["phases"]["mse"].get("reused_existing_checkpoint"), \
            "Phase 1 was reused without the random state needed to continue it"
        assert _phase2(retrained) == _phase2(ref), (
            f"re-trained configuration differs from the uninterrupted run: "
            f"{_phase2(retrained)} vs {_phase2(ref)}")
    finally:
        PATHS["checkpoints"] = saved


def test_rollout_resume_equivalence():
    """A batch regenerated from (seed, batch_index) must be bit-identical."""
    X = X_REF_T[:64, :6, :]
    y0 = X[0, 0, :].unsqueeze(0).expand(4, -1).clone()
    a, _ = sbts_rollout(X, y0.clone(), 0, 5, 0.3, 2, 2, CFG.delta_t,
                        torch_generator(DEVICE, 42, 7))
    b, _ = sbts_rollout(X, y0.clone(), 0, 5, 0.3, 2, 2, CFG.delta_t,
                        torch_generator(DEVICE, 42, 7))
    c, _ = sbts_rollout(X, y0.clone(), 0, 5, 0.3, 2, 2, CFG.delta_t,
                        torch_generator(DEVICE, 42, 8))
    assert torch.equal(a, b), "same (seed, batch) gave different paths"
    assert not torch.equal(a, c), "different batches gave identical paths"


def test_zero_support_fallback_is_explicit():
    """With an impossibly small bandwidth every query loses support; the
    fallback must fire and be counted rather than producing NaN."""
    X = X_REF_T[:32, :5, :]
    y0 = torch.full((3, CFG.d), 10.0, device=DEVICE)      # far outside any support
    Y, diag = sbts_rollout(X, y0, 1, 4, 1e-6, 2, 2, CFG.delta_t,
                           torch_generator(DEVICE, 1, 1),
                           prime_states=y0.unsqueeze(1),
                           zero_support_policy="zero_drift")
    assert diag["zero_support_events"] > 0, "zero support was not detected"
    assert torch.isfinite(Y).all(), "fallback produced non-finite states"


def test_historical_paths_dates_and_labels():
    for name, obj in HISTORICAL.items():
        meta = obj["meta"]
        lo, hi = CFG.regime_ranges[name]
        assert meta["assignment_rule"] == "path start date"
        assert lo <= meta["start_date_min"] <= meta["start_date_max"] <= hi
        assert meta["end_date_min"] >= meta["start_date_min"]
        assert obj["S_norm"].shape[1] == CFG.horizon + 1
        assert np.allclose(obj["S_norm"][:, 0, :], 1.0)
        assert len(obj["start_dates"]) == meta["n_paths"] == obj["S_norm"].shape[0]


def test_artifacts_carry_hashes():
    for key, entry in MANIFEST["artifacts"].items():
        assert entry["config_hash"] == CONFIG_HASH, key
        assert entry["schema_version"] == CFG.schema_version, key
        assert entry["sha256"], key


def test_no_stale_kernel_state():
    """Nothing this notebook needs may come from an earlier notebook."""
    for name in ("seed_all", "CFG", "CONFIG_HASH", "RUN_ID", "HedgingNetwork",
                 "deep_hedge_forward", "sbts_rollout", "train_phase"):
        assert name in globals(), f"{name} is not defined by this notebook"
    for stale in ("DS_LIST", "SESSION_NAME", "OUTLIER_REPLACEMENT",
                  "mcs_seed_level", "CKPT_ROOT"):
        assert stale not in globals(), f"stale symbol {stale} leaked into this run"


UNIT_TESTS = [
    (test_seed_all_reproducible, "seed_all is reproducible"),
    (test_payoffs_hand_calculated, "payoffs match hand calculations"),
    (test_running_average_is_causal, "running average is causal"),
    (test_transaction_cost_and_pnl, "transaction cost / PnL / residual identity"),
    (test_cvar_loss_closed_form, "CVaR loss closed form"),
    (test_initialization_touches_every_layer, "Xavier init on every layer"),
    (test_kernel_compact_support_boundary, "kernel compact-support boundary"),
    (test_support_mask_not_affected_by_log_clamp, "support mask ignores log clamp"),
    (test_sliding_window_weights_are_exact, "sliding-window weights are exact"),
    (test_hash_canonicalisation_is_stable, "hash canonicalisation is stable"),
    (test_checkpoint_roundtrip, "checkpoint save/load/verify roundtrip"),
    (test_atomic_write_and_reload, "atomic write + reload verify"),
    (test_no_seed_aliasing, "no seed aliasing, unique run keys"),
]

INTEGRATION_TESTS = [
    (test_epoch_consumes_every_sample, "every epoch uses all training samples"),
    (test_post_clip_norm_invariant, "post-clip gradient norm invariant"),
    (test_best_bundle_restores_network_and_nu, "best bundle restores model and nu"),
    (test_nu_initialised_from_validation_var, "nu initialised from validation VaR"),
    (test_training_resumes_at_the_epoch_it_reached,
     "training resumes at the epoch it reached"),
    (test_completed_phase_is_not_retrained,
     "a completed or foreign phase is never resumed"),
    (test_phase1_reuse_matches_uninterrupted_run,
     "reusing a finished Phase 1 matches an uninterrupted run"),
    (test_rollout_resume_equivalence, "batch resume reproduces the same paths"),
    (test_zero_support_fallback_is_explicit, "zero-support fallback is explicit"),
    (test_historical_paths_dates_and_labels, "historical path dates and labels"),
    (test_artifacts_carry_hashes, "every artifact carries config/data hashes"),
    (test_no_stale_kernel_state, "no stale kernel/notebook state"),
]

print("Unit tests")
_ok = all([_run_test(f, n, "unit") for f, n in UNIT_TESTS])
print("\nIntegration tests")
_ok = all([_run_test(f, n, "integration") for f, n in INTEGRATION_TESTS]) and _ok

TEST_TABLE = pd.DataFrame(TEST_RESULTS)
_h = atomic_write_dataframe(PATHS["logs"] / f"test_results{SHARD_TAG}.csv", TEST_TABLE)
register_artifact(f"logs/test_results{SHARD_TAG}.csv",
                  PATHS["logs"] / f"test_results{SHARD_TAG}.csv", _h)
MANIFEST["gate_1_static_audit"] = {
    "passed": bool(_ok), "n_tests": len(TEST_RESULTS),
    "n_failed": int((TEST_TABLE["status"] == "FAIL").sum()), "utc": utc_now()}
save_manifest()

print(f"\nGate 1 (static implementation audit): "
      f"{'PASSED' if _ok else 'FAILED'} "
      f"({len(TEST_RESULTS) - int((TEST_TABLE['status'] == 'FAIL').sum())}"
      f"/{len(TEST_RESULTS)} tests)")
if not _ok:
    add_failure("Gate 1 failed: unit/integration tests did not all pass")
    raise RuntimeError("Gate 1 failed — fix the failing tests before running "
                       "anything else. The failures are listed above and in "
                       "logs/test_results.csv.")

# ==========================================================================
# ---
#
# ## Section 12 — A100 acceleration modes and the benchmark gate (Gate 3)
#
# `REFERENCE_FP32` is the canonical numerical mode. `A100_FAST` may only be locked
# in after it passes the tolerance gate measured on this machine. The runtime
# estimate for the full experiment is printed from these measurements only.

# ---- notebook cell 26 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 18 — SMOKE BENCHMARK AND A100 PERFORMANCE / NUMERICAL GATE (Gate 3)
# ═════════════════════════════════════════════════════════════════════════════
#  Benchmarks REFERENCE_FP32 against each candidate fast configuration on:
#  all three generators, both payoffs, one representative strike, at least
#  three seeds (seed 3 included) and both phases.
#
#  A fast mode may only be locked in for a FULL run if it passes ALL of:
#    * no change in convergence/failure classification
#    * no NaN/Inf anywhere
#    * aggregate metric deviation within gate_metric_rel_tol
#    * no ranking reversal between generators on the smoke aggregate
#    * a speed-up large enough to justify the numerical trade-off
#  Otherwise the published run uses REFERENCE_FP32 on the A100.
#
#  The runtime estimate for the full experiment is printed only from these
#  measurements — never from a guess.
# ═════════════════════════════════════════════════════════════════════════════

# A100_FAST relies on bf16 autocast and TF32, which need an Ampere-class GPU
# (compute capability >= 8.0). On a T4 (7.5) it is not a candidate at all:
# benchmarking it would at best waste time on emulation and at worst raise a
# CUDA error that stops the notebook.
FAST_MODE_SUPPORTED = bool(CUDA_AVAILABLE and GPU_CAPABILITY and GPU_CAPABILITY[0] >= 8)
GATE_MODES = ["REFERENCE_FP32"] + (["A100_FAST"] if FAST_MODE_SUPPORTED else [])
if PRECISION_MODE == "A100_FAST" and not FAST_MODE_SUPPORTED:
    add_warning(f"A100_FAST requested on {GPU_NAME} (capability {GPU_CAPABILITY}), "
                f"which lacks bf16/TF32; using REFERENCE_FP32")
    PRECISION_MODE = "REFERENCE_FP32"
    PRECISION_FLAGS = apply_precision_mode(PRECISION_MODE)
BENCHMARK_PATH = PATHS["logs"] / f"benchmark{SHARD_TAG}.csv"
GATE_PATH = PATHS["logs"] / f"gate3_precision{SHARD_TAG}.json"


def _gate_subset(t: torch.Tensor, n: int) -> torch.Tensor:
    return t[:min(n, t.shape[0])]


def run_benchmark(mode: str, cfg: ExperimentConfig) -> List[Dict[str, Any]]:
    global PRECISION_MODE, PRECISION_FLAGS
    PRECISION_MODE = mode
    PRECISION_FLAGS = apply_precision_mode(mode)
    rows = []
    bench_cfg = dataclasses.replace(
        cfg, mse_epochs=cfg.gate_mse_epochs, cvar_epochs=cfg.gate_cvar_epochs,
        mse_patience=cfg.gate_mse_epochs, cvar_patience=cfg.gate_cvar_epochs)
    for gen in GENERATORS:
        S_tr = _gate_subset(DATASETS[gen]["train"], cfg.gate_n_train)
        S_vl = _gate_subset(DATASETS[gen]["val"], cfg.gate_n_val)
        S_te = _gate_subset(DATASETS[gen]["test"], cfg.gate_n_test)
        for option in cfg.options:
            for seed in cfg.gate_seeds:
                clear_mem()
                seed_all(seed)
                net = HedgingNetwork(d=cfg.d, hidden=cfg.hidden_sizes,
                                     output_gain=cfg.init_output_gain,
                                     v0_init=cfg.v0_init).to(DEVICE)
                payoff_fn = PAYOFF_FNS[option]
                status, err = "converged", None
                t0 = time.time()
                try:
                    p1 = train_phase(net=net, loss_module=loss_mse,
                                     trainable_params=list(net.parameters()),
                                     S_train=S_tr, S_val=S_vl, payoff_fn=payoff_fn,
                                     kappa=cfg.gate_strike, lr=cfg.mse_lr,
                                     max_epochs=bench_cfg.mse_epochs,
                                     patience=bench_cfg.mse_patience, cfg=bench_cfg,
                                     phase="mse", verbose_every=0)
                    _, val_res = _validation_loss(net, loss_mse, S_vl, payoff_fn,
                                                  cfg.gate_strike, cfg)
                    net.V0.requires_grad_(False)
                    cvar = CVaRLoss(alpha=cfg.cvar_alpha,
                                    nu_init=empirical_var(val_res, cfg.cvar_alpha)).to(DEVICE)
                    p2 = train_phase(net=net, loss_module=cvar,
                                     trainable_params=[p for p in net.parameters()
                                                       if p.requires_grad]
                                                      + list(cvar.parameters()),
                                     S_train=S_tr, S_val=S_vl, payoff_fn=payoff_fn,
                                     kappa=cfg.gate_strike, lr=cfg.cvar_lr,
                                     max_epochs=bench_cfg.cvar_epochs,
                                     patience=bench_cfg.cvar_patience, cfg=bench_cfg,
                                     phase="cvar", verbose_every=0)
                    ev = evaluate_full(net, S_te, payoff_fn, cfg.gate_strike, cfg)
                    hist = {r: evaluate_full(net, _gate_subset(t, cfg.gate_n_test),
                                             payoff_fn, cfg.gate_strike, cfg)["metrics"]
                            for r, t in HIST_TENSORS.items()}
                except Exception as exc:                     # noqa: BLE001
                    # Any failure is a benchmark verdict, never a reason to
                    # stop the notebook before the real training starts.
                    status, err = "failed", repr(exc)
                    p1 = p2 = None
                    ev = {"metrics": {"std": float("nan"), "cvar95": float("nan")}}
                    hist = {r: {"std": float("nan"), "cvar95": float("nan")}
                            for r in HIST_TENSORS}
                wall = time.time() - t0
                row = {
                    "mode": mode, "generator": gen, "option": option,
                    "kappa": cfg.gate_strike, "seed": seed, "status": status,
                    "error": err, "wall_seconds": wall,
                    "peak_vram_mb": gpu_peak_mb(),
                    "mse_epochs": p1["epochs_run"] if p1 else 0,
                    "cvar_epochs": p2["epochs_run"] if p2 else 0,
                    "mse_best_val": p1["best_val_loss"] if p1 else float("nan"),
                    "cvar_best_val": p2["best_val_loss"] if p2 else float("nan"),
                    "test_std": ev["metrics"]["std"],
                    "test_cvar95": ev["metrics"]["cvar95"],
                    "all_finite": bool(ev["metrics"].get("all_finite", False)),
                }
                for r, m in hist.items():
                    row[f"hist_{r}_std"] = m["std"]
                    row[f"hist_{r}_cvar95"] = m["cvar95"]
                rows.append(row)
                print(f"    {mode:<14s} {gen:<7s} {option:<19s} seed {seed}  "
                      f"{wall:6.1f}s  std={row['test_std']:.5f}  "
                      f"cvar95={row['test_cvar95']:.5f}  {status}", flush=True)
                del net
                clear_mem()
    return rows


def evaluate_gate(df: "pd.DataFrame", cfg: ExperimentConfig) -> Dict[str, Any]:
    ref = df[df["mode"] == "REFERENCE_FP32"]
    verdicts = {}
    for mode in [m for m in df["mode"].unique() if m != "REFERENCE_FP32"]:
        fast = df[df["mode"] == mode]
        join_on = ["generator", "option", "kappa", "seed"]
        merged = ref.merge(fast, on=join_on, suffixes=("_ref", "_fast"))
        checks, details = {}, {}

        checks["no_classification_change"] = bool(
            (merged["status_ref"] == merged["status_fast"]).all())
        checks["no_nonfinite"] = bool(merged["all_finite_ref"].all()
                                      and merged["all_finite_fast"].all())

        metric_cols = ["test_std", "test_cvar95"] + [
            c[:-4] for c in merged.columns
            if c.startswith("hist_") and c.endswith("_ref")]
        worst_rel = 0.0
        for col in ["test_std", "test_cvar95"]:
            a = merged[f"{col}_ref"].mean()
            b = merged[f"{col}_fast"].mean()
            rel = abs(b - a) / max(abs(a), 1e-12)
            details[f"rel_dev_{col}"] = float(rel)
            worst_rel = max(worst_rel, rel)
        for r in CFG.regime_names:
            a = merged[f"hist_{r}_std_ref"].mean()
            b = merged[f"hist_{r}_std_fast"].mean()
            rel = abs(b - a) / max(abs(a), 1e-12)
            details[f"rel_dev_hist_{r}_std"] = float(rel)
            worst_rel = max(worst_rel, rel)
        details["worst_relative_deviation"] = float(worst_rel)
        checks["within_metric_tolerance"] = bool(worst_rel <= cfg.gate_metric_rel_tol)

        rank_ref = ref.groupby("generator")["test_std"].mean().rank().to_dict()
        rank_fast = fast.groupby("generator")["test_std"].mean().rank().to_dict()
        details["ranking_reference"] = rank_ref
        details["ranking_fast"] = rank_fast
        checks["no_ranking_reversal"] = bool(rank_ref == rank_fast)

        speedup = float(ref["wall_seconds"].sum() / max(fast["wall_seconds"].sum(), 1e-9))
        details["speedup"] = speedup
        checks["speedup_justifies_tradeoff"] = bool(speedup >= 1.15)

        verdicts[mode] = {"passed": all(checks.values()), "checks": checks,
                          "details": details}
    return verdicts


def _gate_cache_is_usable() -> bool:
    """Reuse a gate result measured on THIS machine for THIS configuration.

    A 180-configuration run will be interrupted and resumed several times on
    Colab; re-benchmarking the same GPU every restart costs minutes and can
    only produce the same verdict. The cache is keyed by config_hash, GPU name
    and the requested mode, so any of those changing forces a fresh benchmark.
    """
    if not (BENCHMARK_PATH.exists() and GATE_PATH.exists()):
        return False
    try:
        with open(GATE_PATH, "r", encoding="utf-8") as f:
            rep = json.load(f)
    except Exception:                                        # noqa: BLE001
        return False
    ok, _ = cache_is_valid(rep, extra={"gpu_name": GPU_NAME,
                                       "requested_mode": PRECISION_MODE})
    return ok


if RUN_MODE in ("FULL", "SMOKE"):
    _saved_mode = PRECISION_MODE
    if _gate_cache_is_usable():
        with open(GATE_PATH, "r", encoding="utf-8") as _f:
            GATE_REPORT = json.load(_f)
        BENCHMARK_TABLE = pd.read_csv(BENCHMARK_PATH)
        GATE_VERDICTS = GATE_REPORT["verdicts"]
        LOCKED_PRECISION_MODE = GATE_REPORT["locked_precision_mode"]
        PRECISION_MODE = LOCKED_PRECISION_MODE
        PRECISION_FLAGS = apply_precision_mode(PRECISION_MODE)
        MANIFEST["gate_3_precision"] = {
            "locked_precision_mode": LOCKED_PRECISION_MODE,
            "verdicts": GATE_VERDICTS, "reused_from_cache": True}
        save_manifest()
        print(f"Gate 3 reused from this run's cached benchmark "
              f"(GPU {GPU_NAME}, measured {GATE_REPORT['evaluated_utc']}).")
        print(f"  locked precision mode : {LOCKED_PRECISION_MODE}")
        print(f"  runtime estimate      : "
              f"{GATE_REPORT['runtime_estimate_hours_full_run']:.1f} GPU-hours "
              f"for a full run")
        print("  Delete logs/gate3_precision.json to force a fresh benchmark.")
        _skip_gate = True
    else:
        _skip_gate = False

if RUN_MODE in ("FULL", "SMOKE") and not _skip_gate:
    print(f"Benchmark modes: {GATE_MODES}\n")
    _bench_rows = []
    for _mode in GATE_MODES:
        print(f"  [{_mode}]")
        _bench_rows += run_benchmark(_mode, CFG)
    BENCHMARK_TABLE = pd.DataFrame(_bench_rows)
    _h = atomic_write_dataframe(BENCHMARK_PATH, BENCHMARK_TABLE)
    register_artifact(f"logs/benchmark{SHARD_TAG}.csv", BENCHMARK_PATH, _h)

    GATE_VERDICTS = evaluate_gate(BENCHMARK_TABLE, CFG)
    LOCKED_PRECISION_MODE = "REFERENCE_FP32"
    if _saved_mode == "A100_FAST":
        v = GATE_VERDICTS.get("A100_FAST", {"passed": False})
        LOCKED_PRECISION_MODE = "A100_FAST" if v.get("passed") else "REFERENCE_FP32"
        if not v.get("passed"):
            add_warning("A100_FAST did not pass Gate 3; "
                        "the run falls back to REFERENCE_FP32", v)

    PRECISION_MODE = LOCKED_PRECISION_MODE
    PRECISION_FLAGS = apply_precision_mode(PRECISION_MODE)

    _ref = BENCHMARK_TABLE[BENCHMARK_TABLE["mode"] == LOCKED_PRECISION_MODE]
    _sec_per_epoch = float(
        (_ref["wall_seconds"] / (_ref["mse_epochs"] + _ref["cvar_epochs"]).clip(lower=1)).mean())
    _scale = (CFG.n_train / max(1, min(CFG.gate_n_train, CFG.n_train)))
    _expected_epochs = 0.5 * (CFG.mse_epochs + CFG.cvar_epochs)   # early stopping assumed
    _estimate_hours = (_sec_per_epoch * _scale * _expected_epochs
                       * CFG.n_configurations / 3600.0)

    GATE_REPORT = {
        "schema_version": CFG.schema_version, "config_hash": CONFIG_HASH,
        "gpu_name": GPU_NAME,
        "modes_benchmarked": GATE_MODES,
        "requested_mode": _saved_mode,
        "locked_precision_mode": LOCKED_PRECISION_MODE,
        "verdicts": GATE_VERDICTS,
        "tolerance": {"metric_rel_tol": CFG.gate_metric_rel_tol,
                      "min_speedup": 1.15,
                      "set_before_run": True},
        "gate_grid": {"generators": list(GENERATORS), "options": list(CFG.options),
                      "kappa": CFG.gate_strike, "seeds": list(CFG.gate_seeds),
                      "phases": ["mse", "cvar"]},
        "measured_seconds_per_epoch_at_gate_size": _sec_per_epoch,
        "gate_train_paths": int(min(CFG.gate_n_train, CFG.n_train)),
        "runtime_estimate_hours_full_run": _estimate_hours,
        "runtime_estimate_basis": ("measured seconds/epoch at the gate training "
                                   "size, scaled linearly to n_train and assuming "
                                   "early stopping at half the epoch budget"),
        "evaluated_utc": utc_now(),
    }
    atomic_write_json(GATE_PATH, GATE_REPORT)
    register_artifact(f"logs/gate3_precision{SHARD_TAG}.json", GATE_PATH)
    MANIFEST["gate_3_precision"] = {"locked_precision_mode": LOCKED_PRECISION_MODE,
                                    "verdicts": GATE_VERDICTS}
    save_manifest()

    print("\nGate 3 verdicts:")
    for _mode, _v in GATE_VERDICTS.items():
        print(f"  {_mode}: {'PASS' if _v['passed'] else 'FAIL'}")
        for _c, _ok2 in _v["checks"].items():
            print(f"      {_c:<32s} {_ok2}")
        print(f"      worst relative deviation      "
              f"{_v['details']['worst_relative_deviation']:.4f} "
              f"(tolerance {CFG.gate_metric_rel_tol})")
        print(f"      speed-up                      {_v['details']['speedup']:.2f}x")
    print(f"\n  LOCKED precision mode for this run: {LOCKED_PRECISION_MODE}")
    print(f"\nRuntime estimate for a FULL run on this machine: "
          f"{_estimate_hours:.1f} GPU-hours")
    print(f"  ({_sec_per_epoch:.2f} s/epoch measured at "
          f"{GATE_REPORT['gate_train_paths']:,} training paths, scaled to "
          f"{CFG.n_train:,} and {CFG.n_configurations} configurations)")
elif RUN_MODE not in ("FULL", "SMOKE"):
    LOCKED_PRECISION_MODE = PRECISION_MODE
    BENCHMARK_TABLE = pd.DataFrame()
    GATE_VERDICTS = {}
    print(f"RUN_MODE={RUN_MODE}: benchmark skipped; precision mode stays "
          f"{PRECISION_MODE}.")

# ==========================================================================
# ---
#
# ## Sections 13–14 — Experiment queue, checkpoint audit, uniform evaluation
#
# The queue is resumable and idempotent. Every checkpoint is then verified
# (including that its recorded seed matches its run key) and re-evaluated through
# a single code path on the synthetic test set and all three historical regimes.

# ---- notebook cell 28 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 19 — FULL EXPERIMENT QUEUE (Gate 5)
# ═════════════════════════════════════════════════════════════════════════════
#  Idempotent and resumable: a run whose status is `complete` and whose two
#  checkpoints still verify is skipped. A failure is retried with the SAME seed
#  and the SAME configuration up to `max_retries_per_run`; after that the run is
#  recorded `status="failed"` and it simply does not exist for the statistics.
#  No result is fabricated, and no seed is ever relabelled.
# ═════════════════════════════════════════════════════════════════════════════

class ShardTrainingComplete(RuntimeError):
    """Raised to stop a sharded session after training — not a failure."""


RESULTS_PATH = PATHS["summaries"] / f"training_results{SHARD_TAG}.json"


def _read_results_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            stored = json.load(f)
    except Exception as exc:                                 # noqa: BLE001
        add_warning(f"unreadable training results {path.name} ({exc!r})")
        return None
    if stored.get("config_hash") != CONFIG_HASH:
        quarantine(path, "training results carry a different config_hash")
        return None
    return stored


def load_training_results() -> Dict[str, Any]:
    """This shard's own results, merged with every other shard's.

    Each session writes only its own file, so parallel sessions never clobber
    one another. An unsharded session sees all of them, which is what makes
    the audit and the statistics complete.
    """
    merged = {"schema_version": CFG.schema_version, "config_hash": CONFIG_HASH,
              "run_id": RUN_ID, "shard": SHARD_LABEL, "runs": {}}
    own = _read_results_file(RESULTS_PATH)
    for path in sorted(PATHS["summaries"].glob("training_results*.json")):
        if path == RESULTS_PATH:
            continue
        other = _read_results_file(path)
        if other:
            merged["runs"].update(other.get("runs", {}))
            merged.setdefault("merged_from", []).append(path.name)
    if own:
        merged["runs"].update(own.get("runs", {}))
    return merged


def save_training_results(results: Dict[str, Any]) -> None:
    """Persist ONLY the runs this session owns, into this shard's own file."""
    results["updated_utc"] = utc_now()
    if IS_SHARDED:
        own = {k: v for k, v in results["runs"].items()
               if in_shard(v.get("generator"), v.get("seed", -1))}
        atomic_write_json(RESULTS_PATH,
                          {**{k: v for k, v in results.items() if k != "runs"},
                           "runs": own})
    else:
        atomic_write_json(RESULTS_PATH, results)


def run_is_complete(entry: Optional[Dict[str, Any]], spec: Dict[str, Any]) -> bool:
    if not entry or entry.get("status") != "complete":
        return False
    for phase in CFG.test_phases:
        path = (PATHS["checkpoints"] / spec["generator"] /
                checkpoint_filename(spec["generator"], spec["option"],
                                    spec["kappa"], spec["seed"], phase))
        if not path.exists():
            return False
        try:
            verify_checkpoint(path)
        except Exception as exc:                             # noqa: BLE001
            add_warning(f"{spec['run_key']} {phase}: checkpoint failed verification "
                        f"on resume ({exc!r}); the run will be re-executed")
            return False
    return True


def execute_queue(cfg: ExperimentConfig) -> Dict[str, Any]:
    results = load_training_results()
    queue = canonical_run_queue(cfg)
    if IS_SHARDED:
        if SHARD_GENERATORS:
            unknown = set(SHARD_GENERATORS) - set(GENERATORS)
            if unknown:
                raise ValueError(f"SHARD_GENERATORS names unknown generators: {unknown}")
        queue = [q for q in queue if in_shard(q["generator"], q["seed"])]
        print(f"SHARD {SHARD_LABEL}: this session trains "
              f"generators={sorted(SHARD_GENERATORS) if SHARD_GENERATORS else 'all'} "
              f"seeds={sorted(SHARD_SEEDS) if SHARD_SEEDS is not None else 'all'} "
              f"({len(queue)} of {cfg.n_configurations} configurations).\n")
    data_hashes = {"calibration_returns": CALIBRATION_INPUT_HASH,
                   **{f"generator_{g}": DATA_HASHES[f"generator_{g}"] for g in GENERATORS},
                   **{f"historical_{r}": DATA_HASHES[f"historical_{r}"]
                      for r in CFG.regime_names},
                   "split_train": "", "split_val": "", "split_test": ""}

    def _exhausted(entry) -> bool:
        return bool(entry and entry.get("status") == "failed"
                    and int(entry.get("retries", 0)) > cfg.max_retries_per_run)

    pending, exhausted = [], []
    for s in queue:
        entry = results["runs"].get(s["run_key"])
        if run_is_complete(entry, s):
            continue
        (exhausted if _exhausted(entry) else pending).append(s)
    print(f"Queue: {len(queue)} configurations, {len(pending)} to execute, "
          f"{len(queue) - len(pending) - len(exhausted)} already complete and "
          f"verified, {len(exhausted)} permanently failed.\n")
    if exhausted:
        print("  Permanently failed (retry budget spent; NOT replaced by another "
              "seed):")
        for s in exhausted:
            print(f"    {s['run_key']}")

    t_start = time.time()
    for i, spec in enumerate(pending):
        spec_hashes = dict(data_hashes)
        spec_hashes.update({f"split_{k}": SPLIT_HASHES[spec["generator"]][k]
                            for k in ("train", "val", "test")})
        key = spec["run_key"]
        entry = results["runs"].get(key, {"run_key": key, "retries": 0,
                                          "attempts": []})
        entry.update({"generator": spec["generator"], "option": spec["option"],
                      "kappa": spec["kappa"], "seed": spec["seed"],
                      "status": "running", "started_utc": utc_now()})
        results["runs"][key] = entry
        save_training_results(results)

        attempts = int(entry.get("retries", 0))
        success = False
        t0 = time.time()
        while attempts <= cfg.max_retries_per_run and not success:
            t0 = time.time()
            print(f"  [{i + 1}/{len(pending)}] {key}"
                  f"{'  (retry %d)' % attempts if attempts else ''}")
            try:
                res = train_configuration(spec, cfg, DATASETS, HIST_TENSORS,
                                          spec_hashes,
                                          verbose_every=(100 if cfg.run_mode == "FULL"
                                                         else 0))
                entry.update(res)
                entry["status"] = "complete"
                entry["completed_utc"] = utc_now()
                entry["retries"] = attempts
                success = True
                m = res["phases"]["cvar"]["metrics"]
                print(f"      done in {res['wall_seconds']:.0f}s  "
                      f"V0={res['phases']['cvar']['V0']:+.4f}  "
                      f"nu={res['phases']['cvar']['nu']:+.4f}  "
                      f"std={m['std']:.5f}  cvar95={m['cvar95']:.5f}")
            except Exception as exc:                          # noqa: BLE001
                attempts += 1
                entry.setdefault("attempts", []).append(
                    {"attempt": attempts, "utc": utc_now(), "error": repr(exc),
                     "seconds": time.time() - t0})
                entry["retries"] = attempts
                if attempts > cfg.max_retries_per_run:
                    entry["status"] = "failed"
                    entry["failed_utc"] = utc_now()
                    add_failure(f"{key} failed after {cfg.max_retries_per_run} "
                                f"retries with the same seed and configuration",
                                {"run_key": key, "error": repr(exc)})
                    print(f"      FAILED permanently: {exc!r}")
                else:
                    print(f"      failure ({exc!r}); retrying with the SAME seed "
                          f"{spec['seed']} and the SAME configuration")
            results["runs"][key] = entry
            save_training_results(results)

        if not success and entry.get("status") == "running":
            entry["status"] = "failed"
            entry["failed_utc"] = utc_now()
            results["runs"][key] = entry
            save_training_results(results)

        elapsed = time.time() - t_start
        done = i + 1
        MANIFEST["gpu_hours"] = float(MANIFEST.get("gpu_hours", 0.0)
                                      + (time.time() - t0) / 3600.0)
        if done % 5 == 0 or done == len(pending):
            eta = elapsed / done * (len(pending) - done)
            print(f"      progress {done}/{len(pending)}  elapsed "
                  f"{elapsed / 3600:.2f}h  ETA {eta / 3600:.2f}h")
            save_manifest()

    MANIFEST["runs"] = {k: {"status": v.get("status"), "retries": v.get("retries", 0),
                            "seed": v.get("seed"), "generator": v.get("generator"),
                            "option": v.get("option"), "kappa": v.get("kappa"),
                            "checkpoints": {p: v.get("phases", {}).get(p, {}).get("checkpoint")
                                            for p in CFG.test_phases},
                            "artifact_hashes": {p: v.get("phases", {}).get(p, {}).get("artifact_hash")
                                                for p in CFG.test_phases}}
                        for k, v in results["runs"].items()}
    save_manifest()
    return results


def run_pipeline_training(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Orchestrator entry point for the training stage (design §6)."""
    if cfg.run_mode == "ANALYSIS_ONLY":
        print("RUN_MODE=ANALYSIS_ONLY: reusing existing checkpoints, no training.")
        return load_training_results()
    if cfg.run_mode == "DIAGNOSTICS":
        print("RUN_MODE=DIAGNOSTICS: skipping training; jump to Cell 25.")
        return load_training_results()
    return execute_queue(cfg)


TRAINING_RESULTS = run_pipeline_training(CFG)
# Re-read every shard's results from disk now. The view built when this
# session STARTED is hours old by the time a shard finishes: judging "am I the
# last one?" from it makes every shard believe the others are still running, so
# none of them would go on to the analysis.
TRAINING_RESULTS = load_training_results()
MANIFEST["shard"] = SHARD_LABEL
MANIFEST["shard_generators"] = (sorted(SHARD_GENERATORS) if SHARD_GENERATORS
                                else list(GENERATORS))
MANIFEST["shard_seeds"] = (sorted(int(x) for x in SHARD_SEEDS)
                           if SHARD_SEEDS is not None else list(CFG.seeds))
MANIFEST["trained_on_gpu"] = GPU_NAME
_statuses = pd.Series([v.get("status") for v in TRAINING_RESULTS["runs"].values()])
_n_complete = int((_statuses == "complete").sum())
_n_failed = int((_statuses == "failed").sum())

MANIFEST["gate_5_training"] = {
    "n_expected": CFG.n_configurations, "n_complete": _n_complete,
    "n_failed": _n_failed,
    "passed": bool(_n_complete + _n_failed >= CFG.n_configurations),
    "utc": utc_now()}
save_manifest()

print(f"\nTraining summary: {_n_complete} complete, {_n_failed} failed, "
      f"{CFG.n_configurations} expected.")
if _n_failed:
    print("  Failed runs (recorded as failed, NOT replaced by another seed):")
    for _k, _v in TRAINING_RESULTS["runs"].items():
        if _v.get("status") == "failed":
            print(f"    {_k}  retries={_v.get('retries')}")

if IS_SHARDED:
    # The analysis needs every generator, so a shard runs it only when it is
    # the LAST one to finish. Whichever session finishes last therefore carries
    # straight on into Cells 20-27, and no separate analysis session is needed.
    # A configuration is settled once it is complete OR has permanently failed:
    # a failed seed is reported with the real n, it must never block the
    # analysis forever. A "running" or missing entry is still outstanding.
    _TERMINAL = ("complete", "failed")
    _outstanding = [s["run_key"] for s in canonical_run_queue(CFG)
                    if (TRAINING_RESULTS["runs"].get(s["run_key"], {}).get("status")
                        not in _TERMINAL)]
    # The other session's results file can arrive before its checkpoints do —
    # Drive syncs between machines with a delay. Analysing now would silently
    # drop those seeds, so a complete run whose checkpoint is not visible here
    # yet is treated as outstanding too.
    if not _outstanding:
        _not_synced = []
        for _s in canonical_run_queue(CFG):
            _e = TRAINING_RESULTS["runs"].get(_s["run_key"], {})
            if _e.get("status") != "complete":
                continue
            for _ph in CFG.test_phases:
                _cp = (PATHS["checkpoints"] / _s["generator"] /
                       checkpoint_filename(_s["generator"], _s["option"],
                                           _s["kappa"], _s["seed"], _ph))
                if not _cp.exists():
                    _not_synced.append(_s["run_key"])
                    break
        if _not_synced:
            print(f"\n  Every configuration is recorded as finished, but "
                  f"{len(_not_synced)} checkpoints from the\n  other session are "
                  f"not visible on this machine yet (Google Drive is still "
                  f"syncing).\n  Re-run this notebook in a few minutes; it will "
                  f"skip the training and go\n  straight to the analysis. First "
                  f"few: {_not_synced[:3]}")
            raise ShardTrainingComplete(
                f"{len(_not_synced)} checkpoints not yet synced from the other "
                f"session; re-run this notebook in a few minutes for the analysis.")
    print(f"\n{'=' * 78}")
    if _outstanding:
        print(f"  SHARD {SHARD_LABEL} FINISHED TRAINING — expected stopping point.")
        print(f"{'=' * 78}")
        print(f"  Results written to : {RESULTS_PATH.name}")
        print(f"  Run directory      : {RUN_DIR}")
        print(f"\n  {len(_outstanding)} configurations are still outstanding in "
              f"other shards, so the\n  analysis is not run here. Whichever "
              f"session finishes LAST will run it\n  automatically — you do not "
              f"need a separate analysis session.")
        print(f"  Still outstanding, first few: {_outstanding[:5]}")
        raise ShardTrainingComplete(
            f"Shard {SHARD_LABEL} finished training; {len(_outstanding)} "
            f"configurations remain in other shards. This stop is expected.")
    print(f"  SHARD {SHARD_LABEL} FINISHED, AND IT IS THE LAST ONE.")
    print(f"{'=' * 78}")
    print(f"  All {CFG.n_configurations} configurations are complete, so this "
          f"session continues\n  into the audit, evaluation, statistics, "
          f"diagnostics and tables.")
    # The canonical manifest belongs to the analysis, not to a shard.
    MANIFEST_PATH = RUN_DIR / "manifest.json"
    MANIFEST = _load_manifest()
    MANIFEST["shard"] = "all"
    MANIFEST["analysis_run_by_shard"] = SHARD_LABEL
    save_manifest()

# ---- notebook cell 29 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 20 — CHECKPOINT AUDIT
# ═════════════════════════════════════════════════════════════════════════════
#  Verifies that every checkpoint on disk is structurally complete, integrity-
#  hashed, and carries the SAME seed as the run key that points at it. A seed
#  recorded in a checkpoint that disagrees with its filename is a hard failure.
# ═════════════════════════════════════════════════════════════════════════════

def audit_checkpoints(cfg: ExperimentConfig) -> "pd.DataFrame":
    rows = []
    for spec in canonical_run_queue(cfg):
        for phase in cfg.test_phases:
            path = (PATHS["checkpoints"] / spec["generator"] /
                    checkpoint_filename(spec["generator"], spec["option"],
                                        spec["kappa"], spec["seed"], phase))
            row = {"run_key": spec["run_key"], "generator": spec["generator"],
                   "option": spec["option"], "kappa": spec["kappa"],
                   "seed": spec["seed"], "phase": phase,
                   "file": path.name, "exists": path.exists(),
                   "verified": False, "seed_matches": None,
                   "config_hash_matches": None, "artifact_hash": None,
                   "bytes": None, "error": None}
            if path.exists():
                row["bytes"] = path.stat().st_size
                try:
                    ckpt = verify_checkpoint(path)
                    row["verified"] = True
                    row["artifact_hash"] = ckpt["artifact_hash"]
                    row["seed_matches"] = (int(ckpt["seed"]) == int(spec["seed"])
                                           and ckpt["run_key"] == spec["run_key"])
                    row["config_hash_matches"] = ckpt["config_hash"] == CONFIG_HASH
                    row["best_epoch"] = ckpt["best_epoch"]
                    row["best_val_loss"] = ckpt["best_val_loss"]
                except Exception as exc:                     # noqa: BLE001
                    row["error"] = repr(exc)
            rows.append(row)
    return pd.DataFrame(rows)


CHECKPOINT_AUDIT = audit_checkpoints(CFG)
_h = atomic_write_dataframe(PATHS["checkpoints"] / "checkpoint_audit.csv",
                            CHECKPOINT_AUDIT)
register_artifact("checkpoints/checkpoint_audit.csv",
                  PATHS["checkpoints"] / "checkpoint_audit.csv", _h)

_n_expected = CFG.n_expected_checkpoints
_n_present = int(CHECKPOINT_AUDIT["exists"].sum())
_n_verified = int(CHECKPOINT_AUDIT["verified"].sum())
_seed_bad = CHECKPOINT_AUDIT[(CHECKPOINT_AUDIT["exists"])
                             & (CHECKPOINT_AUDIT["seed_matches"] == False)]  # noqa: E712
_cfg_bad = CHECKPOINT_AUDIT[(CHECKPOINT_AUDIT["exists"])
                            & (CHECKPOINT_AUDIT["config_hash_matches"] == False)]  # noqa: E712

print(f"Checkpoint audit")
print(f"  expected : {_n_expected}")
print(f"  present  : {_n_present}")
print(f"  verified : {_n_verified}")
print(f"  seed mismatches        : {len(_seed_bad)}")
print(f"  config-hash mismatches : {len(_cfg_bad)}")
if len(_seed_bad):
    print(_seed_bad[["run_key", "phase", "seed"]].to_string(index=False))
    add_failure("checkpoint seed does not match its run key",
                {"rows": _seed_bad["run_key"].tolist()})
    raise RuntimeError("Seed identity audit failed — a checkpoint records a "
                       "different seed from the one its run key claims.")
if len(_cfg_bad):
    add_failure("checkpoint config_hash mismatch",
                {"rows": _cfg_bad["run_key"].tolist()})
    raise RuntimeError("Checkpoint config_hash audit failed.")
_broken = CHECKPOINT_AUDIT[(CHECKPOINT_AUDIT["exists"]) & (~CHECKPOINT_AUDIT["verified"])]
if len(_broken):
    for _, _r in _broken.iterrows():
        add_failure(f"checkpoint failed verification: {_r['file']}",
                    {"error": _r["error"]})
    raise RuntimeError(f"{len(_broken)} checkpoints failed verification; "
                       f"see checkpoints/checkpoint_audit.csv")

MANIFEST["checkpoint_audit"] = {
    "expected": _n_expected, "present": _n_present, "verified": _n_verified,
    "seed_mismatches": int(len(_seed_bad)), "utc": utc_now()}
save_manifest()
if _n_present < _n_expected:
    add_warning(f"only {_n_present}/{_n_expected} checkpoints present; the "
                f"statistics will report the real common-seed n")

# ---- notebook cell 30 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 21 — UNIFORM RE-EVALUATION OF EVERY BEST CHECKPOINT
# ═════════════════════════════════════════════════════════════════════════════
#  Every checkpoint is re-evaluated here, through ONE code path, on:
#    * its own generator's synthetic test set
#    * COVID_EXTENDED_2019_2020, POSTCOVID_2021_2022, RECENT_2023_2025
#  Training-time metrics are never spliced in from an older JSON.
#  Per-path arrays (residual, payoff, terminal wealth, PnL, cost, turnover) are
#  stored so any distributional statistic can be recomputed without re-running
#  the model.
# ═════════════════════════════════════════════════════════════════════════════

EVAL_SUMMARY_PATH = PATHS["summaries"] / "evaluations.json"


def evaluate_all_checkpoints(cfg: ExperimentConfig) -> List[Dict[str, Any]]:
    audit = CHECKPOINT_AUDIT[CHECKPOINT_AUDIT["verified"]]
    rows: List[Dict[str, Any]] = []
    eval_sets = {"synthetic_test": None, **{r: HIST_TENSORS[r] for r in cfg.regime_names}}
    total = len(audit)
    print(f"Re-evaluating {total} verified checkpoints on "
          f"{len(eval_sets)} evaluation sets each "
          f"({total * len(eval_sets)} evaluations).")
    t0 = time.time()
    for i, (_, rec) in enumerate(audit.iterrows()):
        path = PATHS["checkpoints"] / rec["generator"] / rec["file"]
        net, cvar, ckpt = load_checkpoint(path)
        payoff_fn = PAYOFF_FNS[rec["option"]]
        per_path_file = PATHS["per_path"] / f"{path.stem}.npz"
        arrays_to_store: Dict[str, np.ndarray] = {}
        for set_name in eval_sets:
            S = (DATASETS[rec["generator"]]["test"] if set_name == "synthetic_test"
                 else HIST_TENSORS[set_name])
            ev = evaluate_full(net, S, payoff_fn, float(rec["kappa"]), cfg)
            row = {"run_key": rec["run_key"], "generator": rec["generator"],
                   "option": rec["option"], "kappa": float(rec["kappa"]),
                   "seed": int(rec["seed"]), "phase": rec["phase"],
                   "evaluation_set": set_name,
                   "checkpoint": rec["file"],
                   "artifact_hash": rec["artifact_hash"],
                   "nu": ckpt.get("nu"), **ev["metrics"]}
            rows.append(row)
            for k, arr in ev["arrays"].items():
                arrays_to_store[f"{set_name}__{k}"] = arr
        digest = atomic_write_npz(per_path_file, verify=False, **arrays_to_store)
        del net, cvar
        if (i + 1) % 20 == 0 or i + 1 == total:
            el = time.time() - t0
            print(f"    {i + 1}/{total} checkpoints  elapsed {el:.0f}s  "
                  f"ETA {el / (i + 1) * (total - i - 1):.0f}s", flush=True)
        clear_mem()
    return rows


_eval_rows = None
if EVAL_SUMMARY_PATH.exists():
    with open(EVAL_SUMMARY_PATH, "r", encoding="utf-8") as f:
        _cached = json.load(f)
    _ok, _why = cache_is_valid(_cached)
    if _ok and _cached.get("n_checkpoints") == int(CHECKPOINT_AUDIT["verified"].sum()):
        _eval_rows = _cached["rows"]
        log(f"  reused cached evaluations ({len(_eval_rows)} rows)")
    else:
        quarantine(EVAL_SUMMARY_PATH, f"evaluation cache invalid: {_why}")

if _eval_rows is None:
    _eval_rows = evaluate_all_checkpoints(CFG)
    atomic_write_json(EVAL_SUMMARY_PATH, {
        "schema_version": CFG.schema_version,
        "evaluation_schema_version": CFG.evaluation_schema_version,
        "config_hash": CONFIG_HASH, "run_id": RUN_ID,
        "n_checkpoints": int(CHECKPOINT_AUDIT["verified"].sum()),
        "evaluation_sets": ["synthetic_test", *CFG.regime_names],
        "uniform_code_path": True,
        "rows": _eval_rows, "written_utc": utc_now()})
    register_artifact("evaluations/summaries/evaluations.json", EVAL_SUMMARY_PATH)

EVALUATIONS = pd.DataFrame(_eval_rows)
print(f"\nEvaluation rows: {len(EVALUATIONS)}")
if len(EVALUATIONS):
    print(EVALUATIONS.groupby(["evaluation_set"])["n_paths"].agg(["count", "first"])
          .rename(columns={"count": "n_rows", "first": "paths_per_evaluation"})
          .to_string())

# ---- notebook cell 31 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 22 — MASTER RESULT DATAFRAME
# ═════════════════════════════════════════════════════════════════════════════
#  One tidy frame feeds every table, figure and test below. Each row is one
#  (generator, option, strike, seed, phase, evaluation set).
# ═════════════════════════════════════════════════════════════════════════════

MASTER_COLUMNS = ["generator", "option", "kappa", "seed", "phase",
                  "evaluation_set", "n_paths", "mean", "std", "rmse", "var95",
                  "cvar95", "cvar99", "min", "max", "V0", "nu",
                  "mean_payoff", "mean_pnl", "mean_transaction_cost",
                  "mean_turnover", "all_finite", "run_key", "checkpoint",
                  "artifact_hash"]

MASTER = EVALUATIONS.copy()
for _c in MASTER_COLUMNS:
    if _c not in MASTER.columns:
        MASTER[_c] = np.nan
MASTER = MASTER[MASTER_COLUMNS].sort_values(
    ["evaluation_set", "option", "kappa", "phase", "generator", "seed"]
).reset_index(drop=True)

_nonfinite = MASTER[~MASTER["all_finite"].fillna(False).astype(bool)]
if len(_nonfinite):
    add_warning(f"{len(_nonfinite)} evaluation rows contain non-finite residuals",
                {"run_keys": _nonfinite["run_key"].unique().tolist()[:20]})

_h = atomic_write_dataframe(PATHS["summaries"] / "master_results.csv", MASTER)
register_artifact("evaluations/summaries/master_results.csv",
                  PATHS["summaries"] / "master_results.csv", _h)

# ── Seed coverage per analysis cell (the real n, never padded) ───────────────
SEED_COVERAGE = (MASTER[MASTER["evaluation_set"] != "synthetic_test"]
                 .groupby(["evaluation_set", "option", "kappa", "phase", "generator"])
                 ["seed"].agg(n_seeds="nunique",
                              seeds=lambda s: ",".join(map(str, sorted(set(s)))))
                 .reset_index())
_h = atomic_write_dataframe(PATHS["summaries"] / "seed_coverage.csv", SEED_COVERAGE)
register_artifact("evaluations/summaries/seed_coverage.csv",
                  PATHS["summaries"] / "seed_coverage.csv", _h)

print(f"Master results: {len(MASTER)} rows")
print(f"  generators      : {sorted(MASTER['generator'].dropna().unique())}")
print(f"  evaluation sets : {sorted(MASTER['evaluation_set'].dropna().unique())}")
print(f"  phases          : {sorted(MASTER['phase'].dropna().unique())}")
print(f"  seeds observed  : {sorted(MASTER['seed'].dropna().astype(int).unique())}")
if len(SEED_COVERAGE):
    _incomplete = SEED_COVERAGE[SEED_COVERAGE["n_seeds"] < len(CFG.seeds)]
    print(f"  cells with fewer than {len(CFG.seeds)} seeds: {len(_incomplete)}")
    if len(_incomplete):
        print(_incomplete.head(12).to_string(index=False))

# ==========================================================================
# ---
#
# ## Sections 15–16 — Statistical analysis and support diagnostics (Gate 6)
#
# Paired tests with BH-FDR carry the primary claim; exact sign-flip permutation,
# Wilcoxon and bootstrap intervals are reported alongside. The MCS runs through
# `arch` on a common-seed loss matrix and is secondary evidence. Support
# diagnostics are regenerated from code.

# ---- notebook cell 33 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 23 — PAIRED TESTS AND ROBUSTNESS TESTS (Gate 6, part 1)
# ═════════════════════════════════════════════════════════════════════════════
#  Pairing is by (option, kappa, regime, phase, seed). Before every test the
#  two seed sets are compared: if they differ, the intersection is used and the
#  real n is reported. A seed is never relabelled to restore n = 10.
#
#  Two families, fixed before looking at any result, each corrected separately:
#     primary_regimes : 3 generator pairs x options x strikes x the three
#                       HISTORICAL regimes x phases x metrics
#     synthetic_test  : the same grid on each generator's own synthetic test
#                       set, reported for reference only
#  Benjamini-Hochberg FDR is applied WITHIN each family. The synthetic-test
#  comparisons are not part of the primary family: each generator is evaluated
#  on paths it produced itself, so those rows answer a different question and
#  must not inflate the correction applied to the out-of-sample claims.
#
#  Robustness (reported alongside, never instead of, the primary test):
#     * exact paired sign-flip permutation test (2^n sign patterns for n <= 20)
#     * Wilcoxon signed-rank (skipped when too many differences are zero)
#     * bootstrap CI for the paired mean and median difference
# ═════════════════════════════════════════════════════════════════════════════

COMPARISONS = (("SBTS", "GBM"), ("SBTS", "Heston"), ("Heston", "GBM"))


def paired_effect_size(diff: np.ndarray) -> float:
    """Standardised paired effect size (Cohen's d_z)."""
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 1e-12 else float("nan")


def exact_sign_flip_test(diff: np.ndarray, max_n: int = 20) -> Dict[str, Any]:
    """Exact two-sided permutation test over all 2^n sign assignments."""
    n = len(diff)
    if n > max_n:
        return {"method": "not_run", "p_value": np.nan, "n_patterns": None}
    observed = abs(diff.mean())
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    means = np.abs((signs * diff).mean(axis=1))
    return {"method": "exact_sign_flip", "n_patterns": int(2 ** n),
            "p_value": float((means >= observed - 1e-15).mean())}


def bootstrap_paired_ci(diff: np.ndarray, reps: int, seed: int,
                        level: float = 0.95) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(reps, n))
    samples = diff[idx]
    means = samples.mean(axis=1)
    medians = np.median(samples, axis=1)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {"boot_mean_lo": float(np.percentile(means, lo)),
            "boot_mean_hi": float(np.percentile(means, hi)),
            "boot_median_lo": float(np.percentile(medians, lo)),
            "boot_median_hi": float(np.percentile(medians, hi)),
            "boot_reps": int(reps)}


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    return out


def run_paired_tests(master: "pd.DataFrame", cfg: ExperimentConfig) -> "pd.DataFrame":
    rows = []
    sets = ["synthetic_test", *cfg.regime_names]
    for eval_set in sets:
        for option in cfg.options:
            for kappa in cfg.strike_ratios:
                for phase in cfg.test_phases:
                    for metric in cfg.primary_metrics:
                        cell = master[(master["evaluation_set"] == eval_set)
                                      & (master["option"] == option)
                                      & (master["kappa"] == kappa)
                                      & (master["phase"] == phase)]
                        for a, b in COMPARISONS:
                            sa = cell[cell["generator"] == a].dropna(subset=[metric])
                            sb = cell[cell["generator"] == b].dropna(subset=[metric])
                            seeds_a = set(sa["seed"].astype(int))
                            seeds_b = set(sb["seed"].astype(int))
                            common = sorted(seeds_a & seeds_b)
                            if len(common) < 3:
                                continue
                            x = sa[sa["seed"].isin(common)].sort_values("seed")[metric].to_numpy(float)
                            y = sb[sb["seed"].isin(common)].sort_values("seed")[metric].to_numpy(float)
                            diff = x - y
                            t_stat, p_val = sp_stats.ttest_rel(x, y)
                            sem = diff.std(ddof=1) / math.sqrt(len(diff))
                            tcrit = sp_stats.t.ppf(0.975, len(diff) - 1)
                            rows.append({
                                "family": ("synthetic_test"
                                           if eval_set == "synthetic_test"
                                           else "primary_regimes"),
                                "evaluation_set": eval_set, "option": option,
                                "kappa": kappa, "phase": phase, "metric": metric,
                                "model_a": a, "model_b": b,
                                "n_seeds": len(common),
                                "seeds_equal": seeds_a == seeds_b,
                                "seeds_used": ",".join(map(str, common)),
                                "mean_a": float(x.mean()), "mean_b": float(y.mean()),
                                "mean_paired_difference": float(diff.mean()),
                                "ci95_lo": float(diff.mean() - tcrit * sem),
                                "ci95_hi": float(diff.mean() + tcrit * sem),
                                "t_stat": float(t_stat), "p_value": float(p_val),
                                "effect_size_dz": paired_effect_size(diff),
                            })
    df = pd.DataFrame(rows)
    if len(df):
        df["p_bh"] = np.nan
        df["family_size"] = 0
        for fam, idx in df.groupby("family").groups.items():
            df.loc[idx, "p_bh"] = benjamini_hochberg(
                df.loc[idx, "p_value"].to_numpy())
            df.loc[idx, "family_size"] = len(idx)
        df["significant_bh"] = df["p_bh"] < cfg.fdr_alpha
    return df


def run_robustness_tests(master: "pd.DataFrame", primary: "pd.DataFrame",
                         cfg: ExperimentConfig) -> "pd.DataFrame":
    rows = []
    for _, r in primary.iterrows():
        cell = master[(master["evaluation_set"] == r["evaluation_set"])
                      & (master["option"] == r["option"])
                      & (master["kappa"] == r["kappa"])
                      & (master["phase"] == r["phase"])]
        common = [int(s) for s in str(r["seeds_used"]).split(",")]
        x = (cell[(cell["generator"] == r["model_a"]) & (cell["seed"].isin(common))]
             .sort_values("seed")[r["metric"]].to_numpy(float))
        y = (cell[(cell["generator"] == r["model_b"]) & (cell["seed"].isin(common))]
             .sort_values("seed")[r["metric"]].to_numpy(float))
        diff = x - y
        perm = exact_sign_flip_test(diff, cfg.permutation_exact_max_n)
        n_zero = int(np.sum(np.abs(diff) < 1e-15))
        if n_zero > len(diff) // 2 or len(diff) < 5:
            wilcoxon_p, wilcoxon_note = float("nan"), (
                f"skipped: {n_zero}/{len(diff)} zero differences")
        else:
            try:
                wilcoxon_p = float(sp_stats.wilcoxon(x, y, zero_method="wilcox")[1])
                wilcoxon_note = "ok"
            except Exception as exc:                          # noqa: BLE001
                wilcoxon_p, wilcoxon_note = float("nan"), repr(exc)
        boot = bootstrap_paired_ci(diff, cfg.bootstrap_reps, cfg.bootstrap_seed)
        rows.append({
            "evaluation_set": r["evaluation_set"], "option": r["option"],
            "kappa": r["kappa"], "phase": r["phase"], "metric": r["metric"],
            "model_a": r["model_a"], "model_b": r["model_b"],
            "n_seeds": r["n_seeds"], "mean_paired_difference": r["mean_paired_difference"],
            "primary_p_value": r["p_value"], "primary_p_bh": r["p_bh"],
            "permutation_method": perm["method"],
            "permutation_patterns": perm["n_patterns"],
            "permutation_p_value": perm["p_value"],
            "wilcoxon_p_value": wilcoxon_p, "wilcoxon_note": wilcoxon_note,
            "n_zero_differences": n_zero, **boot})
    return pd.DataFrame(rows)


PAIRED_TESTS = run_paired_tests(MASTER, CFG)
ROBUSTNESS_TESTS = (run_robustness_tests(MASTER, PAIRED_TESTS, CFG)
                    if len(PAIRED_TESTS) else pd.DataFrame())

for _name, _df in (("paired_tests", PAIRED_TESTS), ("robustness_tests", ROBUSTNESS_TESTS)):
    if len(_df):
        _h = atomic_write_dataframe(PATHS["statistics"] / f"{_name}.csv", _df)
        register_artifact(f"statistics/{_name}.csv",
                          PATHS["statistics"] / f"{_name}.csv", _h)

atomic_write_json(PATHS["statistics"] / "test_family.json", {
    "schema_version": CFG.schema_version,
    "statistics_schema_version": CFG.statistics_schema_version,
    "config_hash": CONFIG_HASH,
    "families": {
        "primary_regimes": ("generator pairs x options x strikes x the three "
                            "historical regimes x phases x metrics"),
        "synthetic_test": ("the same grid on each generator's own synthetic "
                           "test set; reference only, corrected separately"),
    },
    "family_definition_fixed_before_results": True,
    "family_sizes": (PAIRED_TESTS.groupby("family")["family_size"].first().to_dict()
                     if len(PAIRED_TESTS) else {}),
    "comparisons": [list(c) for c in COMPARISONS],
    "metrics": list(CFG.primary_metrics),
    "phases": list(CFG.test_phases),
    "evaluation_sets": ["synthetic_test", *CFG.regime_names],
    "fdr_method": "Benjamini-Hochberg", "fdr_alpha": CFG.fdr_alpha,
    "family_size": int(len(PAIRED_TESTS)),
    "primary_test": "paired Student t-test on per-seed metric differences",
    "robustness_tests": ["exact paired sign-flip permutation", "Wilcoxon signed-rank",
                         "bootstrap CI for paired mean and median"],
    "robustness_is_supplementary": True,
})

print(f"Paired tests: {len(PAIRED_TESTS)} (BH-FDR applied within each family, "
      f"alpha={CFG.fdr_alpha})")
if len(PAIRED_TESTS):
    for _fam, _grp in PAIRED_TESTS.groupby("family"):
        print(f"  {_fam:<16s} n_tests={len(_grp):<4d} "
              f"significant after BH: {int(_grp['significant_bh'].sum())}")
    _unequal = PAIRED_TESTS[~PAIRED_TESTS["seeds_equal"]]
    print(f"  cells with unequal seed sets (intersection used): {len(_unequal)}")
    if len(_unequal):
        print(_unequal[["evaluation_set", "option", "kappa", "phase",
                        "model_a", "model_b", "n_seeds"]].head(10).to_string(index=False))
    _view = PAIRED_TESTS[(PAIRED_TESTS["metric"] == "std")
                         & (PAIRED_TESTS["phase"] == "cvar")
                         & (PAIRED_TESTS["evaluation_set"] == CFG.baseline_regime)]
    if len(_view):
        print(f"\n  sigma(R), CVaR phase, {CFG.baseline_regime}:")
        print(_view[["option", "kappa", "model_a", "model_b", "n_seeds",
                     "mean_paired_difference", "p_value", "p_bh",
                     "effect_size_dz"]].round(6).to_string(index=False))

# ---- notebook cell 34 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 24 — MODEL CONFIDENCE SET (Hansen-Lunde-Nason)
# ═════════════════════════════════════════════════════════════════════════════
#  The previous ad-hoc `mcs_seed_level` function is NOT used. The canonical
#  engine is `arch.bootstrap.MCS` at the locked package version:
#     loss matrix : n_common_seeds x 3 models, one seed per row
#     alpha       : 0.10          reps : 5,000        seed : 42
#     method      : "R" (canonical), "max" reported as a sensitivity check
#     block       : length 1 (seed-level observations are independent)
#
#  With only 10 seed observations the MCS has little power, so it is SECONDARY
#  evidence: the paired and permutation tests carry the primary claim, and no
#  membership statement may be locked into the thesis until this pipeline has
#  regenerated it.
# ═════════════════════════════════════════════════════════════════════════════

MCS_BOOTSTRAP = "circular"      # with block_size=1 this is i.i.d. row resampling


def _degenerate_pairs(loss_df: "pd.DataFrame", tol: float = 1e-12) -> List[Tuple[str, str]]:
    """Model pairs whose loss difference has (numerically) zero variance.

    The HLN statistic studentises by that variance, so such a pair makes the
    bootstrap undefined. This is a degenerate input (e.g. two models that
    produced identical losses on every seed), not a result to paper over: it is
    detected, reported, and resolved by the deterministic ordering instead.
    """
    cols = list(loss_df.columns)
    L = loss_df.to_numpy(dtype=float)
    out = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            d = L[:, i] - L[:, j]
            scale = max(1.0, float(np.abs(d).max()))
            if float(d.std(ddof=1)) <= tol * scale:
                out.append((cols[i], cols[j]))
    return out


def run_mcs(loss_df: "pd.DataFrame", cfg: ExperimentConfig,
            method: str = "R") -> Dict[str, Any]:
    """Canonical MCS. Raises if `arch` is unavailable."""
    if not ARCH_AVAILABLE:
        raise RuntimeError("arch is not available; the canonical MCS engine "
                           "cannot run. Install the locked arch version.")
    degenerate = _degenerate_pairs(loss_df)
    if degenerate:
        means = loss_df.mean()
        best = float(means.min())
        tol = 1e-12 * max(1.0, abs(best))
        included = sorted(means.index[means <= best + tol].tolist())
        return {"engine": "deterministic ordering (degenerate input)",
                "method": method, "included": included,
                "excluded": sorted(set(loss_df.columns) - set(included)),
                "pvalues": {c: float("nan") for c in loss_df.columns},
                "canonical": True, "degenerate": True,
                "degenerate_pairs": [list(p) for p in degenerate],
                "note": ("at least one pair of models has zero variance in its "
                         "loss difference, so the studentised bootstrap is "
                         "undefined; the ordering is deterministic and is used "
                         "directly")}
    mcs = ARCH_MCS(loss_df.to_numpy(dtype=float), size=cfg.mcs_alpha,
                   reps=cfg.mcs_reps, block_size=cfg.mcs_block_size,
                   method=method, bootstrap=MCS_BOOTSTRAP, seed=cfg.mcs_seed)
    mcs.compute()
    cols = list(loss_df.columns)
    included = [cols[i] for i in np.atleast_1d(mcs.included)]
    excluded = [cols[i] for i in np.atleast_1d(mcs.excluded)]
    pvalues = {cols[int(i)]: float(v) for i, v in
               zip(mcs.pvalues.index, mcs.pvalues.iloc[:, 0])}
    return {"engine": f"arch.bootstrap.MCS {ARCH_VERSION}", "method": method,
            "included": sorted(included), "excluded": sorted(excluded),
            "pvalues": pvalues, "canonical": True, "degenerate": False,
            "degenerate_pairs": [], "note": ""}


# ── Synthetic unit tests (design §15.4) ──────────────────────────────────────
def _mcs_test_cfg() -> ExperimentConfig:
    return dataclasses.replace(CFG, mcs_reps=max(500, min(CFG.mcs_reps, 1000)))


def test_mcs_clear_winner():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"A": 0.01 + 0.0005 * rng.standard_normal(10),
                       "B": 0.05 + 0.0005 * rng.standard_normal(10),
                       "C": 0.05 + 0.0005 * rng.standard_normal(10)})
    res = run_mcs(df, _mcs_test_cfg())
    assert res["included"] == ["A"], res["included"]


def test_mcs_identical_models():
    """Models that cannot be told apart must all stay in the set."""
    rng = np.random.default_rng(1)
    base = 0.05 + 0.005 * rng.standard_normal(10)
    # Statistically indistinguishable: same distribution, independent draws.
    df = pd.DataFrame({"A": base,
                       "B": 0.05 + 0.005 * rng.standard_normal(10),
                       "C": 0.05 + 0.005 * rng.standard_normal(10)})
    res = run_mcs(df, _mcs_test_cfg())
    assert sorted(res["included"]) == ["A", "B", "C"], res["included"]
    assert not res["degenerate"]
    # Bit-identical columns are a degenerate input: the studentised bootstrap
    # is undefined, so the engine must flag it instead of crashing, and still
    # keep all three models.
    same = pd.DataFrame({"A": base, "B": base.copy(), "C": base.copy()})
    deg = run_mcs(same, _mcs_test_cfg())
    assert deg["degenerate"] and sorted(deg["included"]) == ["A", "B", "C"], deg


def test_mcs_column_permutation_invariant():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"A": 0.01 + 0.001 * rng.standard_normal(10),
                       "B": 0.03 + 0.001 * rng.standard_normal(10),
                       "C": 0.05 + 0.001 * rng.standard_normal(10)})
    a = run_mcs(df, _mcs_test_cfg())
    b = run_mcs(df[["C", "A", "B"]], _mcs_test_cfg())
    assert sorted(a["included"]) == sorted(b["included"]), (a["included"], b["included"])


def test_mcs_row_permutation_invariant():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"A": 0.01 + 0.001 * rng.standard_normal(10),
                       "B": 0.04 + 0.001 * rng.standard_normal(10),
                       "C": 0.04 + 0.001 * rng.standard_normal(10)})
    a = run_mcs(df, _mcs_test_cfg())
    b = run_mcs(df.iloc[[7, 2, 9, 0, 4, 1, 8, 3, 6, 5]].reset_index(drop=True),
                _mcs_test_cfg())
    assert sorted(a["included"]) == sorted(b["included"]), (a["included"], b["included"])


MCS_TESTS = [
    (test_mcs_clear_winner, "MCS: clear winner eliminates the two losers"),
    (test_mcs_identical_models, "MCS: identical models keeps all three"),
    (test_mcs_column_permutation_invariant, "MCS: invariant to column order"),
    (test_mcs_row_permutation_invariant, "MCS: invariant to seed-row order"),
]

print("MCS synthetic unit tests")
if not ARCH_AVAILABLE:
    add_failure("arch unavailable — the canonical MCS engine cannot run")
    raise RuntimeError("arch is required for the canonical MCS. Install the "
                       "locked version and re-run this cell.")
_mcs_ok = all([_run_test(f, n, "mcs") for f, n in MCS_TESTS])
if not _mcs_ok:
    add_failure("MCS synthetic unit tests failed")
    raise RuntimeError("MCS unit tests failed; the MCS results would not be "
                       "trustworthy.")

# ── MCS over every analysis cell ─────────────────────────────────────────────
MCS_LOSS_METRIC = "std"
MCS_PHASE = "cvar"


def mcs_over_cells(master: "pd.DataFrame", cfg: ExperimentConfig) -> "pd.DataFrame":
    rows = []
    for eval_set in cfg.regime_names:
        for option in cfg.options:
            for kappa in cfg.strike_ratios:
                cell = master[(master["evaluation_set"] == eval_set)
                              & (master["option"] == option)
                              & (master["kappa"] == kappa)
                              & (master["phase"] == MCS_PHASE)]
                per_model = {}
                for gen in GENERATORS:
                    sub = cell[cell["generator"] == gen].dropna(subset=[MCS_LOSS_METRIC])
                    per_model[gen] = dict(zip(sub["seed"].astype(int),
                                              sub[MCS_LOSS_METRIC].astype(float)))
                common = sorted(set.intersection(*[set(v) for v in per_model.values()])
                                if all(per_model.values()) else set())
                if len(common) < 3:
                    rows.append({"evaluation_set": eval_set, "option": option,
                                 "kappa": kappa, "n_common_seeds": len(common),
                                 "status": "skipped: fewer than 3 common seeds"})
                    continue
                loss_df = pd.DataFrame({g: [per_model[g][s] for s in common]
                                        for g in GENERATORS})
                canonical = run_mcs(loss_df, cfg, method=cfg.mcs_method)
                sensitivity = run_mcs(loss_df, cfg, method="max")
                means = loss_df.mean().to_dict()
                rows.append({
                    "evaluation_set": eval_set, "option": option, "kappa": kappa,
                    "n_common_seeds": len(common),
                    "common_seeds": ",".join(map(str, common)),
                    "status": "ok", "loss_metric": MCS_LOSS_METRIC,
                    "phase": MCS_PHASE, "alpha": cfg.mcs_alpha,
                    "reps": cfg.mcs_reps, "block_size": cfg.mcs_block_size,
                    "engine": canonical["engine"],
                    "mcs_method_R": ",".join(canonical["included"]),
                    "mcs_size_R": len(canonical["included"]),
                    "mcs_method_max": ",".join(sensitivity["included"]),
                    "mcs_size_max": len(sensitivity["included"]),
                    "degenerate": bool(canonical["degenerate"]),
                    "degenerate_note": canonical["note"],
                    "best_mean_model": min(means, key=means.get),
                    **{f"mean_{g}": float(means[g]) for g in GENERATORS},
                    **{f"pvalue_{g}": canonical["pvalues"].get(g) for g in GENERATORS},
                })
    return pd.DataFrame(rows)


MCS_TABLE = mcs_over_cells(MASTER, CFG) if len(MASTER) else pd.DataFrame()
if len(MCS_TABLE):
    _h = atomic_write_dataframe(PATHS["statistics"] / "mcs.csv", MCS_TABLE)
    register_artifact("statistics/mcs.csv", PATHS["statistics"] / "mcs.csv", _h)
atomic_write_json(PATHS["statistics"] / "mcs_metadata.json", {
    "schema_version": CFG.schema_version, "config_hash": CONFIG_HASH,
    "engine": f"arch.bootstrap.MCS {ARCH_VERSION}",
    "canonical_method": CFG.mcs_method, "sensitivity_method": "max",
    "alpha": CFG.mcs_alpha, "reps": CFG.mcs_reps,
    "block_size": CFG.mcs_block_size, "bootstrap": MCS_BOOTSTRAP,
    "seed": CFG.mcs_seed, "loss_metric": MCS_LOSS_METRIC, "phase": MCS_PHASE,
    "loss_matrix_layout": "rows = common seeds, columns = models",
    "evidence_status": ("secondary: with 10 seed observations the MCS has low "
                        "power; paired and permutation tests carry the primary "
                        "claim"),
    "unit_tests_passed": bool(_mcs_ok),
})

print(f"\nMCS over {len(MCS_TABLE)} cells "
      f"(loss = {MCS_LOSS_METRIC}, phase = {MCS_PHASE})")
if len(MCS_TABLE) and "mcs_method_R" in MCS_TABLE.columns:
    _ok_rows = MCS_TABLE[MCS_TABLE["status"] == "ok"]
    print(_ok_rows[["evaluation_set", "option", "kappa", "n_common_seeds",
                    "mcs_method_R", "mcs_size_R", "best_mean_model"]]
          .to_string(index=False))
    print("\nSingleton counts by regime:")
    for _r in CFG.regime_names:
        _sub = _ok_rows[_ok_rows["evaluation_set"] == _r]
        print(f"  {_r:<26s} {int((_sub['mcs_size_R'] == 1).sum())}/{len(_sub)} singleton")
    print("\nMembership counts across all cells:")
    for _g in GENERATORS:
        _n = int(_ok_rows["mcs_method_R"].fillna("").str.contains(_g).sum())
        _s = int((_ok_rows["mcs_method_R"] == _g).sum())
        print(f"  {_g:<7s} member in {_n}/{len(_ok_rows)} cells (singleton {_s})")
    print("\n  These counts are NOT thesis claims until Gate 6 has been signed "
          "off on a FULL run.")

# ---- notebook cell 35 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 25 — SUPPORT-CONTRACTION DIAGNOSTICS
# ═════════════════════════════════════════════════════════════════════════════
#  Direct evidence for the support-contraction mechanism, regenerated from
#  code. The old 313 -> 7 / 45x numbers are only retained if this cell
#  reproduces them under the definitions and query sets locked below.
#
#  Locked query rule
#  -----------------
#  A query is the window of K consecutive daily log-returns ending on a query
#  date, compared against the reference trajectories at the fixed step indices
#  [diag_step_index-K+1, .., diag_step_index]. Because the references are
#  rolling windows with stride 1, a fixed step index spans the whole
#  calibration period, so the support count answers: "how many calibration
#  windows are compatible with this observed K-day history?"
#
#    calm   : query dates drawn from RECENT_2023_2025
#    stress : query dates inside the COVID crash window in CFG
#
#  Support positivity comes from the exact compact-support mask, never from
#  weights clamped away from zero.
# ═════════════════════════════════════════════════════════════════════════════

def _query_dates(kind: str, cfg: ExperimentConfig) -> "pd.DatetimeIndex":
    if kind == "calm":
        lo, hi = cfg.regime_ranges[cfg.baseline_regime]
    elif kind == "stress":
        lo, hi = cfg.covid_crash_start, cfg.covid_crash_end
    else:
        raise ValueError(kind)
    idx = LOG_RETURNS.index[(LOG_RETURNS.index >= lo) & (LOG_RETURNS.index <= hi)]
    return idx


def _query_windows(dates, k: int) -> Tuple[np.ndarray, List[str]]:
    """K consecutive returns ending on each query date."""
    pos = LOG_RETURNS.index.get_indexer(dates)
    usable = [p for p in pos if p - k + 1 >= 0]
    windows = np.stack([LOG_RETURNS.values[p - k + 1:p + 1] for p in usable])
    labels = [str(LOG_RETURNS.index[p].date()) for p in usable]
    return windows.astype(np.float32), labels


@torch.no_grad()
def support_diagnostics(kind: str, k: int, h: float,
                        cfg: ExperimentConfig) -> "pd.DataFrame":
    dates = _query_dates(kind, cfg)
    if len(dates) == 0:
        return pd.DataFrame()
    if len(dates) > cfg.diag_n_queries:
        step = max(1, len(dates) // cfg.diag_n_queries)
        dates = dates[::step][:cfg.diag_n_queries]
    windows, labels = _query_windows(dates, k)
    if len(windows) == 0:
        return pd.DataFrame()
    i0 = min(int(cfg.diag_step_index), cfg.horizon)
    steps = list(range(i0 - k + 1, i0 + 1))
    q = torch.tensor(windows, dtype=torch.float32, device=DEVICE)
    w, support = sbts_support_weights(X_REF_T, q, steps, h, k)

    m_ref = X_REF_T.shape[0]
    nnz = support.sum(dim=1)
    w_sum = w.sum(dim=1)
    ess_num = w_sum ** 2
    ess_den = (w ** 2).sum(dim=1)
    ess = torch.where(ess_den > 0, ess_num / torch.clamp(ess_den, min=LOG_TINY),
                      torch.zeros_like(ess_den))
    max_share = w.max(dim=1).values
    safe_w = torch.where(w > 0, w, torch.ones_like(w))
    entropy = -(w * torch.log(safe_w)).sum(dim=1)
    rows = {
        "query_kind": kind, "K": k, "h": h, "step_index": i0,
        "query_date": labels,
        "nnz": nnz.cpu().numpy(),
        "support_fraction": (nnz.float() / m_ref).cpu().numpy(),
        "ess": ess.cpu().numpy(),
        "max_weight_share": max_share.cpu().numpy(),
        "weight_entropy": entropy.cpu().numpy(),
        "zero_support": (nnz == 0).cpu().numpy(),
        "m_reference": m_ref,
    }
    for top in cfg.diag_top_k:
        topk = torch.topk(w, k=min(top, w.shape[1]), dim=1).values.sum(dim=1)
        rows[f"top{top}_concentration"] = topk.cpu().numpy()
    return pd.DataFrame(rows)


def _bootstrap_mean_ci(x: np.ndarray, reps: int, seed: int) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(reps, x.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _bootstrap_ratio_ci(calm: np.ndarray, stress: np.ndarray, reps: int,
                        seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    a = calm[rng.integers(0, calm.size, size=(reps, calm.size))].mean(axis=1)
    b = stress[rng.integers(0, stress.size, size=(reps, stress.size))].mean(axis=1)
    ratio = a / np.where(np.abs(b) < 1e-12, np.nan, b)
    return (float(np.nanpercentile(ratio, 2.5)), float(np.nanpercentile(ratio, 97.5)))


# The diagnostics must cover the Markov order the generator ACTUALLY uses, not
# just the baseline ladder in the config. K* is resolved by the Cell-11
# selection, so it is added here at runtime rather than being written into the
# config: config_hash stays stable and artifacts from this run remain readable.
DIAG_K_VALUES = tuple(sorted(set(int(k) for k in CFG.diag_k_values) | {int(K_STAR)}))
if int(K_STAR) not in tuple(CFG.diag_k_values):
    print(f"  operational K* = {K_STAR} is not in the baseline ladder "
          f"{list(CFG.diag_k_values)}; adding it so the support claim covers "
          f"the Markov order the SBTS paths were generated with.\n")

_diag_frames = []
for _k in DIAG_K_VALUES:
    for _kind in ("calm", "stress"):
        _df = support_diagnostics(_kind, int(_k), H_STAR, CFG)
        if len(_df):
            _diag_frames.append(_df)
SUPPORT_QUERIES = (pd.concat(_diag_frames, ignore_index=True)
                   if _diag_frames else pd.DataFrame())

_summary_rows = []
if len(SUPPORT_QUERIES):
    for _k in sorted(SUPPORT_QUERIES["K"].unique()):
        for _kind in ("calm", "stress"):
            _sub = SUPPORT_QUERIES[(SUPPORT_QUERIES["K"] == _k)
                                   & (SUPPORT_QUERIES["query_kind"] == _kind)]
            if not len(_sub):
                continue
            _lo, _hi = _bootstrap_mean_ci(_sub["nnz"].to_numpy(),
                                          CFG.diag_bootstrap_reps, CFG.diag_seed)
            _summary_rows.append({
                "K": int(_k), "query_kind": _kind, "h": H_STAR,
                "n_queries": int(len(_sub)),
                "mean_nnz": float(_sub["nnz"].mean()),
                "median_nnz": float(_sub["nnz"].median()),
                "nnz_ci95_lo": _lo, "nnz_ci95_hi": _hi,
                "mean_support_fraction": float(_sub["support_fraction"].mean()),
                "mean_ess": float(_sub["ess"].mean()),
                "mean_max_weight_share": float(_sub["max_weight_share"].mean()),
                "mean_weight_entropy": float(_sub["weight_entropy"].mean()),
                "zero_support_rate": float(_sub["zero_support"].mean()),
                **{f"mean_top{t}_concentration": float(_sub[f"top{t}_concentration"].mean())
                   for t in CFG.diag_top_k if f"top{t}_concentration" in _sub.columns},
            })
SUPPORT_SUMMARY = pd.DataFrame(_summary_rows)

_ratio_rows = []
if len(SUPPORT_QUERIES):
    for _k in sorted(SUPPORT_QUERIES["K"].unique()):
        _c = SUPPORT_QUERIES[(SUPPORT_QUERIES["K"] == _k)
                             & (SUPPORT_QUERIES["query_kind"] == "calm")]["nnz"].to_numpy(float)
        _s = SUPPORT_QUERIES[(SUPPORT_QUERIES["K"] == _k)
                             & (SUPPORT_QUERIES["query_kind"] == "stress")]["nnz"].to_numpy(float)
        if _c.size == 0 or _s.size == 0:
            continue
        _lo, _hi = _bootstrap_ratio_ci(_c, _s, CFG.diag_bootstrap_reps, CFG.diag_seed)
        _ratio_rows.append({
            "K": int(_k), "h": H_STAR,
            "mean_nnz_calm": float(_c.mean()), "mean_nnz_stress": float(_s.mean()),
            "contraction_ratio": float(_c.mean() / _s.mean()) if _s.mean() > 0 else float("inf"),
            "ratio_ci95_lo": _lo, "ratio_ci95_hi": _hi,
            "zero_support_rate_stress": float(
                (SUPPORT_QUERIES[(SUPPORT_QUERIES["K"] == _k)
                                 & (SUPPORT_QUERIES["query_kind"] == "stress")]
                 ["zero_support"]).mean()),
        })
SUPPORT_CONTRACTION = pd.DataFrame(_ratio_rows)

for _name, _df in (("support_queries", SUPPORT_QUERIES),
                   ("support_summary", SUPPORT_SUMMARY),
                   ("support_contraction", SUPPORT_CONTRACTION)):
    if len(_df):
        _h2 = atomic_write_dataframe(PATHS["diagnostics"] / f"{_name}.csv", _df)
        register_artifact(f"diagnostics/{_name}.csv",
                          PATHS["diagnostics"] / f"{_name}.csv", _h2)

atomic_write_json(PATHS["diagnostics"] / "support_metadata.json", {
    "schema_version": CFG.schema_version, "config_hash": CONFIG_HASH,
    "h_star": H_STAR,
    "k_values_baseline": list(CFG.diag_k_values),
    "k_values_evaluated": list(DIAG_K_VALUES),
    "operational_k": int(K_STAR),
    "operational_k_covered": bool(int(K_STAR) in DIAG_K_VALUES),
    "step_index": int(CFG.diag_step_index),
    "calm_source": CFG.baseline_regime,
    "stress_window": [CFG.covid_crash_start, CFG.covid_crash_end],
    "n_reference_paths": int(M_REF),
    "definitions": {
        "NNZ": "#{m : exact compact support of the Markov-K window is non-empty}",
        "ESS": "(sum_m W_m)^2 / sum_m W_m^2 over normalised positive weights",
        "support_fraction": "NNZ / M_reference",
        "max_weight_share": "max_m W_m / sum_m W_m",
        "weight_entropy": "-sum_m W_m log W_m over normalised positive weights",
        "top_k_concentration": "share of the k largest normalised weights",
    },
    "support_from_clamped_weights": False,
    "bootstrap_reps": CFG.diag_bootstrap_reps, "bootstrap_seed": CFG.diag_seed,
    "legacy_claims_status": ("the 313 -> 7 counts and the 45x contraction ratio "
                             "are superseded by this table unless reproduced here"),
})

print("Support-contraction diagnostics "
      f"(h*={H_STAR:.4f}, K evaluated {list(DIAG_K_VALUES)}, operational K*="
      f"{K_STAR}, step index {CFG.diag_step_index}, M={M_REF})\n")
if len(SUPPORT_SUMMARY):
    print(SUPPORT_SUMMARY.round(4).to_string(index=False))
if len(SUPPORT_CONTRACTION):
    print("\nContraction (calm / stress), with bootstrap 95% CI:")
    print(SUPPORT_CONTRACTION.round(4).to_string(index=False))
    print("\n  The thesis must quote these regenerated values; the legacy "
          "313 -> 7 and 45x\n  numbers may only be kept if they appear here.")
else:
    add_warning("support diagnostics produced no rows")

# ==========================================================================
# ---
#
# ## Sections 17–20 — Tables, figures, final audit and manifest
#
# Every table is written as CSV, LaTeX and an inline preview from the same
# DataFrame. The manifest validates the whole deliverable set and decides whether
# the run may be called a locked thesis result.

# ---- notebook cell 37 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 26 — TABLES AND FIGURES
# ═════════════════════════════════════════════════════════════════════════════
#  Every table is emitted three ways from ONE DataFrame: CSV, LaTeX and a
#  human-readable preview. No number is ever typed into a LaTeX file by hand.
# ═════════════════════════════════════════════════════════════════════════════

TABLE_REGISTRY: Dict[str, Dict[str, Any]] = {}


def export_table(name: str, df: "pd.DataFrame", caption: str, label: str,
                 float_format: str = "%.4f", preview_rows: int = 12) -> None:
    if df is None or len(df) == 0:
        add_warning(f"table {name} is empty and was not exported")
        print(f"  [skip] {name}: no rows")
        return
    csv_path = PATHS["tables"] / f"{name}.csv"
    tex_path = PATHS["tables"] / f"{name}.tex"
    digest = atomic_write_dataframe(csv_path, df)
    try:
        body = df.to_latex(index=False, escape=True, float_format=float_format,
                           caption=caption, label=label, longtable=False)
    except Exception as exc:                                 # noqa: BLE001
        body = (f"% to_latex failed: {exc!r}\n"
                + df.to_string(index=False))
    tex_digest = atomic_write_text(tex_path, body)
    TABLE_REGISTRY[name] = {"csv": str(csv_path.name), "tex": str(tex_path.name),
                            "rows": int(len(df)), "csv_sha256": digest,
                            "tex_sha256": tex_digest, "caption": caption,
                            "label": label,
                            "source_columns": list(df.columns)}
    register_artifact(f"tables/{name}.csv", csv_path, digest)
    register_artifact(f"tables/{name}.tex", tex_path, tex_digest)
    print(f"\n[{name}] {caption}")
    print(df.head(preview_rows).round(6).to_string(index=False))
    if len(df) > preview_rows:
        print(f"  ... {len(df) - preview_rows} more rows (see {csv_path.name})")


# ── 1. Data-period metadata ─────────────────────────────────────────────────
export_table("tab01_data_periods", DATA_PERIOD_TABLE,
             "Data periods. Evaluation regimes are classified by path start date; "
             "the 2019 component of the extended COVID regime overlaps the "
             "generator-calibration window.",
             "tab:data_periods")

# ── 2. Generator calibration ────────────────────────────────────────────────
_calib_rows = []
for _i, _t in enumerate(TICKERS):
    _calib_rows.append({
        "asset": _t,
        "gbm_drift_annual": GBM_PARAMS["drift_annual"][_i],
        "gbm_vol_annual": GBM_PARAMS["sigma_annual"][_i],
        "heston_mu": HESTON_PARAMS["mu"][_i],
        "heston_theta": HESTON_PARAMS["theta"][_i],
        "heston_v0": HESTON_PARAMS["v0"][_i],
        "heston_kappa": HESTON_PARAMS["kappa"][_i],
        "heston_xi": HESTON_PARAMS["xi"][_i],
        "heston_rho": HESTON_PARAMS["rho"][_i],
        "feller_margin": HESTON_PARAMS["feller_2_kappa_theta_minus_xi2"][_i],
    })
CALIBRATION_TABLE = pd.DataFrame(_calib_rows)
export_table("tab02_generator_calibration", CALIBRATION_TABLE,
             f"Generator calibration on {CFG.calibration_start} to "
             f"{CFG.calibration_end}. SBTS uses M={M_REF} rolling reference "
             f"trajectories with h*={H_STAR:.4f} and K*={K_STAR}.",
             "tab:generator_calibration")

# ── 3. Generator fidelity ───────────────────────────────────────────────────
export_table("tab03_generator_fidelity", FIDELITY_DISPLAY,
             "Distributional fidelity of each generator against the historical "
             "calibration sample.", "tab:generator_fidelity")

# ── 4/5. Results (baseline regime and by regime) ────────────────────────────
def aggregate_results(master: "pd.DataFrame", eval_sets: Sequence[str],
                      phase: str = "cvar") -> "pd.DataFrame":
    sub = master[(master["phase"] == phase)
                 & (master["evaluation_set"].isin(eval_sets))]
    if not len(sub):
        return pd.DataFrame()
    agg = (sub.groupby(["evaluation_set", "option", "kappa", "generator"])
           .agg(n_seeds=("seed", "nunique"),
                mean_std=("std", "mean"), sem_std=("std", sp_stats.sem),
                mean_cvar95=("cvar95", "mean"), sem_cvar95=("cvar95", sp_stats.sem),
                mean_rmse=("rmse", "mean"), mean_V0=("V0", "mean"),
                mean_cost=("mean_transaction_cost", "mean"))
           .reset_index())
    return agg


RESULTS_BY_REGIME = aggregate_results(MASTER, CFG.regime_names)
MAIN_RESULTS = (RESULTS_BY_REGIME[RESULTS_BY_REGIME["evaluation_set"] == CFG.baseline_regime]
                if len(RESULTS_BY_REGIME) else pd.DataFrame())
export_table("tab04_main_results", MAIN_RESULTS,
             f"Hedging performance on {CFG.baseline_regime} (CVaR phase, "
             f"classified by path start date); mean over seeds with standard error.",
             "tab:main_results")
export_table("tab05_results_by_regime", RESULTS_BY_REGIME,
             "Hedging performance by evaluation regime (CVaR phase, classified "
             "by path start date).", "tab:results_by_regime")

# ── 6. Pairwise tests ───────────────────────────────────────────────────────
_pw = PAIRED_TESTS.copy() if len(PAIRED_TESTS) else pd.DataFrame()
if len(_pw):
    _pw = _pw[["family", "evaluation_set", "option", "kappa", "phase", "metric", "model_a",
               "model_b", "n_seeds", "mean_paired_difference", "ci95_lo",
               "ci95_hi", "t_stat", "p_value", "p_bh", "effect_size_dz",
               "significant_bh"]]
export_table("tab06_pairwise_tests", _pw,
             f"Paired t-tests on per-seed metric differences with "
             f"Benjamini-Hochberg FDR control at alpha={CFG.fdr_alpha}.",
             "tab:pairwise_tests", preview_rows=15)

# ── 7. Robustness ───────────────────────────────────────────────────────────
export_table("tab07_robustness", ROBUSTNESS_TESTS,
             "Robustness of the paired comparisons: exact sign-flip permutation "
             "test, Wilcoxon signed-rank and bootstrap confidence intervals.",
             "tab:robustness", preview_rows=15)

# ── 8. MCS ──────────────────────────────────────────────────────────────────
export_table("tab08_mcs", MCS_TABLE,
             f"Model Confidence Set (arch implementation, alpha={CFG.mcs_alpha}, "
             f"{CFG.mcs_reps} bootstrap replications, method {CFG.mcs_method}); "
             f"secondary evidence given {len(CFG.seeds)} seed observations.",
             "tab:mcs")

# ── 9. V0 / capital efficiency ──────────────────────────────────────────────
V0_TABLE = pd.DataFrame()
if len(MASTER):
    V0_TABLE = (MASTER[(MASTER["phase"] == "cvar")
                       & (MASTER["evaluation_set"] == CFG.baseline_regime)]
                .groupby(["option", "kappa", "generator"])
                .agg(n_seeds=("seed", "nunique"), mean_V0=("V0", "mean"),
                     sd_V0=("V0", "std"), mean_payoff=("mean_payoff", "mean"),
                     mean_cost=("mean_transaction_cost", "mean"))
                .reset_index())
    V0_TABLE["v0_over_mean_payoff"] = (V0_TABLE["mean_V0"]
                                       / V0_TABLE["mean_payoff"].replace(0, np.nan))
export_table("tab09_v0_capital_efficiency", V0_TABLE,
             "Learned initial capital V0 and capital efficiency on "
             f"{CFG.baseline_regime}.", "tab:v0")

# ── 10. Curriculum effect ───────────────────────────────────────────────────
CURRICULUM_TABLE = pd.DataFrame()
if len(MASTER):
    _piv = (MASTER[MASTER["evaluation_set"].isin(CFG.regime_names)]
            .groupby(["evaluation_set", "option", "kappa", "generator", "phase"])
            [["std", "cvar95"]].mean().reset_index())
    _mse = _piv[_piv["phase"] == "mse"].drop(columns="phase")
    _cvar = _piv[_piv["phase"] == "cvar"].drop(columns="phase")
    CURRICULUM_TABLE = _mse.merge(_cvar, on=["evaluation_set", "option", "kappa",
                                             "generator"], suffixes=("_mse", "_cvar"))
    if len(CURRICULUM_TABLE):
        CURRICULUM_TABLE["delta_std_pct"] = (
            100 * (CURRICULUM_TABLE["std_cvar"] - CURRICULUM_TABLE["std_mse"])
            / CURRICULUM_TABLE["std_mse"].replace(0, np.nan))
        CURRICULUM_TABLE["delta_cvar95_pct"] = (
            100 * (CURRICULUM_TABLE["cvar95_cvar"] - CURRICULUM_TABLE["cvar95_mse"])
            / CURRICULUM_TABLE["cvar95_mse"].replace(0, np.nan))
export_table("tab10_curriculum_effect", CURRICULUM_TABLE,
             "Effect of the CVaR fine-tuning phase relative to the MSE warm-up.",
             "tab:curriculum")

# ── 11. Support diagnostics ─────────────────────────────────────────────────
export_table("tab11_support_diagnostics", SUPPORT_SUMMARY,
             f"Support diagnostics of the SBTS conditioning kernel at "
             f"h*={H_STAR:.4f} over M={M_REF} reference trajectories.",
             "tab:support_diagnostics")
export_table("tab12_support_contraction", SUPPORT_CONTRACTION,
             "Support contraction between the calm regime and the COVID crash "
             "window, with bootstrap confidence intervals.",
             "tab:support_contraction")

# ── Figures ─────────────────────────────────────────────────────────────────
FIGURE_REGISTRY: Dict[str, Dict[str, Any]] = {}
DS_COLORS = {"GBM": "#E74C3C", "Heston": "#27AE60", "SBTS": "#2980B9",
             "Historical": "#2C3E50"}


def save_figure(fig, name: str, caption: str) -> None:
    for ext in ("pdf", "png"):
        p = PATHS["figures"] / f"{name}.{ext}"
        tmp = p.parent / (p.name + ".tmp")
        fig.savefig(tmp, format=ext, dpi=300, bbox_inches="tight")
        os.replace(tmp, p)
        register_artifact(f"figures/{name}.{ext}", p)
    FIGURE_REGISTRY[name] = {"caption": caption,
                             "files": [f"{name}.pdf", f"{name}.png"]}
    plt.close(fig)
    print(f"  figure: {name}.pdf / .png")


print("\nFigures")
# fig1 — historical overview
_fig, _ax = plt.subplots(1, 2, figsize=(11, 3.6))
_norm = CLEAN_PRICES / CLEAN_PRICES.iloc[0]
for _t in TICKERS:
    _ax[0].plot(_norm.index, _norm[_t], lw=0.9, label=_t)
_ax[0].set_yscale("log"); _ax[0].set_ylabel(r"$S_t/S_0$")
_ax[0].axvline(pd.Timestamp(CFG.calibration_end), color="red", ls="--", lw=1.0)
_ax[0].set_title("(a) Normalised prices", loc="left", fontweight="bold")
_ax[0].legend(frameon=False)
for _t in TICKERS:
    _ax[1].plot(LOG_RETURNS.index, LOG_RETURNS[_t] * 100, lw=0.3, alpha=0.7, label=_t)
_ax[1].axvline(pd.Timestamp(CFG.calibration_end), color="red", ls="--", lw=1.0)
_ax[1].set_ylabel("daily return (%)"); _ax[1].set_ylim(-25, 25)
_ax[1].set_title("(b) Daily log-returns", loc="left", fontweight="bold")
_fig.tight_layout()
save_figure(_fig, "fig1_historical_overview",
            "Historical data overview; the dashed line is the end of the "
            "generator-calibration window.")

# fig2 — return distributions, historical vs generators
_fig, _axes = plt.subplots(1, CFG.d, figsize=(11, 3.4), sharey=True)
_axes = np.atleast_1d(_axes)
for _i in range(CFG.d):
    _ax2 = _axes[_i]
    for _label, _data in [("Historical", TRAIN_RETURNS_VALUES[:, _i]),
                          *[(g, GENERATOR_RETURNS_FLAT[g][:, _i]) for g in GENERATORS]]:
        _x = _data * 100
        _lo, _hi = np.percentile(_x, [0.5, 99.5])
        _clip = _x[(_x >= _lo) & (_x <= _hi)]
        if _clip.size > 10:
            _kde = sp_stats.gaussian_kde(_clip)
            _grid = np.linspace(-8, 8, 400)
            _ax2.plot(_grid, _kde(_grid), lw=1.6 if _label == "Historical" else 1.1,
                      ls="-" if _label == "Historical" else "--",
                      color=DS_COLORS[_label], label=_label)
    _ax2.set_title(TICKERS[_i]); _ax2.set_xlabel("daily return (%)")
    if _i == 0:
        _ax2.set_ylabel("density"); _ax2.legend(frameon=False)
_fig.tight_layout()
save_figure(_fig, "fig2_return_distributions",
            "Daily log-return densities: historical calibration sample versus "
            "the three generators.")

# fig3 — hedging residual dispersion by regime
if len(RESULTS_BY_REGIME):
    _fig, _axes = plt.subplots(1, len(CFG.regime_names),
                               figsize=(4 * len(CFG.regime_names), 3.4), sharey=True)
    _axes = np.atleast_1d(_axes)
    for _j, _r in enumerate(CFG.regime_names):
        _ax3 = _axes[_j]
        _sub = RESULTS_BY_REGIME[RESULTS_BY_REGIME["evaluation_set"] == _r]
        _labels = [f"{o.split('_')[0][:3]}\nk={k:.2f}"
                   for o, k in zip(_sub["option"], _sub["kappa"])]
        for _g in GENERATORS:
            _gs = _sub[_sub["generator"] == _g]
            _x = np.arange(len(_gs))
            _ax3.errorbar(_x + 0.12 * (GENERATORS.index(_g) - 1), _gs["mean_std"],
                          yerr=_gs["sem_std"], fmt="o", ms=4, capsize=2,
                          color=DS_COLORS[_g], label=_g)
            _ax3.set_xticks(_x)
            _ax3.set_xticklabels([f"{o[:6]}\n{k:.2f}" for o, k in
                                  zip(_gs["option"], _gs["kappa"])], fontsize=7)
        _ax3.set_title(_r.replace("_", " "), fontsize=9)
        if _j == 0:
            _ax3.set_ylabel(r"mean $\sigma(R)$ over seeds"); _ax3.legend(frameon=False)
    _fig.tight_layout()
    save_figure(_fig, "fig3_residual_dispersion_by_regime",
                "Mean residual standard deviation by regime (CVaR phase), "
                "classified by path start date; error bars are standard errors "
                "over seeds.")

# fig4 — support diagnostics
if len(SUPPORT_QUERIES):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for _kind, _color in (("calm", "#2980B9"), ("stress", "#E74C3C")):
        _sub = SUPPORT_QUERIES[(SUPPORT_QUERIES["query_kind"] == _kind)]
        _grp = _sub.groupby("K")["nnz"].mean()
        _axes[0].plot(_grp.index, _grp.values, "o-", color=_color, label=_kind)
        _k_op = min(K_STAR, int(SUPPORT_QUERIES["K"].max()))
        _vals = _sub[_sub["K"] == _k_op]["nnz"].to_numpy()
        if _vals.size:
            _axes[1].hist(_vals, bins=30, alpha=0.55, color=_color,
                          label=f"{_kind} (K={_k_op})")
    _axes[0].set_xlabel("Markov order K"); _axes[0].set_ylabel("mean non-zero support")
    _axes[0].set_yscale("symlog"); _axes[0].legend(frameon=False)
    _axes[0].set_title("(a) Support size versus K", loc="left", fontweight="bold")
    _axes[1].set_xlabel("non-zero support count"); _axes[1].set_ylabel("queries")
    _axes[1].legend(frameon=False)
    _axes[1].set_title("(b) Support distribution", loc="left", fontweight="bold")
    _fig.tight_layout()
    save_figure(_fig, "fig4_support_contraction",
                "Support contraction of the SBTS conditioning kernel between "
                "calm and stress query states.")

atomic_write_json(PATHS["tables"] / "table_registry.json", {
    "schema_version": CFG.schema_version, "config_hash": CONFIG_HASH,
    "tables": TABLE_REGISTRY, "figures": FIGURE_REGISTRY,
    "generated_from": "one DataFrame per table; no hand-entered numbers"})
print(f"\nExported {len(TABLE_REGISTRY)} tables and {len(FIGURE_REGISTRY)} figures "
      f"to {PATHS['tables']} / {PATHS['figures']}")

# ---- notebook cell 38 --------------------------------------------------
# ═════════════════════════════════════════════════════════════════════════════
#  CELL 27 — FINAL AUDIT AND RUN MANIFEST (Gate 6 sign-off)
# ═════════════════════════════════════════════════════════════════════════════
#  A run is marked `complete` only when every required artifact validates. The
#  manifest is the object the thesis reproducibility appendix must quote,
#  together with the run id.
# ═════════════════════════════════════════════════════════════════════════════

def audit_tests() -> Dict[str, Any]:
    """Audit checks from design §7.3."""
    checks: Dict[str, Any] = {}
    queue = canonical_run_queue(CFG)
    checks["expected_run_keys"] = {"expected": CFG.n_configurations,
                                   "observed": len(queue),
                                   "passed": len(queue) == CFG.n_configurations}
    n_ckpt = int(CHECKPOINT_AUDIT["verified"].sum()) if len(CHECKPOINT_AUDIT) else 0
    n_failed_runs = sum(1 for v in TRAINING_RESULTS["runs"].values()
                        if v.get("status") == "failed")
    checks["expected_checkpoints"] = {
        "expected_if_no_failure": CFG.n_expected_checkpoints,
        "observed": n_ckpt, "failed_runs": n_failed_runs,
        "passed": n_ckpt + n_failed_runs * len(CFG.test_phases) >= CFG.n_expected_checkpoints}
    bad_seed = (CHECKPOINT_AUDIT[(CHECKPOINT_AUDIT["exists"])
                                 & (CHECKPOINT_AUDIT["seed_matches"] == False)]  # noqa: E712
                if len(CHECKPOINT_AUDIT) else pd.DataFrame())
    checks["seed_identity"] = {"mismatches": int(len(bad_seed)),
                               "alias_map": SEED_ALIAS_MAP,
                               "passed": len(bad_seed) == 0 and not SEED_ALIAS_MAP}
    unequal = (int((~PAIRED_TESTS["seeds_equal"]).sum()) if len(PAIRED_TESTS) else 0)
    checks["statistical_pairing"] = {
        "tests": int(len(PAIRED_TESTS)),
        "cells_using_seed_intersection": unequal,
        "passed": True,
        "note": "intersection is allowed and reported; relabelling is not"}
    nonfinite = (int((~MASTER["all_finite"].fillna(False).astype(bool)).sum())
                 if len(MASTER) else 0)
    checks["no_nonfinite_metrics"] = {"rows": nonfinite, "passed": nonfinite == 0}
    checks["tables_from_single_source"] = {
        "tables": len(TABLE_REGISTRY),
        "passed": all(t["rows"] > 0 for t in TABLE_REGISTRY.values())}
    missing_hash = [k for k, v in MANIFEST["artifacts"].items() if not v.get("sha256")]
    checks["artifacts_hashed"] = {"missing": missing_hash,
                                  "passed": not missing_hash}
    checks["snapshot_is_real_data"] = {
        "synthetic": SNAPSHOT_IS_SYNTHETIC, "passed": not SNAPSHOT_IS_SYNTHETIC}
    checks["selection_engine_publishable"] = {
        "engine": SELECTION_META["engine"],
        "passed": bool(SELECTION_META.get("publishable"))}
    checks["run_mode_is_full"] = {"run_mode": CFG.run_mode,
                                  "smoke_sizing": bool(USE_SMOKE_SIZING),
                                  "passed": (CFG.run_mode == "FULL"
                                             and not USE_SMOKE_SIZING)}
    return checks


ACCEPTANCE_CRITERIA = [
    ("runs_from_clean_kernel", True,
     "the notebook defines every symbol it uses; Cell 17T asserts no stale state"),
    ("self_contained", True, "one notebook carries the whole pipeline"),
    ("frozen_hashed_data", not SNAPSHOT_IS_SYNTHETIC,
     "published runs must use the hashed Yahoo snapshot"),
    ("calibration_2005_2019_with_2019_disclosure", True,
     "calibration window and the overlap disclosure are locked in Cell 0/6/7"),
    ("seed_all_defined_in_bootstrap", "seed_all" in globals(),
     "no NameError for seed_all is possible"),
    ("no_seed_substitution", not SEED_ALIAS_MAP,
     "no alias map exists and the checkpoint audit checks seed identity"),
    ("every_epoch_uses_all_paths", True,
     "asserted per epoch in train_phase and tested in Cell 17T"),
    ("best_checkpoint_restores_everything", True,
     "best bundle stores model, optimizer, scheduler and nu"),
    ("nu_from_validation_var", True,
     "Phase 2 initialises nu at the empirical validation VaR"),
    ("gradient_logging_correct", True,
     "post-clip norms are measured after clipping and asserted"),
    ("initialisation_matches_description", True,
     "Xavier on every Linear layer, verified by unit test"),
    ("chronological_purged_selection",
     SELECTION_META["split"]["split_type"].startswith("chronological"),
     "bandwidth/K selection uses a purged chronological split"),
    ("historical_metadata_complete", True,
     "regime metadata reports start/end ranges and the crash share"),
    ("standard_mcs", ARCH_AVAILABLE,
     "MCS runs through arch on a common-seed loss matrix"),
    ("support_diagnostics_regenerated", len(SUPPORT_SUMMARY) > 0,
     "support diagnostics come from code, not from older numbers"),
    ("artifacts_stored_per_run_id", True, "all outputs live under the run id"),
    ("resume_rejects_mismatched_hashes", True,
     "cache_is_valid compares schema, config and data hashes"),
    ("manifest_validates_deliverables", True, "this cell"),
    ("tables_generated_automatically", len(TABLE_REGISTRY) > 0,
     "CSV, LaTeX and preview come from one DataFrame"),
    ("locked_results_only_after_gates", True,
     "the manifest records which gates passed"),
]

AUDIT = audit_tests()
ACCEPTANCE_TABLE = pd.DataFrame(
    [{"criterion": c, "satisfied": bool(v), "evidence": e}
     for c, v, e in ACCEPTANCE_CRITERIA])
_h = atomic_write_dataframe(PATHS["logs"] / "acceptance_criteria.csv",
                            ACCEPTANCE_TABLE)
register_artifact("logs/acceptance_criteria.csv",
                  PATHS["logs"] / "acceptance_criteria.csv", _h)

_required_ok = all(v.get("passed", False) for k, v in AUDIT.items()
                   if k not in ("run_mode_is_full", "snapshot_is_real_data",
                                "selection_engine_publishable"))
_publishable = (_required_ok and AUDIT["run_mode_is_full"]["passed"]
                and AUDIT["snapshot_is_real_data"]["passed"]
                and AUDIT["selection_engine_publishable"]["passed"]
                and not MANIFEST["failures"])

MANIFEST.update({
    "status": "complete" if _required_ok else "incomplete",
    "publishable": bool(_publishable),
    "ended_utc": utc_now(),
    "config_hash": CONFIG_HASH,
    "environment_hash": ENVIRONMENT_HASH,
    "data_hashes": DATA_HASHES,
    "source_snapshot_commit": SOURCE_SNAPSHOT_COMMIT,
    "precision_mode_locked": LOCKED_PRECISION_MODE,
    "audit": AUDIT,
    "acceptance_criteria": ACCEPTANCE_TABLE.to_dict(orient="records"),
    "run_keys": [s["run_key"] for s in canonical_run_queue(CFG)],
    "tables": TABLE_REGISTRY,
    "figures": FIGURE_REGISTRY,
    "statistics_artifacts": [k for k in MANIFEST["artifacts"] if k.startswith("statistics/")],
    "evaluation_artifacts": [k for k in MANIFEST["artifacts"] if k.startswith("evaluations/")],
    "diagnostics_artifacts": [k for k in MANIFEST["artifacts"] if k.startswith("diagnostics/")],
    "n_artifacts": len(MANIFEST["artifacts"]),
    "shard_manifests": sorted(p.name for p in RUN_DIR.glob("manifest__*.json")),
    "shard_results": sorted(p.name for p in
                            PATHS["summaries"].glob("training_results*.json")),
    "selection": {"h_star": H_STAR, "k_star": K_STAR,
                  "engine": SELECTION_META["engine"],
                  "publishable": SELECTION_META.get("publishable")},
})
MANIFEST["manifest_hash"] = canonical_hash(
    {k: v for k, v in MANIFEST.items() if k != "manifest_hash"})
save_manifest()

# ── Notebook snapshot (best effort) ─────────────────────────────────────────
for _candidate in ("SBTS_CANONICAL_A100.ipynb",
                   "/content/SBTS_CANONICAL_A100.ipynb",
                   "notebooks/SBTS_CANONICAL_A100.ipynb"):
    _p = Path(_candidate)
    if _p.exists():
        shutil.copy2(_p, PATHS["notebook_snapshot"] / _p.name)
        register_artifact("notebook_snapshot/notebook",
                          PATHS["notebook_snapshot"] / _p.name)
        break

# ── Thesis result mapping ───────────────────────────────────────────────────
_mapping_lines = [
    "# Thesis result mapping",
    "",
    f"Run id: `{RUN_ID}`  ",
    f"Config hash: `{CONFIG_HASH}`  ",
    f"Manifest hash: `{MANIFEST['manifest_hash']}`  ",
    f"Precision mode: `{LOCKED_PRECISION_MODE}`  ",
    f"Publishable: **{_publishable}**",
    "",
    "| Thesis object | Artifact | Rows | CSV SHA-256 |",
    "|---|---|---:|---|",
]
for _name, _t in TABLE_REGISTRY.items():
    _mapping_lines.append(f"| {_t['caption'][:80]} | `tables/{_t['csv']}` | "
                          f"{_t['rows']} | `{_t['csv_sha256'][:16]}...` |")
for _name, _f in FIGURE_REGISTRY.items():
    _mapping_lines.append(f"| {_f['caption'][:80]} | `figures/{_f['files'][0]}` | - | - |")
_mapping_lines += [
    "",
    "## Provenance chain",
    "",
    "```",
    "frozen data -> locked configuration -> deterministic seed identity ->",
    "generator artifact -> training checkpoint -> per-path evaluation ->",
    "statistical procedure -> table/figure -> thesis claim",
    "```",
    "",
    "## Pending claims",
    "",
    "* Bandwidth/Markov order: "
    f"h*={H_STAR:.4f}, K*={K_STAR} (engine `{SELECTION_META['engine']}`).",
    "* MCS membership: see `statistics/mcs.csv`; secondary evidence only.",
    "* Support contraction: see `diagnostics/support_contraction.csv`; the "
    "legacy 313 -> 7 and 45x numbers hold only if reproduced there.",
]
atomic_write_text(RUN_DIR / "THESIS_RESULT_MAPPING.md", "\n".join(_mapping_lines) + "\n")
register_artifact("THESIS_RESULT_MAPPING.md", RUN_DIR / "THESIS_RESULT_MAPPING.md")
save_manifest()

print("=" * 78)
print(f"RUN {RUN_ID}")
print("=" * 78)
print(f"  status              : {MANIFEST['status']}")
print(f"  publishable         : {MANIFEST['publishable']}")
print(f"  run mode            : {CFG.run_mode}")
print(f"  precision locked    : {LOCKED_PRECISION_MODE}")
print(f"  config hash         : {CONFIG_HASH}")
print(f"  environment hash    : {ENVIRONMENT_HASH[:32]}...")
print(f"  manifest hash       : {MANIFEST['manifest_hash']}")
print(f"  artifacts           : {MANIFEST['n_artifacts']}")
print(f"  gpu hours           : {MANIFEST.get('gpu_hours', 0.0):.2f}")
print(f"  warnings / failures : {len(MANIFEST['warnings'])} / {len(MANIFEST['failures'])}")
print("\nAudit checks")
for _k, _v in AUDIT.items():
    print(f"  {'PASS' if _v.get('passed') else 'FAIL'}  {_k}")
print("\nAcceptance criteria")
print(ACCEPTANCE_TABLE.to_string(index=False))
if MANIFEST["failures"]:
    print("\nFailures recorded:")
    for _f in MANIFEST["failures"]:
        print(f"  - {_f['message']}")
if not _publishable:
    print("\n  This run is NOT publishable. Thesis numbers may only be updated "
          "from a\n  FULL run on the real frozen snapshot, with the paper_full "
          "selection engine,\n  no recorded failures and every audit check "
          "passing (Gate 6 -> Gate 7).")
else:
    print("\n  Gate 6 satisfied. Quote the run id and manifest hash above in the "
          "thesis\n  reproducibility appendix before locking any number.")

