# MCS recalculation design for the legacy thesis results

**Repository:** `Khanh-VuQuoc/ARTICLE_SBTS`  
**Implementation branch:** `claude/mcs-legacy-results-v1`  
**Status:** implementation specification for Claude  
**Scope:** recalculate the Model Confidence Set (MCS) only; do not retrain models or replace the legacy descriptive results

## 1. Decision locked by the thesis owner

1. The legacy result set remains the primary source for the thesis.
2. The divergent run
   `SBTS / asian_worst_of_put / kappa=0.95 / canonical seed=3`
   is replaced by the converged retry at actual seed 10.
3. Seed 10 is a valid replacement replicate for descriptive summaries, provided that the replacement is disclosed and its actual identity is preserved in metadata.
4. No checkpoint training, path generation, OOS evaluation, main result table, or thesis conclusion is to be changed in this ticket.
5. The existing hand-written MCS calculation is superseded. MCS must be recalculated from the legacy per-seed CSV with explicit seed alignment and a maintained implementation of Hansen--Lunde--Nason MCS.

This design preserves the owner's seed-replacement decision. It does **not** treat SBTS actual seed 10 as if it were actual seed 3 when constructing the MCS loss matrix.

## 2. Why a separate MCS pass is required

The current `notebooks/article_4_3period.ipynb`:

- loads the SBTS seed-10 checkpoint for the failed canonical seed 3;
- stores the resulting metrics under the label `seed=3`; and
- sends rows sorted by this canonical label to a custom MCS routine.

That procedure hides the actual replicate identity. For MCS, every row of the `T x k` loss matrix must refer to one common observational unit across all `k` models. In the affected configuration, GBM and Heston have actual seed 3 while SBTS has actual seed 10. These three values must not occupy one paired row.

The correction is narrow:

- retain all ten legacy observations for descriptive means;
- preserve both `canonical_seed` and `actual_seed`;
- use the intersection of actual seeds when forming each MCS matrix;
- therefore use nine common seeds in the three affected period cells and ten common seeds in the other fifteen cells.

## 3. Authoritative input and prohibited fallbacks

### 3.1 Primary input

The notebook must read the legacy file:

```text
MyDrive/ARTICLE_SBTS/article_results_3p/per_seed_metrics_3p.csv
```

The input path must be configurable at the top of the notebook, but the notebook must print the resolved absolute path and SHA-256 hash before analysis.

### 3.2 Required columns

```text
ds, option, kappa, seed, phase, period, std, cvar95, mean, V0
```

If `file_seed` already exists, use it as `actual_seed` after validating it. Otherwise derive `actual_seed` with the single approved mapping below.

```python
SEED_REPLACEMENTS = {
    ("SBTS", "asian_worst_of_put", 0.95, 3): 10,
}
```

Definitions:

- `canonical_seed`: the existing `seed` column, used to identify the intended ten-run design;
- `actual_seed`: the checkpoint/run that produced the numeric observation;
- `replacement_used`: Boolean equal to `canonical_seed != actual_seed`.

Never overwrite either seed column after construction.

### 3.3 No fallback

The implementation must fail loudly if the legacy CSV is absent. It must not silently load:

- any file from `canonical_runs/20260922T103440Z__ee6dcd72`;
- any newly generated checkpoint evaluation;
- any previous MCS CSV; or
- any CSV selected by glob order or modification time.

## 4. Frozen MCS estimand

| Parameter | Locked value |
|---|---|
| Phase | `cvar` |
| Loss | `std` of hedging residual `R` |
| Models and column order | `GBM`, `Heston`, `SBTS` |
| Cells | 3 periods x 2 options x 3 strikes = 18 |
| Test size | `alpha = 0.10` |
| Primary implementation | `arch.bootstrap.MCS` |
| Primary method | `R` |
| Bootstrap | circular block bootstrap |
| Block size | 1, because seed replicates are treated as independent |
| Bootstrap replications | 50,000 |
| RNG policy | deterministic cell-specific seed derived with SHA-256 |
| Sensitivity method | `max`, with otherwise identical settings |

Use a maintained `arch` release whose installed version is printed and written to metadata. Do not copy the current custom `mcs_seed_level` function into the new notebook.

The `arch.bootstrap.MCS` input must be a pandas DataFrame with one row per common `actual_seed` and columns in the frozen model order.

## 5. Exact alignment algorithm

For each `(period, option, kappa)` cell:

1. Filter to `phase == "cvar"` and `loss == std`.
2. Validate exactly one finite value per `(ds, actual_seed)`.
3. Create the actual-seed set for each model.
4. Compute the sorted intersection across `GBM`, `Heston`, and `SBTS`.
5. Record seeds excluded from each model and the reason.
6. Pivot only the common actual seeds to a loss matrix.
7. Assert matrix columns are exactly `GBM`, `Heston`, `SBTS`.
8. Assert there are no missing, duplicated, infinite, or NaN losses.
9. Run primary `MCS(method="R")`.
10. Run sensitivity `MCS(method="max")`.
11. Store included/excluded models, elimination p-values, common-seed count, actual-seed list, and model means computed on the common matrix.

Expected alignment:

| Cell group | Expected common actual seeds | `n_common` |
|---|---|---:|
| `asian_worst_of_put`, `kappa=0.95`, each of 3 periods | `0,1,2,4,5,6,7,8,9` | 9 |
| All other 15 cells | `0,1,2,3,4,5,6,7,8,9` | 10 |

If any observed cell differs from these expectations, stop before running MCS and print a compact audit table.

## 6. Deterministic bootstrap seed

Do not use Python's built-in `hash()`, because it is process-dependent. Use a stable mapping such as:

```python
import hashlib

def cell_rng_seed(period: str, option: str, kappa: float, method: str) -> int:
    key = f"mcs-v2|{period}|{option}|{kappa:.2f}|{method}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)
```

The resolved RNG seed for every run must be present in the output CSV and metadata JSON.

## 7. Reference implementation pattern

Claude should implement pure helper functions in the notebook before the execution cells:

```python
def load_legacy_metrics(path: Path) -> tuple[pd.DataFrame, dict]: ...

def attach_actual_seed(
    df: pd.DataFrame,
    replacements: dict[tuple[str, str, float, int], int],
) -> pd.DataFrame: ...

def build_mcs_loss_matrix(
    df: pd.DataFrame,
    period: str,
    option: str,
    kappa: float,
) -> tuple[pd.DataFrame, dict]: ...

def run_arch_mcs(
    losses: pd.DataFrame,
    method: str,
    alpha: float,
    reps: int,
    seed: int,
) -> dict: ...

def compare_with_legacy_mcs(
    corrected: pd.DataFrame,
    legacy_path: Path | None,
) -> pd.DataFrame: ...
```

Primary call:

```python
from arch.bootstrap import MCS

mcs = MCS(
    losses,
    size=0.10,
    reps=50_000,
    block_size=1,
    method="R",
    bootstrap="circular",
    seed=cell_seed,
)
mcs.compute()
```

The code must read `included`, `excluded`, and `pvalues` from the computed object. Do not infer membership merely from the smallest mean loss.

## 8. Notebook design

Create:

```text
notebooks/article_mcs_recalculation.ipynb
```

Required cell order:

1. **Scope and locked decisions** -- explain that the notebook consumes legacy results and recalculates MCS only.
2. **Environment** -- imports, package versions, constants, paths.
3. **Input provenance** -- existence check, SHA-256, row/column counts.
4. **Seed provenance** -- build and display `canonical_seed`, `actual_seed`, `replacement_used`.
5. **Hard validation** -- schema, finite values, duplicates, 18-cell coverage, expected seed sets.
6. **Loss-matrix builder** -- pure function plus one affected-cell display.
7. **Synthetic smoke tests** -- clear winner, indistinguishable models, unequal/missing seeds, duplicate seed rejection.
8. **Primary MCS** -- method `R` for all 18 cells.
9. **Sensitivity MCS** -- method `max` for all 18 cells.
10. **Legacy comparison** -- compare memberships if the old MCS CSV is available; never use it as input.
11. **Output writer** -- CSV, JSON, Markdown report, optional LaTeX table.
12. **Final gate** -- print PASS/FAIL against all acceptance criteria.

The notebook must not import training networks, load `.pt` checkpoints, download market data, or invoke GPU code.

## 9. Output contract

Write all new files under a new directory so the legacy artifacts remain immutable:

```text
MyDrive/ARTICLE_SBTS/article_results_3p/mcs_recalculated_v2/
```

Required files:

```text
mcs_corrected_method_R.csv
mcs_sensitivity_method_max.csv
mcs_pvalues_long.csv
mcs_alignment_audit.csv
mcs_comparison_with_legacy.csv
mcs_metadata.json
MCS_RECALCULATION_REPORT.md
tab_mcs_corrected.tex
```

Minimum columns for `mcs_corrected_method_R.csv`:

```text
period, option, kappa, loss, phase, alpha, method, bootstrap,
block_size, reps, rng_seed, n_common, common_actual_seeds,
included_models, excluded_models, mcs_size,
gbm_mean_common, heston_mean_common, sbts_mean_common,
gbm_mean_all_available, heston_mean_all_available, sbts_mean_all_available,
input_sha256, arch_version
```

`mcs_alignment_audit.csv` must have one row per cell/model and include:

```text
period, option, kappa, ds, canonical_seeds, actual_seeds,
common_actual_seeds, excluded_actual_seeds, n_available, n_common,
replacement_count, alignment_status
```

`mcs_metadata.json` must record:

- timestamp in UTC;
- source path and SHA-256;
- git commit SHA when available;
- Python, NumPy, pandas, and `arch` versions;
- full seed-replacement mapping;
- all MCS parameters;
- all cell-specific RNG seeds;
- total cells, `n_common` distribution, and final gate status;
- explicit statement: `paired_t_tests_recomputed = false`;
- explicit statement: `descriptive_results_changed = false`.

## 10. Comparison and reporting rules

The report must separate three concepts:

1. **Legacy descriptive estimates:** unchanged, retain all ten available replicates including SBTS actual seed 10.
2. **Corrected MCS estimand:** calculated on common actual seeds; `n=9` for the three affected cells and `n=10` otherwise.
3. **Sensitivity result:** method `max`; disagreements with primary method `R` must be listed, not hidden.

The report must contain a before/after cell table if the old MCS file is present. It must not force membership to match the prior thesis narrative. Any changed cell is a result to report, not an implementation failure.

The LaTeX output must display `n=9` for the affected cells or include a table note that identifies them explicitly. It must not state that every MCS cell uses ten common seeds.

## 11. Acceptance criteria

The implementation is accepted only if all checks pass:

- [ ] The input SHA-256 and resolved path are recorded.
- [ ] Exactly 18 primary MCS cells are produced.
- [ ] Exactly 3 cells have `n_common=9`.
- [ ] Exactly 15 cells have `n_common=10`.
- [ ] The affected common seed set is exactly `0,1,2,4,5,6,7,8,9`.
- [ ] No matrix row pairs GBM/Heston actual seed 3 with SBTS actual seed 10.
- [ ] No duplicate `(cell, ds, actual_seed)` occurs.
- [ ] All loss values supplied to MCS are finite.
- [ ] Model order is identical in all cells.
- [ ] Re-running with the same environment yields identical CSV content.
- [ ] Synthetic clear-winner and tie smoke tests behave as expected.
- [ ] Primary and sensitivity outputs are both saved.
- [ ] The old MCS artifact is never overwritten.
- [ ] No training/evaluation notebook or checkpoint is changed.
- [ ] Descriptive result tables are byte-for-byte untouched.

## 12. Out of scope and scientific boundary

This ticket does not:

- retrain seed 3;
- rerun GBM, Heston, or SBTS;
- alter the ten-replicate descriptive means;
- modify thesis conclusions automatically;
- recalculate the 216 paired t-tests;
- certify that the legacy paired t-tests are corrected by this MCS change.

The last point is important: this task fixes the MCS construction only. It must not claim that a corrected MCS also repairs any separate paired-test alignment issue.

## 13. Files Claude may change

Allowed:

```text
notebooks/article_mcs_recalculation.ipynb
requirements.txt                         # add `arch`; no unrelated upgrades
MCS_LEGACY_RESULTS_RECALCULATION_DESIGN.md  # only status/checklist updates
```

Optional, only if needed for automated smoke tests:

```text
tests/test_mcs_recalculation.py
```

Forbidden:

```text
notebooks/article_data.ipynb
notebooks/article_gbm.ipynb
notebooks/article_heston_sbts.ipynb
notebooks/article_4_3period.ipynb
README.md
all legacy output CSV/TEX/figure files
```

## 14. Handoff instructions for Claude

1. Work only on branch `claude/mcs-legacy-results-v1`.
2. Implement the standalone notebook described above.
3. Run only synthetic/local smoke tests if the Google Drive legacy CSV is unavailable.
4. Do not fabricate real-data outputs.
5. Leave the notebook ready for one full Colab run against the locked Drive CSV.
6. Commit implementation and provide:
   - changed-file list;
   - smoke-test evidence;
   - remaining Colab command/run step;
   - any deviation from this specification.
7. Do not merge the branch and do not update thesis prose until the corrected real-data MCS output has been reviewed.

## 15. Definition of done

The task is complete when Claude has delivered a standalone, deterministic, audited MCS notebook; the notebook has passed synthetic smoke tests; and one Colab execution against the legacy CSV has produced all required artifacts with the final gate marked `PASS`.
