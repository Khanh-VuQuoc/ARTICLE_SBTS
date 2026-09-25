"""Synthetic smoke tests for notebooks/article_mcs_recalculation.ipynb.

The helper code under test is executed straight from the notebook cells
tagged ``mcs-constants`` and ``mcs-helpers`` so there is no second copy.
The end-to-end test runs the whole notebook against a *synthetic* legacy
directory; it never touches Google Drive or real thesis data.

Run:  pytest tests/test_mcs_recalculation.py -q
"""
import json
import os
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "article_mcs_recalculation.ipynb"


def _tagged_source(nb, tag):
    cells = [c for c in nb.cells if tag in c.metadata.get("tags", [])]
    assert len(cells) == 1, f"expected one cell tagged {tag!r}"
    return cells[0].source


@pytest.fixture(scope="module")
def h():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    ns = {}
    exec(_tagged_source(nb, "mcs-constants"), ns)
    exec(_tagged_source(nb, "mcs-helpers"), ns)
    return ns


def make_legacy_csv(h, path, rng_seed=0):
    """Synthetic per_seed_metrics_3p.csv shaped like the legacy file.

    SBTS / asian_worst_of_put / 0.95 carries canonical seed 3 (numerics of
    the actual-seed-10 retry), exactly as the legacy pipeline stored it.
    """
    rng = np.random.default_rng(rng_seed)
    level = {"GBM": 0.060, "Heston": 0.050, "SBTS": 0.040}
    rows = []
    for period in h["PERIODS"]:
        for option in h["OPTIONS"]:
            for kappa in h["KAPPAS"]:
                for ds in h["MODEL_ORDER"]:
                    for seed in h["CANONICAL_SEEDS"]:
                        for phase in ("mse", "cvar"):
                            v = level[ds] * (1.0 + 0.08 * rng.standard_normal())
                            rows.append({"ds": ds, "option": option, "kappa": kappa,
                                         "seed": seed, "phase": phase, "period": period,
                                         "std": v, "cvar95": 2 * v, "cvar99": 3 * v,
                                         "rmse": v, "max": 4 * v, "mean": 0.0, "V0": 0.1})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_synthetic_smoke_suite(h):
    res = h["run_synthetic_smoke_tests"]()
    assert res["passed"].all(), res.to_string()


def test_cell_rng_seed_is_hash_based(h):
    import hashlib
    key = "mcs-v2|Recent_2023_25|basket_asian_call|1.00|R"
    expect = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % 2**32
    assert h["cell_rng_seed"]("Recent_2023_25", "basket_asian_call", 1.0, "R") == expect


def test_attach_keeps_seed_column_and_flags_replacement(h, tmp_path):
    p = tmp_path / "per_seed_metrics_3p.csv"
    make_legacy_csv(h, p)
    raw, info = h["load_legacy_metrics"](p)
    df = h["attach_actual_seed"](raw, h["SEED_REPLACEMENTS"])
    assert (df["seed"] == raw["seed"]).all()
    rep = df[df["replacement_used"]]
    # 3 periods x 2 phases
    assert len(rep) == 6
    assert set(rep["ds"]) == {"SBTS"} and set(rep["canonical_seed"]) == {3}
    assert set(rep["actual_seed"]) == {10}
    assert len(info["sha256"]) == 64 and Path(info["path"]).is_absolute()


def test_file_seed_column_validated(h, tmp_path):
    p = tmp_path / "per_seed_metrics_3p.csv"
    make_legacy_csv(h, p)
    raw, _ = h["load_legacy_metrics"](p)
    good = raw.copy()
    good["file_seed"] = good["seed"]
    mask = ((good["ds"] == "SBTS") & (good["option"] == "asian_worst_of_put")
            & (good["kappa"] == 0.95) & (good["seed"] == 3))
    good.loc[mask, "file_seed"] = 10
    df = h["attach_actual_seed"](good, h["SEED_REPLACEMENTS"])
    assert df["replacement_used"].sum() == 6
    bad = good.copy()
    bad.loc[bad.index[0], "file_seed"] = 99
    with pytest.raises(ValueError, match="file_seed"):
        h["attach_actual_seed"](bad, h["SEED_REPLACEMENTS"])


def test_alignment_on_full_grid(h, tmp_path):
    p = tmp_path / "per_seed_metrics_3p.csv"
    make_legacy_csv(h, p)
    raw, _ = h["load_legacy_metrics"](p)
    df = h["attach_actual_seed"](raw, h["SEED_REPLACEMENTS"])
    assert h["validate_legacy_metrics"](df, h["SEED_REPLACEMENTS"])["passed"].all()
    n = {}
    for period in h["PERIODS"]:
        for option in h["OPTIONS"]:
            for kappa in h["KAPPAS"]:
                L, a = h["build_mcs_loss_matrix"](df, period, option, kappa)
                assert list(L.columns) == ["GBM", "Heston", "SBTS"]
                n[(period, option, kappa)] = a["n_common"]
    assert sorted(n.values()).count(9) == 3 and sorted(n.values()).count(10) == 15
    assert {c for c, v in n.items() if v == 9} == set(h["AFFECTED_CELLS"])


def test_missing_input_fails_loudly(h, tmp_path):
    with pytest.raises(FileNotFoundError, match="No fallback"):
        h["load_legacy_metrics"](tmp_path / "absent.csv")


def test_notebook_end_to_end_on_synthetic_legacy_dir(h, tmp_path):
    nbclient = pytest.importorskip("nbclient")
    root = tmp_path / "ARTICLE_SBTS"
    legacy = root / "article_results_3p"
    legacy.mkdir(parents=True)
    make_legacy_csv(h, legacy / "per_seed_metrics_3p.csv")
    # Stand-ins for legacy artifacts that must stay byte-for-byte untouched
    old_mcs = pd.DataFrame([
        {"period": p, "option": o, "kappa": k, "mcs": "SBTS", "mcs_size": 1}
        for p in h["PERIODS"] for o in h["OPTIONS"] for k in h["KAPPAS"]])
    old_mcs.to_csv(legacy / "mcs_3p.csv", index=False)
    (legacy / "tab_main_results.tex").write_text("% legacy table\n")
    before = {f.name: f.read_bytes() for f in legacy.iterdir()}

    nb = nbformat.read(NOTEBOOK, as_version=4)
    env = {"MCS_ARTICLE_ROOT": str(root), "MCS_GIT_COMMIT": "synthetic-test"}
    old_env = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        nbclient.NotebookClient(nb, timeout=900, kernel_name="python3",
                                resources={"metadata": {"path": str(tmp_path)}}).execute()
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    out = legacy / "mcs_recalculated_v2"
    for name in h["OUTPUT_FILES"]:
        assert (out / name).is_file(), name
    meta = json.loads((out / "mcs_metadata.json").read_text())
    assert meta["final_gate"]["status"] == "PASS"
    assert meta["n_common_distribution"] == {"9": 3, "10": 15}
    assert meta["paired_t_tests_recomputed"] is False
    assert meta["descriptive_results_changed"] is False
    assert len(meta["rng_seeds"]) == 36

    primary = pd.read_csv(out / "mcs_corrected_method_R.csv")
    aff = primary[primary["n_common"] == 9]
    assert len(aff) == 3 and set(aff["common_actual_seeds"]) == {"0,1,2,4,5,6,7,8,9"}
    assert r"$^{\dagger}$" in (out / "tab_mcs_corrected.tex").read_text()
    audit = pd.read_csv(out / "mcs_alignment_audit.csv")
    assert len(audit) == 54 and (audit["alignment_status"] == "ok").all()

    after = {f.name: f.read_bytes() for f in legacy.iterdir() if f.is_file()}
    assert after == before
