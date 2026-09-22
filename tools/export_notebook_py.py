#!/usr/bin/env python3
"""Regenerate the audit .py snapshot of a notebook.

Deliverable 2 of the canonical design is a plain-Python export of
`SBTS_CANONICAL_A100.ipynb` that can be diffed and reviewed without a Jupyter
front end. Markdown cells become comment blocks; code cells are emitted in
order, so running the export is equivalent to running every cell top to bottom.

    python3 tools/export_notebook_py.py notebooks/SBTS_CANONICAL_A100.ipynb
"""
import json
import sys
from pathlib import Path


def export(nb_path: Path) -> Path:
    nb = json.loads(nb_path.read_text())
    out = ["#!/usr/bin/env python3",
           f'"""Audit snapshot exported from {nb_path.name}.',
           "",
           "This file is generated; edit the notebook, not this export.",
           '"""', ""]
    for i, cell in enumerate(nb["cells"]):
        body = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            out.append("# " + "=" * 74)
            out += [f"# {line}".rstrip() for line in body.split("\n")]
            out.append("")
        else:
            out.append(f"# ---- notebook cell {i} " + "-" * 50)
            out.append(body)
            out.append("")
    dst = nb_path.with_suffix(".py")
    dst.write_text("\n".join(out) + "\n")
    return dst


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    src = Path(sys.argv[1])
    dst = export(src)
    print(f"wrote {dst}")
