# -*- coding: utf-8 -*-
"""
Run the public BRO CPT boundary-reporting workflow.

Before running the raw-data stages, obtain the public BRO CPT/CPTu GeoPackage
from the official data provider and either set BRO_CPT_GPKG to the local
GeoPackage path or edit the relevant CFG path inside the pipeline script.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from datetime import datetime

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    "BRO_CPT_Paper1_BoundaryConfidence_Pipeline_v1_9.py",
    "BRO_CPT_Paper1_Postprocess_v1_0.py",
    "11_make_publication_figures_data_driven_polished_v1_3.py",
    "12_prepare_CG_submission_figure_package_v1_0.py",
]

def _iter_steps(steps):
    if tqdm is None:
        return steps
    return tqdm(steps, desc="[BRO-CPT] Workflow", unit="script")

def main() -> None:
    print("[BRO-CPT] Repository root:", ROOT)
    print("[BRO-CPT] Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    for script_name in _iter_steps(STEPS):
        script_path = ROOT / "scripts" / script_name
        if not script_path.exists():
            print(f"[BRO-CPT] SKIP missing: {script_path}")
            continue
        print(f"[BRO-CPT] RUN {script_name}")
        runpy.run_path(str(script_path), run_name="__main__")
    print("[BRO-CPT] DONE")

if __name__ == "__main__":
    main()
