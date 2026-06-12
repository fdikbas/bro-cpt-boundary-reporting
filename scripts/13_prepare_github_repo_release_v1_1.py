
# -*- coding: utf-8 -*-
"""
Prepare a clean GitHub release repository for BRO CPT Paper-1.

v1.1 cleanup changes relative to v1.0
------------------------------------
1. Only final/public script versions are copied.
   - Old interim figure-polish scripts such as v1_0, v1_1, v1_2 are not copied.
   - v1_3 is retained as the final polished figure generator.

2. Missing optional files are written to a separate report:
   - docs/missing_optional_files_report.csv
   - They are not mixed into release_manifest.csv.

3. Public documentation avoids local Windows paths.
   - docs/repository_setup_commands.txt uses generic shell commands.
   - release_manifest.csv stores repository-relative destinations and source labels,
     not absolute local paths.

4. Safety scan remains strict.
   - Raw BRO GeoPackage/cache/database-like files stop ZIP creation.

Run with Spyder/F5 or:

runfile('<PROJECT_ROOT>/bro_cpt_paper1_v2_0/13_prepare_github_repo_release_v1_1.py',
        wdir='<PROJECT_ROOT>/bro_cpt_paper1_v2_0')

Output:
outputs_paper1_v2/BRO_CPT_BoundaryReporting_GitHubRelease_v1_1/
outputs_paper1_v2/BRO_CPT_BoundaryReporting_GitHubRelease_v1_1.zip

Version: v1.1, 2026-06-11
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs_paper1_v2"

RAW_DIR = OUTPUTS_ROOT / "paper1_raw"
ANALYSIS_DIR = OUTPUTS_ROOT / "paper1_analysis"
FIGURE_PACKAGE_DIR = OUTPUTS_ROOT / "CG_submission_figure_package_v1"

RELEASE_NAME = "BRO_CPT_BoundaryReporting_GitHubRelease_v1_1"
RELEASE_DIR = OUTPUTS_ROOT / RELEASE_NAME
RELEASE_ZIP = OUTPUTS_ROOT / f"{RELEASE_NAME}.zip"

OVERWRITE = True
MAX_DERIVED_FILE_MB = 25.0
INCLUDE_PROFILE_EXAMPLES = False

GITHUB_REPO_URL = "https://github.com/fdikbas/bro-cpt-boundary-reporting"
ZENODO_D0I_PLACEHOLDER = "[Zenodo DOI to be added after archiving]"

# Final/public scripts only. Do not auto-copy every numbered script, because the
# development folder may contain interim versions.
FINAL_SCRIPT_CANDIDATES = [
    # Main/final analysis workflow candidates, copied if present.
    "BRO_CPT_Paper1_BoundaryConfidence_Pipeline_v1_9.py",
    "BRO_CPT_Paper1_Postprocess_v1_0.py",
    "10_make_publication_figures_full_singlefile.py",
    "11_make_publication_figures_data_driven_polished_v1_3.py",
    "12_prepare_CG_submission_figure_package_v1_0.py",
    "13_prepare_github_repo_release_v1_1.py",
]

# Optional files are recorded separately if missing.
OPTIONAL_SCRIPT_CANDIDATES = [
    "BRO_CPT_Paper1_BoundaryConfidence_Pipeline_v1_5.py",
]

DERIVED_CANDIDATES = [
    RAW_DIR / "boundary_catalog.csv",
    RAW_DIR / "synthetic_validation_results.csv",
    RAW_DIR / "sensitivity_results.csv",
    RAW_DIR / "realdata_profiles_meta.csv",
    ANALYSIS_DIR / "baseline_boundary_counts_by_cpt.csv",
    FIGURE_PACKAGE_DIR / "fig5_baseline_comparison_data_used.csv",
    FIGURE_PACKAGE_DIR / "figure_generation_manifest.csv",
    FIGURE_PACKAGE_DIR / "figure_manifest_CG.csv",
    FIGURE_PACKAGE_DIR / "figure_captions_CG.txt",
    FIGURE_PACKAGE_DIR / "figure_captions_CG.md",
]

OPTIONAL_DERIVED_CANDIDATES = [
    RAW_DIR / "preparation_summary.csv",
    RAW_DIR / "feature_selection.csv",
    RAW_DIR / "missingness_summary.csv",
    ANALYSIS_DIR / "baseline_counts_by_cpt.csv",
    ANALYSIS_DIR / "boundary_counts_by_cpt.csv",
]

FIGURE_FILE_NAMES = [
    "Figure_1_workflow.pdf",
    "Figure_1_workflow.png",
    "Figure_2_spatial_distribution.pdf",
    "Figure_2_spatial_distribution.png",
    "Figure_3_representative_CPT_profiles.pdf",
    "Figure_3_representative_CPT_profiles.png",
    "Figure_4_synthetic_validation.pdf",
    "Figure_4_synthetic_validation.png",
    "Figure_5_baseline_comparison.pdf",
    "Figure_5_baseline_comparison.png",
    "Figure_6_sensitivity_analysis.pdf",
    "Figure_6_sensitivity_analysis.png",
]

FORBIDDEN_SUFFIXES = {
    ".gpkg", ".sqlite", ".db", ".mdb", ".accdb", ".parquet", ".feather",
    ".h5", ".hdf", ".hdf5", ".las", ".laz",
}
FORBIDDEN_NAME_PARTS = [
    "brocptvolledigeset",
    "cache",
    "raw_external",
]


# =============================================================================
# Repository text
# =============================================================================

def _readme_text() -> str:
    return f"""# Scale-aware CPT boundary reporting with uncertainty bands

This repository contains the reproducible code and derived-output structure for the manuscript:

**Scale-aware boundary reporting for cone penetration test profiles using uncertainty bands and confidence scores**

Target journal: *Computers and Geotechnics*

## Overview

Cone penetration test (CPT/CPTu) profiles often contain multiple depth transitions that can depend on filtering scale, local noise, and boundary-detection parameters. This repository supports a scale-aware reporting framework that summarizes CPT boundary evidence using:

- estimated boundary depth,
- P10-P90 uncertainty band,
- scale-stability/confidence score,
- synthetic recovery metrics,
- real-data baseline comparison,
- sensitivity analysis,
- publication-ready figures.

## Repository contents

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── .gitignore
├── MANUSCRIPT_DECLARATIONS.md
├── REPOSITORY_STRUCTURE.md
├── scripts/
├── data/
│   ├── raw_external/
│   ├── interim/
│   └── DO_NOT_UPLOAD_RAW_BRO_DATA.txt
├── derived_results/
├── figures/
├── docs/
└── tests/
```

## Data policy

Raw third-party BRO CPT/CPTu data are **not redistributed** in this repository. Users should obtain the raw public data from the official source and place them locally under:

```text
data/raw_external/
```

Only code, documentation, final figures, and small derived result tables intended for reproducibility/audit are included here.

## Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate bro-cpt-boundary-reporting
```

Using pip on Windows:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Using pip on Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the workflow

The recommended single entry point is:

```bash
python scripts/00_run_all_workflow.py
```

The numbered scripts are organized in execution order. Some scripts may require the local raw BRO CPT/CPTu data to be placed under `data/raw_external/`.

## Final figures

Final figure files prepared for the manuscript are included under `figures/`. Vector PDF files are recommended as the primary journal-submission files, with PNG files included as high-resolution backups.

## Reproducibility and archiving

After final manuscript submission or acceptance, create a versioned GitHub release and archive it with Zenodo or another persistent repository. Add the final DOI to this README, `CITATION.cff`, and the manuscript's Code availability section.

## Repository URL

{GITHUB_REPO_URL}

## License

Code is released under the MIT License unless otherwise stated. Data remain subject to the terms of the original data providers.
"""


def _declarations_text() -> str:
    return f"""# Manuscript declaration text

These text blocks are prepared for direct use in the manuscript or submission system. Replace bracketed placeholders before submission.

## Code availability

The code used to implement the scale-aware CPT boundary reporting workflow, generate the derived result tables, run the synthetic validation and baseline comparisons, perform sensitivity analysis, and reproduce the publication figures is available in a public GitHub repository:

**Repository:** {GITHUB_REPO_URL}

A versioned archive of the code will be deposited in Zenodo upon acceptance or final submission:

**Archived version:** {ZENODO_D0I_PLACEHOLDER}

The repository contains the analysis scripts, figure-generation scripts, environment files, and documentation required to reproduce the reported derived results. Raw third-party BRO CPT/CPTu data are not redistributed in the repository; users should obtain the raw data from the official public data source and follow the instructions in the repository README.

## Data availability

This study uses public BRO CPT/CPTu data for the Netherlands. The raw CPT/CPTu data are not redistributed with this article or in the GitHub repository because they should be obtained directly from the official data provider and remain subject to the provider's terms of use.

The repository associated with this study provides instructions for obtaining and preparing the raw input data and includes derived result files where redistribution is permitted, including the boundary catalog, synthetic validation summaries, baseline comparison summaries, sensitivity analysis summaries, figure-generation manifests, and figure-caption files. These derived products are sufficient to verify the numerical values shown in the final figures and to audit the reported comparisons.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the author used ChatGPT (OpenAI) to support language editing, code drafting and debugging, organization of the reproducible workflow, preparation of figure-generation scripts, and drafting of figure captions and manuscript declaration text. After using this tool, the author reviewed, edited, corrected, and verified the content as needed and takes full responsibility for the content of the published article.

No generative-AI-created image was used as a scientific data source. The final scientific figures were generated from the described code and the study's data-derived outputs.

## Declaration of competing interest

The author declares that there are no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Acknowledgements

The author acknowledges the public availability of the BRO CPT/CPTu data used in this study and the open-source Python scientific computing ecosystem used for data processing, analysis, and visualization.
"""


TEXT_FILES = {
    "README.md": _readme_text,
    "MANUSCRIPT_DECLARATIONS.md": _declarations_text,
    "REPOSITORY_STRUCTURE.md": lambda: """# Repository structure

## `scripts/`

Final public workflow and support scripts.

## `data/raw_external/`

Local-only directory for raw BRO CPT/CPTu files. Do not commit raw third-party data unless redistribution is explicitly permitted.

## `data/interim/`

Temporary or intermediate files generated during preprocessing. Usually excluded from version control.

## `derived_results/`

Small reproducible result tables, such as boundary catalogs, baseline count summaries, sensitivity summaries, and figure audit files.

## `figures/`

Final manuscript figures and figure manifests. Prefer vector PDF files for journal submission and PNG files as raster backups.

## `docs/`

Supplementary documentation, setup commands, and notes for users.

## `tests/`

Minimal smoke tests and validation checks.
""",
    "requirements.txt": lambda: """numpy>=1.26
pandas>=2.1
matplotlib>=3.8
scipy>=1.11
scikit-learn>=1.3
geopandas>=0.14
shapely>=2.0
fiona>=1.9
pyproj>=3.6
tqdm>=4.66
Pillow>=10.0
openpyxl>=3.1
ruptures>=1.1
""",
    "environment.yml": lambda: """name: bro-cpt-boundary-reporting
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - numpy
  - pandas
  - matplotlib
  - scipy
  - scikit-learn
  - geopandas
  - shapely
  - fiona
  - pyproj
  - tqdm
  - pillow
  - openpyxl
  - pip
  - pip:
      - ruptures
""",
    ".gitignore": lambda: """# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.pytest_cache/
.ipynb_checkpoints/

# Environments
.venv/
venv/
env/
.env
*.conda

# Local raw and large data
data/raw_external/*
data/interim/*
*.gpkg
*.sqlite
*.db
*.mdb
*.accdb
*.parquet
*.feather
*.h5
*.hdf
*.hdf5
*.zip
*.7z
*.rar

# Keep README files in otherwise ignored directories
!data/raw_external/README.md
!data/interim/README.md

# Large/generated outputs
outputs/
logs/
tmp/

# OS/editor files
.DS_Store
Thumbs.db
.vscode/
.idea/
""",
    "LICENSE": lambda: """MIT License

Copyright (c) 2026 Fatih Dikbaş

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the \"Software\"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.

THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND.
""",
    "CITATION.cff": lambda: f"""cff-version: 1.2.0
message: \"If you use this software, please cite the associated manuscript and archived software release.\"
title: \"Scale-aware CPT boundary reporting with uncertainty bands\"
authors:
  - family-names: \"Dikbaş\"
    given-names: \"Fatih\"
version: \"1.0.0\"
date-released: \"2026-06-11\"
repository-code: \"{GITHUB_REPO_URL}\"
license: \"MIT\"
""",
    "data/README.md": lambda: """# Data directory

Place raw external BRO CPT/CPTu data here only for local analysis. Raw third-party data are intentionally excluded from this public repository.
""",
    "data/raw_external/README.md": lambda: """# Raw external data

This folder is for local raw data only. Do not upload raw BRO CPT/CPTu files to GitHub unless the data provider's license explicitly permits redistribution.
""",
    "data/interim/README.md": lambda: """# Interim data

This folder is for temporary intermediate outputs. It is ignored by Git by default.
""",
    "data/DO_NOT_UPLOAD_RAW_BRO_DATA.txt": lambda: """Do not upload raw BRO CPT/CPTu data to a public repository unless the original data license and redistribution terms explicitly permit this.

Recommended public sharing:
- code
- README and documentation
- environment files
- small derived result tables where redistribution is permitted
- final figures
- manifests and captions

Recommended non-public/local only:
- raw GeoPackage or bulk CPT/CPTu data downloads
- large caches
- temporary/interim processing files
""",
    "derived_results/README.md": lambda: """# Derived results

This folder contains small derived result tables copied from the local analysis workflow where available.
""",
    "figures/README.md": lambda: """# Figures

This folder contains final manuscript figures copied from the submission-ready figure package. Use vector PDF files as primary journal-submission files whenever possible.
""",
    "docs/repository_setup_commands.txt": lambda: f"""# Suggested GitHub setup commands

# Run these commands from inside the generated release folder.

git init
git add .
git commit -m \"Initial reproducibility release for BRO CPT boundary reporting\"

git branch -M main
git remote add origin {GITHUB_REPO_URL}.git
git push -u origin main

git tag v1.0.0
git push origin v1.0.0
""",
    "tests/README.md": lambda: """# Tests

Recommended minimal checks:
1. No raw third-party data are present.
2. Required scripts exist.
3. Derived output tables contain required columns.
4. Final figure files exist.
5. `git status` does not include raw data, cache files, or large private files.
""",
}


# =============================================================================
# Helpers
# =============================================================================

def _progress(items: Iterable, desc: str):
    items = list(items)
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, unit="item")


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0)


def _rel_to_release(path: Path) -> str:
    try:
        return str(path.relative_to(RELEASE_DIR)).replace("\\", "/")
    except Exception:
        return path.name


def _source_label(path: Path) -> str:
    # Public-safe label: no absolute local Windows/Linux paths in the manifest.
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except Exception:
        try:
            return str(path.relative_to(SCRIPT_DIR)).replace("\\", "/")
        except Exception:
            return path.name


def _manifest_row(action: str, dst: Path, src: Optional[Path] = None, status: str = "ok", note: str = "") -> Dict[str, str]:
    if dst.exists() and dst.is_file():
        size = f"{_size_mb(dst):.3f}"
        sha = _sha256(dst)
    else:
        size = ""
        sha = ""
    return {
        "action": action,
        "status": status,
        "source_label": "" if src is None else _source_label(src),
        "destination": _rel_to_release(dst),
        "size_mb": size,
        "sha256": sha,
        "note": note,
    }


def _missing_row(kind: str, src: Path, note: str = "") -> Dict[str, str]:
    return {
        "kind": kind,
        "source_label": _source_label(src),
        "note": note,
    }


def _write_text(path: Path, text: str, rows: List[Dict[str, str]], note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    rows.append(_manifest_row("write_text", path, note=note))


def _copy_file(src: Path, dst: Path, rows: List[Dict[str, str]], missing_rows: List[Dict[str, str]], note: str = "", optional: bool = False) -> bool:
    if not src.exists() or not src.is_file():
        missing_rows.append(_missing_row("optional" if optional else "required", src, note=note))
        if not optional:
            rows.append({
                "action": "copy",
                "status": "missing_required",
                "source_label": _source_label(src),
                "destination": _rel_to_release(dst),
                "size_mb": "",
                "sha256": "",
                "note": note,
            })
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    rows.append(_manifest_row("copy", dst, src=src, note=note))
    return True


def _safe_copy_small(src: Path, dst: Path, rows: List[Dict[str, str]], missing_rows: List[Dict[str, str]], note: str = "", optional: bool = False) -> bool:
    if not src.exists() or not src.is_file():
        missing_rows.append(_missing_row("optional" if optional else "required", src, note=note))
        if not optional:
            rows.append({
                "action": "copy_small",
                "status": "missing_required",
                "source_label": _source_label(src),
                "destination": _rel_to_release(dst),
                "size_mb": "",
                "sha256": "",
                "note": note,
            })
        return False
    if _size_mb(src) > MAX_DERIVED_FILE_MB:
        rows.append({
            "action": "copy_small",
            "status": "skipped_too_large",
            "source_label": _source_label(src),
            "destination": _rel_to_release(dst),
            "size_mb": f"{_size_mb(src):.3f}",
            "sha256": "",
            "note": f"{note}; larger than MAX_DERIVED_FILE_MB={MAX_DERIVED_FILE_MB}",
        })
        return False
    return _copy_file(src, dst, rows, missing_rows, note=note, optional=optional)


def _is_forbidden_public_file(path: Path) -> bool:
    rel = str(path.relative_to(RELEASE_DIR)).replace("\\", "/").lower()
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return True
    if any(part.lower() in rel for part in FORBIDDEN_NAME_PARTS):
        if rel.endswith("readme.md"):
            return False
        return True
    return False


def _write_manifest(rows: List[Dict[str, str]]) -> None:
    manifest_path = RELEASE_DIR / "release_manifest.csv"
    fields = ["action", "status", "source_label", "destination", "size_mb", "sha256", "note"]
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_missing_report(missing_rows: List[Dict[str, str]], rows: List[Dict[str, str]]) -> None:
    report_path = RELEASE_DIR / "docs" / "missing_optional_files_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["kind", "source_label", "note"]
    with report_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(missing_rows)
    rows.append(_manifest_row("write_text", report_path, note="missing optional/required source report"))


def _write_repo_docs(rows: List[Dict[str, str]]) -> None:
    for rel, maker in TEXT_FILES.items():
        _write_text(RELEASE_DIR / rel, maker(), rows, note="repository documentation")


def _make_runner(rows: List[Dict[str, str]]) -> None:
    runner = """# -*- coding: utf-8 -*-
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
    # Edit this list after selecting the final public workflow scripts.
]

def _iter_steps(steps):
    if tqdm is None:
        return steps
    return tqdm(steps, desc="[BRO-CPT] Workflow", unit="script")

def main() -> None:
    print("[BRO-CPT] Repository root:", ROOT)
    print("[BRO-CPT] Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if not STEPS:
        print("[BRO-CPT] No public workflow steps configured yet. Edit scripts/00_run_all_workflow.py.")
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
"""
    _write_text(RELEASE_DIR / "scripts" / "00_run_all_workflow.py", runner, rows, note="public workflow runner")


def _copy_scripts(rows: List[Dict[str, str]], missing_rows: List[Dict[str, str]]) -> None:
    public_dir = RELEASE_DIR / "scripts"
    public_dir.mkdir(parents=True, exist_ok=True)
    _make_runner(rows)

    for name in FINAL_SCRIPT_CANDIDATES:
        src = SCRIPT_DIR / name
        dst = public_dir / name
        _safe_copy_small(src, dst, rows, missing_rows, note="final public script", optional=False)

    for name in OPTIONAL_SCRIPT_CANDIDATES:
        src = SCRIPT_DIR / name
        dst = public_dir / name
        _safe_copy_small(src, dst, rows, missing_rows, note="optional legacy script", optional=True)


def _copy_derived_results(rows: List[Dict[str, str]], missing_rows: List[Dict[str, str]]) -> None:
    dst_dir = RELEASE_DIR / "derived_results"
    dst_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    for src in DERIVED_CANDIDATES:
        if src in seen:
            continue
        seen.add(src)
        _safe_copy_small(src, dst_dir / src.name, rows, missing_rows, note="derived/audit result", optional=False)

    for src in OPTIONAL_DERIVED_CANDIDATES:
        _safe_copy_small(src, dst_dir / src.name, rows, missing_rows, note="optional derived result", optional=True)

    if INCLUDE_PROFILE_EXAMPLES:
        prof_dir = RAW_DIR / "profiles_realdata"
        if prof_dir.exists():
            dst_prof = dst_dir / "profiles_realdata_examples"
            dst_prof.mkdir(parents=True, exist_ok=True)
            for src in sorted(prof_dir.glob("*.csv"))[:12]:
                _safe_copy_small(src, dst_prof / src.name, rows, missing_rows, note="optional profile example", optional=True)


def _copy_figures(rows: List[Dict[str, str]], missing_rows: List[Dict[str, str]]) -> None:
    dst_dir = RELEASE_DIR / "figures"
    dst_dir.mkdir(parents=True, exist_ok=True)

    for fname in FIGURE_FILE_NAMES:
        src = FIGURE_PACKAGE_DIR / fname
        dst = dst_dir / fname
        _copy_file(src, dst, rows, missing_rows, note="final Computers and Geotechnics figure", optional=False)

    for fname in ["figure_captions_CG.txt", "figure_captions_CG.md", "figure_manifest_CG.csv", "README_CG_figure_package.txt"]:
        src = FIGURE_PACKAGE_DIR / fname
        dst = RELEASE_DIR / "docs" / fname
        _safe_copy_small(src, dst, rows, missing_rows, note="figure package documentation", optional=True)


def _safety_scan(rows: List[Dict[str, str]]) -> List[Path]:
    bad: List[Path] = []
    for p in RELEASE_DIR.rglob("*"):
        if p.is_file() and _is_forbidden_public_file(p):
            bad.append(p)

    if bad:
        for p in bad:
            rows.append({
                "action": "safety_scan",
                "status": "forbidden_detected",
                "source_label": "",
                "destination": _rel_to_release(p),
                "size_mb": f"{_size_mb(p):.3f}",
                "sha256": "",
                "note": "Forbidden raw/large-data-like file detected in release folder",
            })
    else:
        rows.append({
            "action": "safety_scan",
            "status": "ok",
            "source_label": "",
            "destination": ".",
            "size_mb": "",
            "sha256": "",
            "note": "No forbidden raw-data-like files detected",
        })
    return bad


def _make_zip(rows: List[Dict[str, str]]) -> None:
    if RELEASE_ZIP.exists() and OVERWRITE:
        RELEASE_ZIP.unlink()

    with zipfile.ZipFile(RELEASE_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(RELEASE_DIR.rglob("*")):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(RELEASE_DIR.parent)))

    rows.append({
        "action": "zip",
        "status": "ok",
        "source_label": ".",
        "destination": RELEASE_ZIP.name,
        "size_mb": f"{_size_mb(RELEASE_ZIP):.3f}",
        "sha256": _sha256(RELEASE_ZIP),
        "note": "public GitHub release package",
    })


def prepare_release() -> None:
    print("[GH-release] PROJECT_ROOT:       ", PROJECT_ROOT)
    print("[GH-release] SCRIPT_DIR:         ", SCRIPT_DIR)
    print("[GH-release] RAW_DIR:            ", RAW_DIR)
    print("[GH-release] ANALYSIS_DIR:       ", ANALYSIS_DIR)
    print("[GH-release] FIGURE_PACKAGE_DIR: ", FIGURE_PACKAGE_DIR)
    print("[GH-release] RELEASE_DIR:        ", RELEASE_DIR)
    print("[GH-release] RELEASE_ZIP:        ", RELEASE_ZIP)

    if RELEASE_DIR.exists() and OVERWRITE:
        shutil.rmtree(RELEASE_DIR)
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    missing_rows: List[Dict[str, str]] = []

    steps = [
        ("write_repo_docs", lambda: _write_repo_docs(rows)),
        ("copy_scripts", lambda: _copy_scripts(rows, missing_rows)),
        ("copy_derived_results", lambda: _copy_derived_results(rows, missing_rows)),
        ("copy_figures", lambda: _copy_figures(rows, missing_rows)),
    ]

    for name, func in _progress(steps, desc="[GH-release] Steps"):
        print(f"[GH-release] {name}")
        func()

    _write_missing_report(missing_rows, rows)
    bad = _safety_scan(rows)
    _write_manifest(rows)

    if bad:
        print("[GH-release] ERROR: Forbidden raw-data-like files were detected:")
        for p in bad:
            print("    ", p)
        print("[GH-release] ZIP was not created. Remove these files or update the safety rules after license review.")
        raise RuntimeError("Safety scan failed; release ZIP not created.")

    _make_zip(rows)
    _write_manifest(rows)

    print("[GH-release] missing optional/required report:")
    print("    ", RELEASE_DIR / "docs" / "missing_optional_files_report.csv")
    print("[GH-release] wrote manifest:")
    print("    ", RELEASE_DIR / "release_manifest.csv")
    print("[GH-release] wrote ZIP:")
    print("    ", RELEASE_ZIP)
    print("[GH-release] DONE")


if __name__ == "__main__":
    prepare_release()
