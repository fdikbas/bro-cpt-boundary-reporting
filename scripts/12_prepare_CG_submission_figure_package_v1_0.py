# -*- coding: utf-8 -*-
"""
Prepare a submission-ready figure package for BRO CPT Paper-1
targeting Computers and Geotechnics.

This script collects the final polished v1.3 figure files, renames them with
journal-friendly names, writes figure captions and a manifest, and creates a ZIP
package ready for manuscript submission.

Run in Spyder/F5 or with:

runfile('<PROJECT_ROOT>/bro_cpt_paper1_v2_0/12_prepare_CG_submission_figure_package_v1_0.py',
        wdir='<PROJECT_ROOT>/bro_cpt_paper1_v2_0')

Version: v1.0, 2026-06-11
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from PIL import Image
except Exception:
    Image = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs_paper1_v2"
SOURCE_FIGURE_DIR = OUTPUTS_ROOT / "paper1_analysis_publication_polished_v1_3"
PACKAGE_DIR = OUTPUTS_ROOT / "CG_submission_figure_package_v1"
PACKAGE_ZIP = OUTPUTS_ROOT / "ComputersGeotechnics_FigurePackage_v1.zip"
OVERWRITE = True

FIGURE_MAP: List[Dict[str, str]] = [
    {"figure": "Figure 1", "source_stem": "fig1_workflow_polished", "target_stem": "Figure_1_workflow", "short_title": "Scale-aware CPT boundary reporting workflow"},
    {"figure": "Figure 2", "source_stem": "fig2_nl_pointmap_polished", "target_stem": "Figure_2_spatial_distribution", "short_title": "Spatial distribution of BRO CPT soundings"},
    {"figure": "Figure 3", "source_stem": "fig3_cpt_panels_polished", "target_stem": "Figure_3_representative_CPT_profiles", "short_title": "Representative CPT profiles with boundary uncertainty bands"},
    {"figure": "Figure 4", "source_stem": "fig4_synthetic_validation_polished", "target_stem": "Figure_4_synthetic_validation", "short_title": "Synthetic validation of boundary recovery"},
    {"figure": "Figure 5", "source_stem": "fig5_baseline_comparison_polished", "target_stem": "Figure_5_baseline_comparison", "short_title": "Baseline comparison of real-data boundary counts"},
    {"figure": "Figure 6", "source_stem": "fig6_sensitivity_heatmaps_polished", "target_stem": "Figure_6_sensitivity_analysis", "short_title": "Sensitivity analysis of the scale-aware framework"},
]

CAPTIONS: Dict[str, str] = {
    "Figure 1": "Workflow of the scale-aware CPT boundary reporting framework. Public BRO CPT/CPTu data are first subjected to quality control, depth-grid regularization, and multi-scale filtering. Candidate boundary evidence is then summarized through boundary locations, P10-P90 uncertainty bands, and scale-stability/confidence scores. The lower branch shows the validation and robustness components used in this study, including synthetic boundary recovery, baseline comparison, and parameter-sensitivity analysis.",
    "Figure 2": "Spatial distribution of the Netherlands CPT soundings used for boundary reporting. Point color represents the mean scale-stability score (SGS) of detected boundaries within each CPT sounding, whereas point size represents the number of detected boundaries per sounding. The inset highlights the densest sampled area. The map is used to document the spatial coverage of the real-data evaluation set rather than to infer regional geotechnical zoning.",
    "Figure 3": "Representative real CPT soundings with data-driven boundary locations and uncertainty bands. Each panel shows measured and smoothed CPT profile variables for a selected BRO CPT sounding, together with detected stratigraphic boundary depths. Horizontal lines indicate estimated boundary locations, and shaded intervals represent the corresponding P10-P90 uncertainty bands. Boundary colors denote confidence classes derived from the scale-aware boundary evidence.",
    "Figure 4": "Synthetic validation of boundary recovery under increasing relative noise levels. Panel (a) reports mean precision, recall, and F1 score for recovered boundaries, and panel (b) reports the matched-boundary mean absolute error (MAE). Points represent mean values at each noise level, and shaded envelopes indicate +/-1 standard deviation across synthetic profiles or replicates. The validation evaluates whether the scale-aware procedure can recover known boundary locations under controlled uncertainty.",
    "Figure 5": "Baseline comparison of detected boundary counts on real-data CPT soundings. Panel (a) shows the distribution of boundary counts per CPT sounding for the scale-aware method, the single-scale baseline, and the PELT baseline using violin plots, box plots, and individual sounding-level values. Panel (b) compares the corresponding mean segmentation density. The scale-aware method produced a lower mean boundary density than both baselines, with mean counts of 5.37, 8.40, and 48.63 boundaries per CPT for the scale-aware, single-scale, and PELT methods, respectively.",
    "Figure 6": "Sensitivity analysis of the scale-aware boundary framework across 24 parameter combinations. Panel (a) shows the synthetic mean F1 score, and panel (b) shows the mean number of real-data boundaries per CPT sounding. Each heatmap cell corresponds to a specific combination of minimum separation distance, peak-prominence threshold, minimum scale support, and matching tolerance. Cell values are read directly from the sensitivity results, and the highlighted cell marks the selected locked setting used for the final reporting workflow.",
}


def _iter_progress(items: Iterable, desc: str):
    items = list(items)
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, unit="file")


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0)


def _png_info(path: Path):
    if Image is None or not path.exists() or path.suffix.lower() != ".png":
        return "", "", ""
    try:
        with Image.open(path) as im:
            w, h = im.size
            dpi = im.info.get("dpi", None)
            dpi_text = f"{dpi[0]:.0f} x {dpi[1]:.0f}" if dpi else ""
            return str(w), str(h), dpi_text
    except Exception:
        return "", "", ""


def _copy_one(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source figure: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_text_files() -> None:
    txt_lines: List[str] = []
    md_lines: List[str] = ["# Figure captions for Computers and Geotechnics submission", ""]
    for item in FIGURE_MAP:
        fig = item["figure"]
        txt_lines.append(f"{fig}. {CAPTIONS[fig]}")
        txt_lines.append("")
        md_lines.append(f"**{fig}.** {CAPTIONS[fig]}")
        md_lines.append("")

    (PACKAGE_DIR / "figure_captions_CG.txt").write_text("\n".join(txt_lines).strip() + "\n", encoding="utf-8")
    (PACKAGE_DIR / "figure_captions_CG.md").write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")

    readme = f"""Computers and Geotechnics - Figure package
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Project root: {PROJECT_ROOT}
Source figure directory: {SOURCE_FIGURE_DIR}
Package directory: {PACKAGE_DIR}

Contents
--------
- Six final figure files as PDF and PNG.
- PDF files are intended as the primary vector/editable submission files.
- PNG files are included as high-resolution raster backups.
- figure_captions_CG.txt and figure_captions_CG.md contain final caption text.
- figure_manifest_CG.csv records file size, image dimensions where applicable, and SHA-256 checksums.

Notes
-----
- This package copies final polished v1.3 figure outputs and does not alter the original files.
- Scientific/data-driven figures must remain traceable to the pipeline outputs.
- Do not include raw third-party BRO data in a public repository unless the license and redistribution terms explicitly permit it.
"""
    (PACKAGE_DIR / "README_CG_figure_package.txt").write_text(readme, encoding="utf-8")


def _write_manifest(rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "figure", "short_title", "file_name", "file_type", "source_file",
        "size_mb", "width_px", "height_px", "png_dpi", "sha256", "caption",
    ]
    with (PACKAGE_DIR / "figure_manifest_CG.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _make_zip() -> None:
    if PACKAGE_ZIP.exists() and OVERWRITE:
        PACKAGE_ZIP.unlink()
    with zipfile.ZipFile(PACKAGE_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(PACKAGE_DIR.rglob("*")):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(PACKAGE_DIR.parent)))


def prepare_package() -> None:
    print("[CG-figpack] PROJECT_ROOT:      ", PROJECT_ROOT)
    print("[CG-figpack] SOURCE_FIGURE_DIR: ", SOURCE_FIGURE_DIR)
    print("[CG-figpack] PACKAGE_DIR:       ", PACKAGE_DIR)
    print("[CG-figpack] PACKAGE_ZIP:       ", PACKAGE_ZIP)

    if not SOURCE_FIGURE_DIR.exists():
        raise FileNotFoundError(
            f"Final v1.3 figure directory not found: {SOURCE_FIGURE_DIR}\n"
            "Run 11_make_publication_figures_data_driven_polished_v1_3.py first."
        )

    if PACKAGE_DIR.exists() and OVERWRITE:
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    copy_jobs: List[Tuple[Path, Path, Dict[str, str], str]] = []
    for item in FIGURE_MAP:
        for ext in [".pdf", ".png"]:
            src = SOURCE_FIGURE_DIR / f"{item['source_stem']}{ext}"
            dst = PACKAGE_DIR / f"{item['target_stem']}{ext}"
            copy_jobs.append((src, dst, item, ext.lower().lstrip(".")))

    for src, dst, item, ext in _iter_progress(copy_jobs, desc="[CG-figpack] Copying"):
        _copy_one(src, dst)
        width, height, dpi_text = _png_info(dst)
        rows.append({
            "figure": item["figure"],
            "short_title": item["short_title"],
            "file_name": dst.name,
            "file_type": ext.upper(),
            "source_file": str(src),
            "size_mb": f"{_file_size_mb(dst):.3f}",
            "width_px": width,
            "height_px": height,
            "png_dpi": dpi_text,
            "sha256": _sha256(dst),
            "caption": CAPTIONS[item["figure"]],
        })
        print(f"[CG-figpack] copied: {dst.name}")

    for src in [
        SOURCE_FIGURE_DIR / "fig5_baseline_comparison_data_used.csv",
        SOURCE_FIGURE_DIR / "figure_generation_manifest.csv",
    ]:
        if src.exists():
            dst = PACKAGE_DIR / src.name
            _copy_one(src, dst)
            rows.append({
                "figure": "Audit",
                "short_title": "Supporting audit file",
                "file_name": dst.name,
                "file_type": dst.suffix.lower().lstrip(".").upper(),
                "source_file": str(src),
                "size_mb": f"{_file_size_mb(dst):.3f}",
                "width_px": "",
                "height_px": "",
                "png_dpi": "",
                "sha256": _sha256(dst),
                "caption": "",
            })
            print(f"[CG-figpack] copied audit file: {dst.name}")

    _write_text_files()
    for extra_name in ["figure_captions_CG.txt", "figure_captions_CG.md", "README_CG_figure_package.txt"]:
        p = PACKAGE_DIR / extra_name
        rows.append({
            "figure": "Package",
            "short_title": "Package text file",
            "file_name": p.name,
            "file_type": p.suffix.lower().lstrip(".").upper(),
            "source_file": "",
            "size_mb": f"{_file_size_mb(p):.3f}",
            "width_px": "",
            "height_px": "",
            "png_dpi": "",
            "sha256": _sha256(p),
            "caption": "",
        })

    _write_manifest(rows)
    _make_zip()

    print("[CG-figpack] wrote captions:")
    print("    ", PACKAGE_DIR / "figure_captions_CG.txt")
    print("    ", PACKAGE_DIR / "figure_captions_CG.md")
    print("[CG-figpack] wrote manifest:")
    print("    ", PACKAGE_DIR / "figure_manifest_CG.csv")
    print("[CG-figpack] wrote ZIP:")
    print("    ", PACKAGE_ZIP)
    print("[CG-figpack] DONE")


if __name__ == "__main__":
    prepare_package()
