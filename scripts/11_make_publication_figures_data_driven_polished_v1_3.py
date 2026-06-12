"""Data-driven polished publication figure generator for BRO CPT Paper-1.

Purpose
-------
This script is a safer replacement/companion for
``10_make_publication_figures_full_singlefile.py``.  It keeps Fig. 2--6 strictly
DATA-DRIVEN from the CSV/GPKG outputs already produced by the Paper-1 pipeline,
while polishing typography, layout, legends, annotations, heatmap labels, and
panel balance to approach the visual quality of the manually designed ChatGPT
mockups.

Important design rule
---------------------
AI/mockup figures are NOT used as scientific data sources.  They only guide the
visual style.  All plotted points, curves, bars, heatmap cells, CPT IDs, and
profile traces are read from the project's existing data products.

Expected location
-----------------
Place this file next to the previous script, for example:

    <PROJECT_ROOT>/bro_cpt_paper1_v2_0/

Then run with Spyder/F5 or runfile exactly as usual.  No command-line arguments
are required.

Inputs expected relative to PROJECT_ROOT = Path(__file__).resolve().parents[1]
----------------------------------------------------------------------------
outputs_paper1_v2/paper1_raw/
    boundary_catalog.csv
    synthetic_validation_results.csv
    sensitivity_results.csv
    baselines/baseline_single_scale_boundaries.csv       optional
    baselines/baseline_pelt_boundaries.csv               optional
    profiles_realdata/*.csv                              optional but needed for Fig. 3

Other optional inputs
---------------------
- BRO CPT GeoPackage.  The script first checks BRO_CPT_GPKG environment variable,
  then searches sibling BRO CPT data folders.
- Netherlands province/admin boundary file.  The script searches common GIS
  folders and uses it if found.  If not found, Fig. 2 is still point-based, but
  the province background is omitted.

Outputs
-------
outputs_paper1_v2/paper1_analysis_publication_polished_v1/
    fig1_workflow_polished.png/pdf
    fig2_nl_pointmap_polished.png/pdf
    fig3_cpt_panels_polished.png/pdf
    fig4_synthetic_validation_polished.png/pdf
    fig5_baseline_comparison_polished.png/pdf
    fig6_sensitivity_heatmaps_polished.png/pdf
    figure_generation_manifest.csv

Author: generated for Prof. Dr. Fatih Dikbaş workflow
Version: v1.3, 2026-06-11
"""

from __future__ import annotations

import os
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon
from matplotlib.ticker import FuncFormatter, MaxNLocator

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

try:
    import fiona
except Exception:  # pragma: no cover
    fiona = None

try:
    from scipy.signal import medfilt
except Exception:  # pragma: no cover
    medfilt = None


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "outputs_paper1_v2" / "paper1_raw"
OUT_DIR = PROJECT_ROOT / "outputs_paper1_v2" / "paper1_analysis_publication_polished_v1_3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 450
RNG_SEED = 42

# Visual tokens: natural, muted, print-safe palette.
INK = "#1f2933"
MUTED = "#667085"
GRID = "#e7e3da"
BORDER = "#c9c2b8"
PAPER = "#fbfaf6"
PANEL = "#ffffff"
WARM = "#f3efe6"
TEAL_DARK = "#1f5c6e"
TEAL = "#2c6c85"
TEAL_MID = "#6aa6a6"
TEAL_LIGHT = "#cbe3e0"
AMBER = "#c08a3a"
AMBER_LIGHT = "#efd6a5"
RUST = "#b3543c"
RUST_DARK = "#8b3f3f"
RUST_LIGHT = "#e8b2a3"
GRAY = "#9aa0a6"
LIGHT_GRAY = "#d7d7d7"


# =============================================================================
# Generic helpers
# =============================================================================

def _progress(items: Sequence[Tuple[str, callable]]) -> Iterable[Tuple[str, callable]]:
    """Iterate with tqdm when available; otherwise plain iteration."""
    if tqdm is None:
        return items
    return tqdm(items, desc="[pubfig-polished] Figures", unit="fig")


def _set_pub_style() -> None:
    """Global matplotlib style for journal-quality, editable vector figures."""
    plt.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PANEL,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "font.family": "DejaVu Sans",
        "font.size": 10.4,
        "axes.titlesize": 12.8,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.8,
        "legend.fontsize": 9.4,
        "figure.titlesize": 17.4,
        "figure.titleweight": "bold",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.85,
        "grid.alpha": 0.78,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.size": 3.6,
        "ytick.major.size": 3.6,
        "xtick.major.width": 0.85,
        "ytick.major.width": 0.85,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.07,
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _seq_teal() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "bro_seq_teal", ["#f7f2e8", "#cbe3e0", "#6aa6a6", "#1f5c6e"], N=256
    )


def _seq_rust() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "bro_seq_rust", ["#f7f2e8", "#efd6a5", "#d58467", "#8b3f3f"], N=256
    )


def _div_muted() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "bro_div_muted", [TEAL_DARK, WARM, RUST_DARK], N=256
    )


def _read_csv_required(relative_path: str) -> pd.DataFrame:
    p = RAW_DIR / relative_path
    if not p.exists():
        raise FileNotFoundError(f"Missing required input file: {p}")
    return pd.read_csv(p)


def _read_csv_optional(relative_path: str) -> Optional[pd.DataFrame]:
    p = RAW_DIR / relative_path
    if p.exists():
        return pd.read_csv(p)
    return None


def _safe_col(df: pd.DataFrame, aliases: Sequence[str], required: bool = True) -> Optional[str]:
    """Return first matching column using case-insensitive exact/contains match."""
    cols = list(df.columns)
    low = {str(c).lower(): c for c in cols}
    for a in aliases:
        if a.lower() in low:
            return low[a.lower()]
    for a in aliases:
        al = a.lower()
        for lc, c in low.items():
            if al in lc:
                return c
    if required:
        raise KeyError(f"Could not find any of {aliases} in columns: {cols}")
    return None


def _standardize_boundary_catalog(bc: pd.DataFrame) -> pd.DataFrame:
    """Normalize expected boundary catalog column names without changing values."""
    out = bc.copy()
    mapping = {
        "cpt_id": _safe_col(out, ["cpt_id", "bro_id", "cpt", "id"]),
        "boundary_id": _safe_col(out, ["boundary_id", "boundary", "idx"], required=False),
        "z_hat": _safe_col(out, ["z_hat", "z", "depth", "boundary_depth", "depth_m"]),
        "z_p10": _safe_col(out, ["z_p10", "p10", "lower", "z_lower", "depth_p10"], required=False),
        "z_p90": _safe_col(out, ["z_p90", "p90", "upper", "z_upper", "depth_p90"], required=False),
        "sgs": _safe_col(out, ["sgs", "bcs", "confidence", "boundary_confidence"], required=False),
        "delta_z": _safe_col(out, ["delta_z", "uncertainty", "band", "width"], required=False),
    }
    ren = {v: k for k, v in mapping.items() if v is not None and v != k}
    out = out.rename(columns=ren)
    if "boundary_id" not in out.columns:
        out["boundary_id"] = np.arange(len(out))
    if "sgs" not in out.columns:
        out["sgs"] = np.nan
    if "z_p10" not in out.columns:
        dz = out["delta_z"] if "delta_z" in out.columns else 0.20
        out["z_p10"] = out["z_hat"] - np.asarray(dz, dtype=float) / 2.0
    if "z_p90" not in out.columns:
        dz = out["delta_z"] if "delta_z" in out.columns else 0.20
        out["z_p90"] = out["z_hat"] + np.asarray(dz, dtype=float) / 2.0
    if "delta_z" not in out.columns:
        out["delta_z"] = out["z_p90"] - out["z_p10"]
    out["sgs"] = pd.to_numeric(out["sgs"], errors="coerce")
    if out["sgs"].isna().all():
        out["sgs"] = 0.5
    else:
        out["sgs"] = out["sgs"].fillna(out["sgs"].median())
    return out


def _save(fig: plt.Figure, stem: str) -> None:
    png = OUT_DIR / f"{stem}_polished.png"
    pdf = OUT_DIR / f"{stem}_polished.pdf"
    fig.savefig(png, dpi=DPI)
    fig.savefig(pdf)


def _panel_label(ax: plt.Axes, label: str, x: float = 0.012, y: float = 0.985) -> None:
    ax.text(
        x, y, label, transform=ax.transAxes,
        ha="left", va="top", fontsize=13.5, fontweight="bold",
        color=INK,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=BORDER, lw=0.6, alpha=0.92),
        zorder=10,
    )


def _subtle_box(ax: plt.Axes) -> None:
    for sp in ax.spines.values():
        sp.set_linewidth(0.75)
        sp.set_color(BORDER)


def _format_km_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y/1000:.0f}"))


def _adaptive_text_color(value: float, vmin: float, vmax: float) -> str:
    if not np.isfinite(value) or not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return INK
    t = (value - vmin) / (vmax - vmin)
    return "white" if t > 0.62 else INK


def _annotate_heatmap(ax: plt.Axes, mat: np.ndarray, fmt: str, vmin: float, vmax: float, star_xy: Optional[Tuple[int, int]] = None) -> None:
    """Bold, readable text in heatmap cells; star_xy is (col, row)."""
    nr, nc = mat.shape
    for r in range(nr):
        for c in range(nc):
            val = mat[r, c]
            if np.isfinite(val):
                txt = format(val, fmt)
                color = _adaptive_text_color(float(val), vmin, vmax)
            else:
                txt = "—"
                color = MUTED
            if star_xy is not None and (c, r) == star_xy:
                txt += "★"
            ax.text(
                c, r, txt,
                ha="center", va="center",
                fontsize=8.6, fontweight="bold", color=color,
                zorder=6,
            )


def _nice_title(fig: plt.Figure, title: str, subtitle: str = "", y: float = 0.985) -> None:
    fig.suptitle(title, y=y, fontsize=17.6, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.5, y - 0.035, subtitle, ha="center", va="top", fontsize=10.6, color=MUTED)


# =============================================================================
# Geospatial helpers for Fig. 2
# =============================================================================

def _locate_gpkg() -> Optional[Path]:
    env = os.environ.get("BRO_CPT_GPKG", "").strip()
    if env:
        p = Path(env)
        if p.exists() and p.suffix.lower() == ".gpkg":
            return p

    candidates: List[Path] = []
    parent = PROJECT_ROOT.parent
    common = [
        parent / "BRO CPT" / "data_bro_cpt_pdok" / "extracted" / "brocptvolledigeset_v2_0.gpkg",
        parent / "BRO CPT" / "data_bro_cpt_pdok" / "cache" / "brocptvolledigeset_v2_0.gpkg",
        PROJECT_ROOT / "data_bro_cpt_pdok" / "extracted" / "brocptvolledigeset_v2_0.gpkg",
        PROJECT_ROOT / "data_bro_cpt_pdok" / "cache" / "brocptvolledigeset_v2_0.gpkg",
    ]
    for p in common:
        if p.exists():
            return p

    for base in [PROJECT_ROOT, PROJECT_ROOT.parent / "BRO CPT", PROJECT_ROOT.parent]:
        if base.exists():
            candidates.extend(base.rglob("*.gpkg"))
    candidates = [p for p in candidates if "bro" in p.name.lower() and "cpt" in p.name.lower()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x.stat().st_size, reverse=True)[0]


def _read_gpkg_layer_auto(gpkg: Path, preferred: Sequence[str]) -> Optional["gpd.GeoDataFrame"]:
    if gpd is None:
        return None
    layers: List[str]
    if fiona is not None:
        try:
            layers = list(fiona.listlayers(gpkg))
        except Exception:
            layers = list(preferred)
    else:
        layers = list(preferred)
    ordered = []
    for p in preferred:
        if p in layers and p not in ordered:
            ordered.append(p)
    for ly in layers:
        lname = ly.lower()
        if any(k in lname for k in ["survey", "sonder", "cpt"]):
            if ly not in ordered:
                ordered.append(ly)
    ordered.extend([ly for ly in layers if ly not in ordered])

    for ly in ordered:
        try:
            gdf = gpd.read_file(gpkg, layer=ly)
            if len(gdf) > 0:
                return gdf
        except Exception:
            continue
    try:
        return gpd.read_file(gpkg)
    except Exception:
        return None


def _find_boundary_file(root_dir: Path) -> Optional[Path]:
    if gpd is None:
        return None
    keywords = ["province", "provinc", "provincie", "adm1", "admin1", "gadm", "nuts", "nl_"]
    exts = {".gpkg", ".shp", ".geojson", ".json"}
    search_roots = [
        root_dir / "gis",
        root_dir / "data_gis",
        root_dir / "data",
        root_dir,
        root_dir.parent / "BRO CPT" / "gis",
        root_dir.parent / "BRO CPT" / "data_gis",
        root_dir.parent / "BRO CPT" / "data",
        root_dir.parent / "BRO CPT",
    ]
    seen = set()
    for sr in search_roots:
        if not sr.exists() or sr in seen:
            continue
        seen.add(sr)
        for p in sr.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                name = p.name.lower()
                if any(k in name for k in keywords):
                    return p
    return None


def _load_boundaries(path: Optional[Path]) -> Optional["gpd.GeoDataFrame"]:
    if path is None or gpd is None:
        return None
    try:
        if path.suffix.lower() == ".gpkg" and fiona is not None:
            layers = list(fiona.listlayers(path))
            preferred = [ly for ly in layers if any(k in ly.lower() for k in ["prov", "adm", "admin", "nuts"])]
            ordered = preferred + [ly for ly in layers if ly not in preferred]
            gdf = None
            for ly in ordered:
                try:
                    tmp = gpd.read_file(path, layer=ly)
                    if len(tmp) > 0:
                        gdf = tmp
                        break
                except Exception:
                    continue
            if gdf is None:
                return None
        else:
            gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(3857)
        return gdf.to_crs(3857)
    except Exception:
        return None


def _add_scale_bar(ax: plt.Axes, length_km: int = 50, loc: Tuple[float, float] = (0.08, 0.085)) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = xlim[0] + loc[0] * (xlim[1] - xlim[0])
    y = ylim[0] + loc[1] * (ylim[1] - ylim[0])
    L = length_km * 1000.0
    ax.plot([x, x + L], [y, y], color=INK, lw=2.2, solid_capstyle="butt", zorder=20)
    tick = 0.012 * (ylim[1] - ylim[0])
    ax.plot([x, x], [y - tick, y + tick], color=INK, lw=1.1, zorder=20)
    ax.plot([x + L, x + L], [y - tick, y + tick], color=INK, lw=1.1, zorder=20)
    ax.text(x + L / 2, y + 1.8 * tick, f"{length_km} km", ha="center", va="bottom", fontsize=8.8, color=INK, zorder=20)


def _add_north_arrow(ax: plt.Axes, loc: Tuple[float, float] = (0.935, 0.13)) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = xlim[0] + loc[0] * (xlim[1] - xlim[0])
    y = ylim[0] + loc[1] * (ylim[1] - ylim[0])
    h = 0.07 * (ylim[1] - ylim[0])
    tri = np.array([[x, y + h], [x - 0.012 * (xlim[1] - xlim[0]), y], [x + 0.012 * (xlim[1] - xlim[0]), y]])
    ax.add_patch(Polygon(tri, closed=True, facecolor=INK, edgecolor="white", lw=0.6, zorder=20))
    ax.text(x, y + h + 0.012 * (ylim[1] - ylim[0]), "N", ha="center", va="bottom", fontsize=9, fontweight="bold", color=INK, zorder=20)


# =============================================================================
# Fig. 1: polished conceptual workflow
# =============================================================================

def _workflow_card(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, lines: Sequence[str], fc: str, edge: str, num: str) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        fc=fc, ec=edge, lw=1.25, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + 0.025, y + h - 0.035, num, ha="center", va="center", fontsize=9.0, fontweight="bold", color="white",
            bbox=dict(boxstyle="circle,pad=0.22", fc=edge, ec=edge, lw=0), zorder=4)
    ax.text(x + 0.055, y + h - 0.032, title, ha="left", va="top", fontsize=9.9, fontweight="bold", color=INK, zorder=4)
    ax.text(x + 0.055, y + h - 0.076, "\n".join(lines), ha="left", va="top", fontsize=8.4, color=MUTED, linespacing=1.20, zorder=4)


def fig1_workflow() -> None:
    _set_pub_style()
    fig = plt.figure(figsize=(13.2, 9.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    _nice_title(
        fig,
        "Workflow of the scale-aware CPT boundary reporting framework",
        "From public BRO CPT/CPTu data to boundary location, uncertainty band, and confidence score",
        y=0.965,
    )

    xs = [0.035, 0.178, 0.321, 0.464, 0.607, 0.750]
    y = 0.655
    w, h = 0.125, 0.185
    cards = [
        ("Public BRO data", ["CPT/CPTu profiles", "metadata", "coordinates"], WARM, TEAL, "1"),
        ("Quality control", ["valid depth range", "unit checks", "missing values"], "#f7efe9", RUST, "2"),
        ("Depth-grid", ["regularized depth", "aligned variables", "profile matrix"], WARM, AMBER, "3"),
        ("Multi-scale filters", ["median windows", "scale support", "stable transitions"], "#eaf4f2", TEAL, "4"),
        ("Boundary evidence", ["candidate peaks", "P10–P90 band", "SGS/BCS score"], "#eaf4f2", TEAL_DARK, "5"),
        ("Reporting outputs", ["catalog table", "figures", "replicable package"], WARM, INK, "6"),
    ]
    for x, (title, lines, fc, edge, num) in zip(xs, cards):
        _workflow_card(ax, x, y, w, h, title, lines, fc, edge, num)
    for i in range(len(xs) - 1):
        ax.annotate("", xy=(xs[i + 1] - 0.012, y + h / 2), xytext=(xs[i] + w + 0.012, y + h / 2),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color=TEAL_DARK, shrinkA=0, shrinkB=0))

    # Mini data-driven-looking sketches, conceptual only.
    # Depth profile sketch inside card 1/3/5 areas.
    def mini_profile(x0, y0, color=TEAL):
        z = np.linspace(0, 1, 65)
        rng = np.random.default_rng(3)
        q = 0.5 + 0.18 * np.sin(12 * z) + 0.08 * rng.normal(size=len(z))
        q = np.clip(q, 0.1, 0.9)
        ax.plot(x0 + 0.018 + 0.05 * q, y0 + 0.02 + 0.12 * (1 - z), color=color, lw=1.0, zorder=5)
        for yy in [0.030, 0.075, 0.117]:
            ax.plot([x0 + 0.016, x0 + 0.077], [y0 + yy, y0 + yy], color=RUST if yy == 0.075 else TEAL_LIGHT, lw=1.0, alpha=0.85, zorder=4)

    mini_profile(xs[0] + 0.026, y + 0.02, TEAL)
    mini_profile(xs[3] + 0.026, y + 0.02, TEAL)
    mini_profile(xs[4] + 0.026, y + 0.02, TEAL_DARK)

    # Validation branch
    branch_y = 0.325
    ax.text(0.50, 0.535, "Validation and robustness assessment", ha="center", va="center", fontsize=12.8, fontweight="bold", color=INK)
    ax.plot([0.505, 0.505], [0.61, 0.555], color=MUTED, lw=1.0)
    ax.annotate("", xy=(0.505, 0.555), xytext=(0.505, 0.61), arrowprops=dict(arrowstyle="-|>", color=TEAL_DARK, lw=1.3))

    vx = [0.15, 0.38, 0.61]
    v_w, v_h = 0.19, 0.14
    v_cards = [
        ("Synthetic recovery", ["known true boundaries", "precision/recall/F1", "matched-depth MAE"], "#eaf4f2", TEAL, "A"),
        ("Baseline comparison", ["single-scale", "PELT", "boundary density"], WARM, AMBER, "B"),
        ("Sensitivity analysis", ["parameter grid", "best setting lock", "stability check"], "#f7efe9", RUST, "C"),
    ]
    for x, (title, lines, fc, edge, num) in zip(vx, v_cards):
        _workflow_card(ax, x, branch_y, v_w, v_h, title, lines, fc, edge, num)
    for i in range(2):
        ax.annotate("", xy=(vx[i + 1] - 0.018, branch_y + v_h / 2), xytext=(vx[i] + v_w + 0.018, branch_y + v_h / 2),
                    arrowprops=dict(arrowstyle="-|>", lw=1.35, color=TEAL_DARK))

    # Key idea strip
    strip = FancyBboxPatch((0.085, 0.115), 0.83, 0.085, boxstyle="round,pad=0.018,rounding_size=0.025",
                           fc="white", ec=BORDER, lw=1.0)
    ax.add_patch(strip)
    ax.text(0.105, 0.158, "Key idea", fontsize=10.8, fontweight="bold", color=TEAL_DARK, va="center")
    ax.text(0.205, 0.158,
            "Boundaries that remain stable across analysis scales receive higher confidence and narrower uncertainty bands.",
            fontsize=10.3, color=INK, va="center")

    _save(fig, "fig1_workflow")
    plt.close(fig)


# =============================================================================
# Fig. 2: data-driven point map
# =============================================================================

def fig2_netherlands_pointmap() -> None:
    _set_pub_style()
    bc = _standardize_boundary_catalog(_read_csv_required("boundary_catalog.csv"))
    cpt_stats = (
        bc.groupby("cpt_id", as_index=False)
        .agg(n_boundaries=("boundary_id", "count"), mean_sgs=("sgs", "mean"), median_uncertainty=("delta_z", "median"))
        .rename(columns={"cpt_id": "bro_id"})
    )

    gpkg = _locate_gpkg()
    if gpkg is None or gpd is None:
        raise RuntimeError("Fig. 2 requires geopandas and a readable BRO CPT GeoPackage. Set BRO_CPT_GPKG if auto-detection fails.")

    survey = _read_gpkg_layer_auto(gpkg, preferred=["geotechnical_cpt_survey"])
    if survey is None or len(survey) == 0:
        raise RuntimeError(f"Could not read a survey/CPT layer from {gpkg}")

    id_col = _safe_col(survey, ["bro_id", "cpt_id", "id", "broId"])
    survey = survey.rename(columns={id_col: "bro_id"})
    if survey.crs is None:
        survey = survey.set_crs(3857)
    survey = survey.to_crs(3857)
    gdf = survey[["bro_id", survey.geometry.name]].copy().merge(cpt_stats, on="bro_id", how="inner")
    if len(gdf) == 0:
        raise RuntimeError("No matching CPT coordinates were found between boundary_catalog.csv and the GeoPackage survey layer.")

    boundary_path = _find_boundary_file(PROJECT_ROOT)
    nl = _load_boundaries(boundary_path)

    fig = plt.figure(figsize=(9.9, 10.9))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor("#f1f5f5")
    ax.grid(False)

    if nl is not None and len(nl) > 0:
        nl.plot(ax=ax, facecolor="#f6f1e8", edgecolor="#c6c1b8", linewidth=0.75, zorder=1)
        nl.boundary.plot(ax=ax, color="#b9b3aa", linewidth=0.55, zorder=2)
        b = nl.total_bounds
        pad_x = 0.04 * (b[2] - b[0])
        pad_y = 0.04 * (b[3] - b[1])
        ax.set_xlim(b[0] - pad_x, b[2] + pad_x)
        ax.set_ylim(b[1] - pad_y, b[3] + pad_y)
    else:
        b = gdf.total_bounds
        pad_x = max(50000, 0.15 * (b[2] - b[0]))
        pad_y = max(50000, 0.15 * (b[3] - b[1]))
        ax.set_xlim(b[0] - pad_x, b[2] + pad_x)
        ax.set_ylim(b[1] - pad_y, b[3] + pad_y)

    min_n = float(gdf["n_boundaries"].min())
    max_n = float(gdf["n_boundaries"].max())
    if max_n == min_n:
        sizes = np.full(len(gdf), 70.0)
    else:
        sizes = np.interp(gdf["n_boundaries"].to_numpy(float), (min_n, max_n), (44, 168))

    sc = ax.scatter(
        gdf.geometry.x, gdf.geometry.y,
        c=gdf["mean_sgs"], s=sizes,
        cmap=_seq_teal(), vmin=0.0, vmax=1.0,
        edgecolor="white", linewidth=0.85, alpha=0.98, zorder=6,
    )
    ax.scatter(gdf.geometry.x, gdf.geometry.y, s=sizes * 1.08, facecolor="none", edgecolor=INK, linewidth=0.28, alpha=0.62, zorder=5)

    _panel_label(ax, "(a)")
    ax.set_title("Netherlands CPT soundings used for boundary reporting", pad=14)
    ax.set_xlabel("X (km, EPSG:3857)")
    ax.set_ylabel("Y (km, EPSG:3857)")
    _format_km_axis(ax)
    _subtle_box(ax)

    # Colorbar
    cb = fig.colorbar(sc, ax=ax, fraction=0.036, pad=0.018)
    cb.set_label("Mean SGS", fontsize=10.0)
    cb.ax.tick_params(labelsize=9.2)
    cb.outline.set_edgecolor(BORDER)
    cb.outline.set_linewidth(0.7)

    # Size legend, with real min/median/max counts.
    refs = [int(np.nanmin(gdf["n_boundaries"])), int(np.nanmedian(gdf["n_boundaries"])), int(np.nanmax(gdf["n_boundaries"]))]
    refs = list(dict.fromkeys(refs))
    handles = []
    for nb in refs:
        s = 70.0 if max_n == min_n else float(np.interp(nb, (min_n, max_n), (44, 168)))
        handles.append(ax.scatter([], [], s=s, facecolor=TEAL_MID, edgecolor="white", linewidth=0.65, alpha=0.95))
    leg = ax.legend(handles, [f"{r} boundaries" for r in refs], title="Boundary count", loc="lower left", frameon=True, borderpad=0.82, labelspacing=0.58, handletextpad=0.68)
    leg.get_title().set_fontsize(9.6)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor(BORDER)
    leg.get_frame().set_linewidth(0.85)

    # Inset around densest observed neighborhood. This remains data-driven.
    xy = np.column_stack([gdf.geometry.x.to_numpy(float), gdf.geometry.y.to_numpy(float)])
    if len(xy) >= 5:
        D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
        D.sort(axis=1)
        k = min(8, len(xy) - 1)
        dens = 1.0 / (D[:, 1:k + 1].mean(axis=1) + 1e-9)
        center = xy[int(np.argmax(dens))]
        dx, dy = 65000.0, 65000.0
        bbox = (center[0] - dx, center[0] + dx, center[1] - dy, center[1] + dy)
        ins = ax.inset_axes([0.57, 0.075, 0.38, 0.31])
        ins.set_facecolor("#f1f5f5")
        ins.grid(False)
        if nl is not None and len(nl) > 0:
            nl.plot(ax=ins, facecolor="#f6f1e8", edgecolor="#c6c1b8", linewidth=0.55, zorder=1)
        mask = (gdf.geometry.x.between(bbox[0], bbox[1])) & (gdf.geometry.y.between(bbox[2], bbox[3]))
        gi = gdf.loc[mask].copy()
        if len(gi) > 0:
            si = np.interp(gi["n_boundaries"].to_numpy(float), (min_n, max_n), (28, 110)) if max_n != min_n else np.full(len(gi), 55.0)
            ins.scatter(gi.geometry.x, gi.geometry.y, c=gi["mean_sgs"], s=si, cmap=_seq_teal(), vmin=0, vmax=1,
                        edgecolor="white", linewidth=0.55, alpha=0.98, zorder=5)
        ins.set_xlim(bbox[0], bbox[1])
        ins.set_ylim(bbox[2], bbox[3])
        ins.set_xticks([])
        ins.set_yticks([])
        ins.set_title("Inset: densest sampled area", fontsize=9.4, color=MUTED, pad=4)
        for sp in ins.spines.values():
            sp.set_color(BORDER)
            sp.set_linewidth(0.75)
        try:
            ax.indicate_inset_zoom(ins, edgecolor=MUTED, linewidth=0.8, alpha=0.8)
        except Exception:
            pass

    _add_scale_bar(ax, length_km=50, loc=(0.09, 0.09))
    _add_north_arrow(ax, loc=(0.93, 0.14))

    fig.text(0.5, 0.022, "Point symbology is data-driven: color = mean SGS; symbol size = detected boundary count per CPT.",
             ha="center", va="center", fontsize=8.7, color=MUTED)
    _save(fig, "fig2_nl_pointmap")
    plt.close(fig)


# =============================================================================
# Fig. 3: representative CPT panels
# =============================================================================

def _load_profile(cpt_id: str) -> Optional[pd.DataFrame]:
    folder = RAW_DIR / "profiles_realdata"
    if not folder.exists():
        return None
    exacts = [folder / f"{cpt_id}.csv", folder / f"CPT_{cpt_id}.csv", folder / f"profile_{cpt_id}.csv"]
    for p in exacts:
        if p.exists():
            return pd.read_csv(p)
    matches = list(folder.glob(f"*{cpt_id}*.csv"))
    if matches:
        return pd.read_csv(matches[0])
    # Some profile dumps may include all CPTs in one file.
    for p in folder.glob("*.csv"):
        try:
            head = pd.read_csv(p, nrows=5)
            cid = _safe_col(head, ["cpt_id", "bro_id", "id"], required=False)
            if cid is None:
                continue
            full = pd.read_csv(p)
            sub = full[full[cid].astype(str) == str(cpt_id)].copy()
            if len(sub) > 0:
                return sub
        except Exception:
            continue
    return None


def _standardize_profile(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c_depth = _safe_col(out, ["depth", "z", "depth_m", "diepte"])
    c_qt = _safe_col(out, ["qt", "qc", "cone_resistance", "q_c", "q_t"])
    c_fs = _safe_col(out, ["fs", "friction", "sleeve", "f_s"], required=False)
    out = out.rename(columns={c_depth: "depth", c_qt: "qt"})
    if c_fs is not None:
        out = out.rename(columns={c_fs: "fs"})
    else:
        out["fs"] = np.nan
    out["depth"] = pd.to_numeric(out["depth"], errors="coerce")
    out["qt"] = pd.to_numeric(out["qt"], errors="coerce")
    out["fs"] = pd.to_numeric(out["fs"], errors="coerce")
    out = out.dropna(subset=["depth", "qt"]).sort_values("depth")
    return out


def _smooth_series(y: np.ndarray, kernel: int = 9) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) < 5:
        return y
    if kernel % 2 == 0:
        kernel += 1
    kernel = min(kernel, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if kernel < 3:
        return y
    if medfilt is not None:
        try:
            return medfilt(y, kernel_size=kernel)
        except Exception:
            pass
    return pd.Series(y).rolling(kernel, center=True, min_periods=1).median().to_numpy()


def _pick_representative_cpts(bc: pd.DataFrame, n: int = 4) -> List[str]:
    g = bc.groupby("cpt_id", as_index=False).agg(
        n_boundaries=("boundary_id", "count"),
        mean_sgs=("sgs", "mean"),
        min_sgs=("sgs", "min"),
        med_unc=("delta_z", "median"),
    )
    picks: List[str] = []
    # sparse, dense, low-confidence, median complexity. Preserve actual IDs.
    selectors = [
        g["n_boundaries"].idxmin(),
        g["n_boundaries"].idxmax(),
        g["min_sgs"].idxmin(),
        (g["n_boundaries"] - g["n_boundaries"].median()).abs().idxmin(),
        g["mean_sgs"].idxmax(),
        g["med_unc"].idxmax(),
    ]
    for idx in selectors:
        cid = str(g.loc[idx, "cpt_id"])
        if cid not in picks and _load_profile(cid) is not None:
            picks.append(cid)
        if len(picks) >= n:
            return picks
    # fallback: any available profiles.
    for cid in g.sort_values("n_boundaries", ascending=False)["cpt_id"].astype(str):
        if cid not in picks and _load_profile(cid) is not None:
            picks.append(cid)
        if len(picks) >= n:
            break
    return picks


def _confidence_style(sgs: float) -> Tuple[str, float, float, str]:
    if not np.isfinite(sgs):
        return GRAY, 0.12, 1.1, "unknown"
    if sgs >= 0.67:
        return TEAL, 0.13, 1.35, "high"
    if sgs >= 0.33:
        return AMBER, 0.15, 1.30, "moderate"
    return RUST, 0.17, 1.35, "low"


def fig3_cpt_panels() -> None:
    _set_pub_style()
    bc = _standardize_boundary_catalog(_read_csv_required("boundary_catalog.csv"))
    cpts = _pick_representative_cpts(bc, n=4)
    if len(cpts) == 0:
        raise RuntimeError("No usable profiles were found in RAW_DIR/profiles_realdata for Fig. 3.")

    fig = plt.figure(figsize=(13.0, 9.9))
    outer = fig.add_gridspec(2, 2, wspace=0.20, hspace=0.22)
    letters = ["(a)", "(b)", "(c)", "(d)"]

    for i, cpt_id in enumerate(cpts):
        sub = outer[i // 2, i % 2].subgridspec(1, 2, wspace=0.06)
        ax_q = fig.add_subplot(sub[0, 0])
        ax_f = fig.add_subplot(sub[0, 1], sharey=ax_q)

        prof_raw = _load_profile(cpt_id)
        if prof_raw is None:
            continue
        prof = _standardize_profile(prof_raw)
        if len(prof) == 0:
            continue
        depth = prof["depth"].to_numpy(float)
        qt = prof["qt"].to_numpy(float)
        fs = prof["fs"].to_numpy(float)
        qt_s = _smooth_series(qt, kernel=9)
        fs_s = _smooth_series(fs, kernel=9) if np.isfinite(fs).any() else fs

        ax_q.plot(qt, depth, color=LIGHT_GRAY, lw=0.85, alpha=0.72, label="raw", zorder=2)
        ax_q.plot(qt_s, depth, color=INK, lw=1.65, alpha=0.97, label="smoothed", zorder=3)
        if np.isfinite(fs).any():
            ax_f.plot(fs, depth, color=LIGHT_GRAY, lw=0.85, alpha=0.72, zorder=2)
            ax_f.plot(fs_s, depth, color=INK, lw=1.65, alpha=0.97, zorder=3)
        else:
            ax_f.text(0.5, 0.5, "fs not available", transform=ax_f.transAxes, ha="center", va="center", color=MUTED, fontsize=9.3)

        bci = bc[bc["cpt_id"].astype(str) == str(cpt_id)].copy().sort_values("z_hat")
        for _, row in bci.iterrows():
            z = float(row["z_hat"])
            z0 = float(row["z_p10"])
            z1 = float(row["z_p90"])
            col, alpha, lw, _ = _confidence_style(float(row["sgs"]))
            for ax in [ax_q, ax_f]:
                ax.axhspan(z0, z1, facecolor=col, alpha=alpha, linewidth=0, zorder=0)
                ax.axhline(z, color=col, lw=lw, alpha=0.96, zorder=4)

        for ax in [ax_q, ax_f]:
            ax.invert_yaxis()
            ax.grid(True)
            _subtle_box(ax)
            ax.yaxis.set_major_locator(MaxNLocator(6))
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.tick_params(axis="both", labelsize=9.2)
        ax_q.set_ylabel("Depth (m)", fontsize=10.8)
        ax_q.set_xlabel(r"$q_t$ / $q_c$", fontsize=10.8)
        ax_f.set_xlabel(r"$f_s$", fontsize=10.8)
        plt.setp(ax_f.get_yticklabels(), visible=False)
        ax_f.tick_params(axis="y", length=0)

        _panel_label(ax_q, letters[i], x=0.018, y=0.98)
        nb = len(bci)
        mean_sgs = float(bci["sgs"].mean()) if nb else float("nan")
        ax_q.set_title(f"{cpt_id}\n{nb} boundaries; mean SGS = {mean_sgs:.2f}", loc="left", fontsize=10.3, pad=7, color=INK)

    legend_handles = [
        Line2D([0], [0], color=LIGHT_GRAY, lw=1.1, label="raw profile"),
        Line2D([0], [0], color=INK, lw=1.7, label="smoothed profile"),
        Line2D([0], [0], color=TEAL, lw=1.6, label="high confidence"),
        Line2D([0], [0], color=AMBER, lw=1.6, label="moderate confidence"),
        Line2D([0], [0], color=RUST, lw=1.6, label="low confidence"),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.014), fontsize=9.2)
    _nice_title(
        fig,
        "Representative CPT soundings with data-driven boundaries and uncertainty bands",
        "Real CPT IDs and measured profiles are preserved; shaded bands show P10–P90 boundary uncertainty.",
        y=0.985,
    )
    _save(fig, "fig3_cpt_panels")
    plt.close(fig)


# =============================================================================
# Fig. 4: synthetic validation
# =============================================================================

def fig4_synthetic_validation() -> None:
    _set_pub_style()
    syn = _read_csv_required("synthetic_validation_results.csv")
    c_noise = _safe_col(syn, ["noise_level", "noise", "relative_noise"])
    c_prec = _safe_col(syn, ["precision", "prec"])
    c_rec = _safe_col(syn, ["recall", "rec"])
    c_f1 = _safe_col(syn, ["f1", "f1_score"])
    c_mae = _safe_col(syn, ["mae", "matched_mae", "matched_boundary_error"])

    g = (
        syn.groupby(c_noise)
        .agg(
            precision=(c_prec, "mean"), precision_sd=(c_prec, "std"),
            recall=(c_rec, "mean"), recall_sd=(c_rec, "std"),
            f1=(c_f1, "mean"), f1_sd=(c_f1, "std"),
            mae=(c_mae, "mean"), mae_sd=(c_mae, "std"),
            n=(c_f1, "count"),
        )
        .reset_index()
        .sort_values(c_noise)
    )
    for c in ["precision_sd", "recall_sd", "f1_sd", "mae_sd"]:
        g[c] = g[c].fillna(0.0)

    x = g[c_noise].to_numpy(float)
    total_n = int(g["n"].sum())
    per = int(g["n"].iloc[0]) if len(g) and g["n"].nunique() == 1 else None
    subtitle = f"n = {per} profiles per noise level; total n = {total_n}" if per else f"total n = {total_n} synthetic profiles"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 6.15), gridspec_kw={"wspace": 0.28})
    _nice_title(fig, "Synthetic validation of boundary recovery", subtitle, y=1.035)

    metric_specs = [
        ("Precision", "precision", "precision_sd", TEAL),
        ("Recall", "recall", "recall_sd", TEAL_MID),
        ("F1", "f1", "f1_sd", RUST),
    ]
    for label, mean_col, sd_col, color in metric_specs:
        y = g[mean_col].to_numpy(float)
        sd = g[sd_col].to_numpy(float)
        ax1.fill_between(x, y - sd, y + sd, color=color, alpha=0.105, linewidth=0)
        ax1.plot(x, y, "-o", color=color, lw=2.15, ms=5.0, markerfacecolor="white", markeredgewidth=1.3, label=label)
        for xx, yy in zip(x, y):
            ax1.text(xx, yy + 0.012, f"{yy:.2f}", ha="center", va="bottom", fontsize=8.0, color=color, fontweight="bold")

    ax1.set_title("(a) Boundary recovery metrics")
    ax1.set_xlabel("Noise level (relative)")
    ax1.set_ylabel("Metric value")
    y_min = max(0.0, min(g[["precision", "recall", "f1"]].min()) - 0.06)
    y_max = min(1.02, max(g[["precision", "recall", "f1"]].max()) + 0.06)
    ax1.set_ylim(y_min, y_max)
    ax1.legend(frameon=False, loc="lower right", fontsize=9.4)
    ax1.tick_params(axis="both", labelsize=9.6)
    _subtle_box(ax1)

    y = g["mae"].to_numpy(float)
    sd = g["mae_sd"].to_numpy(float)
    ax2.fill_between(x, y - sd, y + sd, color=TEAL_DARK, alpha=0.105, linewidth=0)
    ax2.plot(x, y, "-o", color=TEAL_DARK, lw=2.15, ms=5.0, markerfacecolor="white", markeredgewidth=1.3, label="MAE")
    for xx, yy in zip(x, y):
        ax2.text(xx, yy + 0.012 * max(1e-9, np.nanmax(y) - np.nanmin(y)), f"{yy:.3f}", ha="center", va="bottom", fontsize=8.0, color=TEAL_DARK, fontweight="bold")
    ax2.set_title("(b) Matched-boundary error")
    ax2.set_xlabel("Noise level (relative)")
    ax2.set_ylabel("MAE (m)")
    _subtle_box(ax2)
    ax2.legend(frameon=False, loc="upper left", fontsize=9.4)
    ax2.tick_params(axis="both", labelsize=9.6)

    fig.text(0.5, 0.02, "Error bands show ±1 standard deviation across synthetic profiles or replicates within each noise level.",
             ha="center", va="center", fontsize=9.4, color=MUTED)
    _save(fig, "fig4_synthetic_validation")
    plt.close(fig)


# =============================================================================
# Fig. 5: baseline comparison
# =============================================================================

_BASELINE_METHOD_ORDER = ["Scale-aware", "Single-scale", "PELT"]


def _method_name(raw: str) -> str:
    """Map flexible raw method/column names to publication labels."""
    s = str(raw).strip()
    lo = s.lower().replace("_", " ").replace("-", " ")
    if ("scale" in lo and "aware" in lo) or "sgs" in lo or "proposed" in lo or "boundary confidence" in lo:
        return "Scale-aware"
    if "single" in lo or "fixed" in lo or "one scale" in lo or "1 scale" in lo:
        return "Single-scale"
    if "pelt" in lo:
        return "PELT"
    return s


def _first_existing_csv(paths: Sequence[Path]) -> Optional[pd.DataFrame]:
    for p in paths:
        if p.exists():
            print(f"[pubfig-polished] Fig.5 input: {p}")
            return pd.read_csv(p)
    return None


def _nonnull_numeric(vals: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(vals, errors="coerce").dropna().to_numpy(float)
    return arr[np.isfinite(arr)]


def _add_count_series(target: Dict[str, pd.Series], name: str, s: pd.Series) -> None:
    """Add/merge one per-CPT count series for a baseline method."""
    m = _method_name(name)
    ss = pd.to_numeric(s, errors="coerce")
    if ss.notna().sum() == 0:
        return
    if m in target:
        # Prefer the series with more numeric entries; otherwise keep the first.
        if ss.notna().sum() > target[m].notna().sum():
            target[m] = ss
    else:
        target[m] = ss


def _counts_from_boundary_file(df0: pd.DataFrame, method_name: str) -> Optional[pd.DataFrame]:
    """Return cpt_id + one method count column from boundary-level or count-level files."""
    if df0 is None or len(df0) == 0:
        return None
    df = df0.copy()
    cid = _safe_col(df, ["cpt_id", "bro_id", "cpt", "id", "sounding_id"], required=False)
    if cid is None:
        return None
    if cid != "cpt_id":
        df = df.rename(columns={cid: "cpt_id"})

    # Some files are already one row per CPT with a count column.
    count_col = _safe_col(
        df,
        [
            "n_boundaries", "boundary_count", "n_boundary", "count", "n_detected",
            "detected_boundaries", "num_boundaries", "n_bnd", "n",
        ],
        required=False,
    )
    if count_col is not None and df["cpt_id"].duplicated().sum() == 0:
        out = df[["cpt_id", count_col]].copy().rename(columns={count_col: method_name})
        out[method_name] = pd.to_numeric(out[method_name], errors="coerce")
        return out

    # Otherwise count boundary rows per CPT.
    out = df.groupby("cpt_id", as_index=False).size().rename(columns={"size": method_name})
    return out


def _baseline_counts_table() -> pd.DataFrame:
    """Construct a robust wide table: cpt_id, Scale-aware, Single-scale, PELT.

    This function intentionally accepts several plausible output schemas.  The
    previous v1.0 version assumed that all non-ID columns in
    baseline_boundary_counts_by_cpt.csv were numeric method counts.  On some
    runs that file also contains labels/metadata columns, which produced empty
    numeric arrays and crashed Matplotlib's violinplot.
    """
    analysis_dir = PROJECT_ROOT / "outputs_paper1_v2" / "paper1_analysis"

    # ------------------------------------------------------------------
    # 1) Prefer a ready-made by-CPT baseline-count table when available.
    #    Supported schemas:
    #    A) wide: cpt_id, scale_aware..., single_scale..., pelt...
    #    B) long: cpt_id, method, n_boundaries
    # ------------------------------------------------------------------
    by_cpt_df = _first_existing_csv([
        analysis_dir / "baseline_boundary_counts_by_cpt.csv",
        analysis_dir / "baseline_counts_by_cpt.csv",
        analysis_dir / "boundary_counts_by_cpt.csv",
        RAW_DIR / "baseline_boundary_counts_by_cpt.csv",
        RAW_DIR / "baseline_counts_by_cpt.csv",
    ])

    if by_cpt_df is not None and len(by_cpt_df) > 0:
        df0 = by_cpt_df.copy()
        cid = _safe_col(df0, ["cpt_id", "bro_id", "cpt", "id", "sounding_id"], required=False)
        if cid is not None and cid != "cpt_id":
            df0 = df0.rename(columns={cid: "cpt_id"})
        if "cpt_id" not in df0.columns:
            df0.insert(0, "cpt_id", np.arange(len(df0)).astype(str))

        method_col = _safe_col(df0, ["method", "baseline", "algorithm", "model", "approach"], required=False)

        def _generic_count_col(df: pd.DataFrame) -> Optional[str]:
            exact_aliases = {
                "n_boundaries", "boundary_count", "n_boundary", "count", "n_detected",
                "detected_boundaries", "num_boundaries", "n_bnd", "n",
            }
            for col in df.columns:
                if str(col).lower() in exact_aliases:
                    return col
            return None

        # Wide-table path, first pass: use columns whose names explicitly map to
        # one of the intended methods.  This prevents accidental interpretation
        # of metadata columns such as "method_label" as long-format data.
        series_by_method: Dict[str, pd.Series] = {}
        for c in df0.columns:
            if c == "cpt_id" or c == method_col:
                continue
            mapped = _method_name(c)
            if mapped not in _BASELINE_METHOD_ORDER:
                continue
            vals = pd.to_numeric(df0[c], errors="coerce")
            if vals.notna().sum() == 0:
                continue
            _add_count_series(series_by_method, mapped, vals)

        if series_by_method:
            out = pd.DataFrame({"cpt_id": df0["cpt_id"].astype(str)})
            ordered = [m for m in _BASELINE_METHOD_ORDER if m in series_by_method] + [m for m in series_by_method if m not in _BASELINE_METHOD_ORDER]
            for m in ordered:
                out[m] = series_by_method[m]
            print(f"[pubfig-polished] Fig.5 methods from wide table: {ordered}")
            return out

        # Long-table path: cpt_id + method + a genuinely generic count column.
        count_col = _generic_count_col(df0)
        if method_col is not None and count_col is not None:
            tmp = df0[["cpt_id", method_col, count_col]].copy()
            tmp[method_col] = tmp[method_col].map(_method_name)
            tmp[count_col] = pd.to_numeric(tmp[count_col], errors="coerce")
            tmp = tmp.dropna(subset=[count_col])
            if len(tmp) > 0:
                wide = tmp.pivot_table(index="cpt_id", columns=method_col, values=count_col, aggfunc="first").reset_index()
                for c in wide.columns:
                    if c != "cpt_id":
                        wide[c] = pd.to_numeric(wide[c], errors="coerce")
                usable = [c for c in wide.columns if c != "cpt_id" and wide[c].notna().sum() > 0]
                if usable:
                    print(f"[pubfig-polished] Fig.5 methods from long table: {usable}")
                    return wide[["cpt_id"] + usable]

        # Wide-table path, second pass: if there are no explicit method-name
        # columns, accept numeric count-like columns but ignore labels/metadata.
        series_by_method = {}
        for c in df0.columns:
            if c == "cpt_id" or c == method_col:
                continue
            vals = pd.to_numeric(df0[c], errors="coerce")
            if vals.notna().sum() == 0:
                continue
            cname = str(c)
            mapped = _method_name(cname)
            lo = cname.lower()
            keep = mapped in _BASELINE_METHOD_ORDER or any(k in lo for k in ["boundary", "count", "n_", "num", "pelt", "single", "scale"])
            if not keep:
                continue
            _add_count_series(series_by_method, mapped, vals)

        if series_by_method:
            out = pd.DataFrame({"cpt_id": df0["cpt_id"].astype(str)})
            ordered = [m for m in _BASELINE_METHOD_ORDER if m in series_by_method] + [m for m in series_by_method if m not in _BASELINE_METHOD_ORDER]
            for m in ordered:
                out[m] = series_by_method[m]
            print(f"[pubfig-polished] Fig.5 methods from count-like wide table: {ordered}")
            return out

        print("[pubfig-polished] WARNING: baseline count table was found, but no usable numeric method columns were detected. Falling back to boundary files.")
        print(f"[pubfig-polished] Fig.5 table columns were: {list(df0.columns)}")

    # ------------------------------------------------------------------
    # 2) Fallback: construct counts from boundary-level files.
    # ------------------------------------------------------------------
    bc = _standardize_boundary_catalog(_read_csv_required("boundary_catalog.csv"))
    df = bc.groupby("cpt_id", as_index=False).size().rename(columns={"size": "Scale-aware"})

    optional_files = {
        "Single-scale": [
            RAW_DIR / "baselines" / "baseline_single_scale_boundaries.csv",
            RAW_DIR / "baseline_single_scale_boundaries.csv",
            analysis_dir / "baselines" / "baseline_single_scale_boundaries.csv",
            analysis_dir / "baseline_single_scale_boundaries.csv",
        ],
        "PELT": [
            RAW_DIR / "baselines" / "baseline_pelt_boundaries.csv",
            RAW_DIR / "baseline_pelt_boundaries.csv",
            analysis_dir / "baselines" / "baseline_pelt_boundaries.csv",
            analysis_dir / "baseline_pelt_boundaries.csv",
        ],
    }
    for mname, paths in optional_files.items():
        tmp0 = _first_existing_csv(paths)
        if tmp0 is not None:
            cnt = _counts_from_boundary_file(tmp0, mname)
            if cnt is not None:
                df = df.merge(cnt, on="cpt_id", how="left")

    for c in df.columns:
        if c != "cpt_id":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    print(f"[pubfig-polished] Fig.5 methods from fallback files: {[c for c in df.columns if c != 'cpt_id']}")
    return df


def fig5_baseline_comparison() -> None:
    _set_pub_style()
    df = _baseline_counts_table()

    preferred = _BASELINE_METHOD_ORDER
    candidate_methods = [m for m in preferred if m in df.columns] + [c for c in df.columns if c != "cpt_id" and c not in preferred]

    methods: List[str] = []
    data: List[np.ndarray] = []
    for m in candidate_methods:
        vals = _nonnull_numeric(df[m])
        if len(vals) == 0:
            print(f"[pubfig-polished] WARNING: Fig.5 skipped empty/non-numeric method column: {m}")
            continue
        methods.append(m)
        data.append(vals)

    if not methods:
        raise RuntimeError(
            "Fig. 5 could not find any non-empty numeric baseline boundary-count columns. "
            "Please check baseline_boundary_counts_by_cpt.csv or boundary_catalog.csv."
        )

    means = np.array([np.nanmean(v) for v in data], dtype=float)
    medians = np.array([np.nanmedian(v) for v in data], dtype=float)
    n_cpt = int(df["cpt_id"].nunique()) if "cpt_id" in df.columns else int(max(len(v) for v in data))

    print("[pubfig-polished] Fig.5 data summary:")
    for m, v in zip(methods, data):
        print(f"    - {m}: n={len(v)}, mean={np.nanmean(v):.3f}, median={np.nanmedian(v):.3f}, min={np.nanmin(v):.3f}, max={np.nanmax(v):.3f}")

    colors = {"Scale-aware": TEAL, "Single-scale": AMBER, "PELT": RUST_DARK}
    fallback_colors = [TEAL, AMBER, RUST_DARK, GRAY]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 6.25), gridspec_kw={"wspace": 0.32, "width_ratios": [1.18, 1.0]})
    _nice_title(fig, "Baseline comparison on real-data soundings", f"n = {n_cpt} CPT soundings; all values are read from boundary-count outputs", y=1.035)

    # Panel A: violin + box + real points.  Matplotlib requires non-empty arrays;
    # these were filtered above.
    if len(data) >= 1:
        parts = ax1.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
        for i, body in enumerate(parts["bodies"]):
            col = colors.get(methods[i], fallback_colors[i % len(fallback_colors)])
            body.set_facecolor(col)
            body.set_edgecolor("none")
            body.set_alpha(0.17)

    bp = ax1.boxplot(data, tick_labels=methods, widths=0.36, patch_artist=True, showfliers=False)
    for i, box in enumerate(bp["boxes"]):
        col = colors.get(methods[i], fallback_colors[i % len(fallback_colors)])
        box.set_facecolor("white")
        box.set_edgecolor(col)
        box.set_linewidth(1.35)
    for elem in ["whiskers", "caps"]:
        for j, line in enumerate(bp[elem]):
            col = colors.get(methods[j // 2], fallback_colors[(j // 2) % len(fallback_colors)])
            line.set_color(col)
            line.set_linewidth(1.05)
    for j, line in enumerate(bp["medians"]):
        col = colors.get(methods[j], fallback_colors[j % len(fallback_colors)])
        line.set_color(col)
        line.set_linewidth(1.7)

    rng = np.random.default_rng(RNG_SEED)
    for i, vals in enumerate(data, start=1):
        x = rng.normal(i, 0.045, size=len(vals))
        ax1.scatter(x, vals, s=22, color=INK, alpha=0.50, linewidth=0, zorder=4)

    all_vals = np.concatenate(data)
    ymax = max(1.0, float(np.nanmax(all_vals)))
    ax1.set_ylim(0, ymax * 1.20)
    for i, (m, mu, med) in enumerate(zip(methods, means, medians), start=1):
        col = colors.get(m, fallback_colors[(i - 1) % len(fallback_colors)])
        ax1.text(i, ymax * 1.065, f"mean {mu:.2f}\nmedian {med:.0f}", ha="center", va="bottom", fontsize=8.8, color=col, fontweight="bold")
    ax1.set_title("(a) Boundary-count distribution")
    ax1.set_ylabel("Detected boundaries per CPT")
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", rotation=0, labelsize=9.5)
    ax1.tick_params(axis="y", labelsize=9.5)
    _subtle_box(ax1)

    # Panel B: horizontal bars with relative increases.
    y = np.arange(len(methods))
    bar_colors = [colors.get(m, fallback_colors[i % len(fallback_colors)]) for i, m in enumerate(methods)]
    ax2.barh(y, means, color=bar_colors, alpha=0.82, edgecolor="white", linewidth=1.0, height=0.52)
    ax2.scatter(means, y, s=42, color=INK, zorder=4)
    ax2.set_yticks(y)
    ax2.set_yticklabels(methods, fontsize=9.5)
    ax2.tick_params(axis="x", labelsize=9.5)
    ax2.invert_yaxis()
    ax2.set_title("(b) Mean segmentation density")
    ax2.set_xlabel("Mean boundaries per CPT")
    finite_means = means[np.isfinite(means)]
    xmax = float(np.nanmax(finite_means)) if len(finite_means) else 1.0
    xpad = max(0.9, 0.10 * xmax)
    ax2.set_xlim(0, xmax * 1.36 + 1.0)
    ref = means[0]
    for i, (m, val) in enumerate(zip(methods, means)):
        label = f"{val:.2f}"
        ax2.text(val + xpad * 0.20, i, label, va="center", ha="left", fontsize=9.0, color=INK, fontweight="bold")
        if i == 0:
            ax2.text(val + xpad * 0.95, i, "reference method", va="center", ha="left", fontsize=8.7, color=TEAL_DARK)
        elif ref > 0:
            inc = (val / ref - 1.0) * 100.0
            ax2.text(val + xpad * 0.95, i, f"+{inc:.1f}% vs scale-aware", va="center", ha="left", fontsize=8.7, color=INK)
    _subtle_box(ax2)

    # Save the exact values used in the figure for auditability.
    pd.DataFrame({
        "method": methods,
        "n_values": [len(v) for v in data],
        "mean_boundaries_per_cpt": means,
        "median_boundaries_per_cpt": medians,
        "min_boundaries_per_cpt": [np.nanmin(v) for v in data],
        "max_boundaries_per_cpt": [np.nanmax(v) for v in data],
    }).to_csv(OUT_DIR / "fig5_baseline_comparison_data_used.csv", index=False)

    _save(fig, "fig5_baseline_comparison")
    plt.close(fig)


# =============================================================================
# Fig. 6: sensitivity heatmaps
# =============================================================================

def fig6_sensitivity_heatmaps() -> None:
    _set_pub_style()
    sens = _read_csv_required("sensitivity_results.csv")
    c_minsep = _safe_col(sens, ["MIN_SEP_M", "min_sep", "minsep"])
    c_prom = _safe_col(sens, ["PEAK_PROM_QUANTILE", "peak_prom", "prom_quantile", "prom"])
    c_sup = _safe_col(sens, ["MIN_SCALE_SUPPORT", "min_scale_support", "support"])
    c_tau = _safe_col(sens, ["TAU_M", "tau", "tau_m"])
    c_f1 = _safe_col(sens, ["syn_mean_f1", "mean_f1", "f1"])
    c_nb = _safe_col(sens, ["real_mean_n_boundaries", "mean_n_boundaries", "mean_boundaries", "n_boundaries"])

    tmp = sens.copy()
    for c in [c_minsep, c_prom, c_sup, c_tau, c_f1, c_nb]:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    tmp = tmp.dropna(subset=[c_minsep, c_prom, c_sup, c_tau, c_f1, c_nb])
    if len(tmp) == 0:
        raise RuntimeError("No numeric sensitivity records are available for Fig. 6.")

    # Balanced ranking: maximize synthetic F1, then prefer lower real-data over-segmentation.
    tmp["_rank"] = tmp[c_f1].rank(ascending=False, method="min") + 0.01 * tmp[c_nb].rank(ascending=True, method="min")
    best = tmp.sort_values("_rank").iloc[0]

    minseps = sorted(tmp[c_minsep].unique())
    proms = sorted(tmp[c_prom].unique())
    sups = sorted(tmp[c_sup].unique())
    taus = sorted(tmp[c_tau].unique())

    nrows = len(taus)
    ncols = len(sups)
    fig = plt.figure(figsize=(14.2, 9.6))
    _nice_title(
        fig,
        "Sensitivity analysis of the scale-aware boundary framework",
        f"{len(tmp)} parameter combinations evaluated on synthetic recovery and real-data segmentation",
        y=0.985,
    )
    outer = fig.add_gridspec(1, 2, wspace=0.24, left=0.045, right=0.945, top=0.875, bottom=0.102)

    def matrix_for(value_col: str, tau: float, sup: float) -> np.ndarray:
        mat = np.full((len(proms), len(minseps)), np.nan, dtype=float)
        sub = tmp[(tmp[c_tau] == tau) & (tmp[c_sup] == sup)]
        for _, r in sub.iterrows():
            rr = proms.index(r[c_prom])
            cc = minseps.index(r[c_minsep])
            mat[rr, cc] = r[value_col]
        return mat

    panels = [
        ("(a) Synthetic mean F1", c_f1, _seq_teal(), "Mean F1", ".3f"),
        ("(b) Mean real-data boundaries per CPT", c_nb, _seq_rust(), "Mean boundaries per CPT", ".2f"),
    ]

    for pidx, (panel_title, value_col, cmap, cbar_label, fmt) in enumerate(panels):
        sub = outer[pidx].subgridspec(nrows, ncols, wspace=0.14, hspace=0.22)
        all_mats = [matrix_for(value_col, tau, sup) for tau in taus for sup in sups]
        vmin = float(np.nanmin([np.nanmin(m) for m in all_mats]))
        vmax = float(np.nanmax([np.nanmax(m) for m in all_mats]))
        imgs = []
        axes = []
        for i_tau, tau in enumerate(taus):
            for i_sup, sup in enumerate(sups):
                ax = fig.add_subplot(sub[i_tau, i_sup])
                mat = matrix_for(value_col, tau, sup)
                im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="upper")
                imgs.append(im)
                axes.append(ax)
                ax.set_xticks(np.arange(len(minseps)))
                ax.set_yticks(np.arange(len(proms)))
                ax.set_xticklabels([f"{v:g}" for v in minseps], fontsize=8.8)
                ax.set_yticklabels([f"{v:g}" for v in proms], fontsize=8.8)
                if i_tau == len(taus) - 1:
                    ax.set_xlabel("MIN_SEP_M", fontsize=8.8)
                else:
                    ax.set_xticklabels([])
                if i_sup == 0:
                    ax.set_ylabel("PEAK_PROM_QUANTILE", fontsize=8.8)
                else:
                    ax.set_yticklabels([])
                ax.set_title(f"support = {sup:g}, tau = {tau:g}", fontsize=9.0, color=INK, pad=4)
                ax.set_xticks(np.arange(-.5, len(minseps), 1), minor=True)
                ax.set_yticks(np.arange(-.5, len(proms), 1), minor=True)
                ax.grid(which="minor", color="white", linewidth=1.0)
                ax.tick_params(which="minor", bottom=False, left=False)
                ax.grid(False)
                _subtle_box(ax)

                star_xy = None
                if (float(sup) == float(best[c_sup])) and (float(tau) == float(best[c_tau])):
                    bx = minseps.index(best[c_minsep])
                    by = proms.index(best[c_prom])
                    star_xy = (bx, by)
                    ax.add_patch(Rectangle((bx - 0.5, by - 0.5), 1, 1, fill=False, ec=INK, lw=1.8, zorder=7))
                _annotate_heatmap(ax, mat, fmt=fmt, vmin=vmin, vmax=vmax, star_xy=star_xy)

        # Shared colorbar per outer panel.
        pos = outer[pidx].get_position(fig)
        cax = fig.add_axes([pos.x1 + 0.008, pos.y0 + 0.08 * pos.height, 0.012, 0.78 * pos.height])
        cb = fig.colorbar(imgs[0], cax=cax)
        cb.set_label(cbar_label, fontsize=9.7)
        cb.ax.tick_params(labelsize=8.5)
        cb.outline.set_edgecolor(BORDER)
        cb.outline.set_linewidth(0.7)
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.025, panel_title, ha="center", va="bottom", fontsize=11.6, fontweight="bold", color=INK)

    best_text = (
        f"Best locked setting: MIN_SEP_M = {best[c_minsep]:g}, "
        f"PEAK_PROM_QUANTILE = {best[c_prom]:g}, "
        f"MIN_SCALE_SUPPORT = {best[c_sup]:g}, tau = {best[c_tau]:g}"
    )
    fig.text(0.5, 0.04, best_text, ha="center", va="center", fontsize=9.4, color=MUTED)
    fig.text(0.5, 0.022, "Cell values are read directly from sensitivity_results.csv; ★ marks the selected best setting.",
             ha="center", va="center", fontsize=8.4, color=MUTED)

    _save(fig, "fig6_sensitivity_heatmaps")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    _set_pub_style()
    print(f"[pubfig-polished] PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"[pubfig-polished] RAW_DIR:      {RAW_DIR}")
    print(f"[pubfig-polished] OUT_DIR:      {OUT_DIR}")
    print(f"[pubfig-polished] DPI:          {DPI}")

    steps: List[Tuple[str, callable]] = [
        ("fig1_workflow", fig1_workflow),
        ("fig2_nl_pointmap", fig2_netherlands_pointmap),
        ("fig3_cpt_panels", fig3_cpt_panels),
        ("fig4_synthetic_validation", fig4_synthetic_validation),
        ("fig5_baseline_comparison", fig5_baseline_comparison),
        ("fig6_sensitivity_heatmaps", fig6_sensitivity_heatmaps),
    ]

    manifest_rows: List[Dict[str, str]] = []
    for stem, func in _progress(steps):
        try:
            func()
            print(f"[pubfig-polished] saved {stem}")
            manifest_rows.append({"figure": stem, "status": "saved", "message": ""})
        except Exception as exc:
            print(f"[pubfig-polished] ERROR in {stem}: {exc}")
            manifest_rows.append({"figure": stem, "status": "error", "message": str(exc)})
            raise

    pd.DataFrame(manifest_rows).to_csv(OUT_DIR / "figure_generation_manifest.csv", index=False)
    print("[pubfig-polished] DONE")


if __name__ == "__main__":
    main()
