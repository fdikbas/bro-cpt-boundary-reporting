"""Publication-grade figure generator (single-file, no local imports).

Generates Fig.1–Fig.6 for the Paper-1 manuscript (Computers and Geotechnics).
- Point-based Netherlands map (no surfaces/interpolation)
- CPT representative panels (qt, fs) with boundaries + uncertainty bands + confidence classes
- Synthetic validation (precision/recall/F1 + MAE)
- Baseline comparison (counts)
- Sensitivity heatmaps

Inputs expected (relative to PROJECT_ROOT):
  outputs_paper1_v2/paper1_raw/
    boundary_catalog.csv
    synthetic_validation_results.csv
    sensitivity_results.csv
    baselines/baseline_single_scale_boundaries.csv  (optional)
    baselines/baseline_pelt_boundaries.csv          (optional)
    profiles_realdata/*.csv                         (optional for Fig.3)

GeoPackage is located automatically (BRO_CPT_GPKG env var preferred).

All figures are saved as PNG (300 dpi) + PDF into:
  outputs_paper1_v2/paper1_analysis_publication/

Author: generated for Prof. Dr. Fatih Dikbaş workflow
"""

from __future__ import annotations

import os
import math
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

# matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

# optional libs
try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    from scipy.signal import medfilt
except Exception:
    medfilt = None

# ---------------------------
# Configuration
# ---------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # ...\BRO CPT Paper1
RAW_DIR = PROJECT_ROOT / "outputs_paper1_v2" / "paper1_raw"
OUT_DIR = PROJECT_ROOT / "outputs_paper1_v2" / "paper1_analysis_publication"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# panel / export
DPI = 300

# ---------------------------
# Style helpers
# ---------------------------

def _set_pub_style() -> None:
    """Global matplotlib style: clean, journal-like."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#4c4c4c",
        "axes.labelcolor": "#2b2b2b",
        "xtick.color": "#2b2b2b",
        "ytick.color": "#2b2b2b",
        "text.color": "#2b2b2b",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.titlesize": 16,
        "axes.grid": True,
        "grid.color": "#e8e8e8",
        "grid.linewidth": 1.0,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,  # IMPORTANT: grid under data
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _natural_diverging() -> LinearSegmentedColormap:
    # muted blue -> warm off-white -> muted rust
    colors = ["#2c6c85", "#f3efe6", "#b3543c"]
    return LinearSegmentedColormap.from_list("muted_div", colors, N=256)


def _natural_sequential() -> LinearSegmentedColormap:
    # pale warm gray -> muted teal
    colors = ["#f3efe6", "#6aa6a6", "#1f5c6e"]
    return LinearSegmentedColormap.from_list("muted_seq", colors, N=256)


def _panel_label(ax, label: str) -> None:
    ax.text(0.01, 0.99, label, transform=ax.transAxes,
            ha="left", va="top", fontsize=18, fontweight="bold")


def _save(fig, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=DPI)
    fig.savefig(OUT_DIR / f"{stem}.pdf")


# ---------------------------
# Data loaders
# ---------------------------

def _load_csv(name: str) -> pd.DataFrame:
    p = RAW_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Missing required input: {p}")
    return pd.read_csv(p)


def _locate_gpkg() -> Optional[Path]:
    """Locate BRO CPT GeoPackage.

    Priority:
      1) env BRO_CPT_GPKG
      2) sibling project roots (BRO CPT)
      3) search under PROJECT_ROOT
    """
    env = os.environ.get("BRO_CPT_GPKG", "").strip()
    if env:
        p = Path(env)
        if p.exists() and p.suffix.lower() == ".gpkg":
            return p

    candidates: List[Path] = []

    # sibling folder heuristic
    parent = PROJECT_ROOT.parent
    for sib in parent.glob("BRO CPT*"):
        cand = sib / "data_bro_cpt_pdok" / "extracted" / "brocptvolledigeset_v2_0.gpkg"
        if cand.exists():
            return cand
        # any gpkg under extracted
        ex = sib / "data_bro_cpt_pdok" / "extracted"
        if ex.exists():
            candidates += list(ex.glob("*.gpkg"))

    # fallback: search in this project
    candidates += list((PROJECT_ROOT / "data_bro_cpt_pdok").rglob("*.gpkg"))
    if not candidates:
        return None

    # choose largest
    candidates = sorted(candidates, key=lambda x: x.stat().st_size, reverse=True)
    return candidates[0]


def _find_province_boundary_file(root_dir: Path) -> Optional[Path]:
    """Auto-detect province boundary dataset under project tree.

    Looks for GPKG/SHP/GeoJSON with province/provincie keywords.
    """
    if gpd is None:
        return None

    keywords = ["province", "provinc", "provincie", "adm1", "admin1", "nuts2", "gadm", "cbs"]
    exts = {".gpkg", ".shp", ".geojson", ".json"}

    # common GIS folders
    search_roots = [
        root_dir / "gis",
        root_dir / "data_gis",
        root_dir,
        root_dir.parent / "BRO CPT" / "gis",
        root_dir.parent / "BRO CPT" / "data_gis",
        root_dir.parent / "BRO CPT",
    ]

    for sr in search_roots:
        if not sr.exists():
            continue
        for p in sr.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                name = p.name.lower()
                if any(k in name for k in keywords):
                    return p
    return None


def _load_province_boundaries(path: Path):
    if gpd is None:
        return None
    try:
        if path.suffix.lower() == ".gpkg":
            # prefer generalized layer names if present
            for layer in ["provincie_gegeneraliseerd", "provincie_niet_gegeneraliseerd", "provinces", "province", "provincie"]:
                try:
                    gdf = gpd.read_file(path, layer=layer)
                    if len(gdf) > 0:
                        break
                except Exception:
                    gdf = None
            if gdf is None:
                gdf = gpd.read_file(path)
        else:
            gdf = gpd.read_file(path)
        # project to EPSG:3857 for metric scale bar
        if gdf.crs is None:
            # if CRS unknown, assume already 3857
            gdf = gdf.set_crs(3857)
        gdf = gdf.to_crs(3857)
        return gdf
    except Exception:
        return None


# ---------------------------
# Figure 1: workflow schematic
# ---------------------------

def fig1_workflow() -> None:
    _set_pub_style()
    fig = plt.figure(figsize=(11.5, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    fig.text(0.5, 0.95, "Workflow of the scale-aware CPT boundary reporting framework",
             ha="center", va="top", fontsize=18, fontweight="bold")
    fig.text(0.5, 0.915, "From public BRO CPT data to boundary location, uncertainty band, and confidence score",
             ha="center", va="top", fontsize=11, color="#5a5a5a")

    # box style
    def box(x, y, w, h, text, fc="#f3efe6", ec="#c8c8c8"):
        b = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.012,rounding_size=0.02",
                          fc=fc, ec=ec, lw=1.2)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10)

    # main row
    y = 0.68
    w = 0.12
    h = 0.09
    xs = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.92]
    texts = [
        "Public\nBRO CPT/CPTu\ndata",
        "Quality control\nand\nstandardization",
        "Depth-grid\nharmonization",
        "Multi-scale\nmedian\nfiltering",
        "Boundary evidence\n& candidate\ndetection",
        "Cross-scale\nboundary\nmatching",
        "Boundary\nreporting\noutputs",
    ]

    fcs = ["#f3efe6", "#f3efe6", "#f3efe6", "#e6f1f1", "#e6f1f1", "#e6f1f1", "#f3efe6"]

    for x, t, fc in zip(xs, texts, fcs):
        box(x, y, w, h, t, fc=fc)

    # arrows
    for i in range(len(xs)-1):
        x0 = xs[i] + w
        x1 = xs[i+1]
        ax.annotate("", xy=(x1-0.01, y+h/2), xytext=(x0+0.01, y+h/2),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="#6b6b6b"))

    # output bullets
    ax.text(xs[-1] + w/2, y-0.02,
            "• Central boundary depth\n• Uncertainty band [P10, P90]\n• Confidence score (BCS/SGS)",
            ha="center", va="top", fontsize=9, color="#2b2b2b")

    # validation branch
    fig.text(0.5, 0.56, "Validation and robustness assessment",
             ha="center", va="center", fontsize=12, fontweight="bold")

    y2 = 0.43
    w2 = 0.17
    h2 = 0.08
    x2s = [0.22, 0.42, 0.62]
    t2 = ["Synthetic\nboundary recovery", "Baseline\ncomparison", "Sensitivity\nanalysis"]
    fc2 = ["#e6f1f1", "#f3efe6", "#f7efe9"]
    for x, t, fc in zip(x2s, t2, fc2):
        box(x, y2, w2, h2, t, fc=fc)
    ax.annotate("", xy=(x2s[0]-0.02, y2+h2/2), xytext=(0.84, y+h/2-0.02),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#6b6b6b"))

    for i in range(2):
        ax.annotate("", xy=(x2s[i+1]-0.01, y2+h2/2), xytext=(x2s[i]+w2+0.01, y2+h2/2),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#6b6b6b"))

    fig.text(0.5, 0.08,
             "Key idea: boundaries that remain stable across analysis scales receive higher confidence and narrower uncertainty bands.",
             ha="center", va="center", fontsize=10, color="#5a5a5a")

    _save(fig, "fig1_workflow")
    plt.close(fig)


# ---------------------------
# Figure 2: Netherlands point-based map
# ---------------------------

def _add_scalebar(ax, length_km=50, location=(0.08, 0.08)):
    # in axes fraction; convert to data coordinates
    x0, y0 = location
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    x = xlim[0] + x0*(xlim[1]-xlim[0])
    y = ylim[0] + y0*(ylim[1]-ylim[0])
    length_m = length_km*1000
    ax.plot([x, x+length_m], [y, y], color="#4c4c4c", lw=2.0, solid_capstyle='butt')
    ax.plot([x, x], [y-0.02*length_m, y+0.02*length_m], color="#4c4c4c", lw=1.2)
    ax.plot([x+length_m, x+length_m], [y-0.02*length_m, y+0.02*length_m], color="#4c4c4c", lw=1.2)
    ax.text(x+length_m/2, y+0.05*length_m, f"{length_km} km", ha="center", va="bottom", fontsize=9, color="#2b2b2b")


def _add_north_arrow(ax, location=(0.92, 0.12)):
    x0, y0 = location
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    x = xlim[0] + x0*(xlim[1]-xlim[0])
    y = ylim[0] + y0*(ylim[1]-ylim[0])
    ax.annotate("N", xy=(x, y), xytext=(x, y-80000),
                ha="center", va="center", fontsize=10, color="#2b2b2b",
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#2b2b2b"))


def fig2_netherlands_pointmap() -> None:
    _set_pub_style()

    bc = _load_csv("boundary_catalog.csv")
    # per CPT stats
    cpt_stats = bc.groupby("cpt_id").agg(
        n_boundaries=("boundary_id", "count"),
        mean_sgs=("sgs", "mean"),
        median_delta_z=("delta_z", "median"),
    ).reset_index().rename(columns={"cpt_id": "bro_id"})

    gpkg = _locate_gpkg()
    if gpkg is None or gpd is None:
        raise RuntimeError("GeoPackage not found or geopandas missing; cannot build map figure.")

    survey_layer = "geotechnical_cpt_survey"
    gdf = gpd.read_file(gpkg, layer=survey_layer)
    # keep only needed
    cols = [c for c in ["bro_id", "standardized_location", "geometry"] if c in gdf.columns]
    gdf = gdf[cols].copy()
    if "geometry" not in gdf.columns:
        # geopandas should provide geometry; but guard
        gdf = gdf.set_geometry("standardized_location")
    # ensure CRS
    if gdf.crs is None:
        gdf = gdf.set_crs(3857)
    gdf = gdf.to_crs(3857)

    gdf = gdf.merge(cpt_stats, on="bro_id", how="inner")

    # load provinces if available
    prov_path = _find_province_boundary_file(PROJECT_ROOT)
    prov_gdf = _load_province_boundaries(prov_path) if prov_path else None

    fig = plt.figure(figsize=(11.5, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(False)

    # background land/water via province dissolve if available
    if prov_gdf is not None and len(prov_gdf) > 0:
        # very subtle land fill
        prov_gdf.plot(ax=ax, facecolor="#f5f2ea", edgecolor="#bfbfbf", linewidth=0.8, alpha=1.0, zorder=1)
        prov_gdf.boundary.plot(ax=ax, color="#cfcfcf", linewidth=0.6, alpha=0.9, zorder=2)
        # extent
        ax.set_xlim(*prov_gdf.total_bounds[[0, 2]])
        ax.set_ylim(*prov_gdf.total_bounds[[1, 3]])
    else:
        # no province file: use points extent + padding
        xb = gdf.total_bounds
        pad_x = 0.15*(xb[2]-xb[0]); pad_y = 0.15*(xb[3]-xb[1])
        ax.set_xlim(xb[0]-pad_x, xb[2]+pad_x)
        ax.set_ylim(xb[1]-pad_y, xb[3]+pad_y)

    # point encoding
    cmap = _natural_sequential()
    sizes = np.interp(gdf["n_boundaries"].values, (gdf["n_boundaries"].min(), gdf["n_boundaries"].max()), (35, 140))
    sc = ax.scatter(gdf.geometry.x, gdf.geometry.y,
                    c=gdf["mean_sgs"], cmap=cmap, vmin=0, vmax=1,
                    s=sizes, edgecolor="#2b2b2b", linewidth=0.4, alpha=0.95, zorder=5)

    # aesthetics
    ax.set_xlabel("X (km, EPSG:3857)")
    ax.set_ylabel("Y (km, EPSG:3857)")

    # ticks in km
    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    ax.set_xticklabels([f"{t/1000:.0f}" for t in ax.get_xticks()])
    ax.set_yticklabels([f"{t/1000:.0f}" for t in ax.get_yticks()])

    # colorbar
    cb = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Mean SGS")

    # legend for size
    # choose 3 reference sizes
    nb_min, nb_med, nb_max = int(gdf["n_boundaries"].min()), int(np.median(gdf["n_boundaries"])), int(gdf["n_boundaries"].max())
    handles = []
    for nb in [nb_min, nb_med, nb_max]:
        s = float(np.interp(nb, (gdf["n_boundaries"].min(), gdf["n_boundaries"].max()), (35, 140)))
        h = ax.scatter([], [], s=s, edgecolor="#2b2b2b", facecolor="#dddddd", linewidth=0.4)
        handles.append(h)
    labels = [f"{nb_min} boundaries", f"{nb_med} boundaries", f"{nb_max} boundaries"]
    leg = ax.legend(handles, labels, title="Boundary count", loc="lower left", frameon=True)
    leg.get_frame().set_edgecolor("#cfcfcf")
    leg.get_frame().set_linewidth(0.8)

    _panel_label(ax, "(a)")

    # inset zoom for Randstad-ish cluster (auto: densest bounding box)
    # compute KDE-ish by counting within coarse grid
    xy = np.vstack([gdf.geometry.x.values, gdf.geometry.y.values]).T
    if len(xy) >= 10:
        # find densest point neighborhood by nearest-neighbor distances
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(8, len(xy))).fit(xy)
        dists, _ = nn.kneighbors(xy)
        density_score = 1 / (np.mean(dists[:, 1:], axis=1) + 1e-9)
        center = xy[np.argmax(density_score)]
        # bbox around center
        dx = 60000; dy = 60000
        bbox = (center[0]-dx, center[0]+dx, center[1]-dy, center[1]+dy)
        inset = ax.inset_axes([0.63, 0.08, 0.33, 0.33])
        inset.grid(False)
        if prov_gdf is not None and len(prov_gdf) > 0:
            prov_gdf.plot(ax=inset, facecolor="#f5f2ea", edgecolor="#cfcfcf", linewidth=0.5, alpha=1.0, zorder=1)
            prov_gdf.boundary.plot(ax=inset, color="#d7d7d7", linewidth=0.4, alpha=1.0, zorder=2)
        # points inside
        mask = (gdf.geometry.x.between(bbox[0], bbox[1])) & (gdf.geometry.y.between(bbox[2], bbox[3]))
        gi = gdf.loc[mask]
        if len(gi) > 0:
            sizes_i = np.interp(gi["n_boundaries"].values, (gdf["n_boundaries"].min(), gdf["n_boundaries"].max()), (25, 110))
            inset.scatter(gi.geometry.x, gi.geometry.y, c=gi["mean_sgs"], cmap=cmap, vmin=0, vmax=1,
                          s=sizes_i, edgecolor="#2b2b2b", linewidth=0.35, alpha=0.95, zorder=5)
        inset.set_xlim(bbox[0], bbox[1]); inset.set_ylim(bbox[2], bbox[3])
        inset.set_xticks([]); inset.set_yticks([])
        inset.set_title("Inset (dense area)", fontsize=9, color="#5a5a5a")
        # outline inset on main
        ax.indicate_inset_zoom(inset, edgecolor="#7a7a7a", linewidth=0.8)

    _add_scalebar(ax, length_km=50)
    _add_north_arrow(ax)

    ax.set_title("Netherlands CPT soundings (point-based)", pad=10)

    _save(fig, "fig2_nl_pointmap")
    plt.close(fig)


# ---------------------------
# Figure 3: representative CPT panels
# ---------------------------

def _pick_representative_cpts(bc: pd.DataFrame, k: int = 4) -> List[str]:
    g = bc.groupby("cpt_id").agg(
        n=("boundary_id", "count"),
        sgs_mean=("sgs", "mean"),
        dz_mean=("delta_z", "mean"),
        sgs_min=("sgs", "min"),
    ).reset_index()
    # pick min n, max n, min sgs_min, median n
    c1 = g.loc[g["n"].idxmin(), "cpt_id"]
    c2 = g.loc[g["n"].idxmax(), "cpt_id"]
    c3 = g.loc[g["sgs_min"].idxmin(), "cpt_id"]
    # median n
    c4 = g.iloc[(g["n"]-g["n"].median()).abs().argsort()[:1]]["cpt_id"].values[0]
    cpts = [c1, c2, c3, c4]
    # unique preserve order
    out=[]
    for c in cpts:
        if c not in out:
            out.append(c)
    return out[:k]


def _load_profile_csv(cpt_id: str) -> Optional[pd.DataFrame]:
    p = RAW_DIR / "profiles_realdata" / f"{cpt_id}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # expected columns: depth, qt, fs, u2 (optional)
    return df


def fig3_cpt_panels() -> None:
    _set_pub_style()
    bc = _load_csv("boundary_catalog.csv")
    cpts = _pick_representative_cpts(bc, 4)

    # colors for confidence classes
    col_hi = "#2c6c85"  # blue
    col_md = "#c08a3a"  # amber
    col_lo = "#b3543c"  # muted rust

    fig = plt.figure(figsize=(11.5, 8))
    gs = fig.add_gridspec(2, 2, wspace=0.18, hspace=0.18)

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for idx, cpt in enumerate(cpts):
        r = idx // 2
        c = idx % 2
        sub = gs[r, c].subgridspec(1, 2, wspace=0.05)
        ax_q = fig.add_subplot(sub[0, 0])
        ax_f = fig.add_subplot(sub[0, 1], sharey=ax_q)

        # load
        df = _load_profile_csv(cpt)
        if df is None or len(df) < 5:
            # placeholder
            for ax in (ax_q, ax_f):
                ax.text(0.5, 0.5, f"{cpt}\nprofile missing", ha="center", va="center")
                ax.set_axis_off()
            continue

        # sanitize
        depth = df["depth"].values
        qt = df["qt"].values
        fs = df["fs"].values

        # smoothed
        if medfilt is not None:
            ksize = 9 if len(depth) >= 50 else 5
            qt_s = medfilt(qt, kernel_size=ksize)
            fs_s = medfilt(fs, kernel_size=ksize)
        else:
            qt_s = pd.Series(qt).rolling(9, center=True, min_periods=1).median().values
            fs_s = pd.Series(fs).rolling(9, center=True, min_periods=1).median().values

        # plot raw (light) + smoothed (main)
        for ax, x_raw, x_s, xlabel in (
            (ax_q, qt, qt_s, r"$q_t$"),
            (ax_f, fs, fs_s, r"$f_s$"),
        ):
            ax.plot(x_raw, depth, color="#c7c7c7", lw=1.0, alpha=0.9, zorder=1, label="raw")
            ax.plot(x_s, depth, color="#2b2b2b", lw=1.6, alpha=0.95, zorder=2, label="smoothed")
            ax.set_xlabel(f"{xlabel}")
            ax.grid(True)
            ax.set_axisbelow(True)

        ax_q.set_ylabel("Depth (m)")
        ax_q.invert_yaxis()

        # boundaries for this CPT
        bci = bc[bc["cpt_id"] == cpt].copy()
        # determine classes by sgs
        def cls(s):
            if s >= 0.67:
                return "high"
            if s >= 0.33:
                return "medium"
            return "low"

        for _, row in bci.iterrows():
            z = row["z_hat"]
            z0 = row["z_p10"]
            z1 = row["z_p90"]
            s = row["sgs"]
            cl = cls(s)
            if cl == "high":
                col = col_hi
                alpha = 0.12
                lw = 1.2
            elif cl == "medium":
                col = col_md
                alpha = 0.14
                lw = 1.2
            else:
                col = col_lo
                alpha = 0.16
                lw = 1.3

            for ax in (ax_q, ax_f):
                ax.axhspan(z0, z1, color=col, alpha=alpha, lw=0, zorder=0)
                ax.axhline(z, color=col, lw=lw, alpha=0.95, zorder=3)

        # panel label + title
        ax_q.text(0.01, 0.99, panel_labels[idx], transform=ax_q.transAxes,
                  ha="left", va="top", fontsize=16, fontweight="bold")
        ax_q.set_title(f"{cpt}", loc="left", fontsize=10, color="#5a5a5a", pad=2)

        # cleaner shared y labels
        plt.setp(ax_f.get_yticklabels(), visible=False)
        ax_f.tick_params(axis='y', length=0)

    # global legend (compact)
    handles = [
        matplotlib.lines.Line2D([], [], color="#c7c7c7", lw=1.0, label="raw"),
        matplotlib.lines.Line2D([], [], color="#2b2b2b", lw=1.6, label="smoothed"),
        matplotlib.lines.Line2D([], [], color="#2c6c85", lw=1.2, label="boundary (high)"),
        matplotlib.lines.Line2D([], [], color="#c08a3a", lw=1.2, label="boundary (medium)"),
        matplotlib.lines.Line2D([], [], color="#b3543c", lw=1.3, label="boundary (low)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Representative CPT soundings with boundaries, uncertainty bands, and confidence", y=0.98)

    _save(fig, "fig3_cpt_panels")
    plt.close(fig)


# ---------------------------
# Figure 4: synthetic validation
# ---------------------------

def fig4_synthetic_validation() -> None:
    _set_pub_style()
    syn = _load_csv("synthetic_validation_results.csv")

    # normalize column names
    # expected: noise_level, precision, recall, f1, mae
    colmap = {c.lower(): c for c in syn.columns}
    def _get(name):
        for k in colmap:
            if name in k:
                return colmap[k]
        raise KeyError(name)

    nl = _get("noise")
    prec = _get("prec")
    rec = _get("rec")
    f1 = _get("f1")
    mae = _get("mae")

    g = syn.groupby(nl).agg(
        precision=(prec, "mean"),
        recall=(rec, "mean"),
        f1=(f1, "mean"),
        mae=(mae, "mean"),
        precision_sd=(prec, "std"),
        recall_sd=(rec, "std"),
        f1_sd=(f1, "std"),
        mae_sd=(mae, "std"),
        n=(prec, "count"),
    ).reset_index().sort_values(nl)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.6), gridspec_kw={"wspace": 0.25})

    colors = {
        "Precision": "#2c6c85",
        "Recall": "#6aa6a6",
        "F1": "#b3543c",
        "MAE": "#2b2b2b",
    }

    ax1.set_title("(a) Boundary recovery metrics")
    x = g[nl].values
    for name, y, ysd in [
        ("Precision", g["precision"].values, g["precision_sd"].values),
        ("Recall", g["recall"].values, g["recall_sd"].values),
        ("F1", g["f1"].values, g["f1_sd"].values),
    ]:
        ax1.plot(x, y, marker='o', lw=2.2, color=colors[name], label=name)
        ax1.fill_between(x, y-ysd, y+ysd, color=colors[name], alpha=0.12, linewidth=0)

    ax1.set_xlabel("Noise level (relative)")
    ax1.set_ylabel("Metric value")
    ax1.set_ylim(0.35, 1.05)
    ax1.legend(frameon=False, loc="lower left")

    ax2.set_title("(b) Matched-boundary error")
    y = g["mae"].values
    ysd = g["mae_sd"].values
    ax2.plot(x, y, marker='o', lw=2.2, color=colors["MAE"], label="MAE")
    ax2.fill_between(x, y-ysd, y+ysd, color=colors["MAE"], alpha=0.12, linewidth=0)
    ax2.set_xlabel("Noise level (relative)")
    ax2.set_ylabel("MAE (m)")
    ax2.legend(frameon=False, loc="upper left")

    fig.suptitle("Synthetic validation (n = 500 profiles per noise level; total n = 2000)", y=1.02)
    for ax in (ax1, ax2):
        ax.grid(True)
        ax.set_axisbelow(True)

    _save(fig, "fig4_synthetic_validation")
    plt.close(fig)


# ---------------------------
# Figure 5: baseline comparison
# ---------------------------

def fig5_baseline_comparison() -> None:
    _set_pub_style()

    # precomputed summary exists in paper1_analysis, but use raw baselines if present
    # We fallback to baseline_boundary_counts_by_cpt.csv if available in paper1_analysis
    analysis_dir = PROJECT_ROOT / "outputs_paper1_v2" / "paper1_analysis"
    by_cpt = analysis_dir / "baseline_boundary_counts_by_cpt.csv"
    if by_cpt.exists():
        df = pd.read_csv(by_cpt)
    else:
        # reconstruct from boundary_catalog and raw baseline files
        bc = _load_csv("boundary_catalog.csv")
        df = bc.groupby("cpt_id").size().reset_index(name="Scale-aware")
        # optional baseline files
        for mname, fname in [("Single-scale", "baselines/baseline_single_scale_boundaries.csv"),
                             ("PELT", "baselines/baseline_pelt_boundaries.csv")]:
            p = RAW_DIR / fname
            if p.exists():
                tmp = pd.read_csv(p)
                tmp = tmp.groupby("cpt_id").size().reset_index(name=mname)
                df = df.merge(tmp, on="cpt_id", how="left")

    # Support both wide and long formats.
    # Wide: columns like [cpt_id, Scale-aware, Single-scale, PELT]
    # Long: columns like [cpt_id, n_boundaries, method]
    data: list[np.ndarray] = []
    methods: list[str] = []

    if {"method", "n_boundaries"}.issubset(set(map(str.lower, df.columns))):
        colmap = {c.lower(): c for c in df.columns}
        c_method = colmap["method"]
        c_nb = colmap["n_boundaries"]

        order = [
            ("scale-aware", "Scale-aware"),
            ("single-scale", "Single-scale"),
            ("pelt", "PELT"),
        ]

        mser = df[c_method].astype(str).str.strip().str.lower()
        for key, disp in order:
            sel = df.loc[mser == key, c_nb]
            vals = pd.to_numeric(sel, errors="coerce").dropna().astype(float).values
            if vals.size == 0:
                continue
            methods.append(disp)
            data.append(vals)

        # If file uses alternate naming (e.g., scale_aware)
        if not data:
            for key, disp in order:
                key2 = key.replace("-", "_")
                sel = df.loc[mser == key2, c_nb]
                vals = pd.to_numeric(sel, errors="coerce").dropna().astype(float).values
                if vals.size == 0:
                    continue
                methods.append(disp)
                data.append(vals)
    else:
        methods = [c for c in df.columns if c != "cpt_id"]
        for m in methods:
            vals = pd.to_numeric(df[m], errors="coerce").dropna().astype(float).values
            data.append(vals)

    if not data:
        raise ValueError("No numeric baseline data found to plot in fig5_baseline_comparison().")

    # Panel (a) box/violin
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.6), gridspec_kw={"wspace": 0.25})
    n_text = "n = 35"
    if "cpt_id" in df.columns:
        try:
            n_text = f"n = {int(pd.Series(df['cpt_id']).nunique())}"
        except Exception:
            pass
    fig.suptitle(f"Baseline comparison on real-data soundings ({n_text})", y=1.02)

    ax1.set_title("(a) Boundary-count distribution")

    # violin backdrop
    parts = ax1.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    cols = ["#2c6c85", "#c08a3a", "#8b3f3f"]
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(cols[i % len(cols)])
        pc.set_alpha(0.18)
        pc.set_edgecolor("none")

    bp = ax1.boxplot(data, tick_labels=methods, showfliers=False, widths=0.35, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor("white")
        patch.set_edgecolor(cols[i % len(cols)])
        patch.set_linewidth(1.4)
    for elem in ['whiskers', 'caps', 'medians']:
        for i, line in enumerate(bp[elem]):
            line.set_color(cols[(i//2) % len(cols)] if elem != 'medians' else cols[i % len(cols)])
            line.set_linewidth(1.2 if elem != 'medians' else 1.6)

    # jittered points
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data, start=1):
        x = rng.normal(i, 0.04, size=len(vals))
        ax1.scatter(x, vals, s=18, color="#2b2b2b", alpha=0.55, zorder=3, linewidth=0)

    ax1.set_ylabel("Detected boundaries per CPT")
    ax1.set_ylim(0, max(140, float(np.nanmax(np.concatenate(data))) * 1.05))
    ax1.grid(True)
    ax1.set_axisbelow(True)

    # annotate mean/median
    for i, m in enumerate(methods, start=1):
        vals = data[i-1]
        ax1.text(i, np.nanpercentile(vals, 98), f"mean {np.mean(vals):.2f}\nmedian {np.median(vals):.0f}",
                 ha="center", va="bottom", fontsize=9, color="#2b2b2b")

    # Panel (b) mean bars
    ax2.set_title("(b) Mean segmentation density")
    means = [float(np.mean(v)) for v in data]
    y = np.arange(len(methods))
    ax2.hlines(y, [0]*len(means), means, color=[cols[i] for i in range(len(means))], lw=8, alpha=0.85)
    ax2.plot(means, y, 'o', color="#2b2b2b", ms=7)
    ax2.set_yticks(y)
    ax2.set_yticklabels(methods)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean boundaries per CPT")
    ax2.grid(True)
    ax2.set_axisbelow(True)

    # relative increase vs scale-aware
    ref = means[0]
    for i, (m, val) in enumerate(zip(methods, means)):
        if i == 0:
            ax2.text(val + 1, i, "reference method", va="center", fontsize=9, color="#2b2b2b")
        else:
            inc = (val/ref - 1) * 100
            ax2.text(val + 1, i, f"+{inc:.0f}% vs scale-aware", va="center", fontsize=9, color="#2b2b2b")

    _save(fig, "fig5_baseline_comparison")
    plt.close(fig)


# ---------------------------
# Figure 6: sensitivity heatmaps
# ---------------------------

def fig6_sensitivity_heatmaps() -> None:
    _set_pub_style()
    sens = _load_csv("sensitivity_results.csv")

    # expected columns
    # MIN_SEP_M, PEAK_PROM_QUANTILE, MIN_SCALE_SUPPORT, TAU_M, syn_mean_f1, real_mean_n_boundaries
    cols = {c.lower(): c for c in sens.columns}
    def col(name):
        for k,v in cols.items():
            if name in k:
                return v
        raise KeyError(name)

    c_minsep = col("min_sep")
    c_prom = col("peak_prom")
    c_sup = col("min_scale_support")
    c_tau = col("tau")
    c_f1 = col("syn_mean_f1")
    c_nb = col("real_mean_n_boundaries")

    # Determine best setting
    # maximize F1, minimize boundaries (tie-break)
    tmp = sens.copy()
    tmp["rank"] = tmp[c_f1].rank(ascending=False, method="min") + 0.01*tmp[c_nb].rank(ascending=True, method="min")
    best = tmp.sort_values("rank").iloc[0]

    # build grids
    minseps = sorted(tmp[c_minsep].unique())
    proms = sorted(tmp[c_prom].unique())
    sups = sorted(tmp[c_sup].unique())
    taus = sorted(tmp[c_tau].unique())

    # figure layout: 2 big panels, each 2x2 of small heatmaps
    fig = plt.figure(figsize=(11.5, 8))
    fig.suptitle("Sensitivity analysis of the scale-aware boundary framework", y=0.98, fontsize=16, fontweight="bold")
    fig.text(0.5, 0.945, "24 parameter combinations evaluated on synthetic recovery and real-data segmentation",
             ha="center", va="top", fontsize=10, color="#5a5a5a")

    # outer gridspec
    outer = fig.add_gridspec(1, 2, wspace=0.18)

    cmap_f1 = LinearSegmentedColormap.from_list("f1", ["#f3efe6", "#6aa6a6", "#1f5c6e"], N=256)
    cmap_nb = LinearSegmentedColormap.from_list("nb", ["#f3efe6", "#e6b08a", "#b3543c"], N=256)

    def draw_panel(gs_cell, value_col, cmap, title, cbar_label):
        sub = outer[gs_cell].subgridspec(2, 2, wspace=0.08, hspace=0.10)
        axes = []
        imgs = []
        for i_tau, tau in enumerate(taus):
            for i_sup, sup in enumerate(sups):
                ax = fig.add_subplot(sub[i_tau, i_sup])
                axes.append(ax)
                # pivot
                d = tmp[(tmp[c_tau]==tau) & (tmp[c_sup]==sup)]
                mat = np.full((len(proms), len(minseps)), np.nan)
                for _, r in d.iterrows():
                    y = proms.index(r[c_prom])
                    x = minseps.index(r[c_minsep])
                    mat[y, x] = r[value_col]
                im = ax.imshow(mat, cmap=cmap, aspect='auto', origin='lower')
                imgs.append(im)

                # annotations
                for yy in range(mat.shape[0]):
                    for xx in range(mat.shape[1]):
                        v = mat[yy, xx]
                        if np.isnan(v):
                            continue
                        txt = f"{v:.3f}" if value_col == c_f1 else f"{v:.2f}"
                        ax.text(xx, yy, txt, ha='center', va='center', fontsize=9,
                                color="#1f1f1f", fontweight='bold' if (abs(v-best[value_col])<1e-9 and
                                                                        minseps[xx]==best[c_minsep] and
                                                                        proms[yy]==best[c_prom] and
                                                                        sup==best[c_sup] and tau==best[c_tau]) else 'normal')

                # ticks
                ax.set_xticks(range(len(minseps)))
                ax.set_xticklabels([str(v) for v in minseps])
                ax.set_yticks(range(len(proms)))
                ax.set_yticklabels([str(v) for v in proms])
                ax.tick_params(axis='both', labelsize=9)

                if i_tau == 1:
                    ax.set_xlabel("MIN_SEP_M")
                else:
                    ax.set_xlabel("")
                if i_sup == 0:
                    ax.set_ylabel("PEAK_PROM_QUANTILE")
                else:
                    ax.set_ylabel("")

                ax.set_title(f"support = {sup}, tau = {tau}", fontsize=9, color="#5a5a5a")
                ax.grid(False)

                # highlight best
                if (sup==best[c_sup]) and (tau==best[c_tau]):
                    bx = minseps.index(best[c_minsep])
                    by = proms.index(best[c_prom])
                    rect = plt.Rectangle((bx-0.5, by-0.5), 1, 1, fill=False, ec="#2b2b2b", lw=1.6)
                    ax.add_patch(rect)
                    ax.scatter([bx], [by], marker='*', s=120, color="#2b2b2b", zorder=5)

        # add shared colorbar
        # place cbar next to rightmost axes in this big panel
        # compute min/max across images
        vmin = np.nanmin([im.get_array() for im in imgs])
        vmax = np.nanmax([im.get_array() for im in imgs])
        for im in imgs:
            im.set_clim(vmin, vmax)
        cax = fig.add_axes([
            0.48 if gs_cell==0 else 0.93,
            0.15,
            0.015,
            0.68
        ])
        cb = fig.colorbar(imgs[0], cax=cax)
        cb.set_label(cbar_label)

        # big panel title
        # position title using fig.text
        if gs_cell == 0:
            fig.text(0.24, 0.89, title, ha='center', va='center', fontsize=12, fontweight='bold')
        else:
            fig.text(0.74, 0.89, title, ha='center', va='center', fontsize=12, fontweight='bold')

    draw_panel(0, c_f1, cmap_f1, "(a) Synthetic mean F1", "Mean F1")
    draw_panel(1, c_nb, cmap_nb, "(b) Mean real-data boundaries per CPT", "Mean boundaries per CPT")

    fig.text(0.5, 0.03,
             "Best locked setting: MIN_SEP_M = 0.80, PEAK_PROM_QUANTILE = 0.95, MIN_SCALE_SUPPORT = 4, tau = 0.25",
             ha='center', va='bottom', fontsize=9, color="#5a5a5a")

    _save(fig, "fig6_sensitivity_heatmaps")
    plt.close(fig)


# ---------------------------
# Main
# ---------------------------

def main():
    print(f"[pubfig] PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"[pubfig] RAW_DIR: {RAW_DIR}")
    print(f"[pubfig] OUT_DIR: {OUT_DIR}")
    # Fig 1–6
    fig1_workflow()
    print("[pubfig] saved fig1")
    fig2_netherlands_pointmap()
    print("[pubfig] saved fig2")
    fig3_cpt_panels()
    print("[pubfig] saved fig3")
    fig4_synthetic_validation()
    print("[pubfig] saved fig4")
    fig5_baseline_comparison()
    print("[pubfig] saved fig5")
    fig6_sensitivity_heatmaps()
    print("[pubfig] saved fig6")
    print("[pubfig] DONE")


if __name__ == "__main__":
    main()
