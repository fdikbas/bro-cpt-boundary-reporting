# -*- coding: utf-8 -*-
"""
BRO CPT Paper-1 Postprocess (v1.0)

Purpose
-------
Generate ALL tables and figures *to disk* (CSV/JSON + PNG/PDF) from an existing
Paper-1 run (boundary_catalog.csv, realdata_profiles_meta.csv, synthetic_validation_results.csv),
and package everything into a single ZIP bundle.

Run (Spyder):
runfile(r"<PROJECT_ROOT>\BRO_CPT_Paper1_Postprocess_v1_0.py",
        wdir=r"<PROJECT_ROOT>")

Inputs (expected under OUTPUT_DIR):
- boundary_catalog.csv
- realdata_profiles_meta.csv
- synthetic_validation_results.csv
- run_config.json   (optional but recommended)
- profiles_realdata\CPT_*.csv  (optional; enables Fig.3 example panels)

Outputs (created under OUTPUT_DIR/paper1_analysis):
- tables: *.csv, *.json
- figures: *.png, *.pdf
- manifest: manifest_sha256.json
- bundle: paper1_results_bundle.zip (includes raw outputs + analysis tables/figs)

"""
from __future__ import annotations

import os, json, hashlib, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CFG = {
    # Root folder containing outputs_paper1_v1
    "ROOT_DIR": None,  # None => auto-detect as directory containing this script
    # Outputs folder name used by the pipeline
    "OUTPUT_DIRNAME": "outputs_paper1_v1",
    # If profiles folder exists, Fig.3 will be produced
    "PROFILES_SUBDIR": "profiles_realdata",
    # Fig.3 panel count
    "N_EXAMPLE_PANELS": 6,
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _pick_examples(cpt_summary: pd.DataFrame, bc: pd.DataFrame) -> list[str]:
    bc = bc.copy()
    # SGS class (fallback cuts)
    high, med = 0.67, 0.33
    bc["sgs_class"] = np.where(bc["sgs"] >= high, "high", np.where(bc["sgs"] >= med, "medium", "low"))

    cs = cpt_summary.copy()
    cs["has_low"] = cs["cpt_id"].isin(bc.loc[bc["sgs_class"] == "low", "cpt_id"].unique())
    cs["has_medium"] = cs["cpt_id"].isin(bc.loc[bc["sgs_class"] == "medium", "cpt_id"].unique())

    picks = []
    # max boundaries
    picks.append(cs.sort_values("n_boundaries", ascending=False).iloc[0]["cpt_id"])
    # min boundaries
    picks.append(cs.sort_values("n_boundaries", ascending=True).iloc[0]["cpt_id"])
    # has low
    lows = cs[cs["has_low"]].sort_values("n_boundaries", ascending=False)["cpt_id"].tolist()
    if lows:
        picks.append(lows[0])
    # max bandwidth
    picks.append(cs.sort_values("bandwidth_max", ascending=False).iloc[0]["cpt_id"])
    # typical
    picks.append(cs.iloc[(cs["n_boundaries"] - cs["n_boundaries"].median()).abs().argsort().iloc[0]]["cpt_id"])
    # another medium if any
    meds = cs[cs["has_medium"]]["cpt_id"].tolist()
    for mid in meds:
        if mid not in picks:
            picks.append(mid)
            break

    uniq = []
    for p in picks:
        if p not in uniq:
            uniq.append(p)
    return uniq[:CFG["N_EXAMPLE_PANELS"]]

def main() -> None:
    root = Path(CFG["ROOT_DIR"]).resolve() if CFG["ROOT_DIR"] else Path(__file__).resolve().parent
    out = root / CFG["OUTPUT_DIRNAME"]
    out.mkdir(parents=True, exist_ok=True)

    # Inputs
    bc_path = out / "boundary_catalog.csv"
    meta_path = out / "realdata_profiles_meta.csv"
    syn_path = out / "synthetic_validation_results.csv"
    run_cfg_path = out / "run_config.json"

    missing = [str(p) for p in [bc_path, meta_path, syn_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    bc = pd.read_csv(bc_path)
    meta = pd.read_csv(meta_path)
    syn = pd.read_csv(syn_path)

    # Derived columns
    bc["bandwidth"] = bc["z_p90"] - bc["z_p10"]

    analysis = out / "paper1_analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    # ---- Tables ----
    mean_metrics = syn.groupby("noise")[["precision", "recall", "f1", "mae", "n_true", "n_pred", "tp", "fp", "fn"]].mean().reset_index()
    mean_metrics.to_csv(analysis / "fig4_synthetic_mean_metrics_by_noise.csv", index=False)

    cpt_summary = bc.groupby("cpt_id").agg(
        n_boundaries=("boundary_id", "count"),
        delta_z_median=("delta_z", "median"),
        delta_z_p90=("delta_z", lambda x: np.quantile(x, 0.9)),
        delta_z_max=("delta_z", "max"),
        bandwidth_median=("bandwidth", "median"),
        bandwidth_p90=("bandwidth", lambda x: np.quantile(x, 0.9)),
        bandwidth_max=("bandwidth", "max"),
        sgs_median=("sgs", "median"),
        support_mean=("support", "mean"),
        support_min=("support", "min"),
    ).reset_index()

    cpt_summary = cpt_summary.merge(
        meta[["bro_id", "depth_max", "csv"]],
        left_on="cpt_id",
        right_on="bro_id",
        how="left",
    ).drop(columns=["bro_id"])

    cpt_summary.to_csv(analysis / "realdata_cpt_level_summary.csv", index=False)

    # SGS/support counts
    high_cut, med_cut = 0.67, 0.33
    if run_cfg_path.exists():
        try:
            rc = json.loads(run_cfg_path.read_text(encoding="utf-8"))
            cuts = rc.get("SGS_CUTS", {})
            high_cut, med_cut = float(cuts.get("high", high_cut)), float(cuts.get("medium", med_cut))
        except Exception:
            pass

    def sgs_class(v: float) -> str:
        if v >= high_cut:
            return "high"
        if v >= med_cut:
            return "medium"
        return "low"

    bc["sgs_class"] = bc["sgs"].apply(sgs_class)
    bc["bandwidth"] = bc["z_p90"] - bc["z_p10"]

    sgs_counts = bc["sgs_class"].value_counts().reindex(["high", "medium", "low"]).fillna(0).astype(int).reset_index()
    sgs_counts.columns = ["sgs_class", "n_boundaries"]
    sgs_counts.to_csv(analysis / "realdata_sgs_class_counts.csv", index=False)

    support_counts = bc["support"].value_counts().sort_index().reset_index()
    support_counts.columns = ["support", "n_boundaries"]
    support_counts.to_csv(analysis / "realdata_support_counts.csv", index=False)

    overall = {
        "n_cpts": int(cpt_summary.shape[0]),
        "n_boundaries": int(bc.shape[0]),
        "boundaries_per_cpt_mean": float(cpt_summary["n_boundaries"].mean()),
        "boundaries_per_cpt_median": float(cpt_summary["n_boundaries"].median()),
        "boundaries_per_cpt_min": int(cpt_summary["n_boundaries"].min()),
        "boundaries_per_cpt_max": int(cpt_summary["n_boundaries"].max()),
        "delta_z_median": float(bc["delta_z"].median()),
        "delta_z_p90": float(np.quantile(bc["delta_z"], 0.9)),
        "delta_z_max": float(bc["delta_z"].max()),
        "bandwidth_median": float(bc["bandwidth"].median()),
        "bandwidth_p90": float(np.quantile(bc["bandwidth"], 0.9)),
        "bandwidth_max": float(bc["bandwidth"].max()),
        "sgs_median": float(bc["sgs"].median()),
        "sgs_min": float(bc["sgs"].min()),
        "support_counts": {int(k): int(v) for k, v in bc["support"].value_counts().sort_index().to_dict().items()},
        "sgs_cuts": {"high": high_cut, "medium": med_cut},
    }
    (analysis / "realdata_overall_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    # ---- Figures (PNG + PDF) ----
    def savefig(base: Path) -> None:
        plt.tight_layout()
        plt.savefig(str(base) + ".png", dpi=300)
        plt.savefig(str(base) + ".pdf")
        plt.close()

    # Fig4: synthetic mean F1
    plt.figure()
    plt.plot(mean_metrics["noise"], mean_metrics["f1"], marker="o")
    plt.xlabel("Noise level (relative)")
    plt.ylabel("Mean F1")
    plt.title("Synthetic boundary recovery (mean F1)")
    savefig(analysis / "fig4_synthetic_meanF1")

    # Real-data boundaries per CPT
    plt.figure(figsize=(10, 4))
    cs = cpt_summary.sort_values("n_boundaries", ascending=False)
    plt.plot(range(len(cs)), cs["n_boundaries"], marker="o")
    plt.xticks(range(len(cs)), cs["cpt_id"], rotation=90, fontsize=6)
    plt.ylabel("Boundaries per CPT")
    plt.title("Real-data: boundaries per CPT (pilot set)")
    savefig(analysis / "fig_realdata_boundaries_per_cpt")

    # Δz histogram
    plt.figure()
    plt.hist(bc["delta_z"].values, bins=30)
    plt.xlabel("Boundary Shift Δz (m)")
    plt.ylabel("Count")
    plt.title("Real-data: Boundary Shift distribution")
    savefig(analysis / "fig_realdata_delta_z_hist")

    # bandwidth histogram
    plt.figure()
    plt.hist(bc["bandwidth"].values, bins=30)
    plt.xlabel("Uncertainty band width (p90−p10) (m)")
    plt.ylabel("Count")
    plt.title("Real-data: uncertainty band width distribution")
    savefig(analysis / "fig_realdata_bandwidth_hist")

    # SGS histogram
    plt.figure()
    plt.hist(bc["sgs"].values, bins=30)
    plt.xlabel("SGS (Boundary Confidence Score)")
    plt.ylabel("Count")
    plt.title("Real-data: SGS distribution")
    savefig(analysis / "fig_realdata_sgs_hist")

    # ---- Fig.3 example panels (optional) ----
    profiles_dir = out / CFG["PROFILES_SUBDIR"]
    if profiles_dir.exists():
        picks = _pick_examples(cpt_summary, bc)
        n = len(picks)
        ncols = 2
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows))
        axes = np.array(axes).reshape(nrows, ncols)

        for idx, cpt in enumerate(picks):
            ax = axes[idx // ncols, idx % ncols]
            row = meta.loc[meta["bro_id"] == cpt]
            if row.empty:
                ax.set_title(f"{cpt} (missing meta)")
                ax.axis("off")
                continue
            csv_name = row.iloc[0]["csv"]
            p = profiles_dir / csv_name
            if not p.exists():
                ax.set_title(f"{cpt} (missing profile)")
                ax.axis("off")
                continue
            df = pd.read_csv(p)

            # expected columns
            cols = {c.lower(): c for c in df.columns}
            depth_col = cols.get("depth", "depth")
            qt_col = cols.get("qt", cols.get("corrected_cone_resistance"))
            fs_col = cols.get("fs", cols.get("local_friction"))

            z = df[depth_col].values
            qt = df[qt_col].values
            fs = df[fs_col].values

            ax.plot(qt, z, label="qt")
            ax.set_xlabel("qt")
            ax.set_ylabel("Depth (m)")
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)

            ax2 = ax.twiny()
            ax2.plot(fs, z, linestyle="--", label="fs")
            ax2.set_xlabel("fs")

            bci = bc[bc["cpt_id"] == cpt].copy()
            if not bci.empty:
                x_min, x_max = np.nanmin(qt), np.nanmax(qt)
                for _, r in bci.iterrows():
                    y1, y2 = float(r["z_p10"]), float(r["z_p90"])
                    if abs(y2 - y1) < 1e-9:
                        ax.axhline(float(r["z_hat"]), linewidth=0.8, alpha=0.35)
                    else:
                        ax.fill_between([x_min, x_max], [y1, y1], [y2, y2], alpha=0.10)
                        ax.axhline(float(r["z_hat"]), linewidth=0.8, alpha=0.35)

                dz_med = float(np.median(bci["delta_z"]))
                bw_med = float(np.median(bci["z_p90"] - bci["z_p10"]))
                sgs_med = float(np.median(bci["sgs"]))
                ax.text(
                    0.02,
                    0.02,
                    f"nB={len(bci)}\nΔz~{dz_med:.2f} m\nBW~{bw_med:.2f} m\nSGS~{sgs_med:.2f}",
                    transform=ax.transAxes,
                    fontsize=9,
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9),
                )

            ax.set_title(cpt)

        for j in range(n, nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")

        fig.suptitle("Example CPT panels with boundaries and uncertainty bands (pilot real-data set)", y=0.995, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(analysis / "fig3_example_cpt_panels.png", dpi=300)
        fig.savefig(analysis / "fig3_example_cpt_panels.pdf")
        plt.close(fig)

    # ---- Manifest for analysis outputs ----
    files = []
    for p in analysis.rglob("*"):
        if p.is_file():
            files.append({"file": p.relative_to(analysis).as_posix(), "sha256": sha256_file(p), "bytes": p.stat().st_size})
    (analysis / "manifest_sha256.json").write_text(json.dumps({"files": files}, indent=2), encoding="utf-8")

    # ---- Bundle ZIP ----
    bundle = out / "paper1_results_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # raw inputs
        for p in [bc_path, meta_path, syn_path, run_cfg_path]:
            if p.exists():
                z.write(p, arcname=f"paper1_raw/{p.name}")
        # analysis folder
        for p in analysis.rglob("*"):
            if p.is_file():
                z.write(p, arcname=f"paper1_analysis/{p.relative_to(analysis).as_posix()}")

    print(f"[done] Analysis outputs: {analysis}")
    print(f"[done] Bundle ZIP: {bundle}")

if __name__ == "__main__":
    main()
