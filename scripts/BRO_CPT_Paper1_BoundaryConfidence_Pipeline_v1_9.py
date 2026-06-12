
# -*- coding: utf-8 -*-
"""
BRO CPT Paper-1 - Boundary Shift + Boundary Confidence (SGS) pipeline (v1.5)

Fixes vs v1.3:
- Real-data extraction from BRO GeoPackage now uses robust SQLite table detection
  (survey -> cone_penetrometer_survey -> cone_penetration_test -> result) instead of geopandas "layer guessing".
- Avoids unsafe int() casting on BRO identifiers (e.g., "CPT000...").
- Writes gpkg_schema_report.txt when detection fails, to make debugging trivial.
- Still supports synthetic validation and produces publication-ready figure-data CSVs.

Run (Spyder):
runfile(r"<PROJECT_ROOT>/BRO_CPT_Paper1_BoundaryConfidence_Pipeline_v1_5.py",
        wdir=r"<PROJECT_ROOT>")
"""

from __future__ import annotations
import os
import re
import json
import time
import hashlib
import zipfile
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.ndimage import median_filter
from scipy.signal import find_peaks

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# -----------------------
# Configuration
# -----------------------
CFG: Dict = {
    # Data root name(s) searched under the script root AND sibling roots.
    "DATA_ROOT": "data_bro_cpt_pdok",
    "CACHE_DIRNAME": "cache",
    "EXTRACTED_DIRNAME": "extracted",
    "OUTPUT_DIRNAME": "outputs_paper1_v1",

    # Optional explicit path (if empty, will auto-discover; can also set env var BRO_CPT_GPKG).
    "GPKG_PATH": "",

    # Real-data sampling
    "MAX_CPTS": 200,
    "RANDOM_SEED": 42,
    "BBOX_WGS84": None,  # [minx,miny,maxx,maxy] or None

    # Required channels in result table (qc/fs are the key ones; u2 optional).
    "REQUIRE_CHANNELS": ["qt", "fs"],

    # Profile constraints
    "MIN_DEPTH_RANGE_M": 8.0,
    "MIN_N_SAMPLES": 200,  # after dropping NaNs in required channels

    # Boundary method parameters
    "SCALES_M": [0.1, 0.2, 0.4, 0.8, 1.6],
    "MIN_SCALE_SUPPORT": 3,
    "MATCH_TOL_M": 0.35,
    "MIN_SEP_M": 0.25,
    "PEAK_PROM_QUANTILE": 0.90,
    "PEAK_PROM_MIN": 0.0,
    "TAU_M": 0.35,
    "SGS_CUTS": {"high": 0.67, "medium": 0.33},

    # Which stages to run
    "RUN_REAL_DATA": True,
    "RUN_SYNTHETIC_VALIDATION": True,

    # Synthetic validation
    "SYNTH_N_PROFILES": 200,
    "SYNTH_NOISE_LEVELS": [0.02, 0.05, 0.10, 0.20],
    "SYNTH_TOL_M": 0.35,

    # Outputs
    "SAVE_FIGURES": True,
    "SAVE_PROFILE_EXAMPLES": 6,
    "OVERWRITE": True,

    # Postprocess (tables+figures+bundle)
    "RUN_POSTPROCESS": True,
    # Optional: set to an external profiles_realdata folder if not under outputs.
    "PROFILES_DIR_OVERRIDE": "",
}

# -----------------------
# Logging / utilities
# -----------------------
def _log(msg: str) -> None:
    print(msg, flush=True)

def _root_dir() -> Path:
    return Path(__file__).resolve().parent

def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(s))[:140]

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# -----------------------
# GeoPackage discovery
# -----------------------
def _candidate_roots(script_root: Path) -> List[Path]:
    roots = [script_root]
    # sibling "BRO CPT" next to "BRO CPT Paper1"
    sib = script_root.parent / "BRO CPT"
    if sib.exists():
        roots.append(sib)
    # heuristic: remove " Paper1"/" Paper-1"/" Paper 1"
    name = script_root.name.lower()
    for token in [" paper1", " paper-1", " paper 1", "_paper1", "-paper1", "paper1"]:
        if token in name:
            base = Path(str(script_root)).parent / re.sub(token.replace(" ", r"\s*"), "", script_root.name, flags=re.IGNORECASE)
            # the regex approach may not yield a valid path name; keep a simpler one:
            base2 = script_root.parent / script_root.name.replace(" Paper1", "").replace(" Paper-1", "").replace(" Paper 1", "")
            if base2.exists():
                roots.append(base2)
    # de-duplicate
    out = []
    seen = set()
    for r in roots:
        rr = r.resolve()
        if rr not in seen:
            out.append(rr)
            seen.add(rr)
    return out

def _find_gpkg(script_root: Path) -> Path:
    # 1) explicit CFG
    if CFG.get("GPKG_PATH"):
        p = Path(CFG["GPKG_PATH"])
        if p.exists():
            return p
    # 2) env var
    env = os.environ.get("BRO_CPT_GPKG", "").strip()
    if env:
        p = Path(env)
        if p.exists():
            return p

    # 3) search likely roots
    data_root_name = CFG["DATA_ROOT"]
    extracted = CFG["EXTRACTED_DIRNAME"]
    cache = CFG["CACHE_DIRNAME"]

    roots = _candidate_roots(script_root)
    gpkg_hits: List[Path] = []

    def add_hits(base: Path):
        if not base.exists():
            return
        gpkg_hits.extend(sorted(base.glob("*.gpkg"), key=lambda x: x.stat().st_size, reverse=True))

    for r in roots:
        dr = r / data_root_name
        # prefer extracted
        add_hits(dr / extracted)
        add_hits(dr / cache)
        # fallback: any gpkg under data_root
        if dr.exists():
            gpkg_hits.extend(sorted(dr.rglob("*.gpkg"), key=lambda x: x.stat().st_size, reverse=True))

    if not gpkg_hits:
        raise FileNotFoundError(f"No .gpkg found under candidate roots for: {script_root}\\{data_root_name}")

    # Prefer canonical filename if present
    for p in gpkg_hits:
        if p.name.lower() == "brocptvolledigeset_v2_0.gpkg":
            return p

    # Else largest
    return gpkg_hits[0]

# -----------------------
# SQLite schema helpers
# -----------------------
def _list_tables(cur) -> List[str]:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [r[0] for r in cur.fetchall()]

def _table_cols(cur, t: str) -> List[Tuple[int, str, str]]:
    # PRAGMA table_info: cid, name, type, notnull, dflt_value, pk
    cur.execute(f"PRAGMA table_info('{t}')")
    return [(int(r[0]), str(r[1]), str(r[2])) for r in cur.fetchall()]

def _write_schema_report(gpkg: Path, out_dir: Path) -> Path:
    report = out_dir / "gpkg_schema_report.txt"
    conn = sqlite3.connect(str(gpkg))
    cur = conn.cursor()
    tables = _list_tables(cur)
    lines = []
    lines.append(f"GeoPackage: {gpkg}")
    lines.append(f"Tables: {len(tables)}")
    lines.append("")
    for t in sorted(tables):
        cols = _table_cols(cur, t)
        lines.append(f"[{t}] ({len(cols)} cols)")
        for _, name, ctype in cols[:200]:
            lines.append(f"  - {name} :: {ctype}")
        if len(cols) > 200:
            lines.append("  ... (truncated)")
        lines.append("")
    conn.close()
    report.write_text("\n".join(lines), encoding="utf-8")
    return report

def _score_table_name(name: str, want: str) -> int:
    n = name.lower()
    sc = 0
    if want == "survey":
        if n == "geotechnical_cpt_survey":
            sc += 1000
        if "cpt" in n and "survey" in n:
            sc += 300
        if "geotechnical" in n:
            sc += 50
    if want == "test":
        if n == "cone_penetration_test":
            sc += 1000
        if "penetration" in n and "test" in n:
            sc += 300
        if "cone" in n and "test" in n:
            sc += 150
    if want == "result":
        if n == "cone_penetration_test_result":
            sc += 1000
        if "penetration" in n and "result" in n:
            sc += 300
        if "cpt" in n and "result" in n:
            sc += 150
    return sc

def _pick_col(cols: List[Tuple[int,str,str]], candidates: List[str]) -> Optional[str]:
    names = [c[1] for c in cols]
    low = [n.lower() for n in names]
    for cand in candidates:
        if cand.lower() in low:
            return names[low.index(cand.lower())]
    return None

def _col_is_numeric(ctype: str) -> bool:
    t = (ctype or "").lower()
    return any(k in t for k in ["real", "float", "double", "numeric", "int", "integer", "decimal"])

# -----------------------
# Detect BRO CPT tables robustly (survey -> cone_penetrometer_survey -> cone_penetration_test -> result)
# -----------------------
def _detect_bro_tables(gpkg: Path, out_dir: Path) -> Dict[str, str]:
    """
    Detect the BRO CPT relational chain in the GeoPackage.

    This GeoPackage schema (brocptvolledigeset_v2_0.gpkg) uses the chain:
        geotechnical_cpt_survey
            -> cone_penetrometer_survey
                -> cone_penetration_test
                    -> cone_penetration_test_result

    We keep detection robust (name+column scoring) but strongly prefer the
    canonical table/column names when present.
    """
    conn = sqlite3.connect(str(gpkg))
    cur = conn.cursor()
    tables = _list_tables(cur)

    def has_cols(t: str, want: List[str]) -> bool:
        cols = [c[1].lower() for c in _table_cols(cur, t)]
        return all(w.lower() in cols for w in want)

    # ---- 1) Survey table (canonical) ----
    survey_table = None
    if "geotechnical_cpt_survey" in tables and has_cols("geotechnical_cpt_survey", ["geotechnical_cpt_survey_pk", "bro_id"]):
        survey_table = "geotechnical_cpt_survey"
    else:
        # fallback scoring
        best = (-1, None)
        for t in tables:
            sc = _score_table_name(t, "survey")
            if sc > best[0]:
                best = (sc, t)
        survey_table = best[1]

    if survey_table is None:
        conn.close()
        raise RuntimeError("Could not locate a CPT survey table.")

    survey_cols = _table_cols(cur, survey_table)
    bro_col = _pick_col(survey_cols, ["bro_id", "broId", "broidentificatie", "identificatie"]) or _pick_col(survey_cols, ["id"])
    survey_pk_col = _pick_col(survey_cols, ["geotechnical_cpt_survey_pk", "survey_pk", "surveyPk", "survey_id", "surveyid"])
    if survey_pk_col is None:
        for _, name, _ in survey_cols:
            nl = name.lower()
            if "survey" in nl and nl.endswith("_pk"):
                survey_pk_col = name
                break

    if bro_col is None or survey_pk_col is None:
        rep = _write_schema_report(gpkg, out_dir)
        conn.close()
        raise RuntimeError(f"Could not detect bro_id/survey_pk columns in survey table '{survey_table}'. See: {rep}")

    # ---- 2) Cone penetrometer survey (links survey -> CPS) ----
    cps_table = None
    if "cone_penetrometer_survey" in tables and has_cols("cone_penetrometer_survey", ["cone_penetrometer_survey_pk", "geotechnical_cpt_survey_fk"]):
        cps_table = "cone_penetrometer_survey"
    else:
        best = (-1, None)
        for t in tables:
            tl = t.lower()
            if "penetrometer" not in tl or "survey" not in tl:
                continue
            cols = _table_cols(cur, t)
            # need one PK and one FK to geotechnical survey
            names = [c[1].lower() for c in cols]
            if not any(n.endswith("_pk") for n in names):
                continue
            if not any(("geotechnical" in n and "survey" in n and n.endswith("_fk")) for n in names):
                continue
            sc = 0
            if tl == "cone_penetrometer_survey":
                sc += 200
            sc += _score_table_name(t, "survey")
            if sc > best[0]:
                best = (sc, t)
        cps_table = best[1]

    if cps_table is None:
        rep = _write_schema_report(gpkg, out_dir)
        conn.close()
        raise RuntimeError(f"Could not locate cone penetrometer survey table linking survey->CPS. See: {rep}")

    cps_cols = _table_cols(cur, cps_table)
    cps_pk_col = _pick_col(cps_cols, ["cone_penetrometer_survey_pk"])
    if cps_pk_col is None:
        for _, name, _ in cps_cols:
            if name.lower().endswith("_pk"):
                cps_pk_col = name
                break
    cps_survey_fk_col = _pick_col(cps_cols, ["geotechnical_cpt_survey_fk"])
    if cps_survey_fk_col is None:
        # fallback: any FK containing both 'geotechnical' and 'survey'
        for _, name, _ in cps_cols:
            nl = name.lower()
            if "geotechnical" in nl and "survey" in nl and nl.endswith("_fk"):
                cps_survey_fk_col = name
                break

    if cps_pk_col is None or cps_survey_fk_col is None:
        rep = _write_schema_report(gpkg, out_dir)
        conn.close()
        raise RuntimeError(f"Could not detect CPS pk/fk columns in '{cps_table}'. See: {rep}")

    # ---- 3) Penetration test table (links CPS -> test) ----
    test_table = None
    if "cone_penetration_test" in tables and has_cols("cone_penetration_test", ["cone_penetration_test_pk", "cone_penetrometer_survey_fk"]):
        test_table = "cone_penetration_test"
    else:
        best = (-1, None)
        for t in tables:
            tl = t.lower()
            if "penetration" not in tl or "test" not in tl:
                continue
            cols = _table_cols(cur, t)
            names = [c[1].lower() for c in cols]
            if not any(n.endswith("_pk") for n in names):
                continue
            if not any(("penetrometer" in n and "survey" in n and n.endswith("_fk")) for n in names):
                continue
            sc = 0
            if tl == "cone_penetration_test":
                sc += 200
            sc += _score_table_name(t, "test")
            if sc > best[0]:
                best = (sc, t)
        test_table = best[1]

    if test_table is None:
        rep = _write_schema_report(gpkg, out_dir)
        conn.close()
        raise RuntimeError(f"Could not locate CPT penetration test table linking CPS->test. See: {rep}")

    test_cols = _table_cols(cur, test_table)
    test_pk_col = _pick_col(test_cols, ["cone_penetration_test_pk", "test_pk"])
    if test_pk_col is None:
        for _, name, _ in test_cols:
            if name.lower().endswith("_pk"):
                test_pk_col = name
                break
    test_cps_fk_col = _pick_col(test_cols, ["cone_penetrometer_survey_fk"])
    if test_cps_fk_col is None:
        for _, name, _ in test_cols:
            nl = name.lower()
            if "penetrometer" in nl and "survey" in nl and nl.endswith("_fk"):
                test_cps_fk_col = name
                break

    if test_pk_col is None or test_cps_fk_col is None:
        rep = _write_schema_report(gpkg, out_dir)
        conn.close()
        raise RuntimeError(f"Could not detect test pk/fk columns in '{test_table}'. See: {rep}")

    # ---- 4) Result/measurement table (links test -> rows) ----
    result_table = None
    if "cone_penetration_test_result" in tables and has_cols("cone_penetration_test_result", ["cone_penetration_test_fk", "depth"]):
        result_table = "cone_penetration_test_result"
    else:
        best = (-1, None)
        for t in tables:
            tl = t.lower()
            if "result" not in tl:
                continue
            cols = _table_cols(cur, t)
            names = [c[1].lower() for c in cols]
            if not any("depth" == n or n.endswith("depth") for n in names):
                continue
            if not any(n.endswith("_fk") and "test" in n for n in names):
                continue
            sc = 0
            if tl == "cone_penetration_test_result":
                sc += 200
            sc += _score_table_name(t, "result")
            if sc > best[0]:
                best = (sc, t)
        result_table = best[1]

    if result_table is None:
        rep = _write_schema_report(gpkg, out_dir)
        conn.close()
        raise RuntimeError(f"Could not locate CPT result/measurement table. See: {rep}")

    res_cols = _table_cols(cur, result_table)
    # FK to test
    test_fk_col = _pick_col(res_cols, ["cone_penetration_test_fk", "test_pk"])
    if test_fk_col is None:
        for _, name, _ in res_cols:
            nl = name.lower()
            if nl.endswith("_fk") and "test" in nl:
                test_fk_col = name
                break

    depth_col = _pick_col(res_cols, ["depth", "penetration_depth", "depth_m", "z"])
    # Prefer corrected_cone_resistance for qt
    qt_col = _pick_col(res_cols, ["corrected_cone_resistance", "cone_resistance", "qc", "tip_resistance", "q_c", "qc_mpa"])
    fs_col = _pick_col(res_cols, ["local_friction", "fs", "sleeve_friction", "f_s", "fs_mpa"])
    u2_col = _pick_col(res_cols, ["pore_pressure_u2", "u2", "u_2", "pore_pressure_2", "u2_kpa", "u2_mpa"])

    if test_fk_col is None or depth_col is None or qt_col is None or fs_col is None:
        rep = _write_schema_report(gpkg, out_dir)
        conn.close()
        raise RuntimeError(f"Could not detect required result columns in '{result_table}'. See: {rep}")

    conn.close()

    return {
        "survey_table": survey_table,
        "bro_col": bro_col,
        "survey_pk_col": survey_pk_col,
        "cps_table": cps_table,
        "cps_pk_col": cps_pk_col,
        "cps_survey_fk_col": cps_survey_fk_col,
        "test_table": test_table,
        "test_pk_col": test_pk_col,
        "test_cps_fk_col": test_cps_fk_col,
        "result_table": result_table,
        "test_fk_col": test_fk_col,
        "depth_col": depth_col,
        "qt_col": qt_col,
        "fs_col": fs_col,
        "u2_col": u2_col or "",
    }

# -----------------------
# Real-data extraction (profiles)
# -----------------------
def _extract_profiles_from_gpkg(gpkg: Path, out_dir: Path) -> Tuple[pd.DataFrame, Path]:
    """
    Extract a random subset of CPT profiles from gpkg into CSVs and return:
    - df_meta: one row per CPT (bro_id, survey_pk, test_pk, n_rows, depth_max)
    - profiles_dir path
    """
    _ensure_dir(out_dir)
    profiles_dir = out_dir / "profiles_realdata"
    _ensure_dir(profiles_dir)

    schema = _detect_bro_tables(gpkg, out_dir)
    _log(f"[schema] Survey table: {schema['survey_table']}")
    _log(f"[schema] Test table: {schema['test_table']}")
    _log(f"[schema] Result table: {schema['result_table']}")

    conn = sqlite3.connect(str(gpkg))
    cur = conn.cursor()

    # Pull survey keys
    survey_table = schema["survey_table"]
    bro_col = schema["bro_col"]
    survey_pk_col = schema["survey_pk_col"]

    # Random subset of surveys
    q = f"SELECT {survey_pk_col}, {bro_col} FROM '{survey_table}'"
    df_sur = pd.read_sql_query(q, conn)
    df_sur = df_sur.dropna(subset=[survey_pk_col, bro_col]).copy()
    df_sur[bro_col] = df_sur[bro_col].astype(str)

    rng = np.random.default_rng(int(CFG["RANDOM_SEED"]))
    if len(df_sur) > int(CFG["MAX_CPTS"]):
        take = rng.choice(df_sur.index.values, size=int(CFG["MAX_CPTS"]), replace=False)
        df_sur = df_sur.loc[take].copy()

        # Map survey -> cone_penetrometer_survey (CPS) -> penetration test
    cps_table = schema["cps_table"]
    cps_pk_col = schema["cps_pk_col"]
    cps_survey_fk_col = schema["cps_survey_fk_col"]

    test_table = schema["test_table"]
    test_pk_col = schema["test_pk_col"]
    test_cps_fk_col = schema["test_cps_fk_col"]

    # --- survey -> CPS ---
    survey_keys = df_sur[survey_pk_col].tolist()
    cps_rows = []
    chunk = 900
    for i in range(0, len(survey_keys), chunk):
        part = survey_keys[i:i+chunk]
        qmarks = ",".join(["?"] * len(part))
        cur.execute(
            f"SELECT {cps_survey_fk_col}, {cps_pk_col} FROM '{cps_table}' WHERE {cps_survey_fk_col} IN ({qmarks})",
            part
        )
        cps_rows.extend(cur.fetchall())

    if not cps_rows:
        rep = _write_schema_report(gpkg, out_dir)
        conn.close()
        raise RuntimeError(f"No survey->CPS mappings found. See schema report: {rep}")

    df_map1 = pd.DataFrame(cps_rows, columns=["survey_key", "cps_key"])
    df_sur2 = df_sur.copy()
    df_sur2["survey_key"] = df_sur2[survey_pk_col].astype(str)
    df_map1["survey_key"] = df_map1["survey_key"].astype(str)

    # If multiple CPS per survey, keep the smallest key (deterministic)
    df_map1["cps_key_num"] = pd.to_numeric(df_map1["cps_key"], errors="coerce")
    df_map1 = df_map1.sort_values(["survey_key", "cps_key_num"]).drop_duplicates("survey_key", keep="first")
    df_sur2 = df_sur2.merge(df_map1[["survey_key", "cps_key"]], on="survey_key", how="left")
    df_sur2 = df_sur2.dropna(subset=["cps_key"]).copy()

    # --- CPS -> penetration test ---
    cps_keys = df_sur2["cps_key"].tolist()
    test_rows = []
    for i in range(0, len(cps_keys), chunk):
        part = cps_keys[i:i+chunk]
        qmarks = ",".join(["?"] * len(part))
        cur.execute(
            f"SELECT {test_cps_fk_col}, {test_pk_col} FROM '{test_table}' WHERE {test_cps_fk_col} IN ({qmarks})",
            part
        )
        test_rows.extend(cur.fetchall())

    if not test_rows:
        rep = _write_schema_report(gpkg, out_dir)
        conn.close()
        raise RuntimeError(f"No CPS->test mappings found. See schema report: {rep}")

    df_map2 = pd.DataFrame(test_rows, columns=["cps_key", "test_key"])
    df_map2["cps_key"] = df_map2["cps_key"].astype(str)
    df_sur2["cps_key"] = df_sur2["cps_key"].astype(str)

    df_map2["test_key_num"] = pd.to_numeric(df_map2["test_key"], errors="coerce")
    df_map2 = df_map2.sort_values(["cps_key", "test_key_num"]).drop_duplicates("cps_key", keep="first")

    df_sur2 = df_sur2.merge(df_map2[["cps_key", "test_key"]], on="cps_key", how="left")
    df_sur2 = df_sur2.dropna(subset=["test_key"]).copy()
# Prepare result table query
    result_table = schema["result_table"]
    test_fk_col = schema["test_fk_col"]
    depth_col = schema["depth_col"]
    qt_col = schema["qt_col"]
    fs_col = schema["fs_col"]
    u2_col = schema["u2_col"] or ""

    cols = [test_fk_col, depth_col, qt_col, fs_col] + ([u2_col] if u2_col else [])
    cols_sql = ", ".join([f'"{c}"' for c in cols])

    # Fetch result rows
    test_keys = df_sur2["test_key"].astype(str).unique().tolist()
    rows = []
    for i in range(0, len(test_keys), 400):
        part = test_keys[i:i+400]
        qmarks = ",".join(["?"] * len(part))
        cur.execute(
            f"SELECT {cols_sql} FROM '{result_table}' WHERE {test_fk_col} IN ({qmarks})",
            part
        )
        rows.extend(cur.fetchall())

    if not rows:
        rep = _write_schema_report(gpkg, out_dir)
        conn.close()
        raise RuntimeError(f"No rows returned from result table. See schema report: {rep}")

    df_res = pd.DataFrame(rows, columns=cols)
    # Make joinable
    df_res[test_fk_col] = df_res[test_fk_col].astype(str)
    df_sur2["test_key"] = df_sur2["test_key"].astype(str)

    # Save profiles
    meta_rows = []
    groups = df_res.groupby(test_fk_col)
    it = groups if tqdm is None else tqdm(groups, total=len(groups), desc="Real CPTs", leave=False)

    label_by_test = dict(zip(df_sur2["test_key"], df_sur2[bro_col].astype(str)))

    for test_key, grp in it:
        cpt_label = label_by_test.get(str(test_key), str(test_key))
        prof = pd.DataFrame({
            "depth": pd.to_numeric(grp[depth_col], errors="coerce"),
            "qt": pd.to_numeric(grp[qt_col], errors="coerce"),
            "fs": pd.to_numeric(grp[fs_col], errors="coerce"),
        })
        if u2_col:
            prof["u2"] = pd.to_numeric(grp[u2_col], errors="coerce")

        prof = prof.dropna(subset=["depth","qt","fs"]).sort_values("depth")
        if prof.empty:
            continue

        # Minimum sample count after required-channel NaN removal
        if len(prof) < int(CFG.get("MIN_N_SAMPLES", 0)):
            continue

        depth_min = float(np.nanmin(prof["depth"].values))
        depth_max = float(np.nanmax(prof["depth"].values))
        if not np.isfinite(depth_min) or not np.isfinite(depth_max) or (depth_max - depth_min) < float(CFG["MIN_DEPTH_RANGE_M"]):
            continue

        safe = _safe_name(cpt_label)
        out_csv = profiles_dir / f"CPT_{safe}.csv"
        prof.to_csv(out_csv, index=False, encoding="utf-8")

        meta_rows.append({
            "bro_id": str(cpt_label),
            "survey_pk": str(df_sur2.loc[df_sur2["test_key"] == str(test_key), "survey_key"].iloc[0]) if (df_sur2["test_key"] == str(test_key)).any() else "",
            "test_pk": str(test_key),
            "n_rows": int(len(prof)),
            "depth_max": depth_max,
            "csv": out_csv.name,
        })

    conn.close()

    df_meta = pd.DataFrame(meta_rows)
    meta_path = out_dir / "realdata_profiles_meta.csv"
    df_meta.to_csv(meta_path, index=False, encoding="utf-8")
    _log(f"[save] {meta_path} (n={len(df_meta)})")
    return df_meta, profiles_dir

# -----------------------
# Boundary extraction core
# -----------------------
def _median_filter_1d(x: np.ndarray, win_samples: int) -> np.ndarray:
    if win_samples < 3:
        return x.astype(float, copy=True)
    # median_filter works for odd/even, but we keep odd for interpretability
    if win_samples % 2 == 0:
        win_samples += 1
    return median_filter(x.astype(float), size=win_samples, mode="nearest")

def _compute_candidates(depth: np.ndarray, qt: np.ndarray, fs: np.ndarray,
                       scales_m: List[float], min_sep_m: float,
                       prom_q: float, prom_min: float) -> Dict[float, Dict]:
    """
    Per-scale boundary candidate detection using aggregated robust gradients.
    Returns dict: scale -> {"peaks_z": array, "peaks_prom": array}
    """
    # Need at least 3 samples for stable gradient-based evidence
    if depth.size < 3 or qt.size < 3 or fs.size < 3:
        return {float(w): {"peaks_idx": np.array([], dtype=int), "peaks_z": np.array([], dtype=float), "peaks_prom": np.array([], dtype=float)}
                for w in scales_m}

    # Estimate dz
    dd = np.diff(depth)
    dz = float(np.nanmedian(dd[np.isfinite(dd) & (dd > 0)])) if dd.size else np.nan
    if not np.isfinite(dz) or dz <= 0:
        dz = 0.02  # fallback

    # Robust normalize gradients to prevent unit domination
    def grad_mag(x: np.ndarray) -> np.ndarray:
        g = np.gradient(x, dz)
        return np.abs(g)

    out = {}
    min_sep_samp = max(1, int(round(min_sep_m / dz)))

    # compute robust scale for qt and fs
    def robust_scale(x):
        m = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - m))
        return float(mad) if np.isfinite(mad) and mad > 0 else 1.0

    qt_scale = robust_scale(qt)
    fs_scale = robust_scale(fs)

    for w in scales_m:
        win = max(3, int(round(w / dz)))
        qt_sm = _median_filter_1d(qt, win)
        fs_sm = _median_filter_1d(fs, win)

        e = grad_mag(qt_sm) / qt_scale + grad_mag(fs_sm) / fs_scale
        e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)

        # prominence threshold
        thr = float(np.quantile(e, prom_q)) if e.size else 0.0
        thr = max(thr, float(prom_min))

        peaks, props = find_peaks(e, distance=min_sep_samp, prominence=thr)
        peaks = peaks.astype(int)
        prom = props.get("prominences", np.array([], dtype=float))

        out[float(w)] = {
            "peaks_idx": peaks,
            "peaks_z": depth[peaks] if peaks.size else np.array([], dtype=float),
            "peaks_prom": prom,
        }
    return out

def _match_across_scales(cands: Dict[float, Dict], match_tol_m: float, min_support: int) -> List[Dict]:
    """
    Simple 1D clustering of candidate depths across scales with tolerance.
    Returns list of matched groups, each has scale->z mapping.
    """
    pts = []
    for w, d in cands.items():
        for z, pr in zip(d["peaks_z"], d["peaks_prom"]):
            pts.append((float(z), float(w), float(pr)))
    if not pts:
        return []
    pts.sort(key=lambda x: x[0])

    clusters = []
    cur = [pts[0]]
    for p in pts[1:]:
        if abs(p[0] - cur[-1][0]) <= match_tol_m:
            cur.append(p)
        else:
            clusters.append(cur)
            cur = [p]
    clusters.append(cur)

    groups = []
    for cl in clusters:
        # per-scale keep highest prominence
        by_scale = {}
        for z, w, pr in cl:
            if (w not in by_scale) or (pr > by_scale[w][1]):
                by_scale[w] = (z, pr)
        if len(by_scale) < int(min_support):
            continue
        groups.append({
            "support": len(by_scale),
            "z_by_scale": {w: by_scale[w][0] for w in by_scale},
        })
    return groups

def _boundary_metrics(groups: List[Dict], tau_m: float) -> List[Dict]:
    rows = []
    for gi, g in enumerate(groups, start=1):
        zs = np.array(list(g["z_by_scale"].values()), dtype=float)
        z_med = float(np.median(zs))
        dz = float(np.median(np.abs(zs - z_med)))
        p10 = float(np.quantile(zs, 0.10))
        p90 = float(np.quantile(zs, 0.90))
        sgs = 1.0 - min(dz / float(tau_m), 1.0)
        rows.append({
            "boundary_id": gi,
            "z_hat": z_med,
            "delta_z": dz,
            "sgs": float(sgs),
            "z_p10": p10,
            "z_p90": p90,
            "support": int(g["support"]),
        })
    return rows

def _process_profile_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # required cols
    for c in ["depth", "qt", "fs"]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {csv_path.name}")
    df = df.dropna(subset=["depth", "qt", "fs"]).sort_values("depth")
    return df

def _compute_profile_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    depth = df["depth"].to_numpy(float)
    qt = df["qt"].to_numpy(float)
    fs = df["fs"].to_numpy(float)

    cands = _compute_candidates(
        depth=depth, qt=qt, fs=fs,
        scales_m=list(CFG["SCALES_M"]),
        min_sep_m=float(CFG["MIN_SEP_M"]),
        prom_q=float(CFG["PEAK_PROM_QUANTILE"]),
        prom_min=float(CFG["PEAK_PROM_MIN"]),
    )
    groups = _match_across_scales(
        cands=cands,
        match_tol_m=float(CFG["MATCH_TOL_M"]),
        min_support=int(CFG["MIN_SCALE_SUPPORT"]),
    )
    rows = _boundary_metrics(groups, tau_m=float(CFG["TAU_M"]))
    return pd.DataFrame(rows)

# -----------------------
# Synthetic validation (minimal but consistent)
# -----------------------
def _make_synth_profile(n: int = 800, dz: float = 0.02, noise: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    z = np.arange(n) * dz
    # 5 regimes with random boundary locations
    cuts = np.sort(rng.choice(np.arange(int(n*0.2), int(n*0.9)), size=4, replace=False))
    bounds = (cuts * dz).tolist()
    # piecewise constants + mild trend
    vals_qt = rng.uniform(2, 15, size=5)
    vals_fs = rng.uniform(0.02, 0.4, size=5)
    qt = np.zeros_like(z)
    fs = np.zeros_like(z)
    start = 0
    for i, cut in enumerate(list(cuts) + [n]):
        qt[start:cut] = vals_qt[i] + 0.05 * z[start:cut]
        fs[start:cut] = vals_fs[i] + 0.005 * z[start:cut]
        start = cut
    # add noise proportional to scale
    qt_noisy = qt * (1 + rng.normal(0, noise, size=n))
    fs_noisy = fs * (1 + rng.normal(0, noise, size=n))
    # sparse spikes
    for _ in range(8):
        j = rng.integers(10, n-10)
        qt_noisy[j] *= rng.uniform(0.5, 1.8)
        fs_noisy[j] *= rng.uniform(0.5, 1.8)
    df = pd.DataFrame({"depth": z, "qt": qt_noisy, "fs": fs_noisy})
    return df, bounds

def _match_pred_to_true(pred: np.ndarray, true: np.ndarray, tol: float) -> Tuple[int,int,int,float]:
    pred = np.sort(np.asarray(pred, float))
    true = np.sort(np.asarray(true, float))
    used = np.zeros(len(true), dtype=bool)
    tp = 0
    errs = []
    for p in pred:
        jbest = None
        dbest = None
        for j, t in enumerate(true):
            if used[j]:
                continue
            d = abs(p - t)
            if d <= tol and (dbest is None or d < dbest):
                dbest = d
                jbest = j
        if jbest is not None:
            used[jbest] = True
            tp += 1
            errs.append(dbest)
    fp = len(pred) - tp
    fn = len(true) - tp
    mae = float(np.mean(errs)) if errs else float("nan")
    return tp, fp, fn, mae

def _synthetic_validation(out_dir: Path) -> Tuple[pd.DataFrame, Path]:
    rows = []
    n_profiles = int(CFG["SYNTH_N_PROFILES"])
    tol = float(CFG["SYNTH_TOL_M"])
    noise_levels = list(CFG["SYNTH_NOISE_LEVELS"])
    rng = np.random.default_rng(int(CFG["RANDOM_SEED"]))

    for nl in noise_levels:
        it = range(n_profiles)
        if tqdm is not None:
            it = tqdm(it, desc=f"Synthetic nl={nl}", leave=False)
        for k in it:
            seed = int(rng.integers(0, 1_000_000_000))
            df, true_bounds = _make_synth_profile(noise=float(nl), seed=seed)
            bd = _compute_profile_boundaries(df)
            pred = bd["z_hat"].to_numpy(float) if not bd.empty else np.array([], float)
            tp, fp, fn, mae = _match_pred_to_true(pred, np.array(true_bounds), tol)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            rows.append({
                "noise": float(nl),
                "tp": int(tp), "fp": int(fp), "fn": int(fn),
                "precision": float(prec), "recall": float(rec), "f1": float(f1),
                "mae": float(mae) if np.isfinite(mae) else np.nan,
                "n_true": int(len(true_bounds)),
                "n_pred": int(len(pred)),
            })

    dfv = pd.DataFrame(rows)
    out_csv = out_dir / "synthetic_validation_results.csv"
    dfv.to_csv(out_csv, index=False, encoding="utf-8")
    _log(f"[save] {out_csv} (n={len(dfv)})")

    # figure (mean F1 vs noise) using matplotlib only if available
    fig_path = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        g = dfv.groupby("noise")["f1"].mean().reset_index()
        plt.figure()
        plt.plot(g["noise"], g["f1"], marker="o")
        plt.title("Synthetic boundary recovery (mean F1)")
        plt.xlabel("Noise level (relative)")
        plt.ylabel("Mean F1")
        fig_path = out_dir / "fig_synth_f1.png"
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close("all")
        _log(f"[save] {fig_path}")
    except Exception as e:
        _log(f"[warn] Could not save fig_synth_f1.png: {e}")

    return dfv, fig_path

# -----------------------
# Main runner
# -----------------------
def main() -> None:
    t0 = time.time()
    root = _root_dir()
    out_dir = root / CFG["OUTPUT_DIRNAME"]
    _ensure_dir(out_dir)

    # Save run config
    cfg_path = out_dir / "run_config.json"
    cfg_path.write_text(json.dumps(CFG, indent=2), encoding="utf-8")
    _log(f"[run] {cfg_path}")

    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "files": []}

    def add_manifest(p: Path):
        if p is None or (not Path(p).exists()):
            return
        pp = Path(p)
        manifest["files"].append({"file": pp.name, "sha256": _sha256(pp), "bytes": pp.stat().st_size})

    # 1) Real data
    if bool(CFG["RUN_REAL_DATA"]):
        try:
            gpkg = _find_gpkg(root)
            _log(f"[run] Using GeoPackage: {gpkg}")
            df_meta, profiles_dir = _extract_profiles_from_gpkg(gpkg, out_dir)

            # boundary catalog for extracted profiles
            cat_rows = []
            csvs = sorted(profiles_dir.glob("CPT_*.csv"))
            if csvs:
                it = csvs if tqdm is None else tqdm(csvs, desc="Boundaries (real)", leave=False)
                for csv in it:
                    try:
                        df = _process_profile_csv(csv)
                        # skip extremely short profiles after NaN removal
                        if df.shape[0] < 3:
                            continue
                        bd = _compute_profile_boundaries(df)
                        if bd.empty:
                            continue
                        bd.insert(0, "cpt_id", csv.stem.replace("CPT_", ""))
                        cat_rows.append(bd)
                    except Exception as _e:
                        # Per-profile failure should not abort batch; keep a small skip log.
                        _log(f"[warn] boundary skipped for {csv.name}: {_e}")
                        continue

                if cat_rows:
                    cat = pd.concat(cat_rows, ignore_index=True)
                else:
                    cat = pd.DataFrame(columns=["cpt_id","boundary_id","z_hat","delta_z","sgs","z_p10","z_p90","support"])

                cat_path = out_dir / "boundary_catalog.csv"
                cat.to_csv(cat_path, index=False, encoding="utf-8")
                _log(f"[save] {cat_path} (n={len(cat)})")
                add_manifest(cat_path)

            add_manifest(out_dir / "realdata_profiles_meta.csv")

        except Exception as e:
            # Ensure schema report exists if possible
            _log(f"[warn] Real-data extraction failed (continuing with synthetic only): {e}")

    # 2) Synthetic validation
    if bool(CFG["RUN_SYNTHETIC_VALIDATION"]):
        dfv, fig = _synthetic_validation(out_dir)
        add_manifest(out_dir / "synthetic_validation_results.csv")
        if fig:
            add_manifest(fig)

    # finalize manifest
    man_path = out_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _log(f"[save] {man_path}")

    _log(f"[done] total time: {time.time()-t0:.1f}s")



# -----------------------
# Postprocess: tables + figures + bundle
# -----------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _postprocess_outputs(out_dir: Path) -> None:
    """
    Create publication-ready summary tables/figures on disk and a single ZIP bundle.
    This is designed to avoid any reliance on external download links.
    """
    bc_path = out_dir / "boundary_catalog.csv"
    meta_path = out_dir / "realdata_profiles_meta.csv"
    syn_path = out_dir / "synthetic_validation_results.csv"
    run_cfg_path = out_dir / "run_config.json"

    if not bc_path.exists():
        _log("[post] boundary_catalog.csv not found; skipping postprocess.")
        return

    bc = pd.read_csv(bc_path)
    bc["bandwidth"] = bc["z_p90"] - bc["z_p10"]

    # SGS cuts
    high_cut, med_cut = 0.67, 0.33
    if run_cfg_path.exists():
        try:
            rc = json.loads(run_cfg_path.read_text(encoding="utf-8"))
            cuts = rc.get("SGS_CUTS", {})
            high_cut = float(cuts.get("high", high_cut))
            med_cut = float(cuts.get("medium", med_cut))
        except Exception:
            pass

    def sgs_class(v: float) -> str:
        if v >= high_cut:
            return "high"
        if v >= med_cut:
            return "medium"
        return "low"

    bc["sgs_class"] = bc["sgs"].apply(sgs_class)

    analysis = out_dir / "paper1_analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    # Matplotlib (for saving figures in postprocess)
    import matplotlib
    matplotlib.use('Agg')  # headless-safe
    import matplotlib.pyplot as plt


    # ---- Tables ----
    if syn_path.exists():
        syn = pd.read_csv(syn_path)
        mean_metrics = syn.groupby("noise")[["precision","recall","f1","mae","n_true","n_pred","tp","fp","fn"]].mean().reset_index()
        mean_metrics.to_csv(analysis / "fig4_synthetic_mean_metrics_by_noise.csv", index=False)

    meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()

    cpt_summary = bc.groupby("cpt_id").agg(
        n_boundaries=("boundary_id","count"),
        delta_z_median=("delta_z","median"),
        delta_z_p90=("delta_z", lambda x: np.quantile(x,0.9)),
        delta_z_max=("delta_z","max"),
        bandwidth_median=("bandwidth","median"),
        bandwidth_p90=("bandwidth", lambda x: np.quantile(x,0.9)),
        bandwidth_max=("bandwidth","max"),
        sgs_median=("sgs","median"),
        support_mean=("support","mean"),
        support_min=("support","min"),
    ).reset_index()

    if not meta.empty and "bro_id" in meta.columns:
        cpt_summary = cpt_summary.merge(
            meta[[c for c in ["bro_id","depth_max","csv"] if c in meta.columns]],
            left_on="cpt_id", right_on="bro_id", how="left"
        ).drop(columns=["bro_id"], errors="ignore")

    cpt_summary.to_csv(analysis / "realdata_cpt_level_summary.csv", index=False)

    sgs_counts = bc["sgs_class"].value_counts().reindex(["high","medium","low"]).fillna(0).astype(int).reset_index()
    sgs_counts.columns = ["sgs_class","n_boundaries"]
    sgs_counts.to_csv(analysis / "realdata_sgs_class_counts.csv", index=False)

    support_counts = bc["support"].value_counts().sort_index().reset_index()
    support_counts.columns = ["support","n_boundaries"]
    support_counts.to_csv(analysis / "realdata_support_counts.csv", index=False)

    overall = {
        "n_cpts": int(cpt_summary.shape[0]),
        "n_boundaries": int(bc.shape[0]),
        "boundaries_per_cpt_mean": float(cpt_summary["n_boundaries"].mean()),
        "boundaries_per_cpt_median": float(cpt_summary["n_boundaries"].median()),
        "boundaries_per_cpt_min": int(cpt_summary["n_boundaries"].min()),
        "boundaries_per_cpt_max": int(cpt_summary["n_boundaries"].max()),
        "delta_z_median": float(bc["delta_z"].median()),
        "delta_z_p90": float(np.quantile(bc["delta_z"],0.9)),
        "delta_z_max": float(bc["delta_z"].max()),
        "bandwidth_median": float(bc["bandwidth"].median()),
        "bandwidth_p90": float(np.quantile(bc["bandwidth"],0.9)),
        "bandwidth_max": float(bc["bandwidth"].max()),
        "sgs_median": float(bc["sgs"].median()),
        "sgs_min": float(bc["sgs"].min()),
        "support_counts": {int(k): int(v) for k,v in bc["support"].value_counts().sort_index().to_dict().items()},
        "sgs_cuts": {"high": high_cut, "medium": med_cut},
    }
    (analysis / "realdata_overall_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    # ---- Figures (PNG+PDF) ----
    def savefig(base: Path) -> None:
        plt.tight_layout()
        plt.savefig(str(base) + ".png", dpi=300)
        plt.savefig(str(base) + ".pdf")
        plt.close()

    # Fig4: synthetic mean F1
    if syn_path.exists():
        plt.figure()
        plt.plot(mean_metrics["noise"], mean_metrics["f1"], marker="o")
        plt.xlabel("Noise level (relative)")
        plt.ylabel("Mean F1")
        plt.title("Synthetic boundary recovery (mean F1)")
        savefig(analysis / "fig4_synthetic_meanF1")

    # Boundaries per CPT
    plt.figure(figsize=(10,4))
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
    profiles_dir = None
    if CFG.get("PROFILES_DIR_OVERRIDE"):
        p = Path(CFG["PROFILES_DIR_OVERRIDE"])
        if p.exists():
            profiles_dir = p
    if profiles_dir is None:
        cand = out_dir / "profiles_realdata"
        if cand.exists():
            profiles_dir = cand

    if profiles_dir is not None and not meta.empty and "csv" in meta.columns:
        # pick examples
        def pick_examples() -> List[str]:
            cs2 = cpt_summary.copy()
            cs2["has_low"] = cs2["cpt_id"].isin(bc.loc[bc["sgs_class"]=="low","cpt_id"].unique())
            cs2["has_medium"] = cs2["cpt_id"].isin(bc.loc[bc["sgs_class"]=="medium","cpt_id"].unique())

            picks = []
            picks.append(cs2.sort_values("n_boundaries", ascending=False).iloc[0]["cpt_id"])
            picks.append(cs2.sort_values("n_boundaries", ascending=True).iloc[0]["cpt_id"])
            lows = cs2[cs2["has_low"]].sort_values("n_boundaries", ascending=False)["cpt_id"].tolist()
            if lows: picks.append(lows[0])
            picks.append(cs2.sort_values("bandwidth_max", ascending=False).iloc[0]["cpt_id"])
            picks.append(cs2.iloc[(cs2["n_boundaries"]-cs2["n_boundaries"].median()).abs().argsort().iloc[0]]["cpt_id"])
            meds = cs2[cs2["has_medium"]]["cpt_id"].tolist()
            for mid in meds:
                if mid not in picks:
                    picks.append(mid); break
            uniq=[]
            for p in picks:
                if p not in uniq:
                    uniq.append(p)
            return uniq[:6]

        picks = pick_examples()
        n = len(picks)
        ncols = 2
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows))
        axes = np.array(axes).reshape(nrows, ncols)

        for idx, cpt in enumerate(picks):
            ax = axes[idx // ncols, idx % ncols]
            row = meta.loc[meta["bro_id"] == cpt]
            if row.empty:
                ax.set_title(f"{cpt} (missing meta)"); ax.axis("off"); continue
            csv_name = row.iloc[0]["csv"]
            p = profiles_dir / csv_name
            if not p.exists():
                ax.set_title(f"{cpt} (missing profile)"); ax.axis("off"); continue

            df = pd.read_csv(p)
            cols = {c.lower(): c for c in df.columns}
            depth_col = cols.get("depth","depth")
            qt_col = cols.get("qt", cols.get("corrected_cone_resistance"))
            fs_col = cols.get("fs", cols.get("local_friction"))

            z = df[depth_col].values
            qt = df[qt_col].values
            fs = df[fs_col].values

            ax.plot(qt, z, label="qt")
            ax.set_xlabel("qt"); ax.set_ylabel("Depth (m)")
            ax.invert_yaxis(); ax.grid(True, alpha=0.3)

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
                        ax.fill_between([x_min, x_max], [y1,y1], [y2,y2], alpha=0.10)
                        ax.axhline(float(r["z_hat"]), linewidth=0.8, alpha=0.35)

                dz_med = float(np.median(bci["delta_z"]))
                bw_med = float(np.median(bci["bandwidth"]))
                sgs_med = float(np.median(bci["sgs"]))
                ax.text(
                    0.02, 0.02,
                    f"nB={len(bci)}\nΔz~{dz_med:.2f} m\nBW~{bw_med:.2f} m\nSGS~{sgs_med:.2f}",
                    transform=ax.transAxes, fontsize=9, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9),
                )
            ax.set_title(cpt)

        for j in range(n, nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")

        fig.suptitle("Example CPT panels with boundaries and uncertainty bands (pilot real-data set)", y=0.995, fontsize=14)
        fig.tight_layout(rect=[0,0,1,0.98])
        fig.savefig(analysis / "fig3_example_cpt_panels.png", dpi=300)
        fig.savefig(analysis / "fig3_example_cpt_panels.pdf")
        plt.close(fig)

    # ---- Manifest for analysis outputs ----
    files = []
    for p in analysis.rglob("*"):
        if p.is_file():
            files.append({"file": p.relative_to(analysis).as_posix(), "sha256": _sha256_file(p), "bytes": p.stat().st_size})
    (analysis / "manifest_sha256.json").write_text(json.dumps({"files": files}, indent=2), encoding="utf-8")

    # ---- Bundle ZIP ----
    bundle = out_dir / "paper1_results_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # raw outputs
        for p in [bc_path, meta_path, syn_path, run_cfg_path]:
            if p.exists():
                z.write(p, arcname=f"paper1_raw/{p.name}")
        # analysis outputs
        for p in analysis.rglob("*"):
            if p.is_file():
                z.write(p, arcname=f"paper1_analysis/{p.relative_to(analysis).as_posix()}")

    _log(f"[post] saved analysis to: {analysis}")
    _log(f"[post] saved bundle zip to: {bundle}")


if __name__ == "__main__":
    main()
    if CFG.get("RUN_POSTPROCESS", True):
        try:
            _postprocess_outputs(_root_dir() / CFG["OUTPUT_DIRNAME"])
        except Exception as e:
            _log(f"[post][warn] Postprocess failed: {e}")