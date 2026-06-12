# Scale-aware CPT boundary reporting with uncertainty bands

This repository contains the public reproducibility package for the manuscript:

**Scale-aware boundary reporting for cone penetration test profiles using uncertainty bands and confidence scores**

Target journal: *Computers and Geotechnics*

## Overview

Cone penetration test (CPT/CPTu) profiles often contain multiple depth transitions whose apparent position may depend on depth resolution, filtering scale, local noise, and boundary-detection parameters. This repository supports a scale-aware CPT boundary reporting workflow that summarizes boundary evidence using estimated boundary depth, P10-P90 uncertainty bands, scale-stability/confidence scores, synthetic boundary recovery metrics, real-data baseline comparison, sensitivity analysis, and publication-ready figures.

## Reproducibility status

This final public package includes the main boundary-confidence pipeline script, the postprocessing script, the final data-driven figure-generation script, the journal figure-package script, final manuscript figures, derived result and audit tables, environment files, manuscript declaration text, and checksum manifests.

Raw third-party BRO CPT/CPTu data are **not redistributed** in this repository. Users should obtain the raw public data from the official data provider and place it locally or set the `BRO_CPT_GPKG` environment variable before running the raw-data pipeline.

## Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate bro-cpt-boundary-reporting
```

Using pip on Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Using pip on Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data preparation

The raw BRO CPT/CPTu GeoPackage is intentionally not included. The main pipeline can read it from the `BRO_CPT_GPKG` environment variable, for example:

```bash
set BRO_CPT_GPKG=C:\path\to\brocptvolledigeset_v2_0.gpkg
python scripts/BRO_CPT_Paper1_BoundaryConfidence_Pipeline_v1_9.py
```

On Linux/macOS:

```bash
export BRO_CPT_GPKG=/path/to/brocptvolledigeset_v2_0.gpkg
python scripts/BRO_CPT_Paper1_BoundaryConfidence_Pipeline_v1_9.py
```

## Principal scripts

```text
scripts/00_run_all_workflow.py
scripts/BRO_CPT_Paper1_BoundaryConfidence_Pipeline_v1_9.py
scripts/BRO_CPT_Paper1_Postprocess_v1_0.py
scripts/11_make_publication_figures_data_driven_polished_v1_3.py
scripts/12_prepare_CG_submission_figure_package_v1_0.py
```

Some scripts may require editing their configuration block depending on the local location of raw and derived data.

## Final figures and derived results

Final manuscript figures are included under `figures/`. Small derived result tables and audit files are included under `derived_results/`.

## Archiving

After final manuscript submission or acceptance, create a versioned GitHub release and archive it with Zenodo or another persistent repository. Add the DOI to this README, `CITATION.cff`, `MANUSCRIPT_DECLARATIONS.md`, and the manuscript Code availability statement.


## Repository URL

https://github.com/fdikbas/bro-cpt-boundary-reporting

## License

Code is released under the MIT License unless otherwise stated. Data remain subject to the original data provider's terms.
