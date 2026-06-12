# Repository structure

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
