# GitHub upload checklist

Repository:

https://github.com/fdikbas/bro-cpt-boundary-reporting

## Before pushing

- Confirm that no raw BRO GeoPackage or cache files are present.
- Confirm that `release_manifest.csv` is present.
- Confirm that `docs/source_code_completeness_report.csv` reports `compile_status=ok` for the public scripts.
- Confirm that final figures are present under `figures/`.
- Confirm that `MANUSCRIPT_DECLARATIONS.md` contains the GitHub repository URL.
- Keep the Zenodo DOI placeholder until an archived release DOI is created.

## Suggested commands

```bash
git init
git add .
git commit -m "Initial reproducibility release for BRO CPT boundary reporting"
git branch -M main
git remote add origin https://github.com/fdikbas/bro-cpt-boundary-reporting.git
git push -u origin main
```

## After pushing

Create a GitHub release/tag and archive it with Zenodo if a permanent DOI is required by the journal or desired for citation.
