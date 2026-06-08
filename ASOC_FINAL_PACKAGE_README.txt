ASOC FINAL PACKAGE

Main files:
1. manuscript_asoc_final.pdf
   - Final compiled manuscript PDF.

2. submission_clean_asoc_final.zip
   - Journal-upload package containing manuscript.tex, references.bib, compiled PDF, separate vector figures, graphical abstract, highlights, cover letter, title page, declarations, and generated tables.

3. research_artifact_asoc_final.zip
   - Reproducibility artifact containing source code, configs, raw and summarized result files, tests, audit scripts, provenance, requirements, and metadata.

4. asoc_separate_figures_final.zip
   - Figure upload package containing the separate vector figures and graphical abstract.

Verification summary:
- pytest tests/test_agents.py tests/test_configs.py tests/test_envs.py tests/test_statistics.py: 31 passed
- scripts/audit_artifact.py: PASS
- scripts/audit_asoc_strong_revision.py: PASS
- PDF pages: 29
- PDF fonts: no Type 3 fonts detected
- Figure_01B old filename removed; MiniGrid figure is Figure_04.
- No PENDING / before journal submission / A reviewer should / MLWA / planned-unreported residue detected in final submission/artifact searches.

Persistent artifact concept DOI used consistently:
https://doi.org/10.5281/zenodo.20578927

Note:
No new Zenodo version-specific DOI was minted in this environment. The package uses the persistent concept DOI consistently and contains no fake DOI placeholder.
