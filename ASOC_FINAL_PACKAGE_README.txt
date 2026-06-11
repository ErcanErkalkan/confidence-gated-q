ASOC STRICT-REVIEW FIXED PACKAGE

Main files in this cleaned revision:
1. manuscript_asoc_final.pdf
   - Recompiled 30-page manuscript after strict-review fixes.

2. paper/manuscript.tex and paper/references.bib
   - Editable LaTeX source and bibliography.

3. paper/figures/ and paper/generated/
   - Separate vector figures and generated LaTeX result tables used by the manuscript.

4. configs/
   - Main audited protocols plus reviewer-response full-run protocol templates:
     - configs/approximate_support_full_reviewer_protocol.json
     - configs/stronger_neural_full_reviewer_protocol.json

5. results/
   - Raw and summarized seed-level results for the executed evidence families.

6. STRICT_REVIEW_FIX_LOG.md
   - Concise record of the strict-review changes and remaining non-fabricated limitations.

Verification summary:
- pytest tests/test_agents.py tests/test_configs.py tests/test_envs.py tests/test_statistics.py: 31 passed
- scripts/audit_artifact.py: PASS
- scripts/audit_asoc_strong_revision.py: PASS
- PDF pages: 30
- PDF fonts: Type 1 Latin Modern; no Type 3 fonts detected
- Render verification: 30 pages rendered successfully

Persistent artifact concept DOI used consistently for this working package:
https://doi.org/10.5281/zenodo.20578927

Note:
No new Zenodo version-specific DOI was minted in this environment. If a journal-specific Zenodo release is created, replace the concept DOI with the version-specific DOI in the final proofs. The package does not fabricate unexecuted stronger-baseline or approximate-support results; it includes executable full-run protocols for those reviewer-requested checks.
