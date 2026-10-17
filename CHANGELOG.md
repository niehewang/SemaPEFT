# Changelog

All notable changes to SemaPEFT are documented here.

## [1.0.0] - 2026-10-17

### Added
- Reference PEFT/LoRA semantic interpretation utilities.
- Cross-runtime semantic extraction for pinned local vLLM and SGLang source snapshots.
- Exact low-rank QR/core certificate and dense oracle.
- Stable P64 certificate profile with float32 storage and float64 certificate computation.
- Proof-carrying repair utilities for exact scale lowering and undeclared-tensor removal.
- Numerical-closure reproduction scripts and corrected process-level systems aggregation.
- Compact validated summaries for runtime semantics, non-scale mismatch, application impact, benchmark coverage, numerical closure, and stable-certificate systems results.
- CPU smoke tests and GitHub Actions CI.
- Apache-2.0 license, citation metadata, contribution/security guidance, and issue/PR templates.

### Reproducibility notes
- The original E3 float32 correctness gate remains failed and is not reclassified.
- P64 is a separately predeclared stable certificate profile.
- Reported systems speedups are certificate-computation results on a single A800 system, not end-to-end inference speedups.
