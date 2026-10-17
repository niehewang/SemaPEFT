# Contributing to SemaPEFT

Thank you for improving SemaPEFT. Contributions should preserve the repository's central distinction between artifact-byte authenticity and runtime execution-semantic integrity.

## Development setup

```bash
conda create -n semapeft python=3.10 -y
conda activate semapeft
pip install -e '.[test]'
pytest -q
```

## Pull requests

Please keep pull requests focused and include:

1. a concise description of the semantic or numerical behavior changed;
2. tests for new behavior or a reproducible failure case;
3. runtime/version/source hashes when the change is runtime-specific;
4. an explicit note if a change alters a certificate boundary, tolerance, aggregation rule, or fail-closed condition.

Do not silently relax numerical tolerances, reinterpret unsupported runtime features as conformant, or convert certificate-computation measurements into end-to-end inference claims.

## Adding runtime support

Runtime adapters should:
- operate from a pinned local source snapshot;
- record version/source hashes;
- isolate unrelated serving initialization when practical;
- return `UNSUPPORTED`/`UNKNOWN` rather than fabricating semantics when extraction is incomplete;
- include a conformance test against the reference PEFT interpretation.

## Data and model artifacts

Do not commit model weights, proprietary datasets, access tokens, private paths, internal experiment handoff documents, or third-party source trees.
