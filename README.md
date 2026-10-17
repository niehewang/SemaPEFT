# SemaPEFT

[![CI](https://github.com/niehewang/SemaPEFT/actions/workflows/ci.yml/badge.svg)](https://github.com/niehewang/SemaPEFT/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.10-blue.svg)](pyproject.toml)

**Release:** v1.0.0 · **License:** Apache-2.0 · **Status:** submission-ready research release


**Same Artifact, Different Model: Verifying Cross-Runtime Semantic Integrity of PEFT Adapters**

Reference implementation and reproducibility code for **SemaPEFT**, a framework for verifying that a parameter-efficient fine-tuning (PEFT) adapter preserves its intended execution semantics across deployment runtimes.

## Why SemaPEFT?

Cryptographic hashes and signatures establish artifact-byte authenticity, but they do **not** establish that two runtimes instantiate the same effective model. Differences in PEFT scaling, rank/alpha patterns, rsLoRA handling, target-module binding, fused layouts, undeclared tensors, or unsupported features can change the realized low-rank operator even when the adapter bytes are unchanged.

SemaPEFT checks the semantic object that matters: the runtime-realized adapter operator.

## What is included

- **Reference PEFT semantics** for LoRA scale, per-module patterns, and rsLoRA.
- **Runtime semantic extraction** from pinned local vLLM/SGLang source snapshots.
- **Exact low-rank certificate** using Gram matrices and thin-QR/core spectral computation.
- **Dense operator oracle** for verification and numerical studies.
- **P64 stable certificate profile**: float32 stored factors, float64 certificate computation.
- **Proof-carrying repair primitives**, including exact scale lowering and undeclared-tensor removal.
- **Corrected systems aggregation** over process-level medians with process-level bootstrap CIs.
- Compact, validated summaries from the completed experiments.

## Installation

```bash
conda create -n semapeft python=3.10 -y
conda activate semapeft
pip install -e .
```

For development/testing:

```bash
pip install -e '.[test]'
pytest -q
```

The final P64 systems measurements were obtained with PyTorch 2.5.1+cu121 on an NVIDIA A800 80GB PCIe GPU. CPU execution is sufficient for the smoke tests.

## Quick start

```python
import math, torch
from semapeft.certificate import p64_compare

g = torch.Generator().manual_seed(11)
d, r = 512, 8
A0 = torch.randn(r, d, generator=g) / math.sqrt(d)
B0 = torch.randn(d, r, generator=g) / math.sqrt(r)
A1 = A0 + 0.05 * torch.randn(r, d, generator=g) / math.sqrt(d)
B1 = B0 + 0.05 * torch.randn(d, r, generator=g) / math.sqrt(r)

result = p64_compare(A0, B0, A1, B1, tolerance=1e-10)
print(result)
```

or simply:

```bash
python experiments/quickstart_certificate.py
```

## Runtime semantic extraction

SemaPEFT can execute the relevant LoRA loader classes/functions from a **local pinned runtime source tree** while isolating unrelated serving initialization:

```bash
python experiments/probe_runtime_semantics.py   --runtime vllm   --version 0.31.0   --runtime-source /path/to/vllm-0.31.0-source   --adapter /path/to/peft-adapter
```

SGLang is supported analogously with `--runtime sglang`.

## Repair examples

Exact scale lowering:

```bash
python experiments/lower_adapter.py   --adapter /path/to/input_adapter   --output /path/to/lowered_adapter   --target-scale 2.0
```

Drop undeclared serialized LoRA tensors:

```bash
python experiments/drop_undeclared_tensors.py   --adapter /path/to/input_adapter   --output /path/to/repaired_adapter
```

Every repair should be followed by runtime loaded-state extraction and re-certification.

## Numerical closure

The public E5A reproduction uses the frozen dimension/rank/module-count grid and compares four numerical routes: LR32, Dense32, LR64-on-float32-inputs, and Dense64-on-float32-inputs.

```bash
python experiments/e5a_numerical_closure.py --device cuda --out e5a_public.csv
```

A real-shape anchor may be supplied separately with `--anchor-npz`.

## Validated result snapshot

The completed experiments support the following restricted claims:

- High-precision low-rank/dense equivalence: **125/125** units passed the predeclared `1e-10` profile; maximum Frobenius and spectral route disagreements were `1.88e-12` and `5.50e-13`.
- Under the stable P64 certificate profile, certificate-computation speedup over dense64 increased strongly with operator dimension. For `r=8`, representative speedups were **44.1×, 136.0×, 476.2×, 1725.6×, and 8281.4×** at `d=256,512,1024,2048,4096`.
- The real `2048×2048, r=16` anchor achieved **1333.6×** certificate-computation speedup and a **54.1×** measured peak working-memory ratio.
- These are **certificate-computation** results on one A800 system, not end-to-end model-loading or inference speedups.

See `results/validated_summary.json` and `results/e5b_selected_rows.csv` for compact machine-readable summaries.

## Repository layout

```text
SemaPEFT/
├── src/semapeft/        # core semantic verification library
├── experiments/         # public reproduction / inspection scripts
├── configs/             # sanitized frozen numerical/system protocols
├── tests/               # CPU smoke tests
├── results/             # compact validated result summaries
├── docs/                # methodology and reproducibility notes
└── provenance/          # hashes linking the release to final experiment scripts
```

## Important scope notes

1. The original E3 float32 correctness gate remains a failure; the P64 profile is a separately predeclared stable certificate stage.
2. No new float32 acceptance threshold is selected from the numerical-forensics results.
3. This repository does not redistribute model weights, datasets, or third-party serving-runtime source trees.
4. Runtime-version claims should always be paired with source/version hashes.

## Citation

If you use SemaPEFT, please cite the software using [`CITATION.cff`](CITATION.cff). GitHub will render the citation metadata automatically once the repository is public. When a paper DOI or preprint identifier is available, the preferred paper citation can be added without changing the software release.

## License

SemaPEFT is released under the [Apache License 2.0](LICENSE). Third-party models, datasets, adapters, and runtime source trees are not redistributed by this repository and remain subject to their original terms.

## Acknowledgements

This project builds on the PEFT ecosystem and evaluates runtime semantics in vLLM and SGLang. We thank the maintainers of the open-source libraries and public adapters/datasets used in the experiments.
