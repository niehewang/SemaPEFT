# Reproducibility

## Tested systems profile

The final stable-certificate systems experiment used Python 3.10, PyTorch 2.5.1+cu121, Ubuntu 22.04.5, and an NVIDIA A800 80GB PCIe GPU. Runtime-semantic experiments covered pinned local source snapshots of vLLM 0.30/0.31 and SGLang 0.5.20/0.5.21.

## Important boundaries

- This repository does not redistribute model weights, public-dataset snapshots, or third-party serving-runtime source trees.
- Runtime semantic extraction expects **local pinned source snapshots** supplied with `--runtime-source`.
- The public numerical scripts use deterministic synthetic factors by default. The real-shape anchor can be supplied separately as an NPZ with the documented six fields.
- Systems speedups refer only to **certificate computation**, not model loading or end-to-end inference.
- E3's original float32 correctness failure remains part of the experiment record; P64 is a separately predeclared stable profile.

## Quick validation

```bash
python -m pip install -e '.[test]'
pytest -q
python experiments/quickstart_certificate.py
```

## Full numerical grid

```bash
python experiments/e5a_numerical_closure.py --device cuda --out e5a_public.csv
```

## Runtime loader semantics

```bash
python experiments/probe_runtime_semantics.py   --runtime vllm --version 0.31.0   --runtime-source /path/to/vllm-0.31.0-source   --adapter /path/to/adapter
```
