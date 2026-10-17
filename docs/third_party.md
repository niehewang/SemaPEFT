# Third-party components

SemaPEFT was evaluated with PEFT adapters, vLLM, SGLang, Hugging Face Transformers, PyTorch, NumPy, and safetensors. This repository does not vendor or redistribute those projects. Users are responsible for obtaining compatible versions under their respective licenses.

The source-isolation utilities execute selected classes/functions from a **local pinned runtime source tree** for semantic inspection while avoiding unrelated top-level service initialization. Source hashes should be recorded in any reproduced experiment.
