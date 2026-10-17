"""SemaPEFT: semantic-integrity verification utilities for PEFT adapters."""
from .certificate import CertificateResult, lowrank_certificate, dense_certificate, compare_certificates, p64_compare
from .peft import lora_scale, canonical_adapter_base, load_lora_pairs, operator_map_reference, operator_gap, lower_adapter_to_scale, drop_undeclared_tensors

__all__ = [
    "CertificateResult", "lowrank_certificate", "dense_certificate", "compare_certificates", "p64_compare",
    "lora_scale", "canonical_adapter_base", "load_lora_pairs", "operator_map_reference", "operator_gap",
    "lower_adapter_to_scale", "drop_undeclared_tensors",
]
__version__ = "1.0.0"
