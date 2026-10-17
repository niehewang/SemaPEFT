# Methodology notes

SemaPEFT separates **artifact authenticity** from **execution-semantic integrity**. A signed or byte-identical PEFT adapter can still realize a different effective operator when a serving runtime interprets PEFT metadata, module binding, scaling, fused layouts, or unsupported features differently.

The core object checked by SemaPEFT is the effective low-rank operator. For a LoRA module with factors `A`, `B` and scale `s`, the effective update is `s B A`. Cross-runtime verification therefore compares a reference interpretation with a target-runtime loaded interpretation rather than only comparing serialized bytes or configuration text.

The low-rank certificate represents an operator difference as `U V`, computes Frobenius quantities through Gram matrices, and computes spectral quantities from a thin-QR compressed core. The stable P64 profile retains the serialized float32 factor values but upcasts them to float64 at the certificate boundary. In the final numerical-closure experiment, LR64 and Dense64 agreed for all 125/125 predeclared units at a `1e-10` relative tolerance.

Repairs are proof-carrying transformations: a repaired artifact is accepted only when its loaded-state operator is re-extracted and the post-repair operator gap returns to the declared tolerance. Examples include exact scale lowering and dropping undeclared serialized tensor modules.
