# Security Policy

## Supported version

Security and correctness fixes are applied to the latest `1.x` release line.

## Reporting a vulnerability or semantic-integrity bug

For issues that could expose credentials, private artifacts, or materially invalidate a semantic-integrity verdict, use GitHub's **Security → Report a vulnerability** workflow when available. If private vulnerability reporting is unavailable, open a minimal public issue requesting a private contact channel and do not include sensitive details.

For ordinary correctness or reproducibility problems, use the bug/reproducibility issue template.

## Scope

SemaPEFT is a research verification tool. A mismatch reported by SemaPEFT is evidence about adapter execution semantics under the inspected runtime/source snapshot; it is not, by itself, evidence of malicious intent or exploitation.
