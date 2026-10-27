# Ethical Guidelines and Safety Standards

DeepFake-AI is committed to responsible research in synthetic media forensics and digital content verification.

## 1. Ethical Principles
- **Defense-Oriented**: The platform is built exclusively to detect, verify, and understand synthetic manipulations.
- **Privacy & Consent**: Benchmark datasets utilized (FF++, Celeb-DF, DFDC, FakeAVCeleb) operate under research licenses with consenting participants or publicly available creative commons footage.
- **Prohibition of Deceptive Generation**: Code in this repository strictly prohibits incorporating generation heads designed to generate deceptive media.

## 2. Forensic Evidentiary Standards
- Automated scores ($P_{fake}$) are intended to assist human forensic examiners and fact-checkers, not replace judicial verification.
- Any legal or evidentiary proceeding must combine spatial-frequency heatmap explanations with cryptographic provenance (e.g. C2PA metadata).
