# Model Card: DeepFake-AI Multi-Domain Detector

## Model Details
- **Developer**: Dar Rehan Rasool / DeepFake-AI Research Initiative
- **Model Date**: October 2026
- **Model Version**: 1.0.0
- **Model Type**: Hybrid Multi-Domain Ensemble (Spatial CNN + 2D-DCT Frequency Stream + Spatio-Temporal ResNet-LSTM)
- **License**: MIT License

## Intended Use
- **Primary Intended Uses**: Verification of digital video and image authenticity for news organizations, legal forensics, academic research, and biometric identity verification.
- **Out-of-Scope Uses**: Automated punitive actions without human review, mass indiscriminate surveillance, or weaponized reverse-engineering for adversarial spoof generation.

## Factors, Risks & Biases
- **Demographic Parity**: Generative artifacts may manifest differently across varying skin tones and facial structures. Benchmarking includes subgroup evaluations across ethnic and age brackets.
- **Lighting & Resolution Sensitivity**: Extreme low-light conditions or low-resolution webcams ($< 100 \times 100$ face crop) may induce false positives due to compression blur.

## Performance Metrics
- **FaceForensics++ (c23)**: AUC 99.6%, EER 1.5%
- **Celeb-DF v2**: AUC 87.5%, EER 12.8%
- **DFDC**: AUC 79.4%, EER 21.5%
- **GenImage Diffusion**: AUC 95.8%, EER 5.1%
