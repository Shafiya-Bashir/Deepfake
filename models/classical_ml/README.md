# Classical Forensic & Machine Learning Models

This directory houses classical statistical and machine learning classifiers trained on extracted forensic handcrafted descriptors:

## Descriptors Used
1. **Local Binary Patterns (LBP)**: Captures subtle skin microtexture variations and blending boundaries.
2. **Gray-Level Co-occurrence Matrix (GLCM)**: Contrast, dissimilarity, homogeneity, energy, and correlation across multi-directional offsets.
3. **Color Moments**: Mean, variance, and skewness across RGB, HSV, and YCbCr color spaces.
4. **Spectral Energy Distribution**: High-frequency energy roll-off ratios computed from 2D Fast Fourier Transform (FFT) and 2D-DCT blocks.

## Classifiers
- **Support Vector Machine (SVM)**: RBF kernel classifier with Platt probability calibration.
- **Random Forest**: Balanced ensemble with out-of-bag error tracking.
- **Gradient Boosted Decision Trees**: High-precision non-linear tree ensembles.
- **Logistic Regression**: Scaled linear baseline for calibration verification.
