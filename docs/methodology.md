# Methodology: Deepfake Detection & Media Forensics Pipeline

## 1. Data Processing & Forensic Hygiene
Raw video and image datasets are processed through a strictly ordered pipeline:
1. **Face Localization & Extraction**:
   - Detect face bounding boxes using MTCNN with confidence threshold $\tau_{det} = 0.95$.
   - Expand bounding boxes by margin ratio $m = 1.3$ to preserve blending boundaries along the jawline and forehead hairline.
   - Resize cropped faces to canonical resolution $224 \times 224$ pixels using bicubic interpolation.
2. **Quality Auditing & Pre-filtering**:
   - Compute Laplacian blur variance $\sigma_{Lap}^2 = \text{Var}(\nabla^2 I)$. Frames with $\sigma_{Lap}^2 < 50$ are flagged as motion-blurred and filtered.
   - Detect dropped or corrupt frames via timestamp and perceptual hash (pHash) analysis.
3. **Data Augmentation Protocol**:
   - Apply realistic compression simulation (JPEG quality $Q \in [40, 95]$).
   - Random horizontal flips ($p=0.5$), subtle color jittering, and Gaussian blur ($p=0.2$) to prevent memorization of compression artifacts.

---

## 2. Model Training Strategies

### 2.1 Multi-Domain Model Architectures
1. **MesoInception4**:
   - Incorporates dilated inception modules to detect mesoscopic image compression patterns at low computational budget.
2. **XceptionNet**:
   - Pretrained on ImageNet, backbone frozen for first 2 epochs, fine-tuned end-to-end with learning rate $\eta = 2 \times 10^{-4}$.
3. **Dual-Stream Spatial-Frequency Network**:
   - Spatial stream: ResNet-34 extracting appearance features $F_s \in \mathbb{R}^{C \times H \times W}$.
   - Frequency stream: 2D-DCT high-pass residual processed by a convolutional branch producing $F_f \in \mathbb{R}^{C \times H \times W}$.
   - Cross-Attention Fusion: Queries derived from spatial features attend to frequency keys and values:
     $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
4. **Spatio-Temporal Recurrent Model**:
   - Sequence length $T = 16$. Frame features extracted via ResNet-18, fed into a 2-layer Bidirectional LSTM ($H = 256$) with dropout $p = 0.3$.

### 2.2 Optimization & Objective Functions
- **Loss Function**: Focal Loss ($\gamma = 2.0$, $\alpha = 0.7$ for Fake) to penalize hard boundary examples and counter class imbalance.
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $\lambda = 0.01$).
- **Learning Rate Schedule**: Cosine Annealing with Warm Restarts ($T_0 = 10, T_{mult} = 2, \eta_{min} = 10^{-6}$).
- **Early Stopping**: Monitored on validation AUC with patience $P = 6$ epochs.

---

## 3. Evaluation & Validation Methodology
- **Identity-Disjoint Splitting**: 70% Train, 15% Validation, 15% Test. No actor identities overlap between splits.
- **Metrics Hierarchy**:
  1. **Area Under ROC Curve (AUC-ROC)**: Threshold-independent assessment of discriminative power.
  2. **Equal Error Rate (EER)**: Operating point where False Positive Rate equals False Negative Rate ($\text{FPR} = \text{FNR}$).
  3. **Accuracy & F1-Score**: Evaluated at calibrated threshold $\tau^* = \arg\max_\tau (J(\tau))$ using Youden's J index.
  4. **Cross-Dataset Generalization AUC**: Zero-shot transfer from FF++ to Celeb-DF v2, DFDC, and GenImage.
