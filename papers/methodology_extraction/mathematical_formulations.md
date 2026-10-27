# Mathematical Formulations in Deepfake Forensics

## 1. Spatial Domain Convolutions & Depthwise Separability

In standard convolutional layers with input tensor $X \in \mathbb{R}^{H \times W \times C_{in}}$ and kernels $K \in \mathbb{R}^{K_h \times K_w \times C_{in} \times C_{out}}$, computational complexity scales as $\mathcal{O}(H \cdot W \cdot K_h \cdot K_w \cdot C_{in} \cdot C_{out})$.

Xception decomposes this into:
1. **Depthwise Convolution**: Applies a single spatial filter per input channel:
   $$\hat{X}_c(i, j) = \sum_{m=-k}^{k} \sum_{n=-k}^{k} K_c(m, n) \cdot X_c(i+m, j+n)$$
2. **Pointwise Convolution**: $1 \times 1$ linear projection across channels:
   $$Y_k(i, j) = \sum_{c=1}^{C_{in}} W_{c, k} \cdot \hat{X}_c(i, j)$$
This decouples cross-channel correlation from spatial correlation, enabling deep feature abstraction at substantially reduced parameter counts.

---

## 2. 2D Discrete Cosine Transform (DCT) & Frequency Forensics

For an $N \times N$ spatial image block $f(x, y)$, the 2D-DCT type-II transformation into frequency coefficients $F(u, v)$ is defined as:
$$F(u, v) = \frac{2}{N} C(u) C(v) \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} f(x, y) \cos\left[\frac{(2x+1)u\pi}{2N}\right] \cos\left[\frac{(2y+1)v\pi}{2N}\right]$$
where:
$$C(u) = \begin{cases} \frac{1}{\sqrt{2}} & \text{if } u = 0 \\ 1 & \text{if } u > 0 \end{cases}$$

### High-Pass Residual Extraction
To isolate high-frequency generator artifacts from semantic image content, we apply an octave frequency bandpass filter $H(u, v)$:
$$R(u, v) = F(u, v) \cdot \mathbb{I}(\sqrt{u^2 + v^2} \ge \tau_{cutoff})$$
The spatial domain residual is recovered via Inverse DCT (IDCT):
$$r(x, y) = \text{IDCT}\{R(u, v)\}$$

---

## 3. Optical Flow & Spatio-Temporal Motion Dynamics

Between consecutive video frames $I_t(x, y)$ and $I_{t+1}(x, y)$, optical flow velocity $(u, v) = (\frac{dx}{dt}, \frac{dy}{dt})$ satisfies the optical flow brightness constancy constraint:
$$\frac{\partial I}{\partial x} u + \frac{\partial I}{\partial y} v + \frac{\partial I}{\partial t} = 0$$

Synthetic facial videos often exhibit high acceleration discontinuities in landmark trajectories:
$$\mathcal{A}_{t} = \left\| \frac{d^2 \mathbf{p}_t}{dt^2} \right\|_2 = \| \mathbf{p}_{t+1} - 2\mathbf{p}_t + \mathbf{p}_{t-1} \|_2$$
where $\mathbf{p}_t \in \mathbb{R}^{68 \times 2}$ denotes detected facial landmark coordinates at frame $t$.

---

## 4. Diffusion Reconstruction Error (DIRE)

Given an input facial image $x_0 \sim q(x)$, the deterministic DDIM forward inversion produces latent noise $z_T$:
$$z_T = \text{DDIM\_Invert}(x_0, T)$$
The deterministic DDIM backward reconstruction yields:
$$\hat{x}_0 = \text{DDIM\_Generate}(z_T, T)$$
The pixel-wise reconstruction residual map is formulated as:
$$\text{DIRE}(x_0) = |x_0 - \hat{x}_0|$$
A binary classification network $\mathcal{F}_\theta$ maps the residual map to synthetic likelihood:
$$\hat{y} = \sigma\left(\mathcal{F}_\theta(\text{DIRE}(x_0))\right)$$

---

## 5. Loss Formulations

### Focal Loss
To address class imbalance between authentic media and scarce synthetic variants:
$$\mathcal{L}_{focal}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
where $p_t = p$ for label $y=1$ and $p_t = 1-p$ for $y=0$, with focusing parameter $\gamma = 2.0$.
