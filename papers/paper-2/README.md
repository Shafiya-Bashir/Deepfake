# Paper 2: FreqForensics-XAI (Frequency Domain & Explainable Edge Forensics)

**Title:** FreqForensics-XAI: Interpretable Multi-Scale Frequency Decomposition and Edge-Deployable Forensics for Generative Media Manipulation  
**Target Journal / Conference:** IEEE Transactions on Multimedia (T-MM) / IEEE ICIP / IEEE Access  
**Format:** IEEE Transactions Two-Column Class (`IEEEtran`)

---

## 👥 Authorship & Research Attribution
* **Dar Rehan Rasool** (Graduate Research Contributor)  
  *Department of Computer Science \& Engineering, Islamic University of Science and Technology (IUST), Kashmir, India*  
  Email: `rehan@iust.ac.in`
* **Dr. Riyaz\*** (Principal Research Supervisor / Senior Author)  
  *Department of Computer Science \& Engineering, Islamic University of Science and Technology (IUST), Kashmir, India*  
  Email: `riyaz.cs@iust.ac.in`
* **Dr. Javad\*** (Principal Research Supervisor / Co-Supervisor)  
  *Department of Computer Science \& Engineering, Islamic University of Science and Technology (IUST), Kashmir, India*  
  Email: `javad.cs@iust.ac.in`  
*\*Equal Senior Authorship and Principal Supervision*

---

## 📄 File Manifest
* `main.tex`: Full IEEEtran LaTeX document with complete equations, tables, citations, and figure references.
* `references.bib`: Complete BibTeX bibliography covering 18 seminal and recent papers in frequency analysis, diffusion models, and model compression.
* `figures/`: High-resolution vector PDF and 300 DPI PNG figures:
  1. `freqforensics_architecture.[pdf/png]`: Multi-scale frequency decomposition and XAI architecture flowchart.
  2. `frequency_power_spectrum_comparison.[pdf/png]`: Azimuthally averaged 2D-DCT radial power spectra ($1/f^2$ law vs. GAN/Diffusion spikes).
  3. `edge_latency_tradeoff.[pdf/png]`: Pareto frontier showing Model Footprint (7.2 MB) vs. Latency (11.4 ms) vs. AUC (0.962).
  4. `cross_generator_generalization.[pdf/png]`: Generalization matrix across StyleGAN2, StyleGAN3, Stable Diffusion, Midjourney, and SadTalker.
  5. `xai_saliency_heatmaps.[pdf/png]`: Explainable AI (XAI) multi-stage spatial-frequency forensic evidence visualizations.

---

## 🛠️ How to Compile
### Option A: Upload to Overleaf (Recommended)
1. Zip this entire `papers/paper-2/` folder.
2. In [Overleaf](https://www.overleaf.com), click **New Project** $\rightarrow$ **Upload Project** $\rightarrow$ select the zip file.
3. Overleaf will compile `main.tex` into a publication-ready PDF automatically!

### Option B: Local TeXLive / MacTeX
```bash
cd papers/paper-2
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
