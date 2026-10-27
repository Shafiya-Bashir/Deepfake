# Paper 1: ST2-SyncNet (Multi-Modal Deepfake Forensics)

**Title:** ST2-SyncNet: Spatio-Temporal Audio-Visual Discrepancy Detection for Multi-Modal Deepfake Video Forensics  
**Target Journal / Conference:** IEEE Transactions on Information Forensics and Security (T-IFS) / IEEE CVPR / IEEE Access  
**Format:** IEEE Transactions Two-Column Class (`IEEEtran`)

---

## 👥 Authorship & Research Attribution
* **Dar Rehan Rasool** (Graduate Research Contributor)  
  *Department of Computer Science & Engineering, Islamic University of Science and Technology (IUST), Kashmir, India*  
  Email: `rehan@iust.ac.in`
* **Dr. Riyaz\*** (Principal Research Supervisor / Senior Author)  
  *Department of Computer Science & Engineering, Islamic University of Science and Technology (IUST), Kashmir, India*  
  Email: `riyaz.cs@iust.ac.in`
* **Dr. Javad\*** (Principal Research Supervisor / Co-Supervisor)  
  *Department of Computer Science & Engineering, Islamic University of Science and Technology (IUST), Kashmir, India*  
  Email: `javad.cs@iust.ac.in`  
*\*Equal Senior Authorship and Principal Supervision*

---

## 📄 File Manifest
* `main.tex`: Full IEEEtran LaTeX document with complete equations, tables, citations, and figure references.
* `references.bib`: Complete BibTeX bibliography covering 20 seminal and recent deepfake, multi-modal, and biometric publications.
* `figures/`: High-resolution vector PDF and 300 DPI PNG figures:
  1. `st2_syncnet_architecture.[pdf/png]`: End-to-end multi-modal architecture with Cross-Modal Cross-Attention (CMCA).
  2. `roc_curves_benchmarks.[pdf/png]`: ROC-AUC curves comparing ST2-SyncNet (0.974) with LipForensics, Xception, MesoNet.
  3. `confusion_matrix_multimodal.[pdf/png]`: Normalized confusion matrix across 5,000 multi-modal test sequences.
  4. `ablation_temporal_windows.[pdf/png]`: Resilience under H.264 video compression (Raw, C23, C40).
  5. `audio_visual_alignment_trace.[pdf/png]`: Temporal viseme-phoneme cross-correlation trace (Authentic $r=0.94$ vs Deepfake $r=0.31$).

---

## 🛠️ How to Compile
### Option A: Upload to Overleaf (Recommended)
1. Zip this entire `papers/paper-1/` folder.
2. In [Overleaf](https://www.overleaf.com), click **New Project** $\rightarrow$ **Upload Project** $\rightarrow$ select the zip file.
3. Overleaf will compile `main.tex` into a publication-ready PDF automatically!

### Option B: Local TeXLive / MacTeX
```bash
cd papers/paper-1
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
