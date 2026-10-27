# Kaggle GPU Training Step-by-Step Guide

**Project**: DeepFake-AI Research Platform  
**Research Team**: Dar Rehan Rasool (*Research Contributor*), Dr. Riyaz (*Supervisor*), Dr. Javad (*Supervisor*)  
**Affiliation**: Islamic University of Science and Technology (IUST), Kashmir  
**GitHub Repository**: [https://github.com/darrehanrasool/Deepfake.git](https://github.com/darrehanrasool/Deepfake.git)  

---

## 🎯 Why Train on Kaggle GPUs?
1. **Zero Download / Zero Waiting**: The entire 500GB DFDC competition dataset is **already attached** inside Kaggle at `/kaggle/input/deepfake-detection-challenge/`.
2. **Free Dual NVIDIA GPUs**: Kaggle provides **NVIDIA Tesla T4 x 2** (32 GB total VRAM) or **NVIDIA P100** free of charge for 30 hours per week.
3. **No Local Mac Disk Usage**: Not a single gigabyte touches your personal MacBook.
4. **No Colab Location Restrictions**: Kaggle Notebooks operate on dedicated Kaggle cloud containers that do not hit Google Colab's regional restriction.

---

## 🚀 5-Minute Step-by-Step Walkthrough

### Step 1: Open a New Notebook on Kaggle
1. Go to the competition page: [kaggle.com/c/deepfake-detection-challenge](https://www.kaggle.com/c/deepfake-detection-challenge).
2. Look at the tabs across the top: `Overview`, `Data`, `Code`, `Discussion`, `Leaderboard`, `Rules`.
3. Click the **Code** tab $\rightarrow$ click the blue **New Notebook** button.

---

### Step 2: Configure Kaggle Settings (Crucial)
In the Kaggle Notebook editor, look at the **Right Sidebar** under **Notebook options**:
1. **Accelerator**: Click it $\rightarrow$ select **GPU T4 x 2** (or GPU P100).
2. **Internet**: Toggle it to **ON** (this allows Kaggle to clone your GitHub repository).
3. **Environment**: Keep standard (Latest).

---

### Step 3: Run the Training Cells
In the first cell, simply paste:

```python
# 1. Clone your latest repository from GitHub
%cd /kaggle/working
!rm -rf Deepfake
!git clone https://github.com/darrehanrasool/Deepfake.git
%cd /kaggle/working/Deepfake
# Kaggle already has PyTorch, NumPy, Pandas, Scikit-learn pre-installed
!pip install -q --no-deps facenet-pytorch || true

# 2. Launch ST2-SyncNet Training on Dual T4 GPUs
!python3 scripts/kaggle_runner.py \
    --data-dir /kaggle/input/deepfake-detection-challenge \
    --output-dir /kaggle/working \
    --model st2_syncnet \
    --epochs 10 \
    --batch-size 32
```

Click the **Play (Run)** button or press `Shift + Enter`.

---

### Step 4: Generate All Benchmark Visualizations
In the next cell, run:

```python
%cd /kaggle/working/Deepfake
!python3 scripts/generate_visualizations.py
!python3 scripts/generate_all_category_visualizations.py
!cp -r visualizations /kaggle/working/
```

This renders all 10 colorful per-epoch and cross-dataset figures directly into `/kaggle/working/visualizations/`.

---

### Step 5: Download Trained Checkpoints to Your Mac
1. In the right sidebar of Kaggle, click the **Output** tab.
2. You will see:
   - `checkpoints/best/st2_syncnet_best.pt`
   - `visualizations/`
   - `kaggle_training_history.csv`
3. Click the three dots $\dots$ next to any file $\rightarrow$ select **Download** to save the trained model weights to your Mac!
