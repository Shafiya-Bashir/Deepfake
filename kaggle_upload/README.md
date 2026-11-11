# Kaggle Upload Package: DeepFake-AI ST2-SyncNet

This directory contains the production-grade assets for submitting the **ST2-SyncNet** deepfake detection model to the Kaggle Deepfake Detection Challenge (DFDC) or private university benchmarks.

---

## 📦 Package Contents

| Filename | Description | Size |
| :--- | :--- | :--- |
| `best_model.pt` | Converged PyTorch weights for **ST2-SyncNet** (`13.08 MB`, SHA-256: `61fe0f94...`) | ~12.5 MB |
| `submission.csv` | Formatted and verified sample submission file compliant with Kaggle DFDC rules | ~1 KB |
| `sample_submission.csv` | Template test manifest for offline pipeline validation | ~1 KB |
| `dataset-metadata.json` | Metadata file for uploading as a private Kaggle Dataset via Web UI or CLI | ~150 B |

---

## 👥 Authorship & Research Attribution
* **Dar Rehan Rasool** (M.Tech Student, Centre for Artificial Intelligence, IUST)
* **Dr. Riyaz** (M.Tech Student, Centre for Artificial Intelligence, IUST)
* **Dr. Javad** (M.Tech Student, Centre for Artificial Intelligence, IUST)
* **Institution:** Islamic University of Science and Technology (IUST), Kashmir, India

---

## 🚀 How to Use on Kaggle

### Method 1: Web Interface Upload (Easiest)
1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) -> Click **"New Dataset"**.
2. Drag and drop `best_model.pt` into the upload box.
3. Title it: **`st2-syncnet-weights`** -> Click **"Create"** (Keep it **Private**).
4. In your Kaggle Notebook, click **"+ Add Input"** on the right sidebar and select your newly created dataset.
5. Model weights will immediately appear at:
   `/kaggle/input/st2-syncnet-weights/best_model.pt`.

### Method 2: Kaggle CLI (Terminal)
```bash
kaggle datasets create -p kaggle_upload/
```
