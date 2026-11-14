# Kaggle to Google Drive Direct Cloud Downloader Guide

**Project**: DeepFake-AI Research Platform  
**Research Team**: Dar Rehan Rasool, Dr. Riyaz Ahmed, Dr. M. Abdul Jawad (M.Tech Students, Centre for Artificial Intelligence, IUST)  
**Affiliation**: Islamic University of Science and Technology (IUST), Kashmir  

---

## 🎯 Executive Overview

Training modern deepfake architectures requires massive benchmark datasets. The primary competition dataset, **Deepfake Detection Challenge (DFDC)** on Kaggle, contains **~472 GB** of high-resolution video files split into 50 parts (`dfdc_train_part_0.zip` to `dfdc_train_part_49.zip`).

To prevent local disk exhaustion on your personal MacBook:
- **Zero local storage used**: The data streams directly from Kaggle servers into your **Google Drive** using Google Colab Pro's 1–2 Gbps cloud connection.
- **Resumable execution**: Each ~9.5 GB chunk is downloaded independently. Already-downloaded chunks are automatically detected and skipped.
- **Ready for University GPU training**: Once in Google Drive, the university GPU cluster can mount or sync the dataset directly next time.

---

## 🛠️ Step-by-Step Instructions

### Step 1: Obtain Your Kaggle API Token (`kaggle.json`)
1. Log into your account on [kaggle.com](https://www.kaggle.com).
2. Go to the competition page: [Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge) and ensure you have clicked **"I Understand and Accept"** under the Rules tab.
3. Click your profile avatar in the upper right corner and select **Settings**.
4. Scroll down to the **API** section and click **Create New Token**.
5. A file named `kaggle.json` will be saved to your Downloads folder.
6. **Upload `kaggle.json` to the root of your Google Drive** (`My Drive / kaggle.json`).

---

### Step 2: Open and Run the Downloader Notebook
You can run the downloader either in your web browser or directly in VSCode using the Google Colab extension:

#### Option A: Run in Web Browser
1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Click **File** -> **Upload Notebook** -> Select `notebooks/01_download_kaggle_to_drive.ipynb` from this repository.
3. In Colab, click **Runtime** -> **Change runtime type** -> Select standard or high-RAM runtime.
4. Run each cell sequentially from top to bottom:
   - **Cell 1**: Installs the Kaggle CLI.
   - **Cell 2**: Prompts you to mount your Google Drive (`/content/drive`).
   - **Cell 3**: Verifies your Kaggle token from Google Drive.
   - **Cell 4**: Select which parts to download (`0-4` for first ~50 GB, or `all` for full 472 GB).
   - **Cell 5**: Executes the download directly into `/content/drive/MyDrive/DeepFake_Datasets/DFDC`.
   - **Cell 6**: Displays the file inventory and storage audit in Google Drive.

#### Option B: Run in VSCode with Google Colab Extension
1. In VSCode, open `notebooks/01_download_kaggle_to_drive.ipynb`.
2. In the top-right corner of the notebook editor, click the **Kernel / Runtime** picker.
3. Connect to your Google Colab runtime.
4. Execute the cells interactively.

---

### Step 3: Command-Line Alternative (via Colab Terminal)
If you prefer running a single Python command in Google Colab terminal:

```bash
# Clone the repository in Colab
git clone https://github.com/darrehanrasool/Deepfake.git
cd Deepfake

# Run the automated downloader (downloads first 5 parts ~50GB by default)
python3 scripts/download_kaggle_to_drive.py --parts 0-4

# Or download all 50 parts (~472GB)
python3 scripts/download_kaggle_to_drive.py --parts all
```

---

## 📊 Chunk Breakdown & Recommended Download Strategy

| Selection | Parts Included | Approximate Size | Download Time (Colab 1 Gbps) | Best For |
| :--- | :---: | :---: | :---: | :--- |
| **Quick Start** | `dfdc_train_part_0.zip` to `dfdc_train_part_4.zip` | ~48 GB | ~8–12 minutes | Initial model validation & rapid prototyping |
| **Phase 1 Benchmark** | `dfdc_train_part_0.zip` to `dfdc_train_part_9.zip` | ~96 GB | ~18–25 minutes | Baseline benchmark comparisons |
| **Complete Dataset** | `dfdc_train_part_0.zip` to `dfdc_train_part_49.zip` | ~472 GB | ~1.5–2.5 hours | Final paper results & SOTA university GPU training |

---

## 🔒 Safety & Quota Reminders
1. **Never download these files to your MacBook**: Keep the destination pointed to Google Drive (`/content/drive/MyDrive/...`).
2. **Auto-Resume**: If Colab disconnects or reaches the idle limit, simply re-run the notebook. Any zip files that finished downloading will be recognized and skipped automatically.
3. **University GPU Next Step**: In your next session, we will connect the university GPU cluster to these downloaded zip archives using `rclone` or virtual streaming, unpack frames on-the-fly, and launch multi-GPU training!
