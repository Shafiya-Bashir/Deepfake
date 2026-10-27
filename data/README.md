# Data Management & Storage Architecture

This directory organizes raw, interim, and processed deepfake detection datasets, splits, and manifests.

## Directory Structure
- `raw/`: Unmodified raw datasets downloaded from institutional repositories:
  - `faceforensics_pp/`: FaceForensics++ video sequences (raw, c23, c40).
  - `celeb_df/`: Celeb-DF v2 high-quality face swap clips.
  - `dfdc/`: Deepfake Detection Challenge selection.
  - `fakeavceleb/`: Audio-video multimodal deepfakes.
  - `genimage/`: Diffusion-synthesized vs real photographic images.
  - `wild_deepfake/`: Real-world internet collected in-the-wild manipulations.
- `interim/`: Intermediate artifacts (face detection bounding boxes, MTCNN crops, aligned face tensors, audio spectrograms).
- `processed/`: Standardized metadata catalogs, normalized features, and clean sample records.
- `splits/`: Identity-disjoint splits:
  - `train/`: Training split (70%)
  - `validation/`: Hyperparameter validation split (15%)
  - `test/`: Unseen evaluation test split (15%)
- `manifests/`: Formal JSON manifests containing SHA-256 integrity checksums, file counts, and licensing metadata.

## Identity Disjoint Splitting Guarantee
To avoid the identity leakage flaw common in academic deepfake research, our split generation guarantees that no actor ID or video source appears in both the training and testing sets.
