# Master Multi-Modal Dataset Catalog & Partition Manifest

**Research Platform**: DeepFake-AI  
**Research Team**: Dar Rehan Rasool (*Graduate Student & Contributor*), Dr. Riyaz (*Supervisor*), Dr. Javad (*Supervisor*)  
**Institution**: Islamic University of Science and Technology (IUST), Kashmir  
**Total Documented Samples**: 600  

---

## 1. Summary of Documented Benchmarks

| Benchmark Dataset | Modality | Samples | Real | Fake | Train | Val | Test | Primary Manipulation Targets |
| :--- | :--- | :---:| :---:| :---:| :---:| :---:| :---:| :--- |
| **DFDC (DeepFake Detection Challenge)** | Multi-Modal (Video + Audio) | 200 | 100 | 100 | 120 | 40 | 40 | DeepFaceLab FaceSwap |
| **FaceForensics++ (FF++)** | Video (MP4) | 120 | 60 | 60 | 72 | 24 | 24 | FaceSwap |
| **Celeb-DF v2** | Video (MP4) | 80 | 40 | 40 | 48 | 16 | 16 | DeepFaceLab Enhanced Face Swapping |
| **FakeAVCeleb** | Multi-Modal (Video + Cloned Voice) | 80 | 40 | 40 | 48 | 16 | 16 | SV2TTS Voice Clone + Wav2Lip |
| **Deepfake Audio (Telecom Fraud / ASVspoof)** | Audio / Voice (WAV) | 70 | 35 | 35 | 42 | 14 | 14 | Tacotron2 Synthesized Speech |
| **GenImage (Diffusion & GAN)** | Static Image (PNG / JPEG) | 50 | 25 | 25 | 30 | 10 | 10 | Stable Diffusion v2.1 |

---

## 2. Strict Identity-Disjoint Protocol
To prevent facial memorization and inflated scores:
- **No actor identity in the Train split ever appears in the Validation or Test splits.**
- The model learns **mathematical manipulation artifacts** (Laplacian seam gradients $\nabla^2 I$, 2D-DCT power roll-off $1/f^2$, EAR blinking velocity $\frac{d(\text{EAR})}{dt}$, and vocoder spectral flatness $\gamma$) rather than individual facial features.

---

## 3. CSV Schema Reference (`data/dataset_catalog_master.csv`)

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `sample_id` | String | Unique identifier per sample |
| `benchmark_dataset` | String | Source dataset (DFDC, FF++, Celeb-DF, FakeAVCeleb, Telecom Fraud, GenImage) |
| `media_modality` | String | Video, Image, Audio/Voice, or Multi-Modal |
| `split` | String | Train (60%), Validation (20%), or Test (20%) |
| `actor_identity_id` | String | Disjoint actor ID proving zero identity leakage across splits |
| `ground_truth_label` | Integer | `0` = Real/Authentic, `1` = Fake/Synthetic |
| `verdict_label` | String | `REAL` or `FAKE` |
| `manipulation_type` | String | Specific generative algorithm or camera recording |
| `resolution_and_format` | String | Frame size, FPS, or audio sampling rate |
| `duration_or_frames` | String | Video length, frame count, or audio duration |
| `s_spatial_boundary` | Float | Measured Laplacian edge gradient variance score |
| `s_spectral_frequency` | Float | Measured 2D-DCT / wavelet azimuthal power spectrum score |
| `s_temporal_biology` | Float | Measured Eye Aspect Ratio (EAR) blink velocity dynamic |
| `s_audio_vocoder` | Float | Measured Spectral Flatness Measure (SFM) & vocoder score |
| `file_relative_path` | String | Relative location in repository storage |

---

*Generated automatically by `scripts/generate_master_dataset_catalog.py`.*
