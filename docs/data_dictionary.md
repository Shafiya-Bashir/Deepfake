# Data Dictionary: DeepFake-AI Schema

This document specifies the exact schemas and data types used in metadata records, interim crops, and evaluation manifests.

## 1. Processed Metadata Schema (`clean_metadata.csv`)

| Field Name | Data Type | Description | Allowed Values / Examples |
|:---|:---:|:---|:---|
| `sample_id` | String | Unique video/image sample identifier | `sample_0001`, `ffpp_c23_042` |
| `dataset` | String | Source benchmark dataset | `faceforensics_pp`, `celeb_df`, `dfdc` |
| `actor_id` | String | Unique identity code of the subject | `actor_001`, `celeb_042` |
| `label` | Integer | Binary classification ground truth | `0` (Real), `1` (Fake) |
| `label_name` | String | Human-readable classification target | `REAL`, `FAKE` |
| `manipulation_type` | String | Specific generative or reenactment technique | `Deepfakes`, `Face2Face`, `original` |
| `compression` | String | Video compression rate factor | `raw`, `c23`, `c40` |
| `frame_count` | Integer | Total decoded video frames | `60`, `120`, `300` |
| `fps` | Float | Video frame rate | `25.0`, `29.97`, `30.0` |
| `face_detected` | Boolean | Whether MTCNN confirmed a valid facial bounding box | `True`, `False` |
| `bounding_box` | List[Int] | Canonical face box `[x1, y1, x2, y2]` | `[45, 60, 210, 225]` |
| `laplacian_blur_score` | Float | Variance of Laplacian filter (sharpness metric) | `> 50.0` (Sharp), `< 50.0` (Blurry) |
| `path` | String | Relative filepath to media artifact | `data/raw/faceforensics_pp/sample_0001.mp4` |

---

## 2. Evaluation Output Schema (`model_benchmark_report.csv`)

| Metric Field | Data Type | Formula / Range | Objective |
|:---|:---:|:---:|:---:|
| `roc_auc` | Float | $[0.0, 1.0]$ | Higher is better (Target $> 0.95$) |
| `eer` | Float | $[0.0, 1.0]$ (Point where $\text{FPR} = \text{FNR}$) | Lower is better (Target $< 0.05$) |
| `accuracy` | Float | $\frac{TP + TN}{Total} \in [0.0, 1.0]$ | Higher is better |
| `precision` | Float | $\frac{TP}{TP + FP} \in [0.0, 1.0]$ | Minimizes false accusations |
| `recall` | Float | $\frac{TP}{TP + FN} \in [0.0, 1.0]$ | Catches all synthetic media |
| `f1_score` | Float | $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$ | Harmonic mean |
| `log_loss` | Float | Cross-entropy loss | Lower is better |
| `brier_score` | Float | Mean squared probability error | Lower is better (Calibration) |
