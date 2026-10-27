"""Data loading, preprocessing, frequency feature engineering, and quality auditing."""

from deepfake_ai.data.dataset_registry import DatasetRegistry
from deepfake_ai.data.dataset_adapter import DeepfakeDatasetAdapter
from deepfake_ai.data.dataset_manifest import load_manifest, save_manifest
from deepfake_ai.data.dataset_downloader import DatasetDownloader
from deepfake_ai.data.preprocessing import FacePreprocessor
from deepfake_ai.data.feature_engineering import FrequencyFeatureExtractor
from deepfake_ai.data.data_splitting import split_identity_disjoint
from deepfake_ai.data.data_quality import audit_data_quality
from deepfake_ai.data.data_validator import validate_dataset_integrity
from deepfake_ai.data.missing_values import handle_missing_frames
from deepfake_ai.data.deduplication import deduplicate_media_samples

__all__ = [
    'DatasetRegistry',
    'DeepfakeDatasetAdapter',
    'load_manifest',
    'save_manifest',
    'DatasetDownloader',
    'FacePreprocessor',
    'FrequencyFeatureExtractor',
    'split_identity_disjoint',
    'audit_data_quality',
    'validate_dataset_integrity',
    'handle_missing_frames',
    'deduplicate_media_samples',
]
