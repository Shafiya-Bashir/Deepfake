"""Registry mapping dataset names to metadata and file structures."""

from typing import Dict, Any, Optional


class DatasetRegistry:
    """Catalog of supported deepfake benchmarks."""
    _REGISTRY = {
        'faceforensics_pp': {
            'display_name': 'FaceForensics++',
            'type': 'video',
            'compressions': ['raw', 'c23', 'c40'],
            'manipulations': ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FaceShifter'],
        },
        'celeb_df': {
            'display_name': 'Celeb-DF (v2)',
            'type': 'video',
            'compressions': ['raw'],
            'manipulations': ['Deepfakes'],
        },
        'dfdc': {
            'display_name': 'DFDC',
            'type': 'video',
            'compressions': ['variable'],
            'manipulations': ['multiple'],
        },
        'fakeavceleb': {
            'display_name': 'FakeAVCeleb',
            'type': 'multimodal',
            'modalities': ['video', 'audio'],
        },
        'genimage': {
            'display_name': 'GenImage',
            'type': 'image',
            'generators': ['StableDiffusion', 'Midjourney', 'DALL-E3'],
        }
    }

    @classmethod
    def get(cls, dataset_id: str) -> Optional[Dict[str, Any]]:
        return cls._REGISTRY.get(dataset_id.lower())

    @classmethod
    def list_datasets(cls) -> list:
        return list(cls._REGISTRY.keys())
