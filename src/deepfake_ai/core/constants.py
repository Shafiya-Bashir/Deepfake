"""Constants and enumerations for DeepFake-AI."""

from enum import Enum, IntEnum
from typing import List


class Label(IntEnum):
    """Binary classification target."""
    REAL = 0
    FAKE = 1


class ManipulationType(str, Enum):
    """Manipulation classifications."""
    ORIGINAL = 'original'
    DEEPFAKES = 'Deepfakes'
    FACE2FACE = 'Face2Face'
    FACESWAP = 'FaceSwap'
    NEURAL_TEXTURES = 'NeuralTextures'
    FACESHIFTER = 'FaceShifter'
    DIFFUSION_SYNTHESIS = 'Diffusion'
    GAN_SYNTHESIS = 'GAN'
    AUDIO_CLONED = 'AudioCloned'
    TALKING_HEAD = 'TalkingHead'


class GeneratorFamily(str, Enum):
    """Generative backbone families."""
    PROGAN = 'ProGAN'
    STYLEGAN = 'StyleGAN'
    STYLEGAN2 = 'StyleGAN2'
    STYLEGAN3 = 'StyleGAN3'
    STABLE_DIFFUSION = 'StableDiffusion'
    MIDJOURNEY = 'Midjourney'
    DALLE3 = 'DALL-E3'
    FLUX = 'FLUX'
    SADTALKER = 'SadTalker'


class CompressionLevel(str, Enum):
    """Compression rate factor profiles."""
    RAW = 'raw'
    C23 = 'c23'
    C40 = 'c40'


SUPPORTED_DATASETS: List[str] = [
    'FaceForensics++',
    'Celeb-DF_v2',
    'DFDC',
    'FakeAVCeleb',
    'GenImage',
    'WildDeepfake',
]

DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_SEED = 42
