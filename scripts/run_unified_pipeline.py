#!/usr/bin/env python3
"""Run single-command end-to-end verification pipeline."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from deepfake_ai.pipeline.unified_pipeline import UnifiedPipeline


def main():
    print('Executing DeepFake-AI Unified Pipeline...')
    pipeline = UnifiedPipeline()
    sample_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = pipeline.predict_image(sample_img)
    print('--- Forensic Verdict ---')
    for k, v in result.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()
