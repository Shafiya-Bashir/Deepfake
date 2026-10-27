"""Dataset downloader and synthetic mock generator for local testing."""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Generates local dataset scaffolds and sample test records."""

    @staticmethod
    def setup_sample_data(base_data_dir: Path):
        raw_dir = base_data_dir / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)

        for ds in ['faceforensics_pp', 'celeb_df', 'dfdc', 'fakeavceleb', 'genimage']:
            ds_dir = raw_dir / ds
            ds_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f'Scaffolded directory: {ds_dir}')
