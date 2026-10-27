"""CLI training dispatcher."""

import logging
from deepfake_ai.core.config import load_config
from deepfake_ai.core.reproducibility import seed_everything

logger = logging.getLogger(__name__)


def main():
    config = load_config()
    seed_everything(config.get('environment.seed', 42))
    logger.info('Training dispatcher initialized.')


if __name__ == '__main__':
    main()
