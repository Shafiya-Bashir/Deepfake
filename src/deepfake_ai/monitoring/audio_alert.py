"""Audio notification triggers for experiment milestones and completions."""

import platform
import os
import logging

logger = logging.getLogger(__name__)


def play_alert(alert_type: str = 'complete'):
    """Plays a system chime or terminal bell."""
    try:
        if platform.system() == 'Darwin':
            sound_map = {
                'complete': 'Glass',
                'error': 'Sosumi',
                'milestone': 'Hero',
                'warning': 'Ping'
            }
            sound = sound_map.get(alert_type, 'Ping')
            os.system(f'afplay /System/Library/Sounds/{sound}.aiff &')
        else:
            print('', end='', flush=True)
    except Exception as e:
        logger.debug(f'Audio alert skipped: {e}')
