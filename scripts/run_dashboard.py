#!/usr/bin/env python3
"""Launch interactive deepfake forensics research dashboard."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from deepfake_ai.dashboard.server import create_dashboard_app


def main():
    parser = argparse.ArgumentParser(description='DeepFake-AI Dashboard')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind')
    args = parser.parse_args()

    server = create_dashboard_app(port=args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('
Dashboard terminated.')


if __name__ == '__main__':
    main()
