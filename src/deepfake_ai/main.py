"""Main Command Line Interface for DeepFake-AI."""

import argparse
import sys
from deepfake_ai.core.config import load_config
from deepfake_ai.hardware.hardware_detector import detect_hardware
from deepfake_ai.research.paper_registry import PaperRegistry


def main():
    parser = argparse.ArgumentParser(description='DeepFake-AI Command Line Interface')
    subparsers = parser.add_subparsers(dest='command', help='Available subcommands')

    # info
    subparsers.add_parser('info', help='Print system, hardware, and configuration info')

    # papers
    subparsers.add_parser('papers', help='List curated research papers in catalog')

    # dashboard
    dash_p = subparsers.add_parser('dashboard', help='Run interactive forensics dashboard')
    dash_p.add_argument('--port', type=int, default=8080, help='Port to bind')

    args = parser.parse_args()

    if args.command == 'info':
        profile = detect_hardware()
        print('=== DeepFake-AI System Diagnostics ===')
        print(f'Device: {profile.recommended_device}')
        print(f'OS: {profile.os_name} {profile.os_version}')
        print(f'RAM: {profile.total_ram_gb} GB Total ({profile.available_ram_gb} GB Available)')
    elif args.command == 'papers':
        registry = PaperRegistry()
        papers = registry.list_all()
        print(f'=== DeepFake-AI Research Catalog ({len(papers)} Papers) ===')
        for i, p in enumerate(papers, 1):
            print(f"{i:02d}. [{p.get('year')}] {p.get('title')} ({p.get('venue')}) - {p.get('category')}")
    elif args.command == 'dashboard':
        from deepfake_ai.dashboard.server import create_dashboard_app
        server = create_dashboard_app(port=args.port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print('
Shutting down dashboard server.')
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
