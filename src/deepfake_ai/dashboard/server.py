"""Lightweight HTTP server for forensic research dashboard."""

import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from deepfake_ai.dashboard.state import DASHBOARD_STATE


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            template_path = Path(__file__).parent / 'templates' / 'index.html'
            with open(template_path, 'rb') as f:
                self.wfile.write(f.read())
        elif self.path == '/api/state':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(DASHBOARD_STATE).encode())
        else:
            super().do_GET()


def create_dashboard_app(port: int = 8080):
    server = HTTPServer(('127.0.0.1', port), DashboardHandler)
    print(f'DeepFake-AI Dashboard running at http://127.0.0.1:{port}')
    return server
