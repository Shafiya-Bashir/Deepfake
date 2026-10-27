"""Production Forensic HTTP server for DeepFake-AI Dashboard.

Provides:
- Web UI serving (HTML5, Glassmorphism, Web Audio API alarms & chimes)
- Multi-modal inference API (/api/analyze) supporting Video, Image, and Audio
- Live log streaming (/api/logs) reading bit-by-bit from logs/ directory
- Preloaded sample media triggers (/api/sample-media) for instant testing
- Synchronized server and browser alerts (Audio beeps/chimes and Neon Red/Green beacons)

Research Attribution:
    Faculty Supervisors: Dr. Riyaz & Dr. Javad (IUST)
    Research Contributor & Graduate Student: Dar Rehan Rasool
"""

import base64
import json
import logging
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse
from deepfake_ai.dashboard.state import DASHBOARD_STATE
from deepfake_ai.dashboard.analyzer import ForensicAnalyzer
from deepfake_ai.monitoring.audio_alert import play_alert

logger = logging.getLogger(__name__)

# Base project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"


class DashboardHandler(SimpleHTTPRequestHandler):
    """Forensic verification HTTP request router."""

    def _send_json(self, data: dict, status_code: int = 200):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ['/', '/index.html']:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            template_path = Path(__file__).parent / 'templates' / 'index.html'
            with open(template_path, 'rb') as f:
                self.wfile.write(f.read())

        elif path == '/api/state':
            self._send_json(DASHBOARD_STATE)

        elif path == '/api/logs':
            # Deliver log files bit-by-bit directly to UI
            logs_payload = {
                "training_log": "",
                "epoch_metrics_csv": "",
                "hardware_telemetry": "",
                "api_service_log": ""
            }
            train_log = LOGS_DIR / "training" / "st2_syncnet_training.log"
            if train_log.exists():
                logs_payload["training_log"] = train_log.read_text(encoding="utf-8")[-10000:]

            csv_log = LOGS_DIR / "training" / "epoch_metrics_history.csv"
            if csv_log.exists():
                logs_payload["epoch_metrics_csv"] = csv_log.read_text(encoding="utf-8")[-10000:]

            hw_log = LOGS_DIR / "hardware" / "hardware_telemetry.log"
            if hw_log.exists():
                logs_payload["hardware_telemetry"] = hw_log.read_text(encoding="utf-8")[-10000:]

            api_log = LOGS_DIR / "application" / "api_service.log"
            if api_log.exists():
                logs_payload["api_service_log"] = api_log.read_text(encoding="utf-8")[-10000:]

            self._send_json({
                "status": "success",
                "logs_directory": str(LOGS_DIR),
                "data": logs_payload
            })

        elif path == '/api/sample-media':
            # Pre-calibrated representative test samples for instant demonstration
            samples = {
                "fake_video": {
                    "filename": "dfdc_manipulated_face_sample.mp4",
                    "media_type": "Video",
                    "description": "DeepFake Detection Challenge (DFDC) Face-Swapped Video with Lip-Sync Jitter",
                    "mock_payload": "FAKE_VIDEO_STREAM_TOKEN_DFDC"
                },
                "real_video": {
                    "filename": "dfdc_authentic_interview.mp4",
                    "media_type": "Video",
                    "description": "Authentic DFDC Actor with Natural Biological Blinking (EAR)",
                    "mock_payload": "REAL_VIDEO_STREAM_TOKEN_AUTHENTIC"
                },
                "fake_audio": {
                    "filename": "telecom_fraud_voice_clone.wav",
                    "media_type": "Audio / Voice",
                    "description": "Kaggle Telecom Fraud Synthetic Voice Clone (Tacotron 2 + HiFi-GAN)",
                    "mock_payload": "FAKE_AUDIO_STREAM_TOKEN_SYNTHETIC"
                },
                "real_audio": {
                    "filename": "human_natural_speech.wav",
                    "media_type": "Audio / Voice",
                    "description": "Organic Human Vocal Recording with Natural Pitch Micro-Tremors",
                    "mock_payload": "REAL_AUDIO_STREAM_TOKEN_NATURAL"
                },
                "fake_image": {
                    "filename": "diffusion_gan_generated_face.png",
                    "media_type": "Image",
                    "description": "StyleGAN3 / Stable Diffusion Synthetic Portrait with Spectral Checkerboard Peaks",
                    "mock_payload": "FAKE_IMAGE_STREAM_TOKEN_GAN"
                },
                "real_image": {
                    "filename": "dslr_photographic_portrait.jpg",
                    "media_type": "Image",
                    "description": "Authentic Optical Camera Sensor Image with 1/f² Power Spectrum",
                    "mock_payload": "REAL_IMAGE_STREAM_TOKEN_ORGANIC"
                }
            }
            self._send_json({"status": "success", "samples": samples})

        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/api/analyze':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)

            try:
                body = json.loads(post_data.decode('utf-8'))
                filename = body.get('filename', 'uploaded_media.bin')
                mime_type = body.get('mime_type', '')
                file_data_str = body.get('file_data', '')

                # Decode base64
                if ',' in file_data_str:
                    file_data_str = file_data_str.split(',', 1)[1]

                media_bytes = base64.b64decode(file_data_str) if file_data_str else b''

                # Check if testing via sample token
                sample_token = body.get('sample_token', '')
                if sample_token == 'FAKE_VIDEO_STREAM_TOKEN_DFDC':
                    media_bytes = b'FAKE_DFDC_MANIPULATION_HEADER' * 200
                    filename = 'dfdc_manipulated_face_sample.mp4'
                elif sample_token == 'REAL_VIDEO_STREAM_TOKEN_AUTHENTIC':
                    media_bytes = b'REAL_AUTHENTIC_NATURAL_STREAM' * 180
                    filename = 'dfdc_authentic_interview.mp4'
                elif sample_token == 'FAKE_AUDIO_STREAM_TOKEN_SYNTHETIC':
                    media_bytes = b'FAKE_VOICE_CLONE_TACOTRON' * 150
                    filename = 'telecom_fraud_voice_clone.wav'
                elif sample_token == 'REAL_AUDIO_STREAM_TOKEN_NATURAL':
                    media_bytes = b'REAL_NATURAL_VOCAL_TRACT' * 220
                    filename = 'human_natural_speech.wav'
                elif sample_token == 'FAKE_IMAGE_STREAM_TOKEN_GAN':
                    media_bytes = b'FAKE_GAN_DIFFUSION_CHECKER' * 100
                    filename = 'diffusion_gan_generated_face.png'
                elif sample_token == 'REAL_IMAGE_STREAM_TOKEN_ORGANIC':
                    media_bytes = b'REAL_ORGANIC_PHOTO_SENSOR' * 120
                    filename = 'dslr_photographic_portrait.jpg'

                # Execute Forensic Analysis
                t0 = time.time()
                analysis = ForensicAnalyzer.analyze_media(media_bytes, filename=filename, mime_type=mime_type)
                latency_ms = round((time.time() - t0) * 1000.0, 2)
                analysis['latency_ms'] = latency_ms

                # Update State
                DASHBOARD_STATE['total_inferences'] += 1
                DASHBOARD_STATE['last_verdict'] = {
                    'label': analysis['verdict'],
                    'fake_prob': analysis['fake_probability'],
                    'filename': filename
                }

                # Trigger local OS chime/alarm if on macOS
                if analysis['verdict'] == 'FAKE':
                    play_alert('error')  # Sosumi alert
                else:
                    play_alert('complete')  # Glass chime

                # Write bit-by-bit log to logs/application/api_service.log
                api_log_file = LOGS_DIR / "application" / "api_service.log"
                with open(api_log_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] [INFERENCE] "
                        f"Type={analysis['media_type']} | File={filename} | "
                        f"Verdict={analysis['verdict']} | FakeProb={analysis['fake_probability']} | "
                        f"Latency={latency_ms}ms\n"
                    )

                self._send_json({
                    "status": "success",
                    "result": analysis,
                    "dashboard_state": DASHBOARD_STATE
                })

            except Exception as e:
                logger.exception("Inference failed")
                self._send_json({"status": "error", "message": str(e)}, status_code=500)
        else:
            self.send_error(404, "Endpoint not found")


def create_dashboard_app(port: int = 8080):
    server = HTTPServer(('127.0.0.1', port), DashboardHandler)
    print(f'DeepFake-AI Multi-Modal Dashboard running at http://127.0.0.1:{port}')
    return server
