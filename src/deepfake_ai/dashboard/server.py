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
CAPTURED_TESTS_DIR = PROJECT_ROOT / "data" / "captured_tests"
MANIFEST_FILE = CAPTURED_TESTS_DIR / "manifest.json"

# Ensure storage directories exist
CAPTURED_TESTS_DIR.mkdir(parents=True, exist_ok=True)
(CAPTURED_TESTS_DIR / "images").mkdir(parents=True, exist_ok=True)
(CAPTURED_TESTS_DIR / "audio").mkdir(parents=True, exist_ok=True)
(CAPTURED_TESTS_DIR / "video").mkdir(parents=True, exist_ok=True)


def save_tested_media(media_bytes: bytes, original_filename: str, media_type: str, verdict: str, fake_prob: float) -> str:
    """Saves live captured and tested media to data/captured_tests/ with audit manifest."""
    if not media_bytes or len(media_bytes) < 32:
        return ""
    # Avoid re-saving simulated mock tokens
    if media_bytes.startswith(b'FAKE_') or media_bytes.startswith(b'REAL_'):
        return ""

    if "Image" in media_type:
        subfolder = "images"
        ext = Path(original_filename).suffix if Path(original_filename).suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp'] else ".jpg"
    elif "Audio" in media_type:
        subfolder = "audio"
        ext = Path(original_filename).suffix if Path(original_filename).suffix.lower() in ['.wav', '.mp3', '.aac', '.ogg', '.flac'] else ".wav"
    else:
        subfolder = "video"
        ext = Path(original_filename).suffix if Path(original_filename).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv'] else ".mp4"

    timestamp_str = time.strftime('%Y%m%d_%H%M%S')
    clean_stem = "".join(c for c in Path(original_filename).stem if c.isalnum() or c in ('_', '-'))
    if not clean_stem:
        clean_stem = "capture"
    save_filename = f"{timestamp_str}_{verdict}_{clean_stem}{ext}"

    target_dir = CAPTURED_TESTS_DIR / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / save_filename
    target_file.write_bytes(media_bytes)

    relative_path = f"data/captured_tests/{subfolder}/{save_filename}"

    # Update persistent manifest
    manifest_data = []
    if MANIFEST_FILE.exists():
        try:
            manifest_data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        except Exception:
            manifest_data = []

    entry = {
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "filename": save_filename,
        "original_filename": original_filename,
        "relative_path": relative_path,
        "media_type": media_type,
        "verdict": verdict,
        "fake_probability": fake_prob,
        "size_bytes": len(media_bytes)
    }
    manifest_data.insert(0, entry)
    manifest_data = manifest_data[:200]
    MANIFEST_FILE.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    return relative_path


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

        elif path == '/api/logs' or path == '/api/dataset-logs':
            # Deliver log files bit-by-bit directly to UI
            logs_payload = {
                "dataset_log": "",
                "training_log": "",
                "overnight_log": "",
                "epoch_metrics_csv": "",
                "hardware_telemetry": "",
                "api_service_log": "",
                "dataset_summary": {
                    "master_catalog": "data/dataset_catalog_master.csv",
                    "total_samples": 9100,
                    "train_samples": 7280,
                    "val_samples": 1820,
                    "best_auc": 0.9995,
                    "best_checkpoint": "checkpoints/best/st2_syncnet_best.pt",
                    "author": "Dar Rehan Rasool (@darrehanrasool)",
                    "supervisors": "Dr. Riyaz* & Dr. Javad* (IUST)"
                }
            }

            # 1. Dataset specific logs
            ds_audit = LOGS_DIR / "datasets" / "dataset_audit.log"
            ds_dl = LOGS_DIR / "datasets" / "kaggle_download.log"
            ds_gdrive = LOGS_DIR / "datasets" / "gdrive_streamer.log"
            ds_entries = []
            if ds_audit.exists():
                ds_entries.append(ds_audit.read_text(encoding="utf-8").strip())
            if ds_dl.exists():
                ds_entries.append(ds_dl.read_text(encoding="utf-8").strip())
            if ds_gdrive.exists():
                ds_entries.append(ds_gdrive.read_text(encoding="utf-8").strip())
            logs_payload["dataset_log"] = "\n".join([e for e in ds_entries if e])

            # 2. Training and overnight telemetry
            train_log = LOGS_DIR / "training" / "st2_syncnet_training.log"
            if train_log.exists():
                logs_payload["training_log"] = train_log.read_text(encoding="utf-8")[-15000:]

            overnight_log = LOGS_DIR / "training" / "overnight_training.log"
            if overnight_log.exists():
                logs_payload["overnight_log"] = overnight_log.read_text(encoding="utf-8")[-15000:]

            csv_log = LOGS_DIR / "training" / "epoch_metrics_history.csv"
            if csv_log.exists():
                logs_payload["epoch_metrics_csv"] = csv_log.read_text(encoding="utf-8")[-15000:]

            hw_log = LOGS_DIR / "hardware" / "hardware_telemetry.log"
            if hw_log.exists():
                logs_payload["hardware_telemetry"] = hw_log.read_text(encoding="utf-8")[-15000:]

            api_log = LOGS_DIR / "application" / "api_service.log"
            if api_log.exists():
                logs_payload["api_service_log"] = api_log.read_text(encoding="utf-8")[-15000:]

            self._send_json({
                "status": "success",
                "logs_directory": str(LOGS_DIR),
                "data": logs_payload
            })

        elif path == '/api/saved-tests':
            manifest_data = []
            if MANIFEST_FILE.exists():
                try:
                    manifest_data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
                except Exception:
                    manifest_data = []
            self._send_json({
                "status": "success",
                "count": len(manifest_data),
                "directory": "data/captured_tests",
                "tests": manifest_data
            })

        elif path == '/api/saved-file':
            from urllib.parse import parse_qs
            query = parse_qs(parsed.query)
            rel_path = query.get('path', [''])[0]
            if rel_path:
                target = PROJECT_ROOT / rel_path
                try:
                    resolved = target.resolve()
                    if CAPTURED_TESTS_DIR.resolve() in resolved.parents and resolved.is_file():
                        suffix = resolved.suffix.lower()
                        content_type = "application/octet-stream"
                        if suffix in ['.jpg', '.jpeg']:
                            content_type = "image/jpeg"
                        elif suffix == '.png':
                            content_type = "image/png"
                        elif suffix in ['.wav', '.wave']:
                            content_type = "audio/wav"
                        elif suffix == '.mp4':
                            content_type = "video/mp4"

                        self.send_response(200)
                        self.send_header('Content-Type', content_type)
                        self.send_header('Content-Length', str(resolved.stat().st_size))
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        with open(resolved, 'rb') as f:
                            self.wfile.write(f.read())
                        return
                except Exception as e:
                    logger.error(f"Error serving saved file: {e}")
            self.send_error(404, "Saved file not found")

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

                # Automatically save tested media to data/captured_tests/
                saved_to_path = save_tested_media(
                    media_bytes=media_bytes,
                    original_filename=filename,
                    media_type=analysis['media_type'],
                    verdict=analysis['verdict'],
                    fake_prob=analysis['fake_probability']
                )
                analysis['saved_to'] = saved_to_path

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
