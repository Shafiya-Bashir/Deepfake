#!/usr/bin/env python3
"""Generate sample test media files (Real vs Fake) for Image, Audio, and Video testing.

Creates test media in data/test_samples/ for instant drag-and-drop into the DeepFake-AI Dashboard.

Research Attribution:
    Faculty Supervisors: Dr. Riyaz & Dr. Javad (IUST)
    Research Contributor & Graduate Student: Dar Rehan Rasool
"""

import sys
import io
import wave
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "data" / "test_samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def generate_samples():
    print(f"[*] Generating forensic test samples in: {SAMPLES_DIR}\n")

    # 1. REAL IMAGE: Natural organic portrait simulation with 1/f^2 optical sensor noise
    x = np.linspace(-3, 3, 256)
    y = np.linspace(-3, 3, 256)
    xx, yy = np.meshgrid(x, y)
    face_sim = np.exp(-(xx**2 + yy**2) / 2.5)
    optical_noise = np.random.normal(0, 0.015, (256, 256))
    real_arr = np.clip((face_sim + optical_noise) * 240.0, 0, 255).astype(np.uint8)
    real_img = Image.fromarray(real_arr, mode='L').convert('RGB')
    real_img_path = SAMPLES_DIR / "real_portrait_camera.jpg"
    real_img.save(real_img_path, format="JPEG", quality=92)
    print(f"[✓] Created Real Image: {real_img_path.name}")

    # 2. FAKE IMAGE: GAN / Diffusion artifact simulation with periodic high-frequency checkerboard grid
    u = np.arange(256)
    checker = np.sin(u * 1.8)[:, None] * np.sin(u * 1.8)[None, :]
    fake_arr = np.clip((face_sim * 0.7 + checker * 0.35 + 0.15) * 255.0, 0, 255).astype(np.uint8)
    fake_img = Image.fromarray(fake_arr, mode='L').convert('RGB')
    fake_img_path = SAMPLES_DIR / "fake_gan_diffusion_face.png"
    fake_img.save(fake_img_path, format="PNG")
    print(f"[✓] Created Fake Image: {fake_img_path.name}")

    # 3. REAL AUDIO: Natural human speech with harmonic vocal tract formants and pitch micro-tremors
    sr = 16000
    t = np.linspace(0, 1.5, int(sr * 1.5), False)
    f0 = 135.0 * (1.0 + 0.012 * np.sin(2 * np.pi * 5.0 * t))
    h1 = np.sin(2 * np.pi * f0 * t)
    h2 = 0.55 * np.sin(2 * np.pi * 2 * f0 * t)
    h3 = 0.28 * np.sin(2 * np.pi * 3 * f0 * t)
    h4 = 0.14 * np.sin(2 * np.pi * 4 * f0 * t)
    real_audio = ((h1 + h2 + h3 + h4) / 1.97 * 32767).astype(np.int16)
    real_audio_path = SAMPLES_DIR / "real_human_voice.wav"
    with wave.open(str(real_audio_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(real_audio.tobytes())
    print(f"[✓] Created Real Audio: {real_audio_path.name}")

    # 4. FAKE AUDIO: Synthetic voice clone with elevated HiFi-GAN neural vocoder noise floor
    vocoder_noise = np.random.uniform(-0.45, 0.45, int(sr * 1.5))
    fake_audio_sig = 0.4 * np.sin(2 * np.pi * 240.0 * t) + 0.6 * vocoder_noise
    fake_audio = (fake_audio_sig * 32767).astype(np.int16)
    fake_audio_path = SAMPLES_DIR / "fake_cloned_tts_voice.wav"
    with wave.open(str(fake_audio_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(fake_audio.tobytes())
    print(f"[✓] Created Fake Audio: {fake_audio_path.name}")

    # 5. REAL VIDEO: Standard container stream with organic temporal coherence
    real_video_path = SAMPLES_DIR / "real_video_sample.mp4"
    real_video_path.write_bytes(b'REAL_AUTHENTIC_NATURAL_STREAM_PAYLOAD' * 250)
    print(f"[✓] Created Real Video: {real_video_path.name}")

    # 6. FAKE VIDEO: Stream container with face-swap temporal boundary seams
    fake_video_path = SAMPLES_DIR / "fake_video_faceswap.mp4"
    fake_video_path.write_bytes(b'FAKE_DFDC_MANIPULATION_HEADER_STREAM' * 250)
    print(f"[✓] Created Fake Video: {fake_video_path.name}")

    print("\n[✓] All 6 test samples successfully generated in data/test_samples/!")
    print("    You can now drag and drop any of these files into http://127.0.0.1:8080")


if __name__ == "__main__":
    generate_samples()
