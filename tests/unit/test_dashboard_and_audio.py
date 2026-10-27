import io
import wave
import numpy as np
import pytest
from PIL import Image
from deepfake_ai.monitoring.audio_alert import play_alert
from deepfake_ai.dashboard.state import DASHBOARD_STATE
from deepfake_ai.dashboard.analyzer import ForensicAnalyzer


@pytest.mark.unit
def test_dashboard_state():
    assert 'active_model' in DASHBOARD_STATE
    play_alert('complete')


@pytest.mark.unit
def test_forensic_analyzer_image():
    img = Image.new('RGB', (128, 128), color='blue')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    res = ForensicAnalyzer.analyze_media(buf.getvalue(), 'test_img.jpg')
    assert res['verdict'] in ['REAL', 'FAKE']
    assert 0.0 <= res['fake_probability'] <= 1.0
    assert 'alert' in res
    assert res['alert']['visual_light'] in ['RED', 'GREEN']
    assert res['alert']['sound'] in ['BEEP', 'CHIME']
    assert 'advanced_metrics' in res
    assert 'explanation_steps' in res['advanced_metrics']


@pytest.mark.unit
def test_forensic_analyzer_audio():
    t = np.linspace(0, 0.5, 8000)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes((np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16).tobytes())
    res = ForensicAnalyzer.analyze_media(wav_buf.getvalue(), 'voice_sample.wav')
    assert res['media_type'] == 'Audio / Voice'
    assert res['verdict'] in ['REAL', 'FAKE']
    assert res['alert']['sound'] in ['BEEP', 'CHIME']


@pytest.mark.unit
def test_forensic_analyzer_video():
    res = ForensicAnalyzer.analyze_media(b'dummy_video_stream_bytes', 'interview.mp4')
    assert res['media_type'] == 'Video'
    assert res['verdict'] in ['REAL', 'FAKE']
    assert res['alert']['visual_light'] in ['RED', 'GREEN']
