import pytest
from deepfake_ai.monitoring.audio_alert import play_alert
from deepfake_ai.dashboard.state import DASHBOARD_STATE


@pytest.mark.unit
def test_dashboard_state():
    assert 'active_model' in DASHBOARD_STATE
    play_alert('complete')
