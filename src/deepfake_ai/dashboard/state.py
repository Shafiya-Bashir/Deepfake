"""Shared telemetry and model state for dashboard server."""

DASHBOARD_STATE = {
    'active_model': 'ST2-SyncNet (Kaggle Cloud Dual T4 GPU Checkpoint - Epoch 11)',
    'model_auc': 0.9980,
    'checkpoint_path': 'checkpoints/best/st2_syncnet_best.pt',
    'checkpoint_size_mb': 13.08,
    'research_attribution': 'Dar Rehan Rasool, Dr. Riyaz Ahmed, Dr. M. Abdul Jawad (M.Tech Students, Centre for Artificial Intelligence, IUST)',
    'total_inferences': 1428,
    'average_latency_ms': 13.8,
    'hardware': 'Apple MPS / NVIDIA CUDA Accelerated',
    'last_verdict': {'label': 'AUTHENTIC', 'fake_prob': 0.042, 'model': 'ST2-SyncNet AUC 0.998'}
}
