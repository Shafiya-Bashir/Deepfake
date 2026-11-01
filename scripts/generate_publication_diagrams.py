#!/usr/bin/env python3
"""Compatibility wrapper for publication diagram generation.

Invokes the primary diagram generator: scripts/generate_top_notch_diagrams.py
Generates 9 Scalable Vector Graphics (.svg) and editable Draw.io (.drawio) models.

Research Attribution:
    Lead Author: Dar Rehan Rasool (@darrehanrasool)
    Supervisors: Dr. Riyaz* & Dr. Javad* (IUST)
"""

import sys
from pathlib import Path
import subprocess

SCRIPT_DIR = Path(__file__).resolve().parent
TARGET = SCRIPT_DIR / "generate_top_notch_diagrams.py"

if __name__ == "__main__":
    print(f"[*] Delegating to primary diagram engine: {TARGET.name}")
    sys.exit(subprocess.call([sys.executable, str(TARGET)]))
