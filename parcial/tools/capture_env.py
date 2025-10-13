#!/usr/bin/env python3
"""Captura información del entorno para reproducibilidad"""
import platform
import sys
from datetime import datetime

print(f"DATE={datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}")
print(f"PYTHON {sys.version.replace(chr(10), ' ')}")

try:
    import numpy as np
    print(f"NUMPY {np.__version__}")
except Exception:
    print("NUMPY none")

try:
    import torch
    print(f"TORCH {torch.__version__}")
except Exception:
    print("TORCH none")

print(f"PLATFORM {platform.platform()}")
