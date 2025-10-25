#!/usr/bin/env python3
"""Configuracion de pytest"""
import pytest
import torch

@pytest.fixture(autouse=True)
def set_seed():
    """Seed fijo para reproducibilidad"""
    torch.manual_seed(42)
