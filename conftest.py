"""
conftest.py — patches model loading for CI environment.

In CI, outputs/best_model.pth does not exist (training was not run).
We patch load_model to return the model with random weights so all
API and architecture tests can run without a trained checkpoint.
"""
import pytest
from unittest.mock import patch
import sys
import os

# Add the project root to Python path
# This lets pytest find app.py, src/, config.py etc.
sys.path.insert(0, os.path.dirname(__file__))
import torch


def _mock_load_model(model, path, device):
    """Return model with random weights instead of loading from disk."""
    model.eval()
    return model


@pytest.fixture(autouse=True)
def patch_model_loading():
    with patch("src.utils.load_model", side_effect=_mock_load_model):
        with patch("app.load_model", side_effect=_mock_load_model):
            yield