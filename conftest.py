"""
conftest.py — patches model loading for CI environment.

In CI, outputs/best_model.pth does not exist (training was not run).
We patch load_model inside the lifespan startup so all API and
architecture tests can run without a trained checkpoint.
"""
import pytest
from unittest.mock import patch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def _mock_load_model(model, path, device):
    """Return model with random weights instead of loading from disk."""
    model.eval()
    return model


@pytest.fixture(autouse=True)
def patch_model_loading():
    with patch("src.utils.load_model", side_effect=_mock_load_model):
        with patch("app.load_model", side_effect=_mock_load_model):
            yield