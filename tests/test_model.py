import torch
import pytest
from src.model import EurosatCNN
from config import NUM_CLASSES


@pytest.fixture
def model():
    m = EurosatCNN(num_classes=NUM_CLASSES)
    m.eval()
    return m


def test_model_output_shape(model):
    x = torch.randn(4, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, NUM_CLASSES), (
        f"Expected (4, {NUM_CLASSES}), got {out.shape}"
    )


def test_model_output_is_logits_not_probs(model):
    """Verify output is raw logits, not softmax probabilities."""
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)

    has_negative = (out < 0).any().item()
    has_greater_than_one = (out > 1).any().item()

    assert has_negative or has_greater_than_one, (
        "Model output looks like probabilities (all values in [0,1]). "
        "Expected raw logits — check if softmax is applied inside forward()."
    )


def test_model_single_image(model):
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, NUM_CLASSES)


def test_softmax_sums_to_one(model):
    x = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(3), atol=1e-5)


def test_model_has_block3(model):
    assert hasattr(model, "block3"), "model must have block3 for GradCAM"
    assert len(model.block3) > 3, "model.block3 must have at least 4 layers"