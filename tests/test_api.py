import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

@pytest.fixture(scope="module")
def client():
    from app import app
    with TestClient(app, raise_server_exceptions=True) as c:  # ← context manager triggers lifespan
        yield c



def test_root_health(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "running" in r.json()["message"].lower()


def test_get_classes(client):
    r = client.get("/classes")
    assert r.status_code == 200
    data = r.json()
    assert "classes" in data
    assert len(data["classes"]) == 10  # EuroSAT has 10 classes


def make_dummy_image_bytes(size=(64, 64)):
    img = Image.new("RGB", size, color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_predict_returns_valid_response(client):
    buf = make_dummy_image_bytes()
    r = client.post("/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "all_scores" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["all_scores"]) == 10


def test_predict_confidence_sums_to_one(client):
    buf = make_dummy_image_bytes()
    r = client.post("/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
    scores = r.json()["all_scores"]
    total = sum(scores.values())
    assert abs(total - 1.0) < 0.01  # softmax must sum to ~1


def test_predict_top_class_matches_confidence(client):
    buf = make_dummy_image_bytes()
    r = client.post("/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
    data = r.json()
    top_class = data["prediction"]
    reported_confidence = data["confidence"]
    score_for_top = data["all_scores"][top_class]
    assert abs(reported_confidence - score_for_top) < 0.001


def test_predict_invalid_file_returns_400(client):
    r = client.post(
        "/predict",
        files={"file": ("bad.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    assert r.status_code == 400


def test_explain_returns_png(client):
    buf = make_dummy_image_bytes()
    r = client.post(
        "/predict/explain/view",
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 1000  # non-empty PNG