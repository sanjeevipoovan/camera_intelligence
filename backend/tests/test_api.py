"""
End-to-End HTTP Integration Tests.

Validates the full pipeline over HTTP:
- Successful analysis
- Rejection of pitch-black images (HTTP 422)
- Rejection of oversized files (HTTP 413)
- Rejection of spoofed formats (HTTP 415)
- Health check uptime (HTTP 200)
"""

import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def generate_test_jpeg(width: int = 400, height: int = 400, is_black: bool = False) -> bytes:
    """Generate in-memory JPEG bytes for HTTP upload."""
    if is_black:
        img = Image.new("RGB", (width, height), color=(0, 0, 0))
    else:
        np_arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(np_arr, mode="RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_endpoint() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_name" in data


def test_analyze_success() -> None:
    jpeg_bytes = generate_test_jpeg(width=300, height=300, is_black=False)
    files = {"file": ("camera.jpg", jpeg_bytes, "image/jpeg")}
    data = {"question": "What is in this image?"}

    response = client.post("/api/v1/analyze", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["success"] is True
    assert "answer" in res_data
    assert res_data["metrics"]["tokens_generated"] > 0


def test_analyze_rejects_pitch_black_photo() -> None:
    black_bytes = generate_test_jpeg(width=300, height=300, is_black=True)
    files = {"file": ("dark.jpg", black_bytes, "image/jpeg")}
    data = {"question": "Can you see anything?"}

    response = client.post("/api/v1/analyze", files=files, data=data)
    assert response.status_code == 422
    assert response.json()["error"] == "IMAGE_TOO_DARK"


def test_analyze_rejects_spoofed_file() -> None:
    fake_bytes = b"NOT_A_REAL_IMAGE_DATA_STREAM"
    files = {"file": ("malicious.jpg", fake_bytes, "image/jpeg")}
    data = {"question": "Is this valid?"}

    response = client.post("/api/v1/analyze", files=files, data=data)
    assert response.status_code == 415
    assert response.json()["error"] == "UNSUPPORTED_MEDIA_TYPE"


def test_analyze_rejects_oversized_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.MAX_IMAGE_SIZE_MB", 0.001)  # ~1 KB
    jpeg_bytes = generate_test_jpeg(width=500, height=500)
    files = {"file": ("large.jpg", jpeg_bytes, "image/jpeg")}
    data = {"question": "Analyze this"}

    response = client.post("/api/v1/analyze", files=files, data=data)
    assert response.status_code == 413
    assert response.json()["error"] == "PAYLOAD_TOO_LARGE"