"""API integration tests for FastAPI backend."""

from fastapi.testclient import TestClient
from web.app import app


client = TestClient(app)


def test_api_presets():
    response = client.get("/api/presets")
    assert response.status_code == 200
    data = response.json()
    assert "presets" in data
    assert len(data["presets"]) > 0


def test_api_index():
    response = client.get("/")
    assert response.status_code == 200
    assert "Türkçe Şive ve Telaffuz Ölçer" in response.text


def test_api_evaluate_endpoint():
    # Evaluate with sample audio
    with open("/tmp/test_tts.mp3", "rb") as f:
        response = client.post(
            "/api/evaluate",
            files={"file": ("test_tts.mp3", f, "audio/mpeg")},
            data={"target_text": "merhaba bugün nasılsınız"}
        )
    assert response.status_code == 200
    res_json = response.json()
    assert "overall_score" in res_json
    assert res_json["overall_score"] > 70.0
    assert "subscores" in res_json
    assert "word_scores" in res_json
