from fastapi.testclient import TestClient

from reduction.gateway.main import app

client = TestClient(app)


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_optimize_endpoint():
    resp = client.post(
        "/v1/pipeline/chat",
        json={"user_message": "hi", "output_format": "toon", "call_provider": False},
    )
    assert resp.status_code == 200
    assert resp.json()["optimized"] is True


def test_encode_toon_endpoint():
    data = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
    resp = client.post("/v1/encode/toon", json=data)
    assert resp.status_code == 200
    assert resp.json()["toon"] == "items[2]{id,name}:\n  1,a\n  2,b"


def test_chat_rejects_bad_output_format():
    resp = client.post("/v1/pipeline/chat", json={"user_message": "hi", "output_format": "xml"})
    assert resp.status_code == 422


def test_metrics_endpoint():
    resp = client.get("/v1/metrics")
    assert resp.status_code == 200
    assert "input_savings_pct" in resp.json()
