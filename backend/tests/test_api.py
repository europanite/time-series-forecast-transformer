from fastapi.testclient import TestClient

from local_ts_forecast.api import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}
