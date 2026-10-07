from app import app

def test_ping():
    client = app.test_client()
    
    # test predict single
    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data

    # test predict batch
    response = client.post(
        "/predict-batch",
        json={"features": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "predictions" in data