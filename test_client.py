import requests
import json

API_URL = "http://localhost:8000"

def predict():
    response = requests.post(
        url=f"{API_URL}/predict", 
        json={"features": [5.1, 3.5, 1.4, 0.2]},
    )

    print(json.dumps(response.json()))

def predict_batch():
    response = requests.post(
        url=f"{API_URL}/predict-batch", 
        json={"features": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]},
    )

    print(json.dumps(response.json()))

if __name__ == "__main__":
    predict()
    predict_batch()