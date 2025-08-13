import requests as req
import time

token = "abc123"

headers = {"Authorization": f"Bearer {token}"}

data = {
    "age": 42,
    "job": "entrepreneur",
    "marital": "married",
    "education": "primary",
    "balance": 558,
    "housing": "yes",
    "duration": 186,
    "campaign": 2,
}

resp = req.post("http://localhost:8000/predict", json=data, headers=headers)

print(resp.status_code)
print(resp.text)
