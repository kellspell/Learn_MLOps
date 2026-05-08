
from locust import HttpUser, task, between

test_payload = {
    "temp_c": 8.75,
    "humidity": 0.83,
    "wind_speed_kmph": 70.0,
    "wind_bearing_degree": 259.0,
    "visibility_km": 15.82,
    "pressure_millibars": 1016.51,
    "current_weather_condition": 1.0
}


class MLServiceUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def test_weather_predictions(self):
        with self.client.post("/predict", json=test_payload, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Unexpected status {response.status_code}: {response.text}")
            elif "prediction" not in response.json():
                response.failure(f"Missing 'prediction' in response: {response.text}")

