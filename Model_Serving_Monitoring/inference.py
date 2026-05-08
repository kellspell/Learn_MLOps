
import requests
import pandas as pd

URL = 'http://weather-fastapi-aci.eastus2.azurecontainer.io/predict'

data = pd.read_csv('sample_inference_data.csv')
data = data.drop(columns=['Timestamp', 'Location', 'Future_weather_condition'])
data.columns = [
    'temp_c', 'humidity', 'wind_speed_kmph',
    'wind_bearing_degree', 'visibility_km',
    'pressure_millibars', 'current_weather_condition'
]

for i, row in data.iterrows():
    payload = row.to_dict()
    try:
        r = requests.post(URL, json=payload, timeout=10)
        r.raise_for_status()
        prediction = r.json().get('prediction', 'N/A')
        print(f'Row {i:>4} | prediction: {prediction}')
    except requests.exceptions.HTTPError as e:
        print(f'Row {i:>4} | HTTP error: {e}')
    except requests.exceptions.ConnectionError:
        print(f'Row {i:>4} | Connection error — is the endpoint reachable?')
    except requests.exceptions.Timeout:
        print(f'Row {i:>4} | Request timed out')
