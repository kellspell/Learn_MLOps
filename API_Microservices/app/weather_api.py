import uvicorn
from fastapi import FastAPI
from variables import WeatherVariables
import numpy
import joblib
import onnxruntime as rt

app = FastAPI()

scaler = joblib.load("artifacts/scaler.pkl")

sess = rt.InferenceSession("artifacts/svc.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

label_map = {0: "Clear", 1: "Rain", 2: "Snow"}


@app.get('/')
def index():
    return {'message': 'Weather prediction service. Access API docs at /docs.'}


@app.post('/predict')
def predict_weather(data: WeatherVariables):
    data = data.dict()

    features = numpy.array([[
        data['temp_c'],
        data['humidity'],
        data['wind_speed_kmph'],
        data['wind_bearing_degree'],
        data['visibility_km'],
        data['pressure_millibars'],
        data['current_weather_condition']
    ]])

    features = scaler.transform(features.reshape(1, 7))
    prediction = sess.run([label_name], {input_name: features.astype(numpy.float32)})[0]

    return {'prediction': label_map.get(int(prediction[0]), 'Unknown')}
