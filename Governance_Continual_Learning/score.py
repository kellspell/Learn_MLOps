import numpy as np
import os
import glob
import joblib
import onnxruntime
from applicationinsights import TelemetryClient
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

# Find your Application Insights instrumentation key in Azure Portal:
# Azure ML workspace → Associated resources → Application Insights → Overview → Instrumentation Key
TELEMETRY_KEY = '29784952-4aa1-4def-9a18-a90b29b2f66a'

LABEL_MAP = {0: 'Clear', 1: 'Rain', 2: 'Snow'}


def init():
    global model, scaler, input_name, label_name, tc

    tc = TelemetryClient(TELEMETRY_KEY)

    scaler_path = glob.glob(
        os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'scaler', '**', '*.pkl'),
        recursive=True
    )[0]
    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        tc.track_event('FileNotFoundException', {'error_message': str(e)}, {'FileNotFoundError': 101})
        tc.flush()

    model_onnx = glob.glob(
        os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'support-vector-classifier', '**', '*.onnx'),
        recursive=True
    )[0]
    try:
        model = onnxruntime.InferenceSession(model_onnx, None)
    except Exception as e:
        tc.track_event('FileNotFoundException', {'error_message': str(e)}, {'FileNotFoundError': 101})
        tc.flush()

    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name


@input_schema('data', NumpyParameterType(np.array([[34.927778, 0.24, 7.3899, 83, 16.1000, 1016.51, 1]])))
@output_schema(NumpyParameterType(np.array([0])))
def run(data):
    try:
        data = scaler.transform(data)
    except Exception as e:
        tc.track_event('ScalingException', {'error_message': str(e)}, {'ScalingError': 301})
        tc.flush()
        return 'error'

    try:
        result = model.run([label_name], {input_name: data.astype(np.float32)})[0]
        return LABEL_MAP.get(int(result[0]), 'Unknown')
    except Exception as e:
        tc.track_event('InferenceException', {'error_message': str(e)}, {'InferenceError': 401})
        tc.flush()
        return 'error' 