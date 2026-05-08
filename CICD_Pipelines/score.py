import numpy as np
import os
import glob
import joblib
import onnxruntime
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

def init():
    global model, scaler, input_name, label_name

    scaler_path = glob.glob(
        os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'scaler', '**', '*.pkl'),
        recursive=True
    )[0]
    scaler = joblib.load(scaler_path)

    model_onnx = glob.glob(
        os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'support-vector-classifier', '**', '*.onnx'),
        recursive=True
    )[0]
    model = onnxruntime.InferenceSession(model_onnx, None)
    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name

@input_schema('data', NumpyParameterType(np.array([[34.927778, 0.24, 7.3899, 83, 16.1000, 1016.51, 1]])))
@output_schema(NumpyParameterType(np.array([0])))
def run(data):
    try:
        data = scaler.transform(data.reshape(1, 7))
        result = model.run([label_name], {input_name: data.astype(np.float32)})[0]
    except Exception as e:
        return str(e)
    return result.tolist()
