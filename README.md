# Weather AI Project — MLOps on Azure ML

End-to-end MLOps project that covers the full machine learning lifecycle: raw sensor data ingestion, feature engineering, training two classifiers, experiment tracking with Azure ML and MLflow, ONNX model export, registration to the Azure ML Model Registry, containerised deployment to both Azure Container Instance (staging) and Azure Kubernetes Service (production), a standalone FastAPI microservice deployed to ACI via Docker, a fully automated CI/CD pipeline on Azure DevOps, load testing with Locust, batch inference monitoring, and a governance scoring endpoint on AKS with Application Insights telemetry.

---

## Prerequisites

### Azure resources required

| Resource | Name used in this project |
|---|---|
| Azure subscription | `f2f70602-65d9-479b-8014-a1c92a06ef0e` |
| Resource group | `Learn_MLOps` |
| Azure ML workspace | `MLOps` (East US 2) |
| Azure Container Registry | created automatically by Azure ML |
| AKS cluster | `port-aks` (Standard_D2s_v3, created by the deployment notebook) |

### Local environment setup

1. **Clone the repository**
   ```bash
   git clone https://kellspell2019@dev.azure.com/kellspell2019/Learn_MLOps/_git/Learn_MLOps
   cd Learn_MLOps
   ```

2. **Create and activate the conda environment**
   ```bash
   conda create -n Mlflow python=3.11 -y
   conda activate Mlflow
   pip install azureml-core azureml-dataset-runtime mlflow scikit-learn \
               skl2onnx onnxruntime pandas numpy matplotlib seaborn gitpython
   ```

3. **Authenticate to Azure ML**
   ```bash
   az login
   ```

4. **Install additional packages for FastAPI deployment**
   ```bash
   pip install azure-mgmt-containerinstance azure-identity
   ```

5. **Run the notebooks in order**
   ```
   1. Model/Dataprocessing_register.ipynb
   2. ML_Pipelines/ML-pipeline.ipynb
   3. Deploy/Deploy_model_ACI.ipynb
   4. Deploy/Deploy_model_AKS.ipynb
   5. Deploy/Deploy_FastAPI_ACI.ipynb
   ```

### Azure DevOps CI/CD setup

1. Push the repo to Azure DevOps.
2. Create a service connection named **`mlops_sp`** (Workload Identity Federation) scoped to the `Learn_MLOps` resource group.
3. Register the pipeline in Azure DevOps pointing to `/azure-pipelines.yml`.
4. The pipeline will trigger automatically on the next push to `main`.

---

## Project Structure

```
Weather-AI-Project/
├── Dataset/
│   ├── weather_dataset_raw.csv          # Original raw data (96,453 rows)
│   ├── weather_dataset_processed.csv    # Cleaned & feature-engineered data
│   └── mlflow.db                        # Local MLflow tracking store
├── Model/
│   └── Dataprocessing_register.ipynb    # Data cleaning & Azure ML registration
├── ML_Pipelines/
│   ├── Data/
│   │   ├── training_data.csv            # 77,160 rows (80 %)
│   │   └── validation_data.csv          # 19,289 rows (20 %)
│   ├── outputs/
│   │   ├── svc.onnx                     # Exported SVM model
│   │   ├── rf.onnx                      # Exported Random Forest model
│   │   └── scaler.pkl                   # Fitted StandardScaler
│   ├── ML-pipeline.ipynb                # Training & registration pipeline
│   └── azure-pipelines.yml              # Azure DevOps CI/CD pipeline definition
├── Deploy/
│   ├── score.py                         # Inference entry script (ONNX runtime)
│   ├── Deploy_model_ACI.ipynb           # ACI deployment notebook
│   ├── Deploy_model_AKS.ipynb           # AKS deployment notebook
│   ├── Deploy_Model_MLflow.ipynb        # MLflow model deployment notebook
│   └── Deploy_FastAPI_ACI.ipynb         # FastAPI microservice ACI deployment
├── API_Microservices/
│   ├── Dockerfile                       # Python 3.11-slim container definition
│   └── app/
│       ├── weather_api.py               # FastAPI app with /predict endpoint
│       ├── variables.py                 # Pydantic request schema
│       ├── requirements.txt             # Container dependencies
│       └── artifacts/
│           ├── svc.onnx                 # SVM model (ONNX)
│           ├── rf.onnx                  # Random Forest model (ONNX)
│           └── scaler.pkl               # Fitted StandardScaler
├── CICD_Pipelines/
│   ├── aml_config/
│   │   └── config.json                  # Azure ML workspace connection config
│   ├── score.py                         # Inference entry script for CI/CD deploy
│   ├── conda_env.yml                    # Conda environment spec for the container
│   ├── InferenceConfig.yml              # Links score.py and conda_env.yml
│   ├── AciDeploymentConfig.yml          # ACI compute & auth settings
│   ├── AksDeploymentConfig.yml          # AKS compute, autoscaling & auth settings
│   └── deploy.py                        # Python deployment script (ACI and AKS)
├── Testing_Security/
│   └── load_test.py                     # Locust load test for the FastAPI endpoint
├── Essentials_Production_Release/
│   └── create_aks_cluster.py            # Verifies / connects to the port-aks production cluster
├── Model_Serving_Monitoring/
│   ├── inference.py                     # Batch inference script against the FastAPI endpoint
│   └── sample_inference_data.csv        # 582-row sample dataset for monitoring & validation
├── Governance_Continual_Learning/
│   ├── score.py                         # AKS scoring script with Application Insights telemetry
│   ├── deploy.py                        # Deploys governance scoring script to port-aks
│   └── conda_env.yml                    # Environment spec including applicationinsights
└── README.md
```

---

## Dataset

| Property | Value |
|---|---|
| Source | Port of Turku, Finland — hourly weather readings |
| Date range | April 2006 onward |
| Raw size | 96,453 rows × 11 columns |
| Processed size | 96,449 rows × 10 columns |

**Raw columns:**

| Column | Type | Description |
|---|---|---|
| S_No | int | Row index (dropped in processing) |
| Timestamp | object → datetime | Hourly timestamp with timezone |
| Location | object | Sensor location string |
| Temperature_C | float | Dry-bulb temperature (°C) |
| Apparent_Temperature_C | float | Feels-like temperature (dropped — 0.99 correlation with Temperature_C) |
| Humidity | float | Relative humidity (0–1) |
| Wind_speed_kmph | float | Wind speed (km/h) |
| Wind_bearing_degrees | int | Wind direction (°) |
| Visibility_km | float | Visibility distance (km) |
| Pressure_millibars | float | Atmospheric pressure (mbar) |
| Weather_conditions | object | Target label: `rain`, `snow`, `clear` |

**Class distribution (raw):**

| Label | Encoded value | Count |
|---|---|---|
| rain | 1 | 82,271 |
| snow | 2 | 10,712 |
| clear | 0 | 3,470 |

---

## Notebook 1 — Data Processing & Registration

**File:** `Dataprocessing_register.ipynb`

### Steps

1. **Data quality assessment** — `df.describe()`, shape, and dtype inspection.

2. **Missing data handling** — `Weather_conditions` had NaN rows filled with forward-fill (`ffill`). No other column had nulls.

3. **Timestamp conversion** — parsed from timezone-aware string to `datetime64`.

4. **Label encoding** — `Weather_conditions` string labels converted to integers using `sklearn.LabelEncoder` → new column `Current_weather_condition`.

5. **Feature engineering** — `Future_weather_condition` created by shifting `Current_weather_condition` forward by **4 rows (4 hours)**, making this a short-horizon weather forecasting task. Rows that produce NaN from the shift are dropped.

6. **Correlation analysis** — Pearson correlation matrix computed and visualised as a seaborn heatmap. Key findings:
   - `Temperature_C` ↔ `Apparent_Temperature_C`: **0.99** (redundant → dropped)
   - `Temperature_C` ↔ `Current_weather_condition`: **−0.58**
   - `Current_weather_condition` ↔ `Future_weather_condition`: **0.83**

7. **Feature selection** — `S_No` (irrelevant index) and `Apparent_Temperature_C` (redundant) removed.

8. **Time-series visualisation** — temperature plotted over the full date range.

9. **Save processed dataset** — written to `Dataset/weather_dataset_processed.csv`.

10. **Azure ML registration** — uploaded to the default Azure ML Datastore and registered as a versioned Tabular Dataset named `processed_weather_data_portofTurku`.

### Libraries used

`pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn.preprocessing.LabelEncoder`, `azureml.core` (Workspace, Dataset), `azureml.data.DataType`

---

## Notebook 2 — ML Pipeline

**File:** `ML_Pipelines/ML-pipeline.ipynb`

### Azure ML Workspace

| Setting | Value |
|---|---|
| Subscription | `f2f70602-65d9-479b-8014-a1c92a06ef0e` |
| Resource group | `Learn_MLOps` |
| Workspace | `MLOps` |
| Region | East US 2 |

### Steps

#### 1. Data ingestion from Azure ML
Loads `processed_weather_data_portofTurku` (latest version) from the registered Azure ML Dataset. Fixes a type-casting issue where Azure ML misreads integer labels as booleans — corrected by remapping `{True→1, False→0, NaN→2}`.

#### 2. Train / validation split
Chronological (index-based) split to preserve temporal ordering:

| Split | Rows | Share |
|---|---|---|
| Training | 77,160 | ~80 % |
| Validation | 19,289 | ~20 % |

Both splits are saved as CSVs, uploaded to the Azure Datastore, and registered as separate versioned datasets (`training_dataset`, `validation_dataset`).

#### 3. Feature selection & scaling

**Input features (X):**

| Feature |
|---|
| Temperature_C |
| Humidity |
| Wind_speed_kmph |
| Wind_bearing_degrees |
| Visibility_km |
| Pressure_millibars |
| Current_weather_condition |

**Target (y):** `Future_weather_condition` (weather 4 hours ahead)

The training set is further split 80/20 into train and test subsets (`sklearn.model_selection.train_test_split`, `random_state=1`). Both are normalised with `sklearn.preprocessing.StandardScaler` (fit on train, transform applied to both).

---

### Model 1 — Support Vector Machine

**Experiment names:** `support-vector-machine` (Azure ML) / `mlflow-support-vector-machine` (MLflow)

Hyperparameter search via `GridSearchCV`:

| Parameter | Values searched |
|---|---|
| kernel | `linear`, `rbf` |
| C | `1`, `10` |

**Best parameters found:** `kernel=rbf`, `C=1.0`

**Test set results:**

| Metric | Value |
|---|---|
| Accuracy | **95.19 %** |
| Precision (macro) | 88.70 % |
| Recall (macro) | 88.59 % |
| F1-score (macro) | 88.64 % |

---

### Model 2 — Random Forest Classifier

**Experiment names:** `random-forest-classifier` (Azure ML) / `mlflow-random-forest-classifier` (MLflow)

**Hyperparameters:** `n_estimators=100`, `max_depth=10`, `random_state=0`

**Test set results:**

| Metric | Value |
|---|---|
| Accuracy | **95.54 %** |
| Precision (macro) | 90.21 % |
| Recall (macro) | 88.28 % |
| F1-score (macro) | 89.21 % |

> Random Forest outperforms SVM on all metrics and trains in ~7 seconds vs ~25 minutes for SVM with grid search.

---

### Experiment Tracking

Both models are tracked in parallel with:

- **Azure ML Experiments** — logs parameters, metrics, dataset name/version, and git commit SHA via `azureml.core.experiment.Experiment`.
- **MLflow** — tracking URI set to the Azure ML workspace MLflow endpoint. Models logged with `mlflow.sklearn.log_model`.

---

### Model Packaging & Registration

Models are serialised to **ONNX** format using `skl2onnx` for cross-platform inference compatibility:

| Artefact | Format | Registered name |
|---|---|---|
| SVM model | ONNX | `support-vector-classifier` |
| Random Forest model | ONNX | `random-forest-classifier` |
| StandardScaler | pickle | `scaler` |

All three are registered in the Azure ML Model Registry via `azureml.core.model.Model.register`, tagged with the dataset name, version, and test accuracy.

### Libraries used

`pandas`, `numpy`, `scikit-learn` (SVC, RandomForestClassifier, GridSearchCV, StandardScaler, metrics), `azureml.core` (Workspace, Dataset, Experiment, Model), `mlflow`, `mlflow.sklearn`, `skl2onnx`, `pickle`, `matplotlib`, `gitpython`

---

## Notebook 3 — Model Deployment

### Deploy to Azure Container Instance (ACI)

**File:** `Deploy/Deploy_model_ACI.ipynb`

Deploys the registered ONNX model and scaler to a lightweight ACI endpoint for staging and testing.

#### Steps

1. **Connect to workspace** — loads `MLOps` workspace via `azureml.core.Workspace`.
2. **Write inference script** — `score.py` is written to disk via `%%writefile`. It loads the scaler and ONNX model on startup (`init()`) and runs predictions on incoming requests (`run(data)`).
3. **Define environment** — creates a fresh Python 3.8 `Environment` with `CondaDependencies`: `numpy`, `onnxruntime`, `joblib`, `scikit-learn`, `inference-schema[numpy-support]`, `azureml-defaults`.
4. **Build inference config** — `InferenceConfig(entry_script='score.py', environment=env)`.
5. **Build deployment config** — `AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1, auth_enabled=False)`.
6. **Load registered models** — fetches `support-vector-classifier` and `scaler` from the Azure ML Model Registry.
7. **Deploy** — `Model.deploy()` builds a Docker image, pushes it to ACR, and starts the container. `service.wait_for_deployment(show_output=True)` streams progress.

#### Endpoint

| Property | Value |
|---|---|
| Type | Azure Container Instance |
| Auth | None (public) |
| Scoring URI | `http://<aci-ip>.eastus2.azurecontainer.io/score` |

---

### Deploy to Azure Kubernetes Service (AKS)

**File:** `Deploy/Deploy_model_AKS.ipynb`

Deploys the same models to a production-grade AKS cluster with autoscaling and key-based authentication.

#### Steps

1–4. Same as ACI notebook (workspace, score.py, environment, inference config).
5. **Provision AKS cluster** — creates or reuses an AKS compute target named `port-aks` with `Standard_D2s_v3` nodes. Detects and recreates any cluster in a `Failed` provisioning state automatically.
6. **Build deployment config** — `AksWebservice.deploy_configuration` with autoscaling (1–3 replicas, 70 % target utilisation), `auth_enabled=True`.
7. **Load registered models** — same as ACI.
8. **Deploy** — `Model.deploy()` targeting the AKS compute, then waits for completion.
9. **Retrieve auth key** — `service.get_keys()` returns the primary key for authenticated requests.

#### Endpoint

| Property | Value |
|---|---|
| Type | Azure Kubernetes Service |
| Auth | Key-based |
| Scoring URI | `http://<aks-ip>:80/api/v1/service/weather-aks-prediction/score` |

---

### Deploy via MLflow

**File:** `Deploy/Deploy_Model_MLflow.ipynb`

Alternative deployment path that uses the MLflow tracking URI to reference the model directly from an MLflow experiment run, then deploys to ACI using the same ONNX inference stack.

#### Steps

1. **Connect to workspace** and set `mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())`.
2. **Reference model artefacts** — points to the MLflow run path for `mlflow-support-vector-machine` experiment.
3. **Write score.py inline** — same ONNX inference logic as the other notebooks, written programmatically via Python instead of `%%writefile`.
4. **Define environment** — same Python 3.8 environment with ONNX dependencies.
5. **Deploy to ACI** — `Model.deploy()` targeting `port-weather-pred` service, North Europe region.
6. **Test the endpoint** — calls `webservice.run()` with a sample payload and prints the prediction.

> This notebook demonstrates the MLflow-native deployment path as an alternative to the AML SDK-only approach used in `Deploy_model_ACI.ipynb`.

---

## Calling the Endpoints

### Input format

Both endpoints accept a JSON body with a `data` key containing a 2D array of 7 features in this order:

| Index | Feature | Example value |
|---|---|---|
| 0 | Temperature_C | `34.93` |
| 1 | Humidity | `0.24` |
| 2 | Wind_speed_kmph | `7.39` |
| 3 | Wind_bearing_degrees | `83` |
| 4 | Visibility_km | `16.10` |
| 5 | Pressure_millibars | `1016.51` |
| 6 | Current_weather_condition | `1` |

**Output:** integer label — `0` = clear, `1` = rain, `2` = snow

---

### ACI endpoint (no auth)

```python
import requests, json

url = "http://<aci-ip>.eastus2.azurecontainer.io/score"
payload = {"data": [[34.927778, 0.24, 7.3899, 83, 16.1000, 1016.51, 1]]}

response = requests.post(url, json=payload)
print(response.json())  # e.g. [1]
```

```bash
curl -X POST http://<aci-ip>.eastus2.azurecontainer.io/score \
     -H "Content-Type: application/json" \
     -d '{"data": [[34.927778, 0.24, 7.3899, 83, 16.1000, 1016.51, 1]]}'
```

---

### AKS endpoint (key auth)

```python
import requests, json

url = "http://<aks-ip>:80/api/v1/service/weather-aks-prediction/score"
api_key = "<primary-key-from-service.get_keys()>"
payload = {"data": [[34.927778, 0.24, 7.3899, 83, 16.1000, 1016.51, 1]]}

response = requests.post(
    url,
    json=payload,
    headers={"Authorization": f"Bearer {api_key}"}
)
print(response.json())  # e.g. [1]
```

```bash
curl -X POST http://<aks-ip>:80/api/v1/service/weather-aks-prediction/score \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <api-key>" \
     -d '{"data": [[34.927778, 0.24, 7.3899, 83, 16.1000, 1016.51, 1]]}'
```

> Retrieve the AKS key from the Azure ML studio (Endpoints → weather-aks-prediction → Consume) or by running `service.get_keys()[0]` in Python.

---

## FastAPI Microservice

**File:** `Deploy/Deploy_FastAPI_ACI.ipynb`

A standalone REST API built with FastAPI that wraps the SVM ONNX model and exposes a `/predict` endpoint. Deployed as a Docker container to ACI — independent of the Azure ML scoring endpoints.

### Application structure

| File | Purpose |
|---|---|
| `weather_api.py` | FastAPI app — loads scaler and ONNX model on startup, exposes `GET /` and `POST /predict` |
| `variables.py` | Pydantic model defining the 7-field request body schema |
| `requirements.txt` | Container dependencies: fastapi, uvicorn, onnxruntime, scikit-learn, joblib, numpy |
| `Dockerfile` | `python:3.11-slim` base, copies `app/`, installs requirements, runs uvicorn on port 80 |
| `artifacts/` | ONNX models and scaler bundled directly into the container image |

### Deployment steps

1. **Connect to workspace** — retrieves the ACR name from `ws.get_details()['containerRegistry']`.
2. **Get ACR credentials** — uses `azure-mgmt-containerregistry` with token extracted from the azureml workspace auth (no Azure CLI required).
3. **Docker login** — authenticates the local Docker daemon to the ACR.
4. **Docker build** — builds `python:3.11-slim` image, installs dependencies, bundles the app and model artifacts.
5. **Docker push** — pushes the image to ACR as `weather-fastapi:latest`.
6. **Create ACI container** — uses `azure-mgmt-containerinstance` to create a public container group with a DNS name label.
7. **Test** — sends a sample POST request and prints the prediction.

### Endpoint

| Property | Value |
|---|---|
| Type | Azure Container Instance (Docker) |
| Auth | None (public) |
| Base URL | `http://weather-fastapi-aci.eastus2.azurecontainer.io` |
| Predict | `POST /predict` |
| Docs | `GET /docs` (Swagger UI) |

### Request / response

```python
import requests

url = "http://weather-fastapi-aci.eastus2.azurecontainer.io/predict"
payload = {
    "temp_c": 34.927778,
    "humidity": 0.24,
    "wind_speed_kmph": 7.3899,
    "wind_bearing_degree": 83,
    "visibility_km": 16.1000,
    "pressure_millibars": 1016.51,
    "current_weather_condition": 1
}
response = requests.post(url, json=payload)
print(response.json())  # {"prediction": "Rain"}
```

**Output labels:** `"Clear"` (0) · `"Rain"` (1) · `"Snow"` (2)

### Difference from Azure ML scoring endpoints

| | Azure ML ACI/AKS endpoints | FastAPI microservice |
|---|---|---|
| Runtime | Azure ML container + inference-schema | Plain Docker + FastAPI |
| Auth | None / key-based | None |
| Model loading | From `AZUREML_MODEL_DIR` at runtime | Bundled into image |
| Use case | Azure ML ecosystem integration | Standalone REST API |

---

## Load Testing

**File:** `Testing_Security/load_test.py`

Locust-based load test that simulates concurrent users hitting the FastAPI `/predict` endpoint to measure throughput, latency, and failure rate under load.

### How to run

```bash
locust -f Testing_Security/load_test.py --host http://weather-fastapi-aci.eastus2.azurecontainer.io
```

Open `http://localhost:8089`, set the number of users and spawn rate, then start the test.

### Baseline results (10 users, 5-minute run)

| Metric | Value |
|---|---|
| Total requests | 102 |
| Failures | 0 |
| Median latency | 200 ms |
| p95 latency | 310 ms |
| p99 latency | 530 ms |
| Max latency | 8,095 ms (cold start) |
| RPS | 0.4 |

The 8,095 ms outlier is a container cold start on the first request after a period of inactivity — expected behaviour for ACI.

---

## Production Release

**File:** `Essentials_Production_Release/create_aks_cluster.py`

Verifies that the `port-aks` production cluster exists and is in a `Succeeded` state before any production deployment. Connects directly to the Azure ML workspace — no `config.json` required.

### What it does

1. Connects to the `MLOps` workspace
2. Looks up the `port-aks` compute target
3. Waits for the cluster to reach `Succeeded` state if it is still provisioning
4. Prints a confirmation that the cluster is ready

### How to run

```bash
python Essentials_Production_Release/create_aks_cluster.py
```

Expected output:
```
MLOps
Learn_MLOps
eastus2
Found existing cluster: port-aks
Cluster status: Succeeded
Cluster ready for production deployments.
```

---

## Model Serving & Monitoring

**File:** `Model_Serving_Monitoring/inference.py`

Batch inference script that reads every row from `sample_inference_data.csv` and sends it to the live FastAPI endpoint. Used for post-deployment validation and monitoring whether the model's predictions remain consistent over time.

### What it does

1. Loads `sample_inference_data.csv` (582 rows, Port of Turku weather data)
2. Drops metadata columns (`Timestamp`, `Location`, `Future_weather_condition`)
3. Renames remaining columns to match the FastAPI `WeatherVariables` schema
4. Sends each row as a named-field POST request to `/predict`
5. Prints the prediction per row — with explicit error handling for HTTP errors, connection failures, and timeouts

### How to run

```bash
cd Model_Serving_Monitoring
python inference.py
```

Sample output:
```
Row    0 | prediction: Rain
Row    1 | prediction: Rain
...
Row  103 | prediction: Clear
Row  104 | prediction: Clear
...
Row  582 | prediction: Rain
```

### Results summary (582 rows)

| Prediction | Count (approx.) |
|---|---|
| Rain | ~490 |
| Clear | ~92 |
| Snow | 0 |

Rain dominates — consistent with the class distribution of the Port of Turku dataset.

---

## Governance & Continual Learning

**Files:** `Governance_Continual_Learning/score.py`, `deploy.py`, `conda_env.yml`

Production-grade scoring entry script deployed to `port-aks` as a separate endpoint (`weather-governance-prediction`). Extends the standard inference logic with **Application Insights telemetry** for observability and governance.

### score.py — what's different from the CI/CD version

| Feature | CI/CD `score.py` | Governance `score.py` |
|---|---|---|
| Model path resolution | `glob` | `glob` |
| Error handling | None | Structured telemetry events |
| App Insights | No | Yes — error codes per category |
| Label output | integer | String (`"Clear"`, `"Rain"`, `"Snow"`) |

### Telemetry events tracked

| Event | Error code | Trigger |
|---|---|---|
| `FileNotFoundException` | 101 | Scaler or ONNX model file not found on startup |
| `ScalingException` | 301 | `scaler.transform()` fails |
| `InferenceException` | 401 | `model.run()` fails |

All events are sent to Application Insights (instrumentation key: `29784952-4aa1-4def-9a18-a90b29b2f66a`) and can be queried in Azure Monitor.

### Deployment

```bash
cd Governance_Continual_Learning
python deploy.py
```

Deploys to `port-aks` with autoscaling (1–3 replicas) and App Insights enabled.

### Endpoint

| Property | Value |
|---|---|
| Cluster | `port-aks` |
| Service name | `weather-governance-prediction` |
| Auth | Key-based |
| Scoring URI | `http://20.7.107.211:80/api/v1/service/weather-governance-prediction/score` |
| Swagger URI | `http://20.7.107.211:80/api/v1/service/weather-governance-prediction/swagger.json` |
| App Insights | Active |

---

## CI/CD Pipeline

### Configuration Files — `CICD_Pipelines/`

| File | Purpose |
|---|---|
| `score.py` | Inference entry script used by the pipeline deployment (identical logic to `Deploy/score.py`) |
| `conda_env.yml` | Conda environment spec (Python 3.8 + inference dependencies) for the deployed container |
| `InferenceConfig.yml` | Binds `score.py` and `conda_env.yml` into an Azure ML inference config |
| `AciDeploymentConfig.yml` | ACI settings: 1 CPU, 1 GB RAM, no auth, App Insights enabled |
| `AksDeploymentConfig.yml` | AKS settings: autoscaling 1–3 replicas, key auth, App Insights enabled |
| `deploy.py` | Python script that authenticates to Azure ML and calls `Model.deploy()` for ACI or AKS based on `--target` argument |
| `aml_config/config.json` | Workspace connection details (subscription, resource group, workspace name) |

### Pipeline Definition — `azure-pipelines.yml`

Azure DevOps YAML pipeline with three sequential stages:

```
push to main (touching CICD_Pipelines/, Deploy/, or azure-pipelines.yml)
  └─► Stage 1: Validate
        └─ pyflakes lint check on score.py
              └─► Stage 2: Deploy to ACI (Staging)
                    └─ pip install azureml-core
                    └─ python deploy.py --target aci
                          └─► Stage 3: Deploy to AKS (Production)
                                └─ Requires manual approval gate
                                └─ python deploy.py --target aks
```

| Stage | Trigger | Auth method |
|---|---|---|
| Validate | Automatic on push | — |
| Deploy ACI | Automatic after Validate passes | `AzureCliAuthentication` via `mlops_sp` service connection |
| Deploy AKS | Manual approval gate → automatic | `AzureCliAuthentication` via `mlops_sp` service connection |

The `mlops_sp` service connection uses **Workload Identity Federation** (OIDC) — no client secret stored anywhere.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data processing | pandas, numpy |
| Machine learning | scikit-learn |
| Experiment tracking | Azure ML Experiments, MLflow |
| Cloud platform | Azure Machine Learning (SDK v1) |
| Model export | ONNX (skl2onnx), pickle |
| Model serving | ONNX Runtime, inference-schema |
| Staging deployment | Azure Container Instance (ACI) |
| Production deployment | Azure Kubernetes Service (AKS) |
| REST API framework | FastAPI + Uvicorn |
| API containerisation | Docker (python:3.11-slim) |
| CI/CD | Azure DevOps Pipelines (YAML) |
| Load testing | Locust |
| Monitoring | Azure Application Insights |
| Batch inference | pandas + requests |
| Visualisation | matplotlib, seaborn |
| Environment | Miniconda (`Mlflow` conda env) |

---

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1 — DATA PREPARATION                                      │
│  Dataprocessing_register.ipynb                                  │
│                                                                 │
│  Raw CSV (96,453 rows)                                          │
│    ├─ Clean nulls, parse timestamps                             │
│    ├─ Encode Weather_conditions → integer labels                │
│    ├─ Engineer Future_weather_condition (4-hour shift)          │
│    ├─ Drop redundant features (S_No, Apparent_Temperature_C)    │
│    └─ Register processed dataset → Azure ML Datastore           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2 — MODEL TRAINING                                        │
│  ML_Pipelines/ML-pipeline.ipynb                                 │
│                                                                 │
│  Azure ML Dataset (processed_weather_data_portofTurku)          │
│    ├─ Chronological 80/20 train/validation split                │
│    ├─ Feature selection (7 features) + StandardScaler           │
│    ├─ Train SVM  (GridSearchCV rbf/linear, C=1/10) → 95.19 %   │
│    ├─ Train Random Forest (n=100, depth=10)       → 95.54 %    │
│    ├─ Log metrics & parameters → Azure ML Experiments + MLflow  │
│    ├─ Export models → ONNX format (skl2onnx)                    │
│    └─ Register artefacts → Azure ML Model Registry              │
│         ├─ support-vector-classifier  (ONNX)                    │
│         ├─ random-forest-classifier   (ONNX)                    │
│         └─ scaler                     (pickle)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3 — MANUAL DEPLOYMENT (notebooks)                         │
│                                                                 │
│  Deploy/Deploy_model_ACI.ipynb  ──────────────────────────────► │
│    ├─ score.py  (ONNX runtime inference entry script)           │
│    ├─ Python 3.8 environment + dependencies                     │
│    └─ ACI endpoint  (staging, no auth)                          │
│         └─ http://<ip>.eastus2.azurecontainer.io/score          │
│                                                                 │
│  Deploy/Deploy_model_AKS.ipynb  ──────────────────────────────► │
│    ├─ Provision port-aks cluster  (Standard_D2s_v3)             │
│    ├─ score.py + Python 3.8 environment                         │
│    └─ AKS endpoint  (production, key auth, autoscaling 1–3)     │
│         └─ http://<ip>:80/api/v1/service/weather-aks-.../score  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4 — FASTAPI MICROSERVICE                                  │
│  Deploy/Deploy_FastAPI_ACI.ipynb                                │
│                                                                 │
│  API_Microservices/                                             │
│    ├─ weather_api.py  (FastAPI + ONNX runtime)                  │
│    ├─ variables.py    (Pydantic request schema)                  │
│    ├─ artifacts/      (svc.onnx + scaler.pkl bundled in image)  │
│    └─ Dockerfile      (python:3.11-slim)                        │
│                                                                 │
│  Deploy steps (no Azure CLI required):                          │
│    ├─ Get ACR credentials via azure-mgmt-containerregistry      │
│    ├─ docker build → docker push → ACR                          │
│    └─ azure-mgmt-containerinstance → create ACI container       │
│         └─ http://weather-fastapi-aci.eastus2.azurecontainer.io │
│              ├─ POST /predict  → {"prediction": "Rain"}         │
│              └─ GET  /docs     → Swagger UI                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           │  code pushed to main
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5 — AUTOMATED CI/CD                                       │
│  azure-pipelines.yml  (Azure DevOps)                            │
│                                                                 │
│  Trigger: push to main touching                                 │
│    CICD_Pipelines/**  │  Deploy/**  │  azure-pipelines.yml      │
│                                                                 │
│  Stage 1 — Validate        (automatic)                          │
│    └─ pyflakes lint → CICD_Pipelines/score.py                   │
│                                                                 │
│  Stage 2 — Deploy ACI      (automatic after Stage 1)            │
│    └─ deploy.py --target aci                                     │
│         ├─ AzureCliAuthentication  (mlops_sp OIDC)              │
│         ├─ Fetch latest registered models from Model Registry    │
│         └─ Redeploy ACI staging endpoint                        │
│                                                                 │
│  Stage 3 — Deploy AKS      (manual approval gate)               │
│    └─ deploy.py --target aks                                     │
│         ├─ AzureCliAuthentication  (mlops_sp OIDC)              │
│         ├─ Fetch latest registered models from Model Registry    │
│         └─ Redeploy AKS production endpoint                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6 — LOAD TESTING                                          │
│  Testing_Security/load_test.py                                  │
│                                                                 │
│  Locust simulates concurrent users against FastAPI /predict     │
│    ├─ Baseline: 102 requests, 0 failures                        │
│    ├─ Median: 200 ms  │  p95: 310 ms  │  p99: 530 ms           │
│    └─ Target: http://weather-fastapi-aci.eastus2.azurecontainer.io │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7 — PRODUCTION RELEASE & MONITORING                       │
│                                                                 │
│  Essentials_Production_Release/create_aks_cluster.py            │
│    └─ Verify port-aks is Succeeded → ready for production       │
│                                                                 │
│  Model_Serving_Monitoring/inference.py                          │
│    ├─ Load sample_inference_data.csv  (582 rows)                │
│    ├─ Send each row to FastAPI /predict                         │
│    └─ Log predictions for drift monitoring                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 8 — GOVERNANCE & CONTINUAL LEARNING                       │
│  Governance_Continual_Learning/                                 │
│                                                                 │
│  score.py  (production entry script with telemetry)             │
│    ├─ ONNX inference  (same as CI/CD)                           │
│    ├─ Application Insights events per error category            │
│    │    ├─ FileNotFoundException  (101)                         │
│    │    ├─ ScalingException       (301)                         │
│    │    └─ InferenceException     (401)                         │
│    └─ String output labels: "Clear" / "Rain" / "Snow"           │
│                                                                 │
│  deploy.py → port-aks                                           │
│    └─ weather-governance-prediction  (key auth, autoscaling)    │
│         └─ http://20.7.107.211:80/api/v1/service/              │
│              weather-governance-prediction/score                │
└─────────────────────────────────────────────────────────────────┘
```
