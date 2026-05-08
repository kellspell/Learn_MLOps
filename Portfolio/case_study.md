# Case Study — Weather Forecast MLOps Pipeline on Azure

## Overview

**Goal:** Build a production-ready MLOps system that takes a weather classification problem from raw data all the way to a live, monitored, automatically redeployed API.

**Dataset:** 96,453 hourly weather readings from the Port of Turku, Finland (temperature, humidity, wind speed, wind bearing, visibility, pressure, weather condition).

**Prediction task:** Classify weather conditions 4 hours ahead — Clear, Rain, or Snow.

---

## Problem Decomposition

The project was broken into 8 sequential stages, each building on the previous:

1. Data processing and Azure ML registration
2. Model training, experiment tracking, and ONNX export
3. Manual deployment to ACI (staging) and AKS (production)
4. FastAPI microservice as the production-facing REST API
5. Automated CI/CD pipeline on Azure DevOps
6. Load testing
7. Production cluster verification and batch inference monitoring
8. Governance scoring with Application Insights telemetry

---

## Stage 1 — Data Processing

**Challenge:** Raw data had NaN rows in the target column, a highly correlated redundant feature (`Apparent_Temperature_C`, 0.99 correlation with `Temperature_C`), and required a custom feature engineering step to create the prediction target.

**Solution:**
- Forward-fill NaN weather condition rows
- Drop `Apparent_Temperature_C` and the row index column
- Label-encode `Weather_conditions` (clear=0, rain=1, snow=2)
- Engineer `Future_weather_condition` by shifting the encoded label forward 4 rows (4 hours)
- Register the processed dataset as a versioned Tabular Dataset in Azure ML Datastore

**Class distribution:** Rain 85%, Snow 11%, Clear 4% — heavily imbalanced toward rain, which matched the Port of Turku climate.

---

## Stage 2 — Model Training

**Approach:** Trained two models to compare performance. Used a chronological 80/20 train/validation split to avoid data leakage from the time series nature of the data.

| Model | Configuration | Accuracy |
|---|---|---|
| Support Vector Machine | GridSearchCV: kernel=rbf/linear, C=1/10 → best: rbf, C=1 | 95.19% |
| Random Forest | n_estimators=100, max_depth=10 | 95.54% |

**Experiment tracking:** Both models were tracked simultaneously in Azure ML Experiments and MLflow (tracking URI set to the Azure ML workspace endpoint), logging parameters, metrics, dataset name/version, and git commit SHA.

**Model packaging:** Both models exported to ONNX using `skl2onnx` for cross-platform inference without a scikit-learn dependency at serving time. The fitted StandardScaler serialised with joblib. All three artefacts registered in the Azure ML Model Registry.

**Decision:** The SVM was chosen as the primary serving model despite slightly lower accuracy because it was trained and optimised with GridSearchCV — making it more carefully tuned. The Random Forest was registered as an alternative.

---

## Stage 3 — Deployment

**ACI (staging):** Lightweight Azure Container Instance for integration testing. No authentication, 1 CPU / 1GB RAM, App Insights enabled.

**AKS (production):** Azure Kubernetes Service cluster (`port-aks`, Standard_D2s_v3). Key-based authentication, autoscaling 1–3 replicas at 70% target utilisation.

**Inference script:** ONNX Runtime `InferenceSession` for inference, joblib for scaler loading, `glob` for model path resolution (resilient to version number changes in `AZUREML_MODEL_DIR`), `inference-schema` decorators for automatic Swagger generation.

---

## Stage 4 — FastAPI Microservice

**Why FastAPI instead of just using Azure ML endpoints?**

Azure ML scoring endpoints accept a `{"data": [[...]]}` format — a raw 2D array. This is fine for internal pipeline use but not suitable for a production web or mobile app that expects a clean, documented API with named fields and human-readable responses.

The FastAPI microservice provides:
- Named-field request body (`temp_c`, `humidity`, `wind_speed_kmph`, etc.)
- String label response (`{"prediction": "Rain"}` vs `[1]`)
- Automatic Swagger UI at `/docs`
- No Azure ML SDK dependency at runtime — just ONNX Runtime and joblib

**Deployment challenge:** The original deployment notebook used `az acr build` and `az container create` (Azure CLI). Azure CLI was not installed on the development machine.

**Solution:** Rewrote the deployment using pure Python SDK:
1. `azure-mgmt-containerregistry` to retrieve ACR credentials
2. Docker build/push via `subprocess`
3. `azure-mgmt-containerinstance` to create the ACI container group

The key engineering problem was authenticating the `azure-mgmt-*` SDKs without Azure CLI. These SDKs require an `azure.core.credentials.TokenCredential`. The solution was an `AzureMLCredential` wrapper that extracts the bearer token from the azureml workspace's own auth object (`ws._auth_object.get_authentication_header()`) and wraps it as an `AccessToken` — bridging the azureml SDK auth cache to the management SDKs.

```python
class AzureMLCredential:
    def get_token(self, *scopes, **kwargs):
        token = self._auth.get_authentication_header()['Authorization'].replace('Bearer ', '')
        return AccessToken(token, int(time.time()) + 3600)
```

---

## Stage 5 — CI/CD Pipeline

**Pipeline:** 3-stage Azure DevOps YAML pipeline.

| Stage | Trigger | What it does |
|---|---|---|
| Validate | Automatic on push | pyflakes lint check on score.py |
| Deploy ACI | Automatic after Validate | Redeploy staging ACI endpoint |
| Deploy AKS | Manual approval gate | Redeploy production AKS endpoint |

**Authentication challenge:** The service connection used Workload Identity Federation (OIDC) — not a service principal secret. This meant `servicePrincipalKey` was not available in the pipeline environment. The fix was `AzureCliAuthentication()` from `azureml-core`, which reads the Azure CLI session injected by the `AzureCLI@2` task.

**Secondary challenge:** `azure-cli-ml` v1 extension was incompatible with Azure CLI 2.85.0 (deprecated). Solution: replaced all CLI-based deployment commands with a `deploy.py` Python script using `azureml-core` SDK directly.

**Third challenge:** `AzureCliAuthentication` requires `azure.cli.core` to be importable in the same Python environment. The Azure DevOps hosted agent's system `az` CLI is separate from the conda Python environment. Fixed by `pip install azure-cli-core` — a lightweight package that provides the `azure.cli.core` module.

---

## Stage 6 — Load Testing

**Tool:** Locust

**Results (10 users, ~5 minutes):**

| Metric | Value |
|---|---|
| Requests | 102 |
| Failures | 0 |
| Median latency | 200ms |
| p95 latency | 310ms |
| p99 latency | 530ms |
| Max latency | 8,095ms |

The 8,095ms outlier was a cold start — the ACI container had been idle and required a warm-up on the first request. All subsequent requests stayed under 600ms at p99.

---

## Stage 7 — Production Release & Monitoring

**Production cluster verification** (`create_aks_cluster.py`): A simple script that connects to the Azure ML workspace and confirms `port-aks` is in `Succeeded` state before any deployment proceeds. Acts as a pre-deployment gate.

**Batch inference monitoring** (`inference.py`): Sends all 582 rows of the sample dataset to the live FastAPI endpoint, logs predictions per row with error handling. Used for post-deployment validation and detecting prediction drift over time.

**Monitoring results:** 582 rows processed, 0 errors. Prediction distribution — Rain: ~490, Clear: ~92, Snow: 0 — consistent with the dataset's class distribution.

---

## Stage 8 — Governance & Continual Learning

Deployed a separate AKS endpoint (`weather-governance-prediction`) with structured Application Insights telemetry. Each error category is tracked as a custom event with a numeric code:

| Event | Code | Meaning |
|---|---|---|
| FileNotFoundException | 101 | Model or scaler file missing on startup |
| ScalingException | 301 | StandardScaler transform failed |
| InferenceException | 401 | ONNX Runtime inference failed |

These events are queryable in Azure Monitor using KQL, enabling alerting on error rate spikes — the first step toward automated model retraining triggers.

---

## Lessons Learned

1. **Tool deprecation is a real production risk.** The Azure CLI ML extension v1 deprecation in CLI 2.85.0 broke the entire original deployment approach. Building on the Python SDK directly is more stable and testable.

2. **Authentication layers compound.** Getting three different auth systems (Azure DevOps OIDC, Azure CLI, azureml-core) to work together required understanding exactly what each layer provides and where it falls short. The `AzureMLCredential` wrapper pattern is reusable across any project using azureml auth with azure-mgmt SDKs.

3. **ONNX is worth the export step.** Decoupling the serving environment from scikit-learn simplifies the container image, reduces cold start time, and makes the model portable to non-Python runtimes.

4. **Load testing reveals cold starts, not just throughput.** The 8,095ms outlier in the Locust results wouldn't appear in unit tests or integration tests — only under real traffic patterns. For ACI, a warm-up request or minimum replica strategy is worth considering in production.

5. **FastAPI for end users, Azure ML endpoints for pipelines.** The two serving approaches complement each other. Azure ML endpoints integrate naturally with the Azure ML ecosystem (Model Registry, App Insights, autoscaling). FastAPI gives end users a clean, documented, framework-agnostic interface.

---

## Skills Demonstrated

- End-to-end ML pipeline design and implementation
- Azure Machine Learning SDK (workspace, datasets, experiments, models, deployments)
- ONNX model export and cross-platform inference
- REST API development with FastAPI and Pydantic
- Docker containerisation and Azure Container Registry
- Azure Kubernetes Service deployment and autoscaling
- Azure DevOps YAML pipelines (multi-stage, manual gates, OIDC auth)
- Load testing with Locust
- Application Insights telemetry and Azure Monitor integration
- Python SDK authentication patterns for Azure management SDKs
- MLflow experiment tracking
