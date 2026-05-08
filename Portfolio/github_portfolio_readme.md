# Weather Forecast MLOps Pipeline — Azure ML

End-to-end MLOps project covering the full machine learning lifecycle from raw data to production deployment, automated CI/CD, load testing, and governance monitoring on Microsoft Azure.

---

## Overview

Trained two weather classification models (SVM and Random Forest) on 96,453 hourly sensor readings from the Port of Turku, Finland, predicting weather conditions 4 hours ahead. Both models were exported to ONNX, registered in Azure ML Model Registry, and deployed to production via a fully automated Azure DevOps pipeline.

A standalone FastAPI microservice was also built, containerised with Docker, and deployed to Azure Container Instance — serving as the production-facing REST API for end users.

---

## Architecture

```
Raw Data (96k rows)
    │
    ▼
Data Processing & Feature Engineering
    │  Label encoding, 4-hour target shift, correlation analysis
    ▼
Model Training (Azure ML + MLflow)
    │  SVM → 95.19%  │  Random Forest → 95.54%
    │  ONNX export → Azure ML Model Registry
    ▼
Deployment
    ├── ACI (staging)     → Azure ML scoring endpoint
    ├── AKS (production)  → Azure ML scoring endpoint (autoscaling)
    └── FastAPI on ACI    → Production REST API (Docker)
    ▼
CI/CD (Azure DevOps)
    │  Validate → Deploy ACI → Deploy AKS (manual gate)
    ▼
Load Testing (Locust) → 0% failure rate, 200ms median latency
    ▼
Governance (Application Insights telemetry on AKS)
```

---

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Support Vector Machine | 95.19% | 88.70% | 88.59% | 88.64% |
| Random Forest | 95.54% | 90.21% | 88.28% | 89.21% |

**Load test (FastAPI endpoint):** 102 requests · 0 failures · 200ms median · 310ms p95

---

## Tech Stack

| Category | Tools |
|---|---|
| ML | scikit-learn, ONNX, skl2onnx, MLflow |
| Cloud | Azure Machine Learning (SDK v1), Azure Container Instance, Azure Kubernetes Service |
| API | FastAPI, Uvicorn, Pydantic, Docker |
| CI/CD | Azure DevOps Pipelines (YAML, 3-stage) |
| Monitoring | Azure Application Insights, Locust |
| Languages | Python 3.11 |

---

## Project Structure

| Folder | Purpose |
|---|---|
| `Model/` | Data processing & Azure ML registration notebook |
| `ML_Pipelines/` | Training pipeline, ONNX export, model registration |
| `Deploy/` | ACI, AKS, MLflow, and FastAPI deployment notebooks |
| `API_Microservices/` | FastAPI app + Dockerfile |
| `CICD_Pipelines/` | Scoring script, conda env, Azure DevOps deploy script |
| `Testing_Security/` | Locust load test |
| `Essentials_Production_Release/` | Production cluster verification |
| `Model_Serving_Monitoring/` | Batch inference monitoring script |
| `Governance_Continual_Learning/` | AKS scoring with App Insights telemetry |

---

## Live Endpoints

| Endpoint | URL |
|---|---|
| FastAPI (predict) | `http://weather-fastapi-aci.eastus2.azurecontainer.io/predict` |
| FastAPI (docs) | `http://weather-fastapi-aci.eastus2.azurecontainer.io/docs` |
| AKS governance | `http://20.7.107.211:80/api/v1/service/weather-governance-prediction/score` |

---

## Key Engineering Decisions

- **ONNX over pickle** — cross-platform inference, no scikit-learn dependency at serving time
- **FastAPI over Azure ML endpoints** — clean named-field API for frontend consumers; Azure ML endpoints used internally
- **Python SDK deployment over Azure CLI** — `azure-cli-ml` v1 extension deprecated in Azure CLI 2.85+; pure SDK approach with `AzureMLCredential` wrapper bridges azureml auth to `azure-mgmt-*` SDKs
- **Workload Identity Federation** — no client secrets stored; OIDC-based service connection for CI/CD
