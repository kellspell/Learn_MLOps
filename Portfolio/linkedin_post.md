Just completed a full end-to-end MLOps project on Microsoft Azure — from raw sensor data to a live production API with automated CI/CD, load testing, and governance monitoring.

Here's what I built:

**The problem:** Predict weather conditions 4 hours ahead using hourly sensor data from the Port of Turku, Finland (96,453 readings).

**The models:** Trained a Support Vector Machine (95.19% accuracy) and a Random Forest (95.54% accuracy) using scikit-learn, exported both to ONNX format for cross-platform inference, and registered them in the Azure ML Model Registry.

**The deployment stack:**
- Staging endpoint on Azure Container Instance (ACI)
- Production endpoint on Azure Kubernetes Service (AKS) with autoscaling (1–3 replicas)
- Standalone FastAPI microservice containerised with Docker, deployed to ACI — this is the production-facing REST API for end users

**The automation:**
Built a 3-stage Azure DevOps CI/CD pipeline (Validate → Deploy ACI → Deploy AKS) that redeploys both endpoints automatically on every push to main, using Workload Identity Federation — no secrets stored anywhere.

**The observability:**
- Load tested the FastAPI endpoint with Locust: 102 requests, 0 failures, 200ms median latency
- Built a batch inference monitoring script that validates predictions across the full dataset
- Deployed a governance scoring endpoint on AKS with Application Insights telemetry — tracking errors by category in Azure Monitor

**The key challenge:** The Azure CLI ML extension (v1) was deprecated in Azure CLI 2.85+, so I built a pure Python SDK deployment approach using a custom AzureMLCredential wrapper to bridge the azureml workspace token to the azure-mgmt-* SDKs — no CLI required.

This project taught me how to think beyond model accuracy and focus on the full production lifecycle: reproducibility, automation, observability, and governance.

Tech: Python · scikit-learn · ONNX · Azure ML · FastAPI · Docker · Azure DevOps · MLflow · Locust · Application Insights

#MLOps #MachineLearning #Azure #Python #FastAPI #DevOps #DataScience #CloudComputing
