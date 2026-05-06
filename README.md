# Weather AI Project — MLOps on Azure ML

End-to-end weather classification pipeline that ingests raw hourly sensor data, cleans and engineers features, trains two classifiers, and registers all artefacts to Azure Machine Learning.

---

## Project Structure

```
Weather-AI-Project/
├── Dataset/
│   ├── weather_dataset_raw.csv          # Original raw data (96,453 rows)
│   └── weather_dataset_processed.csv    # Cleaned & feature-engineered data
├── ML_Pipelines/
│   ├── Data/
│   │   ├── training_data.csv            # 77,160 rows (80 %)
│   │   └── validation_data.csv          # 19,289 rows (20 %)
│   ├── outputs/
│   │   ├── svc.onnx                     # Exported SVM model
│   │   ├── rf.onnx                      # Exported Random Forest model
│   │   └── scaler.pkl                   # Fitted StandardScaler
│   └── ML-pipeline.ipynb                # Training & registration pipeline
├── Dataprocessing_register.ipynb        # Data cleaning & Azure registration
├── mlflow.db                            # Local MLflow tracking store
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

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data processing | pandas, numpy |
| Machine learning | scikit-learn |
| Experiment tracking | Azure ML Experiments, MLflow |
| Cloud platform | Azure Machine Learning (SDK v1) |
| Model export | ONNX (skl2onnx), pickle |
| Visualisation | matplotlib, seaborn |
| Environment | Miniconda (`Mlflow` conda env) |

---

## Workflow Summary

```
Raw CSV
  └─► Dataprocessing_register.ipynb
        ├─ Clean & encode labels
        ├─ Engineer Future_weather_condition (4-hr shift)
        ├─ Drop redundant features
        ├─ Save processed CSV
        └─ Register to Azure ML Datastore
              └─► ML_Pipelines/ML-pipeline.ipynb
                    ├─ Load from Azure ML Dataset
                    ├─ Chronological train/validation split → register both
                    ├─ Feature selection + StandardScaler
                    ├─ Train SVM (GridSearchCV) → log to AzureML + MLflow
                    ├─ Train Random Forest → log to AzureML + MLflow
                    ├─ Export both models to ONNX
                    ├─ Save scaler as pickle
                    └─ Register all artefacts to Azure ML Model Registry
```
