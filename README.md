# ⚡ Electricity Load Forecasting using MLOps & DVC

An end-to-end MLOps pipeline for Electricity Load Forecasting using XGBoost, DVC, and DVCLive. The project automates data ingestion, preprocessing, feature engineering, model training, and evaluation with reproducible pipelines and experiment tracking.

## 🚀 Tech Stack

* Python
* Pandas
* Scikit-learn
* XGBoost
* DVC
* DVCLive
* Git & GitHub
* Yaml

## 📂 Pipeline Stages

1. Data Ingestion
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Model Evaluation

## 📊 Features

* Hour and day lag feature engineering
* Experiment tracking with DVCLive
* Reproducible ML pipelines using DVC
* Automated workflow with `dvc repro`

## ▶️ Run Pipeline

```bash
dvc repro
```

## 🧪 Run Experiments

```bash
dvc exp run
dvc exp show
```

## 📁 Project Structure

```bash
data/
models/
reports/
mycode/src/
dvc.yaml
params.yaml
```

## 📈 Evaluation Metrics

* MAE
* MSE
* R2 Score

## 👨‍💻 Author

Harsh Kumar
