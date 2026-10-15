# Real-Time Fraud Detection System

An end-to-end fraud detection system that scores financial transactions in real time using behavioral deep learning models. Built entirely with free, open-source tools.

> **Status:** 🚧 In progress. Project setup and data download are complete. Data ingestion (Step 1) is next.

---

## Overview

Fraud is rare, changes over time, and often only shows up as a change in a user's normal behavior. This project combines:

- **Behavioral features** built with PySpark, such as how a transaction compares to a user's usual activity
- **Supervised models** (XGBoost baseline and a PyTorch Transformer) that learn from labeled fraud
- **Anomaly scoring** to flag unusual patterns that labels may miss
- **Real-time scoring** of streamed transactions through Kafka and a FastAPI model service
- **Explainability** with SHAP, showing why each transaction was flagged

---

## Planned Architecture

```
PaySim CSV ──► PySpark ingestion ──► Delta Lake (raw + features)
                                          │
                                          ▼
                            Model training (XGBoost, Transformer,
                            anomaly detection) tracked in MLflow
                                          │
                                          ▼
Kafka (transaction stream) ──► FastAPI model service ──► Fraud scores
                                                            │
                                                            ▼
                                              Dashboard + SHAP explanations
```

---

## Tech Stack ($0 budget)

| Component | Tool |
|---|---|
| Environment & dependencies | uv |
| Data processing | PySpark |
| Storage | Delta Lake (local) |
| Model training | PyTorch, XGBoost, scikit-learn (Google Colab / Kaggle for free GPUs) |
| Experiment tracking | MLflow (local) |
| Streaming | Apache Kafka (Docker) |
| Model serving | FastAPI + Docker |
| Explainability | SHAP |
| Demo | Streamlit / Hugging Face Spaces |

No paid cloud services are used. The serving layer is designed so it could be moved to a managed platform such as AWS SageMaker.

---

## Dataset

**PaySim: Synthetic Financial Datasets for Fraud Detection**
https://www.kaggle.com/datasets/ealaxi/paysim1

PaySim simulates mobile money transactions based on one month of real financial logs from a mobile money service in an African country.

- About 6.3 million transactions
- 744 time steps (`step` column), each representing one hour
- Fraud occurs only in `TRANSFER` and `CASH_OUT` transactions

### Data leakage note

Transactions detected as fraud are cancelled in the simulation, so the balance columns (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`) reflect the fraud label. These columns are **excluded** from modeling to avoid leakage.

---

## Getting Started

### Prerequisites

- [uv](https://docs.astral.sh/uv/)
- Java 17 (required by PySpark), for example [Eclipse Temurin](https://adoptium.net/)
- A free [Kaggle](https://www.kaggle.com/) account with an API token

Check Java is installed:

```bash
java -version
```

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone the repository and install dependencies

```bash
git clone <your-repo-url>
cd fraud-detection
uv sync
```

`uv sync` creates a virtual environment and installs the exact versions in `uv.lock`.

### 3. Download the data

Place your Kaggle API token at `~/.kaggle/kaggle.json`, then run:

```bash
uvx kaggle datasets download -d ealaxi/paysim1 -p data/raw --unzip
```

### 4. Run code

```bash
uv run python src/ingest.py   # coming in Step 1
uv run jupyter lab            # for exploration notebooks
```

---

## Project Structure

```
fraud-detection/
├── data/
│   ├── raw/          # PaySim CSV (not committed)
│   └── delta/        # Delta Lake tables (not committed)
├── notebooks/        # Exploratory analysis
├── src/
│   └── ingest.py     # Step 1: load, filter, save to Delta Lake
├── pyproject.toml
├── uv.lock
└── README.md
```

The `data/` folder is listed in `.gitignore` to keep large files out of the repository.

---

## Roadmap

- [x] Project setup with uv
- [x] Download PaySim dataset
- [ ] **Step 1:** Ingest data with PySpark, filter to `TRANSFER` and `CASH_OUT`, drop leaky columns, save to Delta Lake
- [ ] **Step 2:** Build behavioral features with PySpark
- [ ] **Step 3:** Train XGBoost baseline, Transformer model, and anomaly detector; track in MLflow
- [ ] **Step 4:** Serve the model with FastAPI and score streamed transactions from Kafka
- [ ] **Step 5:** Build a dashboard with SHAP explanations and deploy a public demo

---

## Citation

E. A. Lopez-Rojas, A. Elmir, and S. Axelsson. "PaySim: A financial mobile money simulator for fraud detection." *28th European Modeling and Simulation Symposium (EMSS)*, 2016.

---

## Author

**Emmanuel Osabutey**
