# Real-Time Fraud Detection System

An end-to-end fraud detection system that scores financial transactions in real time using behavioral deep learning models. Built entirely with free, open-source tools.

> **Status:** 🚧 In progress. Steps 1 and 2 are complete. In Step 3, the XGBoost baseline catches 44% of test-set fraud with 46% precision (PR-AUC 0.44, 45x better than random).

---

## Overview

Fraud is rare, changes over time, and often only shows up as a change in an account's normal behavior. This project combines:

- **Behavioral features** built with PySpark, such as how a transaction compares to an account's usual activity
- **Supervised models** (XGBoost baseline and a PyTorch Transformer) that learn from labeled fraud
- **Anomaly scoring** to flag unusual patterns that labels may miss
- **Real-time scoring** of streamed transactions through Kafka and a FastAPI model service
- **Explainability** with SHAP, showing why each transaction was flagged

---

## Architecture

```
PaySim CSV ──► Step 1: PySpark ingestion ──► Delta Lake (transactions)
                                                    │
                                                    ▼
                                  Step 2: behavioral features ──► Delta Lake (features)
                                                    │
                                                    ▼
                                  Step 3: model training (XGBoost, Transformer,
                                  anomaly detection) tracked in MLflow
                                                    │
                                                    ▼
Kafka (transaction stream) ──► Step 4: FastAPI model service ──► Fraud scores
                                                                      │
                                                                      ▼
                                                  Step 5: dashboard + SHAP explanations
```

---

## Tech Stack ($0 budget)

| Component | Tool |
|---|---|
| Environment & dependencies | uv (Python 3.11) |
| Data processing | PySpark (runs on Java 17) |
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

PaySim simulates mobile money transactions based on one month of real financial logs from a mobile money service in an African country. Real bank data is private, so a realistic simulation is the best public option.

- 6,362,620 transactions
- 744 time steps (`step` column), each representing one hour
- Fraud occurs only in `TRANSFER` and `CASH_OUT` transactions

### Data leakage note

Transactions detected as fraud are cancelled in the simulation, so the balance columns (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`) give away the fraud label. They are **removed** in Step 1. A model trained on them would look nearly perfect in testing and fail on real data.

---

## Getting Started

### Prerequisites

- [uv](https://docs.astral.sh/uv/)
- Java 17 (required by PySpark), for example [Eclipse Temurin](https://adoptium.net/)
- A free [Kaggle](https://www.kaggle.com/) account with an API token
- **Windows only:** `winutils.exe` and `hadoop.dll` (see below)

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install Java 17

```powershell
# Windows
winget install EclipseAdoptium.Temurin.17.JDK
```

On macOS or Linux, install Temurin 17 from [adoptium.net](https://adoptium.net/) or your package manager.

**Windows: set `JAVA_HOME` and `PATH`.** Run this in PowerShell (one line), then open a new terminal:

```powershell
$jdk = (Get-ChildItem "C:\Program Files\Eclipse Adoptium" -Directory | Where-Object Name -like "jdk-17*" | Select-Object -First 1).FullName; [Environment]::SetEnvironmentVariable("JAVA_HOME", $jdk, "User"); $p = [Environment]::GetEnvironmentVariable("Path", "User"); [Environment]::SetEnvironmentVariable("Path", "$p;$jdk\bin", "User")
```

Check it works:

```bash
java -version   # should show openjdk version "17..."
```

### 3. Windows only: add the Hadoop helper files

Spark needs two small files to write data on Windows.

1. Create the folder `C:\hadoop\bin`.
2. Download `winutils.exe` and `hadoop.dll` from the newest `hadoop-3.3.x/bin` folder at [cdarlint/winutils](https://github.com/cdarlint/winutils) into that folder.

The scripts find this folder automatically. To use a different location, set the `HADOOP_HOME` environment variable.

### 4. Clone the repository and install dependencies

```bash
git clone https://github.com/tettehosabutey49/fraud-detection-project.git
cd fraud-detection-project
uv sync
```

`uv sync` creates a virtual environment and installs the exact versions in `uv.lock`, including PySpark, Delta Lake and `pandas<3` (PySpark does not fully support pandas 3 yet).

### 5. Download the data

Place your Kaggle API token at `~/.kaggle/kaggle.json`, then run:

```bash
uvx kaggle datasets download -d ealaxi/paysim1 -p data/raw --unzip
```

### 6. Run the pipeline

```bash
uv run python src/ingest.py     # Step 1: clean data -> data/delta/transactions
uv run python src/features.py   # Step 2: features   -> data/delta/features
uv run python src/train.py      # Step 3: models     -> models/, reports/, MLflow
```

To browse model runs in MLflow:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000 in your browser.

The first run takes a few extra minutes while Spark downloads the Delta Lake libraries.

### Troubleshooting

| Problem | Fix |
|---|---|
| `Java not found and JAVA_HOME environment variable is not set` | Install Java 17 and set `JAVA_HOME` (step 2), then open a new terminal |
| `'java' is not recognized` | Java's `bin` folder is not on `PATH` (step 2) |
| `winutils.exe not found` | Add the Hadoop helper files (step 3) |
| `Failed to delete ... spark-...` errors after `Done.` | Harmless. Windows locks a temp file while Spark shuts down; the data is already saved |
| `PySpark does not yet fully support pandas >=3.0.0` | Run `uv add "pandas<3"` |
| `cannot import name 'service' from 'google.protobuf'` | uv installed an old MLflow because of a pyarrow limit. Run `uv remove pyarrow mlflow`, then `uv add "mlflow>=3" pyarrow "protobuf<7"` |

---

## Step 1: Data Ingestion (complete)

`src/ingest.py` reads the raw CSV with a fixed schema, keeps only `TRANSFER` and `CASH_OUT`, removes the leaky balance columns, runs data quality checks, and saves a Delta table partitioned by transaction type.

**Results**

| Type | Transactions | Fraud | Fraud rate |
|---|---|---|---|
| TRANSFER | 532,909 | 4,097 | 0.769% |
| CASH_OUT | 2,237,500 | 4,116 | 0.184% |
| **Total** | **2,770,409** | **8,213** | **0.296%** |

All data quality checks passed: no missing values, no negative amounts, all hours between 1 and 744, and labels only 0 or 1.

**Insights**

- **Fraud is rare:** about 1 in 340 transactions. Models are judged on precision and recall, not accuracy.
- **Transfers are about 4x riskier** than cash-outs.
- **Fraud splits almost evenly** between transfers and cash-outs, pointing to a two-step pattern: move money out of a victim's account, then cash it out.
- **Senders almost never repeat** (2,768,630 unique senders), but **receivers do** (509,565 unique, about 5 transactions each). So behavior features are built around receiver accounts.

---

## Step 2: Behavioral Features (complete)

`src/features.py` adds 14 features to every transaction and saves them to `data/delta/features`. Every history feature uses **earlier hours only**, so the model never sees the future. Transactions in the same hour have no known order, so the current hour is left out.

| Feature | Meaning |
|---|---|
| `hour_of_day` | Hour of the day (0-23) |
| `is_transfer` | 1 for TRANSFER, 0 for CASH_OUT |
| `log_amount` | Amount on a log scale |
| `dest_count_{1,6,24}h` | Transactions the receiver got in the previous 1, 6 and 24 hours |
| `dest_amount_{1,6,24}h` | Money the receiver got in the previous 1, 6 and 24 hours |
| `dest_count_prior` | All earlier transactions to the receiver |
| `dest_mean_prior` | Receiver's average earlier amount (-1 if none) |
| `dest_first_seen` | 1 if the receiver has no earlier history |
| `amount_vs_dest_mean` | This amount divided by the receiver's average (-1 if none) |
| `hours_since_dest_last` | Hours since the receiver last got money (-1 if none) |

The data is then split **by time**: the oldest 70% of rows for training, the next 15% for validation and the newest 15% for testing. This mirrors real life, where a model is trained on the past and used on the future.

**Results**

All rows kept and no missing feature values.

| Split | Hours | Rows | Fraud | Fraud rate |
|---|---|---|---|---|
| Train | 1–323 | 1,951,895 | 3,643 | 0.187% |
| Valid | 324–377 | 410,437 | 560 | 0.136% |
| Test | 378–743 | 408,077 | 4,010 | 0.983% |

| Median value | Normal | Fraud |
|---|---|---|
| Receiver never seen before (% of transactions) | 19.4% | 67.3% |
| Receiver's earlier transactions | 3 | 0 |
| Amount | 171,029 | 440,963 |
| Average hour of day | 14.5 | 11.4 |

**Insights**

- **New receivers are the strongest signal.** 67.3% of fraud goes to an account with no history, versus 19.4% of normal transactions. Fraudsters mostly use fresh accounts rather than repeat "mule" accounts.
- **Fraud amounts are about 2.6x larger** at the median.
- **Night hours look extremely risky** (40–58% fraud between 2am and 5am). This is partly a simulator quirk: normal customers almost stop at night, while PaySim's fraud runs at a steady pace. Step 3 will test models with and without `hour_of_day`.
- **The test set has about 5x more fraud than training**, because normal activity is concentrated in the first ~400 hours while fraud is spread evenly. Model scores will be compared against each split's base fraud rate.
- **Recent-activity features look weak**: the median receiver count in the last 24 hours is 0 for both fraud and normal transactions.

---

## Step 3: Modeling (in progress)

`src/train.py` trains the first models and compares them fairly.

- **Two simple baselines** set the bar a model must beat: PaySim's own rule (`isFlaggedFraud`), and "flag every transfer to a brand-new receiver".
- **XGBoost, trained twice:** with all 14 features, and without `hour_of_day`, to measure how much the model relies on PaySim's night-time quirk.
- **Early stopping on validation PR-AUC** decides how many trees to use.
- **The alert threshold is chosen on the validation set**, then precision and recall are reported on the untouched test set.
- **Metrics:** PR-AUC, lift (PR-AUC divided by the base fraud rate, so splits with different fraud rates can be compared), ROC-AUC, precision and recall.
- Every run is logged to **MLflow**. Models are saved to `models/` and results to `reports/step3_results.csv`.

**Results on the test set** (hours 378–743, base fraud rate 0.98%)

| Model | Alerts | Fraud caught | Precision | PR-AUC | Lift |
|---|---|---|---|---|---|
| PaySim's own rule | 13 | 13 / 4,010 (0.3%) | 100% | 0.013 | 1.3x |
| Rule: transfer to a new receiver | 14,180 | 2,005 / 4,010 (50%) | 14.1% | 0.076 | 7.7x |
| **XGBoost, all features** | **3,862** | **1,780 / 4,010 (44%)** | **46.1%** | **0.439** | **44.6x** |
| XGBoost, without `hour_of_day` | 738 | 714 / 4,010 (18%) | 96.7% | 0.401 | 40.8x |

**Insights**

- **XGBoost catches nearly as much fraud as the best rule with about a quarter of the alerts**, so far less of a fraud team's time goes to honest customers.
- **The model relies on the Step 2 findings:** `dest_first_seen` drives 50% of its decisions and `is_transfer` 19%.
- **It is not just exploiting the night-time quirk:** removing `hour_of_day` lowers test PR-AUC only from 0.44 to 0.40.
- **The alert threshold does not transfer perfectly** from validation (0.14% fraud) to test (0.98% fraud). In production, the threshold would be set by alert capacity and reviewed as fraud rates change.
- **56% of fraud is still missed.** About a third of fraud goes to receivers with history, which the current features cannot separate from normal payments.

Next in Step 3: the Transformer sequence model and an anomaly detector. With senders appearing once and receivers about 5 times, sequences are short, so the Transformer is treated as an experiment to compare against XGBoost.

---

## Project Structure

```
fraud-detection-project/
├── data/                 # not committed (see .gitignore)
│   ├── raw/              # PaySim CSV
│   └── delta/
│       ├── transactions/ # Step 1 output, partitioned by type
│       └── features/     # Step 2 output, partitioned by split
├── notebooks/            # Exploratory analysis
├── src/
│   ├── spark_utils.py    # Shared Spark + Delta Lake setup
│   ├── ingest.py         # Step 1: load, clean, check, save
│   ├── features.py       # Step 2: behavioral features + time split
│   └── train.py          # Step 3: baselines + XGBoost, logged to MLflow
├── models/               # saved models (not committed)
├── reports/              # results tables and feature importance
├── mlflow.db             # MLflow run history (not committed)
├── LICENSE
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Roadmap

- [x] Project setup with uv
- [x] Download PaySim dataset
- [x] **Step 1:** Ingest data with PySpark, filter to `TRANSFER` and `CASH_OUT`, drop leaky columns, save to Delta Lake
- [x] **Step 2:** Build receiver behavior features with PySpark and split the data by time
- [ ] **Step 3:** Train XGBoost baseline, Transformer model, and anomaly detector; track in MLflow *(XGBoost and baselines built)*
- [ ] **Step 4:** Serve the model with FastAPI and score streamed transactions from Kafka
- [ ] **Step 5:** Build a dashboard with SHAP explanations and deploy a public demo

---

## Limitations

- PaySim is simulated. Results may not carry over fully to real bank data, where fraudsters adapt.
- The pipeline only keeps `TRANSFER` and `CASH_OUT`, so receiver history does not include other transaction types.
- The fraud rate shifts over time (0.19% in training, 0.98% in testing), so scores on the test set are not directly comparable to training.
- Some strong signals, such as night-time fraud, may be artifacts of the simulator.

---

## Citation

E. A. Lopez-Rojas, A. Elmir, and S. Axelsson. "PaySim: A financial mobile money simulator for fraud detection." *28th European Modeling and Simulation Symposium (EMSS)*, 2016.

---

## License

See the [LICENSE](LICENSE) file.

---

## Author

**Emmanuel Osabutey**
[LinkedIn](#) · [GitHub](https://github.com/tettehosabutey49)