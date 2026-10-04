# 🦟 An Explainable AI (XAI) Enabled Hybrid LSTM-XGBoost Framework for Dengue Outbreak Forecasting

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dengue26.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An **Early Warning Decision Support System (EWDSS)** designed for public health surveillance in Bangladesh. This framework combines deep learning temporal sequence extraction with gradient boosted decision boundaries and game-theoretic model interpretability to forecast weekly dengue fever outbreaks.

🌐 **Live Interactive System:** [https://dengue26.streamlit.app/](https://dengue26.streamlit.app/)

---

## 📌 Executive Summary & Key Highlights

* **Hybrid Architecture:** Fuses 64-dimensional latent temporal sequence embeddings extracted from a Deep **LSTM** network into an optimized **XGBoost Regressor**.
* **Biological Feature Engineering:** Integrates 1–4 week delayed biological lags encoding Aedes mosquito Extrinsic (EIP) and Intrinsic (IIP) Incubation Periods, rolling climate averages, and trigonometric cyclical encoding ($\sin/\cos$ month transformation).
* **Model Explainability (XAI):** Implements **TreeSHAP** (SHapley Additive exPlanations) for global and local epidemiological feature contribution analysis.
* **Prospective Real-World Validation:** Tested against late-2026 Director General of Health Services (DGHS) surveillance data, achieving **98.25%** and **93.82%** predictive accuracy in real-time outbreak conditions.

---

## 📊 Model Performance Metrics

Evaluated on an unseen test dataset spanning strictly chronological time-series splits (2024–2026) to guarantee zero temporal data leakage:

| Metric | Score / Value | Interpretation |
| :--- | :--- | :--- |
| **Coefficient of Determination ($R^2$)** | **0.9346** | Explains 93.46% of epidemic variance |
| **Root Mean Squared Error (RMSE)** | **666.83** Cases | Robust peak outbreak error margin |
| **Mean Absolute Error (MAE)** | **501.75** Cases | Average weekly prediction error |
| **Prospective Accuracy (Sept 2026)** | **98.25%** | 11,800 actual vs 12,006 predicted |
| **Prospective Accuracy (Oct 2026)** | **93.82%** | 12,657 actual vs 11,875 predicted |

---

## 🏗️ System Architecture & Workflow

```text
 Meteorological & Epidemiological Data (2018–2026)
                 │
                 ▼
 ⚙️ Feature Engineering (1–4 Wk Biological Lags + Cyclical Month Encoding)
                 │
                 ▼
 🔒 Strict Chronological Train/Test Split (80/20) & MinMaxScaler
                 │
                 ▼
 🧠 LSTM Network Layer ────► 64-dim Latent Sequence Embedding Extraction
                 │                                  │
                 └──────────────────┬───────────────┘
                                    ▼
                         🌲 XGBoost Regressor
                                    │
                 ┌──────────────────┴──────────────────┐
                 ▼                                     ▼
 🔮 Predictive Output & Risk Level           🔍 TreeSHAP Explainability
