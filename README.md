# Credit Card Fraud Detection 🚨💳

A full pipeline for detecting credit card fraud using XGBoost and Graph Neural Networks (GNNs), with real-time API and dashboard.

## 📌 Project Highlights
- 📊 EDA & preprocessing on imbalanced Kaggle dataset
- ⚙️ XGBoost model with high precision & recall
- 🧠 GCN & GraphSAGE for fraud ring detection
- 🚀 FastAPI backend for live predictions
- 📈 Streamlit dashboard for user interaction

## 📁 Project Structure
- `notebooks/` – Exploratory and training notebooks
- `api/` – FastAPI endpoint for predictions
- `streamlit_app/` – Dashboard to interact with predictions
- `models/` – Trained model files
- `data/` – Sample dataset or link to download

## 🛠️ Tech Stack
`Python`, `XGBoost`, `SMOTE`, `PyTorch Geometric`, `FastAPI`, `Streamlit`, `Sklearn`, `Matplotlib`

## 🚀 Run Locally
```bash
git clone https://github.com/<your-username>/Credit_Card_Fraud_Detection.git
cd Credit_Card_Fraud_Detection
pip install -r requirements.txt
uvicorn api.fastapi_app:app --reload
streamlit run streamlit_app/dashboard.py
