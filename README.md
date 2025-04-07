
---

## 📊 Dataset Overview

- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Transactions: 284,807
- Fraud Cases: 492 (~0.17%)
- Features: 30 anonymized + Time + Amount + Class

---

## 🔍 Phase 1: Data Preprocessing & EDA

- Normalized `Time` and `Amount` using `StandardScaler`
- Visualized class distribution, PCA projection, scatter plots, and heatmaps
- Handled class imbalance using **SMOTE** and **undersampling**

---

## ⚙️ Phase 2: Baseline Model with XGBoost

XGBoost was selected for its performance on imbalanced, tabular data.

**Metrics (Fraud Class):**
- Precision: 91.86%
- Recall: 80.61%
- F1-Score: 85.87%
- AUC-ROC: 0.9699

📌 XGBoost provided strong baseline performance with a good balance of precision and recall.

---

## 🧠 Phase 3: Advanced Detection with Graph Neural Networks (GNNs)

### Why GNN?
Traditional models treat transactions in isolation. GNNs allow the model to learn from **relationships** between users, cards, and merchants — critical for detecting **fraud rings** and **collaborative fraud**.

### Graph Construction:
- Nodes: Users, Cards, Merchants, Transactions
- Edges: Entity interactions (transactions, shared cards)

### GNN Implementation:
- Used `PyTorch Geometric` and `NetworkX`
- Built a **GraphSAGE** model with **class weighting**
- Trained to detect relational and hidden fraud patterns

📊 **GraphSAGE Results:**

| Class        | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Non-Fraud (0)| 1.000     | 0.9988 | 0.9994   |
| Fraud (1)    | 0.760     | 1.0000 | 0.8636   |

- ✅ **Recall (Fraud)**: 1.000 → All frauds detected
- ✅ **Weighted Avg F1**: 0.9989
- ✅ **Overall Accuracy**: 99.88%

🧠 GraphSAGE GNN detected complex patterns that boosted recall without overfitting.

---

## 🚀 Phase 4: Real-Time Deployment

### 🔧 FastAPI Backend
- Created a RESTful API for serving model predictions
- Accepts transaction data and returns prediction + probability
- Lightweight and scalable

### 📊 Streamlit Dashboard
- Upload or input new transaction data
- Get real-time fraud predictions
- Visualize confusion matrix, ROC curve, and model performance
- Built for analysts, testers, and stakeholders

---

## 🛠️ Tech Stack

- **Languages**: Python
- **ML Models**: XGBoost, GraphSAGE (GNN)
- **Libraries**: Pandas, Scikit-learn, Imbalanced-learn, PyTorch Geometric, NetworkX
- **Deployment**: FastAPI, Uvicorn
- **Visualization**: Streamlit, Matplotlib, Seaborn

---

## 📈 Model Comparison

| Model               | Precision | Recall | F1-Score |
|---------------------|-----------|--------|----------|
| Logistic Regression | 82.9%     | 64.3%  | 72.4%    |
| Decision Tree       | 74.0%     | 75.5%  | 74.8%    |
| Random Forest       | 95.2%     | 81.6%  | 87.9%    |
| SVM                 | 94.4%     | 69.4%  | 80.0%    |
| **XGBoost**         | **91.9%** | 80.6%  | 85.9%    |
| **GraphSAGE GNN**   | 76.0%     | **100%**| **86.4%** |

---

## 💻 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/kamanaHarikrishna/Credit_Card_Fraud_Detection.git
cd Credit_Card_Fraud_Detection

# Install dependencies
pip install -r requirements.txt

# Start FastAPI
uvicorn api.fastapi_app:app --reload

# Run Streamlit dashboard
streamlit run streamlit_app/dashboard.py
