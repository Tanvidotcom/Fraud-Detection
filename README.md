# 🔍 Fraud Detection System with Explainable AI

## Project Overview
An end-to-end machine learning system to detect credit card fraud in real time.
Built using LightGBM, SHAP explainability, and a live Streamlit dashboard.


## 🚀 Live Dashboard
🔗 (https://fraud-detection-tanvi-zcbojz5ybh9rappywedadwk.streamlit.app/) 

## Project Structure
```
Fraud-Detection/
├── data/
│   ├── train_transaction.csv
│   └── train_identity.csv
├── dashboard/
│   ├── app.py
│   ├── model.pkl
│   ├── scaler.pkl
│   └── risk_score.csv
├── charts/
│   ├── class_imbalance.png
│   ├── shap_summary.png
│   ├── confusion_matrices.png
│   ├── roc_pr_curves.png
│   ├── threshold_f1.png
│   ├── risk_tiers.png
│   └── fraud_by_hour.png
├── analysis.ipynb
├── requirements.txt
├── summary.md
└── README.md
```

## ⚙️ Setup Instructions
1. Clone this repository
2. Create a virtual environment:
   python -m venv venv
   venv\Scripts\Activate.ps1
4. Install dependencies
   pip install -r requirements.txt

## 📓 How to Run Notebook
jupyter notebook
Open `analysis.ipynb` and run all cells top to bottom.

## 📊 How to Run Dashboard Locally
streamlit run dashboard/app.py

## 🤖 Models Used
| Model | Type |
|---|---|
| LightGBM | Best model — highest PR-AUC |
| XGBoost | Comparison model |
| Isolation Forest | Unsupervised baseline |

## 📈 Key Results
- **Best Model:** LightGBM
- **Metric Used:** PR-AUC (more reliable than accuracy for imbalanced data)
- **Class Imbalance:** Handled using SMOTE
- **Fraud Rate:** 3.5% of all transactions
- **Risk Tiers:** Critical 🔴 | Suspicious 🟡 | Clear 🟢

## 🔍 Explainability
SHAP values used to explain every prediction:
- Global feature importance plot
- Waterfall plots for fraud, borderline and legitimate transactions
- Plain English explanation for each case

## 💡 Key Fraud Signals (from SHAP)
1. Unusually high transaction amount
2. Transactions between 11pm and 4am
3. Unfamiliar device or browser

## 🛡️ Fraud Prevention Policies
1. Auto-block transactions above 3x cardholder average — require OTP
2. Step-up authentication for all transactions between 11pm and 5am

## 💰 Estimated Annual Savings
Assuming 80% detection rate, 100,000 fraud cases/year, avg fraud $150:
**Savings = 0.80 × 100,000 × $150 = $12,000,000 annually**

## ⚠️ Model Limitations
- Trained on a sample due to memory constraints
- New fraud patterns not in training data may be missed
- SMOTE generates synthetic samples not real fraud cases

## 🛠️ Tools & Libraries
| Tool | Purpose |
|---|---|
| Python | Main language |
| LightGBM | Primary classifier |
| XGBoost | Comparison model |
| SHAP | Explainable AI |
| SMOTE | Class imbalance handling |
| Streamlit | Live dashboard |
| Plotly | Interactive charts |
| Pandas / NumPy | Data processing |
