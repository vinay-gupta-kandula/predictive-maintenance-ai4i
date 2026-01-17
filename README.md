# AI-Driven Predictive Maintenance System (AI4I 2020)

## 📌 Project Overview

This project develops an **end-to-end AI-driven predictive maintenance system** using the **AI4I 2020 industrial dataset**.
The objective is to **predict machine failure risk**, interpret the drivers of failure using explainable AI, and quantify the **business impact** of deploying such a system in an industrial setting.

The solution includes:

* Rigorous exploratory data analysis (EDA)
* Methodologically correct feature engineering
* Robust model training and validation for imbalanced data
* SHAP-based explainability
* Cost–benefit analysis tied to real business decisions
* An interactive Streamlit dashboard for decision-makers

This repository is **fully reproducible** and structured to reflect **industry-grade data science workflows**.

---

## 🎯 Project Objectives

1. **Predict machine failure risk** using sensor and operational data
2. Handle **severely imbalanced classification** correctly
3. Prevent **data leakage** throughout the pipeline
4. Provide **interpretable insights** using SHAP
5. Quantify **financial value** via cost–benefit analysis
6. Deliver results through an **interactive dashboard**

---

## 📂 Repository Structure

```
predictive-maintenance-ai4i/
│
├── dashboard/
│   └── app.py                     # Streamlit interactive dashboard
│
├── reports/
│   ├──executive_summary.pdf
│   ├── video_demonstration.txt  
│   ├── correlation_heatmap.png
│   └── top_at_risk_machines.png
│
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv            # Original dataset
│   └── processed/
│       └── ai4i2020_features.csv  # Engineered features (generated)
│
├── models/
│   ├── best_model.joblib           # Trained XGBoost model
│   ├── feature_list.json           # Final model feature list
│   ├── imputer.joblib              # Median imputer
│   ├── threshold.txt               # Optimized decision threshold
│   └── test_idx.npy                # Validation indices
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   ├── 04_SHAP_Insights.ipynb
│   └── 05_Cost_Benefit_Analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── utils.py
│
├── requirements.txt
├── .gitignore
└── README.md

```

---

## 🧪 Dataset Description

* **Dataset**: AI4I 2020 Predictive Maintenance Dataset
* **Samples**: 10,000 machines
* **Target**: `Machine_failure` (binary)
* **Class imbalance**: ~3.4% failures

### Important Design Decision

Each machine has **one independent observation** (no cycles or time history).
Therefore:

* **No artificial lag or rolling features are created**
* The problem is framed as **static failure risk prediction**
* This avoids fabricated temporal signals and ensures methodological correctness

---

## 🔍 Notebooks Overview

### **01_EDA.ipynb — Exploratory Data Analysis**

* Data structure and distributions
* Missing value verification
* Outlier analysis
* Class imbalance analysis
* Correlation and sensor behavior visualization

**Key finding**:
Raw sensor values alone are not sufficient → advanced feature engineering required.

---

### **02_Feature_Engineering.ipynb**

* Physics-based interaction features
* Load, wear, stress, and thermal proxies
* Nonlinear transformations (log, squared terms)
* Categorical binning and risk indicators
* Explicit leakage prevention
* Identifier columns (`UDI`, `Product_ID`) retained **only for dashboard use**

**Final model feature count**: **19**

---

### **03_Model_Training.ipynb**

* Models evaluated:

  * Random Forest
  * XGBoost
* Validation strategy:

  * Time-aware split for consistency
* Imbalance handling:

  * Class weighting
  * SMOTE (train-only)
* Metrics reported:

  * F1-score
  * Precision-Recall AUC
  * ROC-AUC
  * MCC
* **Threshold optimization** for F1

**Final model**: **XGBoost**

**Final performance (validation set)**:

* Optimized F1-score: **0.83**
* Precision-Recall AUC: **~0.78**

---

### **04_SHAP_Insights.ipynb**

* SHAP explainability for failure class only
* Global importance (beeswarm & bar plots)
* Local explanations (waterfall plots)

**Key drivers of failure**:

* Load and energy features
* Torque–wear interactions
* Thermal stress indicators
* Product type differences

---

### **05_Cost_Benefit_Analysis.ipynb**

* Business-aligned cost assumptions:

  * False Positive (preventive maintenance): **$500**
  * False Negative (unplanned breakdown): **$50,000**
* Evaluation on **hold-out validation set only**
* Uses optimized decision threshold

**Results**:

* Cost without model: **$1,600,000**
* Cost with model: **$450,000**
* **Net savings: $1,150,000**
* **Cost reduction: ~72%**

---

## 📊 Interactive Dashboard

The Streamlit dashboard enables:

* Machine-level failure probability inspection
* Risk categorization (Low / Medium / High)
* Business recommendations
* Expected cost calculation
* Probability smoothing to avoid unrealistic 0% / 100% certainty

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

## ⚙️ Environment Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/vinay-gupta-kandula/predictive-maintenance-ai4i
cd predictive-maintenance-ai4i
```

### 2️⃣ Create Virtual Environment


```bash
python -m venv .venv
```

**Activate the virtual environment**

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate
```

**Windows (CMD)**

```cmd
.venv\Scripts\activate.bat
```

**Linux / macOS**

```bash
source .venv/bin/activate
```


### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Reproduce the Analysis

Run notebooks **in order**:

1. `01_EDA.ipynb`
2. `02_Feature_Engineering.ipynb`
3. `03_Model_Training.ipynb`
4. `04_SHAP_Insights.ipynb`
5. `05_Cost_Benefit_Analysis.ipynb`

All models and artifacts will be regenerated automatically.

---

## 📈 Key Results Summary

| Aspect          | Result             |
| --------------- | ------------------ |
| Final Model     | XGBoost            |
| Optimized F1    | **0.83**           |
| PR-AUC          | **~0.78**          |
| Missed Failures | 9 (validation set) |
| Net Savings     | **$1.15M**         |
| Cost Reduction  | **~72%**           |

---

# Reports

This folder contains final project deliverables.

- **executive_summary.pdf**: Business and management-level summary
- **correlation_heatmap.png**: Feature relationship analysis
- **top_at_risk_machines.png**: Dashboard output highlighting high-risk machines
- **video_demonstration.txt**: Link to 3–5 minute project walkthrough video

## Executive Report
- `executive_summary.pdf`

## Video Demonstration (3–5 minutes)
The video link is provided in:
- `video_demonstration.txt`

The video covers:
- End-to-end methodology walkthrough
- Interactive dashboard demonstration
- Key findings and business recommendations


---

## 🧠 Key Strengths of This Project

✔ Methodologically correct (no fabricated time features)
✔ Strong imbalance handling
✔ Clear leakage prevention
✔ Business-aligned evaluation
✔ Explainable AI integration
✔ Production-ready dashboard

---

## 👤 Author

**Vinay Gupta Kandula**
AI-Driven Predictive Maintenance Project




