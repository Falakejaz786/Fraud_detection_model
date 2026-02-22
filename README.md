# 💳 Credit Card Fraud Detection 

## 📌 Project Objective

This project aims to detect fraudulent credit card transactions using machine learning on a **highly imbalanced dataset**.

Fraud detection presents unique challenges:

* Fraud cases are extremely rare
* False negatives are very costly
* ROC-AUC can be misleading
* Interpretability is important for trust

The goal was to:

1. Build baseline models
2. Tune advanced models
3. Compare them using appropriate metrics
4. Select the best model based on fraud-focused evaluation
5. Explain predictions using SHAP

---

## 📊 Dataset

The dataset used is the **Credit Card Fraud Detection dataset** available on:

👉 Kaggle

Search for:

> Credit Card Fraud Detection

⚠️ The file `creditcard.csv` is not included in this repository due to GitHub file size limitations.
Please download it from Kaggle and place it in the project root directory.

---

# ⚙️ Project Workflow

## 1️⃣ Baseline Models

The following models were trained initially:

* Logistic Regression
* Random Forest
* XGBoost

### 📈 Baseline Results

| Model               | ROC-AUC | PR-AUC     |
| ------------------- | ------- | ---------- |
| Logistic Regression | 0.9605  | 0.7414     |
| Random Forest       | 0.9529  | **0.8542** |
| XGBoost             | 0.9832  | 0.7270     |

### 🔎 Observation

* XGBoost had the highest ROC-AUC
* Random Forest had the highest PR-AUC

Since the dataset is highly imbalanced, **PR-AUC is more meaningful than ROC-AUC**.

👉 At baseline stage, **Random Forest performed best in terms of fraud detection quality.**

---

## 2️⃣ Model Tuning

Next, advanced tuning was performed on:

* XGBoost (with `scale_pos_weight`)
* LightGBM (with `class_weight="balanced"`)

### 📈 Tuned Model Results

| Model          | ROC-AUC | PR-AUC     |
| -------------- | ------- | ---------- |
| XGBoost_tuned  | 0.9826  | 0.8079     |
| LightGBM_tuned | 0.9790  | **0.8269** |

---

# 🏆 Final Model Selection

Although XGBoost had slightly higher ROC-AUC, **LightGBM achieved higher PR-AUC (0.8269)**.

Since fraud detection prioritizes:

* Precision
* Recall
* Minority class performance

👉 **LightGBM_tuned was selected as the final model.**

---

# ❓ Why LightGBM Won

LightGBM performed better because:

* It handled class imbalance effectively using `class_weight="balanced"`
* It learned complex non-linear patterns
* It maintained better precision at higher recall levels
* It showed more stable performance in PR curves

Although XGBoost ranked samples slightly better overall (ROC), LightGBM was stronger at correctly identifying fraud cases under imbalance.

---

# 📊 Evaluation Strategy

The following were generated:

* ROC curves
* Precision-Recall curves
* Confusion matrices
* Probability distribution plots

Key insight:

> ROC curves appeared optimistic due to class imbalance. PR curves gave a more realistic view of fraud detection performance.

---

# 🔍 Model Explainability

To make the model interpretable:

* Feature importance plots were generated
* SHAP (SHapley Additive Explanations) was used
* Individual fraud cases were explained using SHAP force plots

SHAP allowed:

* Understanding which features increased fraud probability
* Local explanation of specific fraud predictions
* Transparent model behavior

---

# 💾 Model Persistence

Final model was saved using `joblib`:

```python
joblib.dump(best_model, "models/lightgbm_model.pkl")
```

This allows reuse without retraining.

---

# 🎯 Key Technical Insights

* PR-AUC is superior to ROC-AUC for imbalanced datasets
* Class weighting improves minority class detection
* LightGBM and XGBoost outperform linear models in fraud detection
* SHAP provides better interpretability than traditional feature importance
* Model selection must align with business objective, not just metric values

---

# 📁 Project Structure

```
fraud-detection/
│
├── baseline.ipynb
├── tuning.ipynb
├── final_model.ipynb
├── plots/
├── models/
├── experiment_results.csv
└── README.md
```

---

# 🚀 Future Improvements

* Threshold optimization (Precision@K or Recall@Precision)
* Stratified K-Fold Cross Validation
* Probability calibration
* Deployment using FastAPI
* Monitoring for concept drift

---

# 🧠 Final Conclusion

This project demonstrates:

* End-to-end ML workflow
* Handling extreme class imbalance
* Model comparison and tuning
* Business-focused metric selection
* Model interpretability using SHAP

The final selected model:
👉 **LightGBM_tuned** based on superior PR-AUC performance.
