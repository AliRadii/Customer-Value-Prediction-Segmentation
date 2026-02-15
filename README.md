# 💰 Customer Lifetime Value Prediction & Customer Segmentation

End-to-end Machine Learning system for predicting **Customer Lifetime Value (CLV)** and segmenting customers using 541,909 real-world e-commerce transactions from the **Online Retail (UCI)** dataset.

This project demonstrates a full ML lifecycle:

- Feature engineering from raw transactional data  
- Supervised regression modeling  
- Unsupervised customer segmentation  
- Deep learning experimentation  
- Production-style Streamlit deployment  

---

## 📊 Dataset

- **Online Retail Dataset (UCI ML Repository)**
- 541,909 transactions
- Period: 2010–2011
- UK-based e-commerce company

⚠️ The dataset is not included in this repository.  
Please download it from the official UCI repository.

---

# 🧠 Project Pipeline

---

## 1️⃣ Data Preprocessing & Feature Engineering

### Data Cleaning
- Removed cancelled invoices (InvoiceNo starting with “C”)
- Removed negative quantities
- Dropped missing CustomerID values

### Customer-Level Aggregation
Transactions were aggregated per customer to build behavioral features.

### Engineered Features
- **Recency** (days since last purchase)
- **Unique products purchased**
- **Total items purchased**
- **Average unit price**
- **Maximum unit price**
- **Most common country** (one-hot encoded)
- **Most common season** (one-hot encoded)

### Target Transformation

To reduce skewness in spending distribution:


Model predicts `log10(CLV)`, then converts back using:


---

# 📈 2️⃣ Supervised Learning – CLV Regression

Multiple regression models were trained and evaluated.

## 🔍 Model Performance Comparison

| Model              | MAE   | RMSE  | R²    |
|-------------------|-------|-------|-------|
| Linear Regression | 0.292 | 0.382 | 0.521 |
| Random Forest     | 0.086 | 0.119 | 0.953 |
| **XGBoost**       | **0.078** | **0.109** | **0.961** |
| DNN               | —     | —     | 0.843 |

### ✅ Best Model: XGBoost

- Highest R² score (0.961)
- Lowest RMSE (0.109)
- Strong generalization performance
- Selected for deployment

---

# 🔍 3️⃣ Unsupervised Learning – Customer Segmentation

Customer segmentation was performed to identify behavioral groups.

### Algorithms Used
- **KMeans (k=5)**
- **Gaussian Mixture Model (GMM)**
- **PCA** for visualization

### Identified Customer Segments
- High-value loyal customers
- Medium-value repeat buyers
- At-risk customers
- Low-frequency shoppers
- Occasional buyers

Segmentation enables targeted marketing and retention strategies.

---

# 🤖 4️⃣ Deep Learning Experiments

Deep learning models were tested to explore nonlinear representations.

### DNN Regressor
- R² = 0.843

### Autoencoder
- Test MSE: 0.0883
- Test RMSE: 0.2972

Random Forest trained on latent features:
- RMSE: 0.327
- R²: 0.649

### Conclusion
While neural networks learned meaningful representations, ensemble tree-based methods achieved superior predictive accuracy for this dataset.

---

# 🚀 Streamlit Deployment

A lightweight Streamlit application was built to:

- Accept customer behavioral inputs
- Apply preprocessing & scaling
- Predict log10(CLV)
- Convert to actual CLV
- Display customer tier classification
- Show feature importance
- Allow prediction history export

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

---

# 📈 2️⃣ Supervised Learning – CLV Regression

Multiple regression models were trained and evaluated.

## 🔍 Model Performance Comparison

| Model              | MAE   | RMSE  | R²    |
|-------------------|-------|-------|-------|
| Linear Regression | 0.292 | 0.382 | 0.521 |
| Random Forest     | 0.086 | 0.119 | 0.953 |
| **XGBoost**       | **0.078** | **0.109** | **0.961** |
| DNN               | —     | —     | 0.843 |

### ✅ Best Model: XGBoost

- Highest R² score (0.961)
- Lowest RMSE (0.109)
- Strong generalization performance
- Selected for deployment

---

# 🔍 3️⃣ Unsupervised Learning – Customer Segmentation

Customer segmentation was performed to identify behavioral groups.

### Algorithms Used
- **KMeans (k=5)**
- **Gaussian Mixture Model (GMM)**
- **PCA** for visualization

### Identified Customer Segments
- High-value loyal customers
- Medium-value repeat buyers
- At-risk customers
- Low-frequency shoppers
- Occasional buyers

Segmentation enables targeted marketing and retention strategies.

---

# 🤖 4️⃣ Deep Learning Experiments

Deep learning models were tested to explore nonlinear representations.

### DNN Regressor
- R² = 0.843

### Autoencoder
- Test MSE: 0.0883
- Test RMSE: 0.2972

Random Forest trained on latent features:
- RMSE: 0.327
- R²: 0.649

### Conclusion
While neural networks learned meaningful representations, ensemble tree-based methods achieved superior predictive accuracy for this dataset.

---

# 🚀 Streamlit Deployment

A lightweight Streamlit application was built to:

- Accept customer behavioral inputs
- Apply preprocessing & scaling
- Predict log10(CLV)
- Convert to actual CLV
- Display customer tier classification
- Show feature importance
- Allow prediction history export

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
