# Customer Lifetime Value Prediction & Customer Segmentation

End-to-end customer analytics project built on the **Online Retail (UCI)** dataset (541,909 transactions).  
The project focuses on **predicting Customer Lifetime Value (CLV)** and **segmenting customers** using behavioral features.

## Project Highlights
- ✅ Feature engineering with **RFM** + product & seasonality signals
- ✅ Supervised learning for **CLV regression** (XGBoost selected for deployment)
- ✅ Unsupervised learning for **customer segmentation** (KMeans/GMM + PCA)
- ✅ Streamlit app for real-time CLV prediction + interpretability (feature importance)

## Dataset
- **Online Retail Dataset (UCI ML Repository)**
- Period: 2010–2011  
- Note: Dataset is not included in this repository. Download from the official source.

## Pipeline Overview

### 1) Preprocessing & Feature Engineering
- Removed cancelled invoices and invalid quantities
- Built customer-level features:
  - **Recency** (days since last purchase)
  - Product behavior: unique products, total items
  - Pricing behavior: avg/max unit price
  - Most common country & season (one-hot)

### 2) Supervised ML (Regression)
Trained multiple models and compared using regression metrics (RMSE/MAE/R²).  
**XGBoost** performed best and is used in deployment.

### 3) Unsupervised Learning (Segmentation)
- **KMeans (k=5)** for behavioral segments
- **Gaussian Mixture Model (GMM)** for soft clustering
- **PCA** for visualization and cluster interpretation

### 4) Deep Learning (Experiments)
Notebook includes deep learning experiments (DNN/Autoencoder) for CLV modeling and representation learning.

## Streamlit App
The Streamlit app loads the trained artifacts from `/models` and predicts **log10(CLV)**, then converts back to CLV:
\`\`\`
CLV = 10 ** prediction
\`\`\`

### Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
