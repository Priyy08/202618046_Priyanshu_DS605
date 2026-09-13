# DS605 Lab 4 — Data Analysis & Streamlit Dashboard

## 🚀 Deployed Streamlit App

**Live App:**  
https://priyy08-202618046-priyanshu-ds605-202618046-lab-4app-1scpfr.streamlit.app/

---

## 🖥️ Screenshots of Deployed App

### Screenshot 1

<p align="center">
  <img src="https://github.com/user-attachments/assets/c49d692f-a2ab-465f-9b41-c1caa536e684" width="95%">
</p>

### Screenshot 2

<p align="center">
  <img src="https://github.com/user-attachments/assets/16686c5c-cfba-4b12-94bf-d9845716636b" width="95%">
</p>

### Screenshot 3

<p align="center">
  <img src="https://github.com/user-attachments/assets/1480bdb5-9103-4b8d-bef4-64ee34c493c4" width="95%">
</p>

---

## 📊 Important Plots

### Plot 1

<p align="center">
  <img src="https://github.com/user-attachments/assets/e3b5c7ff-36dd-477f-bcf5-d2ef84531ed0" width="85%">
</p>

### Plot 2

<p align="center">
  <img src="https://github.com/user-attachments/assets/bf799d35-d7f6-47fd-83eb-5271cb921b92" width="85%">
</p>

### Plot 3

<p align="center">
  <img src="https://github.com/user-attachments/assets/44eba71b-5f1b-41ea-a08c-d7f78b32a115" width="85%">
</p>

### Plot 4

<p align="center">
  <img src="https://github.com/user-attachments/assets/13029fa5-4309-45d8-adc4-e019b0dbf5fe" width="85%">
</p>

### Plot 5

<p align="center">
  <img src="https://github.com/user-attachments/assets/469f3b05-cff1-42fb-9f70-63b141363220" width="85%">
</p>


## 📈 Final HistGradient Boosting Model Results

### Model Performance

| Metric | Final Result |
|---|---:|
| **MAE** | **$45.38** |
| **RMSE** | **$88.03** |
| **R² Score** | **0.4446** |

---

## 🔍 Overfitting Analysis

| Metric | Score |
|---|---:|
| **Train R²** | **0.6864** |
| **Test R²** | **0.6518** |
| **R² Gap** | **0.0346** |

The small **R² gap of 0.0346** between training and testing performance indicates that the final HistGradient Boosting model does not show significant overfitting and generalizes reasonably well to unseen data.

---

## ⚙️ Tuned HistGradient Boosting Performance

### Training Performance

| Metric | Score |
|---|---:|
| **MAE** | **0.2750** |
| **RMSE** | **0.3698** |
| **R² Score** | **0.6864** |

### Testing Performance

| Metric | Score |
|---|---:|
| **MAE** | **0.2939** |
| **RMSE** | **0.3939** |
| **R² Score** | **0.6518** |

### 📊 Final Model Summary

The tuned HistGradient Boosting model achieved an **R² score of 0.6518 on the test set**, explaining approximately **65.18% of the variance** in the target variable. The relatively small difference between training R² (**0.6864**) and testing R² (**0.6518**) suggests good generalization with limited overfitting.
