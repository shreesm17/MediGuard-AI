#  MediGuard-AI

### Privacy-Preserving Heart Risk Prediction & Medical Loan Approval System

An intelligent, full-stack AI application that combines **healthcare risk prediction**, **Explainable AI (XAI)**, and **privacy-preserving financial decision-making** to enable **accessible and transparent medical support**.

---

## 🚀 What This Project Does

MediGuard AI allows users to:

* Input clinical health data
* Get a **heart disease risk score (0–100)**
* Understand **why the model made that prediction (XAI)**
* Receive **personalized medical recommendations**
* Apply for a **loan based on health risk**
* Ensure **zero exposure of raw medical data**

---

## 🧠 Core Innovations

### 1. Explainable AI (XAI)

* Displays **Top Risk Factors** influencing prediction
* Example from system:

  * Chest Pain Type → highest impact
  * Age, Cholesterol, Vessels → key contributors
* Builds **trust, transparency, and interpretability**

---

### 2. Privacy-Preserving Intelligence 🔐

* Applies **Differential Privacy** to risk scores
* Ensures:

  * Raw clinical data is **never shared**
  * Only **noise-protected risk score** is used
* Includes **Audit Log System**:

  * Tracks every step of data usage
  * Demonstrates compliance mindset

---

### 3. AI-Driven Financial Decision System 💰

* Loan approval based on:

  * Income
  * Credit score
  * Health risk score
* Outputs:

  * Loan amount
  * EMI
  * Interest rate
  * Tenure

---

## 📊 System Output (Real Example)

* ❤️ Risk Score: **32/100 (LOW)**
* 💡 Recommendation: Preventive care + lifestyle maintenance
* 💰 Estimated Cost: ₹15,000
* 🏦 Loan Approved: ₹18,000
* 📉 EMI: ₹818/month @ 8.5%

---

## 🏗️ Architecture Overview

```
User Input (Frontend)
        ↓
FastAPI Backend
        ↓
ML Model (Scikit-learn)
        ↓
Differential Privacy Layer
        ↓
Explainability Engine (Feature Importance)
        ↓
Loan Decision Engine
        ↓
UI Dashboard (Charts + Audit Logs)
```

---

## 🛠️ Tech Stack

* **Backend:** FastAPI
* **Machine Learning:** Scikit-learn
* **Frontend:** HTML, CSS, Bootstrap
* **Visualization:** Chart.js
* **Data Processing:** Pandas, NumPy

---

## 📂 Project Structure

```
MediGuard-AI/
│── app.py                # API & backend logic
│── model.py              # ML + loan logic
│── index.html            # UI dashboard
│── scaler.pkl            # Feature scaling (excluded)
│── disease_model.pkl     # Trained model (excluded)
│── README.md
```

---

## ⚙️ How to Run

```bash
pip install fastapi uvicorn pandas numpy scikit-learn
uvicorn app:app --reload
```

Open:

```
http://127.0.0.1:8000
```

---

## 🔍 Key Features Visible in UI

* Clinical data input panel
* Risk score visualization (circular gauge)
* AI explanation chart (Top Risk Factors)
* Treatment cost estimation
* Loan approval dashboard
* Privacy audit log

---

## 📌 Real-World Impact

* 🏥 Helps early detection of heart disease
* 💳 Enables **financial accessibility for treatment**
* 🔐 Promotes **ethical AI with privacy guarantees**
* 📊 Demonstrates **interpretable AI in critical domains**

---

## 🌟 Why This Project Stands Out

* Combines **Healthcare + FinTech + AI**
* Implements **Explainable AI + Differential Privacy**
* End-to-end **production-style system**
* Focuses on **real-world usability and trust**

---

## 🚀 Future Enhancements

* SHAP-based deep explainability
* Federated learning integration
* Deployment (Render / AWS)
* Integration with real hospital APIs

---

## 👩‍💻 Developed By

**Shreejaa S M**


---
