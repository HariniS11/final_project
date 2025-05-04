# 🧠 Predicting Term Deposit Subscription

*A Streamlit-based Machine Learning Web Application*

---

## 📌 Problem Statement

A Portuguese banking institution launched several direct marketing campaigns via phone calls to promote term deposit products. The goal is to improve marketing efficiency by **predicting which clients are likely to subscribe to a term deposit**.

By building a machine learning model that classifies clients as likely to **subscribe ("yes")** or **not ("no")**, the bank can target campaigns more effectively, saving time and cost. This model is deployed as an **interactive web application using Streamlit** to empower marketing teams with real-time prediction capabilities.

---

## 📂 Dataset Overview

The dataset contains customer and campaign-related attributes from previous marketing efforts.

### 🔑 Features:

* **Numerical:** `age`, `balance`, `day`, `duration`
* **Categorical:** `job`, `marital`, `education`, `default`, `housing`, `loan`, `month`
* **Target Variable:** `y` (yes/no)

---

## ⚙️ Data Preprocessing

| Column Type                             | Preprocessing Technique |
| --------------------------------------- | ----------------------- |
| `age`, `balance`, `day`, `duration`     | MinMax Scaling          |
| `marital`, `default`, `housing`, `loan` | One-Hot Encoding        |
| `education`, `month`, `y`               | Label Encoding          |

* **Target variable:** `y` is encoded for binary classification.
* **Imbalanced Dataset Handling:** Applied **ROSE (Random Over-Sampling Examples)** technique to balance the classes.

---

## 🤖 Model & Performance

* **Algorithm Used:** [XGBoost](https://xgboost.readthedocs.io/en/stable/)
* **Train/Test Split:** Performed after ROSE sampling
* **Evaluation Metrics on Test Set:**

  * **Accuracy:** `86.17%`
  * **Precision:** `44.66%` *(Lower due to test data imbalance)*
  * **Recall:** `75.93%`
  * **F1 Score:** `56.24%`
  * **ROC-AUC Score:** `82%`

---

## 🌐 Deployment

* **Frontend:** Built using **Streamlit** for interactive prediction
* **Deployment:** Hosted on **AWS EC2 Instance**
* **Remote Access:** Configured using **Mobatek MobaXterm**

---

## 🚀 How to Run Locally

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/term-deposit-predictor.git
   cd term-deposit-predictor
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run Streamlit App**

   ```bash
   streamlit run app.py
   ```

---

## 🧪 Future Improvements

* Apply **SMOTE** or **ADASYN** for enhanced class balancing
* Incorporate **SHAP** or **LIME** for model explainability
* Tune hyperparameters using cross-validation
* Add logging and enhanced UI for better UX

---

## 👤 Author

**Your Name**
[LinkedIn](https://www.linkedin.com/in/yourprofile) • [GitHub](https://github.com/yourusername)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
