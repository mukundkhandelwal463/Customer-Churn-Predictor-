# 📊 Customer Churn Intelligence & Prediction Hub

An AI-powered, interactive **Customer Churn Analytics & Prediction Dashboard** built with **Streamlit**, **Scikit-Learn**, and **Plotly**. This application enables telecom and subscription businesses to predict customer churn in real time, analyze retention trends with exploratory data visualizations, inspect machine learning diagnostics, and process customer batches.

---

## ✨ Features & Capabilities

### 1. 🔮 Live Churn Predictor
- **Real-Time Risk Scoring**: Calculates exact churn probability percentages.
- **Interactive Risk Gauge Meter**: Plotly-powered visual gauge indicating risk zones (*Low Risk*, *Moderate Risk*, *High Risk*).
- **Local Feature Impact Analysis**: Bar chart highlighting the exact customer attributes that drove the prediction up or down.
- **Dynamic Retention Playbook**: Generates actionable, tailored customer retention strategies (contract upgrade incentives, onboarding touchpoints, high-value alerts).
- **Customer Persona Presets**: 1-click presets (*High-Risk Solo User*, *Loyal 2-Year Family*, *Moderate-Risk Account*) for instant testing.

### 2. 📊 Exploratory Data Analysis & Insights (EDA)
- **Executive KPIs**: Total customer count, overall churn rate, retained accounts, and average lifetime spend.
- **Interactive Plotly Visualizations**:
  - Donut chart of overall churn proportion.
  - Churn breakdown by contract type (Month-to-month vs. Annual vs. Two-year).
  - Tenure distributions comparing churned vs. retained subscribers.
  - Total Charges box-plots across contract types.

### 3. 📈 Machine Learning Diagnostics & Weights
- **Model Scorecard**: Evaluates Accuracy (~77.6%), ROC-AUC (~83.8%), Precision, Recall, and F1-Score on test data.
- **Confusion Matrix Heatmap**: Interactive heatmap detailing True Positives, False Positives, True Negatives, and False Negatives.
- **Receiver Operating Characteristic (ROC) Curve**: Shows AUC classification performance against random guessing.
- **Standardized Feature Weights**: Horizontal bar chart exposing the logistic regression coefficients.

### 4. 📁 Batch CSV Prediction & Export
- Upload any customer CSV dataset or test with sample records.
- Generates batch predictions with churn risk probabilities and risk tier tags.
- Download enriched prediction results as a CSV with one click.

---

## 📂 Project Structure

```text
Customer-Churn-Predictor/
├── logistic_reg_project.py   # Main Streamlit Dashboard Application
├── churn.csv                 # Telco customer dataset (7,043 rows)
├── requirements.txt          # Python dependencies (streamlit, scikit-learn, plotly, pandas)
├── .gitignore                # Git ignore configuration
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/mukundkhandelwal463/Customer-Churn-Predictor-.git
cd Customer-Churn-Predictor-
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Dashboard
```bash
streamlit run logistic_reg_project.py
```
*(Or `python -m streamlit run logistic_reg_project.py`)*

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** (Interactive Web UI)
- **Plotly** (Interactive dynamic charts and gauges)
- **Scikit-Learn** (Logistic Regression, Data Preprocessing, Metrics)
- **Pandas & NumPy** (Data manipulation and batch processing)

---

## 📬 Contact & Socials

Created by **Mukund Khandelwal**
- [LinkedIn Profile](https://www.linkedin.com/posts/mukund-khandelwal-6a8663283_machinelearning-logisticregression-streamlit-activity-7353860195720028183-NMnj?utm_source=share&utm_medium=member_desktop&rcm=ACoAAET5diABs7bbZlDnVTGZ4DnPgeKxnEmHsgA)
- [GitHub Repository](https://github.com/mukundkhandelwal463/Customer-Churn-Predictor-)

---

## 📝 License
This project is open-source under the **MIT License**.
