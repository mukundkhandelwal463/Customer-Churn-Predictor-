# ------------- Logistic Regression Churn Prediction ---------------

import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load and train model (cached for performance)
@st.cache_resource
def load_and_train_model():
    csv_path = os.path.join(os.path.dirname(__file__), "churn.csv")
    data = pd.read_csv(csv_path)

    # Select required columns
    columns = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "Contract", "TotalCharges", "Churn"
    ]
    data = data[columns].copy()

    # Clean and preprocess
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].mean())

    # Manual mapping
    map_dicts = {
        'gender': {'Female': 0, 'Male': 1},
        'Partner': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1},
        'PhoneService': {'No': 0, 'Yes': 1},
        'MultipleLines': {'No': 0, 'Yes': 1, 'No phone service': 2},
        'Contract': {'Month-to-month': 1, 'One year': 2, 'Two year': 3},
        'Churn': {'No': 0, 'Yes': 1}
    }
    for col, mapping in map_dicts.items():
        data[col] = data[col].map(mapping)

    data['SeniorCitizen'] = data['SeniorCitizen'].astype(int)

    # Split and scale
    X = data.drop("Churn", axis=1)
    y = data["Churn"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_test = scaler.transform(x_test)

    # Train model
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return lr_model, scaler, acc

LR, scaler, accuracy = load_and_train_model()

# Prediction function
def predict_churn(gender, senior_citizen, partner, dependents, tenure, phone_service, multiline, contract, total_charge):
    gender_val = 1 if gender == 'Male' else 0
    senior_val = 1 if senior_citizen == 'Yes' else 0
    partner_val = 1 if partner == 'Yes' else 0
    dependents_val = 1 if dependents == 'Yes' else 0
    phone_val = 1 if phone_service == 'Yes' else 0
    multiline_dict = {'No': 0, 'Yes': 1, 'No phone service': 2}
    contract_dict = {'Month-to-month': 1, 'One year': 2, 'Two year': 3}

    multiline_val = multiline_dict[multiline]
    contract_val = contract_dict[contract]

    df = pd.DataFrame([{
        'gender': gender_val,
        'SeniorCitizen': senior_val,
        'Partner': partner_val,
        'Dependents': dependents_val,
        'tenure': float(tenure),
        'PhoneService': phone_val,
        'MultipleLines': multiline_val,
        'Contract': contract_val,
        'TotalCharges': float(total_charge)
    }])

    df_scaled = scaler.transform(df)
    result = LR.predict(df_scaled)
    return "Customer is likely to CHURN ❌" if result[0] == 1 else "Customer is NOT likely to churn ✅"

# Tips for Churn Prevention
churn_tips_data = {
    "Tips": [
        "Identify the Reasons",
        "Improve Communication",
        "Enhance Experience",
        "Offer Incentives",
        "Personalize Interactions",
        "Monitor Engagement",
        "Predictive Analytics",
        "Feedback Loop",
        "Training & Development",
        "Competitive Analysis"
    ]
}

retention_tips_data = {
    "Tips": [
        "Exceptional Customer Service",
        "Loyalty Programs",
        "Regular Communication",
        "High-Quality Service",
        "Resolve Issues Quickly",
        "Build Relationships",
        "Provide Value",
        "Simplify Processes",
        "Stay Responsive",
        "Show Appreciation"
    ]
}

# Streamlit App
st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("📊 Logistic Regression → Customer Churn Predictor 🔍")
st.markdown(f"**Model Accuracy:** `{accuracy * 100:.2f}%`")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ['Female', 'Male'])
    SeniorCitizen = st.selectbox("Senior Citizen", ['No', 'Yes'])
    Partner = st.selectbox("Have Partner", ['No', 'Yes'])
    Dependents = st.selectbox("Dependent", ['No', 'Yes'])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=1, step=1)

with col2:
    PhoneService = st.selectbox("Phone Service", ['No', 'Yes'])
    MultipleLines = st.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])
    Contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=29.85, step=1.0)

if st.button("🔮 Predict Churn"):
    try:
        result = predict_churn(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, Contract, TotalCharges)
        if "CHURN" in result:
            st.error(result)
            st.markdown("### 🛑 Tips to Prevent Churn")
            st.dataframe(pd.DataFrame(churn_tips_data), height=350)
        else:
            st.success(result)
            st.markdown("### 🌟 Tips for Retaining Customers")
            st.dataframe(pd.DataFrame(retention_tips_data), height=350)
    except Exception as e:
        st.error(f"Prediction error: {e}")
