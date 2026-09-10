# ==============================================================================
# 📊 Customer Churn Predictor & Analytics Dashboard
# ==============================================================================

import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# ------------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom modern CSS styling
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-delta-positive {
        font-size: 0.85rem;
        color: #ef4444;
    }
    .metric-delta-negative {
        font-size: 0.85rem;
        color: #10b981;
    }

    /* Result Banner */
    .risk-banner-high {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.05) 100%);
        border-left: 5px solid #ef4444;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 14px 0;
    }
    .risk-banner-low {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.05) 100%);
        border-left: 5px solid #10b981;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 14px 0;
    }
    .risk-banner-medium {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.05) 100%);
        border-left: 5px solid #f59e0b;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 14px 0;
    }

    /* Subheader badges */
    .hero-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. Data Loading & Model Training Pipeline
# ------------------------------------------------------------------------------
@st.cache_data
def load_raw_dataset():
    csv_path = os.path.join(os.path.dirname(__file__), "churn.csv")
    df = pd.read_csv(csv_path)
    # Ensure numeric TotalCharges for EDA
    tc_num = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = tc_num.fillna(tc_num.mean())
    return df

@st.cache_resource
def train_churn_model():
    raw_df = load_raw_dataset()
    columns = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "Contract", "TotalCharges", "Churn"
    ]
    data = raw_df[columns].copy()

    tc_num = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"] = tc_num.fillna(tc_num.mean())

    map_dicts = {
        "gender": {"Female": 0, "Male": 1},
        "Partner": {"No": 0, "Yes": 1},
        "Dependents": {"No": 0, "Yes": 1},
        "PhoneService": {"No": 0, "Yes": 1},
        "MultipleLines": {"No": 0, "Yes": 1, "No phone service": 2},
        "Contract": {"Month-to-month": 1, "One year": 2, "Two year": 3},
        "Churn": {"No": 0, "Yes": 1}
    }
    for col, mapping in map_dicts.items():
        data[col] = data[col].map(mapping)

    data["SeniorCitizen"] = data["SeniorCitizen"].astype(int)

    X = data.drop("Churn", axis=1)
    y = data["Churn"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_test = scaler.transform(x_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_test": y_test,
        "y_prob": y_prob,
        "feature_names": X.columns.tolist(),
        "coefficients": model.coef_[0]
    }

    return model, scaler, metrics, map_dicts

# Initialize data and model
raw_data = load_raw_dataset()
lr_model, scaler, model_metrics, mappings = train_churn_model()


# ------------------------------------------------------------------------------
# 3. Sidebar Controls & Presets
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/combo-chart.png", width=64)
    st.title("Churn Intelligence")
    st.caption("AI-Powered Telco Retention System")
    st.markdown("---")

    st.subheader("⚡ Quick Customer Presets")
    preset = st.selectbox(
        "Load Sample Customer Profile:",
        [
            "Custom Input",
            "🚨 High-Risk (New, Month-to-Month, Solo)",
            "🛡️ Low-Risk (Long-term, 2-Year Contract)",
            "⚖️ Moderate-Risk (Family Plan, 1-Year)"
        ]
    )

    st.markdown("---")
    st.subheader("⚙️ Classification Settings")
    decision_threshold = st.slider(
        "Decision Threshold for Churn:",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Lower threshold to flag more at-risk customers proactively."
    )

    st.markdown("---")
    st.markdown("""
    **Model Specs:**
    - Algorithm: `Logistic Regression`
    - Trained on: `7,043 Customer Records`
    - Accuracy: **`{:.2f}%`**
    - ROC-AUC: **`{:.2f}%`**
    """.format(model_metrics["accuracy"] * 100, model_metrics["roc_auc"] * 100))


# Pre-fill values based on chosen preset
preset_values = {
    "gender": "Female",
    "senior": "No",
    "partner": "No",
    "dependents": "No",
    "tenure": 1,
    "phone": "Yes",
    "multiline": "No",
    "contract": "Month-to-month",
    "total_charges": 65.50
}

if preset == "🚨 High-Risk (New, Month-to-Month, Solo)":
    preset_values = {
        "gender": "Female",
        "senior": "Yes",
        "partner": "No",
        "dependents": "No",
        "tenure": 1,
        "phone": "Yes",
        "multiline": "No",
        "contract": "Month-to-month",
        "total_charges": 85.00
    }
elif preset == "🛡️ Low-Risk (Long-term, 2-Year Contract)":
    preset_values = {
        "gender": "Male",
        "senior": "No",
        "partner": "Yes",
        "dependents": "Yes",
        "tenure": 60,
        "phone": "Yes",
        "multiline": "Yes",
        "contract": "Two year",
        "total_charges": 4850.00
    }
elif preset == "⚖️ Moderate-Risk (Family Plan, 1-Year)":
    preset_values = {
        "gender": "Female",
        "senior": "No",
        "partner": "Yes",
        "dependents": "No",
        "tenure": 14,
        "phone": "Yes",
        "multiline": "Yes",
        "contract": "One year",
        "total_charges": 950.00
    }


# ------------------------------------------------------------------------------
# 4. Main App Layout & Tabs
# ------------------------------------------------------------------------------
st.markdown('<div class="hero-badge">AI / ML Retention Engine</div>', unsafe_allow_html=True)
st.title("📊 Customer Churn Intelligence & Prediction Hub")
st.markdown("Predict individual customer churn probability, discover dataset trends with interactive visual analytics, and inspect machine learning model diagnostics.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Live Churn Predictor",
    "📊 Dataset Analytics & Insights",
    "📈 Model Diagnostics & Weights",
    "📁 Batch CSV Prediction"
])


# ==============================================================================
# TAB 1: Live Churn Predictor
# ==============================================================================
with tab1:
    st.subheader("🔮 Individual Customer Risk Assessment")
    st.write("Configure the customer's demographics, subscription options, and billing metrics to predict their likelihood to churn.")

    with st.form("churn_prediction_form"):
        col_demo, col_service, col_billing = st.columns(3)

        with col_demo:
            st.markdown("##### 👤 Demographics")
            gender_input = st.selectbox(
                "Gender",
                ["Female", "Male"],
                index=["Female", "Male"].index(preset_values["gender"])
            )
            senior_input = st.selectbox(
                "Senior Citizen",
                ["No", "Yes"],
                index=["No", "Yes"].index(preset_values["senior"])
            )
            partner_input = st.selectbox(
                "Has Partner",
                ["No", "Yes"],
                index=["No", "Yes"].index(preset_values["partner"])
            )
            dependents_input = st.selectbox(
                "Has Dependents",
                ["No", "Yes"],
                index=["No", "Yes"].index(preset_values["dependents"])
            )

        with col_service:
            st.markdown("##### 📱 Services & Contract")
            phone_input = st.selectbox(
                "Phone Service",
                ["No", "Yes"],
                index=["No", "Yes"].index(preset_values["phone"])
            )
            multiline_input = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"],
                index=["No", "Yes", "No phone service"].index(preset_values["multiline"])
            )
            contract_input = st.selectbox(
                "Contract Plan",
                ["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(preset_values["contract"])
            )

        with col_billing:
            st.markdown("##### 💳 Billing & Tenure")
            tenure_input = st.number_input(
                "Tenure (Months with Company)",
                min_value=0,
                max_value=120,
                value=int(preset_values["tenure"]),
                step=1
            )
            total_charges_input = st.number_input(
                "Total Charges Incurred ($)",
                min_value=0.0,
                max_value=20000.0,
                value=float(preset_values["total_charges"]),
                step=10.0
            )
            # Estimated monthly charge calculation
            est_monthly = total_charges_input / max(tenure_input, 1)
            st.caption(f"💡 Estimated Monthly Bill: **${est_monthly:.2f}/mo**")

        predict_btn = st.form_submit_button("🚀 Run Churn Analysis", use_container_width=True, type="primary")

    # Prediction Logic & Visualizations
    # Perform prediction immediately or when submitted
    gender_num = 1 if gender_input == "Male" else 0
    senior_num = 1 if senior_input == "Yes" else 0
    partner_num = 1 if partner_input == "Yes" else 0
    dependents_num = 1 if dependents_input == "Yes" else 0
    phone_num = 1 if phone_input == "Yes" else 0
    multiline_num = {"No": 0, "Yes": 1, "No phone service": 2}[multiline_input]
    contract_num = {"Month-to-month": 1, "One year": 2, "Two year": 3}[contract_input]

    input_df = pd.DataFrame([{
        "gender": gender_num,
        "SeniorCitizen": senior_num,
        "Partner": partner_num,
        "Dependents": dependents_num,
        "tenure": float(tenure_input),
        "PhoneService": phone_num,
        "MultipleLines": multiline_num,
        "Contract": contract_num,
        "TotalCharges": float(total_charges_input)
    }])

    input_scaled = scaler.transform(input_df)
    churn_prob = lr_model.predict_proba(input_scaled)[0, 1]
    is_churn = churn_prob >= decision_threshold

    st.markdown("---")
    st.subheader("🎯 Prediction Results & Risk Diagnostics")

    res_col1, res_col2 = st.columns([1.1, 1.4])

    with res_col1:
        # Gauge Chart for Risk Probability
        gauge_color = "#ef4444" if churn_prob >= 0.60 else ("#f59e0b" if churn_prob >= 0.35 else "#10b981")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_prob * 100,
            number={"suffix": "%", "font": {"size": 42, "color": gauge_color}},
            title={"text": "<b>Churn Probability</b>", "font": {"size": 20, "color": "#f8fafc"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": gauge_color, "thickness": 0.35},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.1)",
                "steps": [
                    {"range": [0, 35], "color": "rgba(16, 185, 129, 0.2)"},
                    {"range": [35, 60], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [60, 100], "color": "rgba(239, 68, 68, 0.2)"}
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 4},
                    "thickness": 0.8,
                    "value": decision_threshold * 100
                }
            }
        ))
        fig_gauge.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc"}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        if churn_prob >= 0.60:
            st.markdown(f"""
            <div class="risk-banner-high">
                <h4 style="color:#ef4444; margin:0 0 6px 0;">🚨 High Churn Risk Detected</h4>
                <p style="margin:0; color:#cbd5e1;">This customer has a <b>{churn_prob*100:.1f}%</b> likelihood of leaving. Immediate retention action is recommended.</p>
            </div>
            """, unsafe_allow_html=True)
        elif churn_prob >= 0.35:
            st.markdown(f"""
            <div class="risk-banner-medium">
                <h4 style="color:#f59e0b; margin:0 0 6px 0;">⚠️ Moderate Churn Risk</h4>
                <p style="margin:0; color:#cbd5e1;">This customer has a <b>{churn_prob*100:.1f}%</b> probability of churn. Consider offering contract loyalty incentives.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-banner-low">
                <h4 style="color:#10b981; margin:0 0 6px 0;">✅ Low Churn Risk</h4>
                <p style="margin:0; color:#cbd5e1;">This customer is stable with only a <b>{churn_prob*100:.1f}%</b> chance of leaving.</p>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        # Local Feature Impact Breakdown
        feature_impact = input_scaled[0] * lr_model.coef_[0]
        impact_df = pd.DataFrame({
            "Feature": [
                "Gender", "Senior Citizen", "Partner", "Dependents",
                "Tenure", "Phone Service", "Multiple Lines", "Contract", "Total Charges"
            ],
            "Impact": feature_impact
        }).sort_values("Impact", ascending=True)

        impact_df["Effect"] = impact_df["Impact"].apply(lambda x: "Increases Churn" if x > 0 else "Promotes Retention")
        colors = impact_df["Impact"].apply(lambda x: "#ef4444" if x > 0 else "#10b981")

        fig_impact = px.bar(
            impact_df,
            x="Impact",
            y="Feature",
            orientation="h",
            color="Effect",
            color_discrete_map={"Increases Churn": "#ef4444", "Promotes Retention": "#10b981"},
            title="Customer Attribute Impact on Churn Score"
        )
        fig_impact.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc"},
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="Log-Odds Contribution"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="")
        )
        st.plotly_chart(fig_impact, use_container_width=True)

    # Tailored Retention Recommendations Section
    st.markdown("#### 💡 Tailored Retention Playbook")
    rec_col1, rec_col2, rec_col3 = st.columns(3)

    with rec_col1:
        if contract_input == "Month-to-month":
            st.info("📋 **Contract Migration:** Customer is on a Month-to-month plan. Offer a 15% discount to upgrade to an Annual or 2-Year Contract.")
        else:
            st.success("📋 **Contract Status:** Secured on long-term contract. Schedule periodic relationship check-ins.")

    with rec_col2:
        if tenure_input <= 6:
            st.warning("⏱️ **Onboarding Experience:** Low tenure (< 6 mos). Trigger an automated CS follow-up call and onboarding tips.")
        else:
            st.success(f"⏱️ **Tenure Loyalty:** Customer has been subscribed for {tenure_input} months. Eligible for VIP loyalty perks.")

    with rec_col3:
        if total_charges_input > 1500 and churn_prob > 0.4:
            st.error("💰 **High-Value Account:** High total spend with elevated risk. Route to Senior Retention Specialist immediately.")
        else:
            st.info("💰 **Value Growth:** Recommend bundle add-ons (Tech Support / Streaming discounts) to build stickiness.")


# ==============================================================================
# TAB 2: Dataset Analytics & Insights (EDA)
# ==============================================================================
with tab2:
    st.subheader("📊 Dataset Overview & Churn Distribution")
    st.write("Exploratory Data Analysis across the Telco customer dataset (7,043 total customer records).")

    # High-level KPIs
    total_cust = len(raw_data)
    churn_count = (raw_data["Churn"] == "Yes").sum()
    retained_count = total_cust - churn_count
    churn_rate = (churn_count / total_cust) * 100
    avg_charges = raw_data["TotalCharges"].mean()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Customers</div>
            <div class="metric-value">{total_cust:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Churn Rate</div>
            <div class="metric-value">{churn_rate:.1f}%</div>
            <div class="metric-delta-positive">{churn_count:,} customers left</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Retained Customers</div>
            <div class="metric-value">{retained_count:,}</div>
            <div class="metric-delta-negative">{100-churn_rate:.1f}% retention</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Lifetime Spend</div>
            <div class="metric-value">${avg_charges:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Row 1 Charts: Churn Distribution & Churn by Contract
    row1_c1, row1_c2 = st.columns(2)

    with row1_c1:
        churn_counts = raw_data["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        churn_counts["Label"] = churn_counts["Churn"].map({"Yes": "Churned ❌", "No": "Retained ✅"})
        fig_donut = px.pie(
            churn_counts,
            values="Count",
            names="Label",
            hole=0.55,
            color="Label",
            color_discrete_map={"Churned ❌": "#ef4444", "Retained ✅": "#10b981"},
            title="Overall Customer Churn Proportion"
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc"},
            height=340
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with row1_c2:
        contract_churn = raw_data.groupby(["Contract", "Churn"]).size().reset_index(name="Count")
        fig_contract = px.bar(
            contract_churn,
            x="Contract",
            y="Count",
            color="Churn",
            barmode="group",
            color_discrete_map={"Yes": "#ef4444", "No": "#10b981"},
            title="Churn Breakdown by Contract Type"
        )
        fig_contract.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc"},
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            height=340
        )
        st.plotly_chart(fig_contract, use_container_width=True)

    # Row 2 Charts: Tenure & Total Charges distribution
    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        fig_tenure = px.histogram(
            raw_data,
            x="tenure",
            color="Churn",
            marginal="box",
            nbins=36,
            color_discrete_map={"Yes": "#ef4444", "No": "#10b981"},
            title="Customer Tenure Distribution (Months)"
        )
        fig_tenure.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc"},
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="Tenure (Months)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            height=340
        )
        st.plotly_chart(fig_tenure, use_container_width=True)

    with row2_c2:
        fig_charges = px.box(
            raw_data,
            x="Contract",
            y="TotalCharges",
            color="Churn",
            color_discrete_map={"Yes": "#ef4444", "No": "#10b981"},
            title="Total Charges Incurred across Contract Types"
        )
        fig_charges.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc"},
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            height=340
        )
        st.plotly_chart(fig_charges, use_container_width=True)


# ==============================================================================
# TAB 3: Model Diagnostics & Feature Weights
# ==============================================================================
with tab3:
    st.subheader("📈 Logistic Regression Diagnostics & Performance")
    st.write("Quantitative evaluation of the classification model on held-out test data (20% split, 1,409 test records).")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Accuracy", f"{model_metrics['accuracy']*100:.2f}%")
    with m2:
        st.metric("ROC-AUC Score", f"{model_metrics['roc_auc']*100:.2f}%")
    with m3:
        st.metric("Precision", f"{model_metrics['precision']*100:.2f}%")
    with m4:
        st.metric("Recall", f"{model_metrics['recall']*100:.2f}%")
    with m5:
        st.metric("F1-Score", f"{model_metrics['f1']*100:.2f}%")

    diag_c1, diag_c2 = st.columns(2)

    with diag_c1:
        # Confusion Matrix Heatmap
        cm = model_metrics["confusion_matrix"]
        cm_labels = [["True Negative (TN)", "False Positive (FP)"], ["False Negative (FN)", "True Positive (TP)"]]
        cm_text = [[f"{val}<br><i>{cm_labels[i][j]}</i>" for j, val in enumerate(row)] for i, row in enumerate(cm)]

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted: Retained (0)", "Predicted: Churned (1)"],
            y=["Actual: Retained (0)", "Actual: Churned (1)"],
            text=cm_text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False
        ))
        fig_cm.update_layout(
            title="Confusion Matrix Heatmap",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc"},
            height=340
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with diag_c2:
        # ROC Curve
        fpr, tpr, _ = roc_curve(model_metrics["y_test"], model_metrics["y_prob"])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"Logistic Regression (AUC = {model_metrics['roc_auc']:.3f})",
            line=dict(color="#6366f1", width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random Guess",
            line=dict(color="#94a3b8", dash="dash")
        ))
        fig_roc.update_layout(
            title="Receiver Operating Characteristic (ROC) Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc"},
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            height=340
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Global Feature Weights / Coefficients
    st.markdown("#### ⚖️ Global Model Coefficients (Feature Weights)")
    coef_df = pd.DataFrame({
        "Feature": model_metrics["feature_names"],
        "Coefficient": model_metrics["coefficients"]
    }).sort_values("Coefficient", ascending=True)

    coef_df["Impact Direction"] = coef_df["Coefficient"].apply(
        lambda x: "Increases Churn (+)" if x > 0 else "Protects Retention (-)"
    )

    fig_coef = px.bar(
        coef_df,
        x="Coefficient",
        y="Feature",
        orientation="h",
        color="Impact Direction",
        color_discrete_map={"Increases Churn (+)": "#ef4444", "Protects Retention (-)": "#10b981"},
        title="Logistic Regression Standardized Feature Weights"
    )
    fig_coef.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="Standardized Coefficient Weight"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title=""),
        height=320
    )
    st.plotly_chart(fig_coef, use_container_width=True)


# ==============================================================================
# TAB 4: Batch CSV Prediction & Export
# ==============================================================================
with tab4:
    st.subheader("📁 Batch CSV Customer Prediction")
    st.write("Upload a CSV file of customer accounts to run churn predictions in bulk and export risk probabilities.")

    batch_c1, batch_c2 = st.columns([1.5, 1])

    with batch_c1:
        uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])

    with batch_c2:
        st.markdown("##### 🧪 Quick Test Options")
        use_sample = st.button("Load 15 Sample Records from Telco Dataset", use_container_width=True)

    batch_df = None
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded CSV with {len(batch_df)} customer rows.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    elif use_sample:
        batch_df = raw_data.sample(15, random_state=42).copy()

    if batch_df is not None:
        required_cols = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "Contract", "TotalCharges"
        ]
        missing_cols = [c for c in required_cols if c not in batch_df.columns]

        if missing_cols:
            st.error(f"The uploaded CSV is missing required columns: {missing_cols}")
        else:
            with st.spinner("Calculating churn risk probabilities across all records..."):
                proc_df = batch_df[required_cols].copy()
                tc_num = pd.to_numeric(proc_df["TotalCharges"], errors="coerce")
                proc_df["TotalCharges"] = tc_num.fillna(tc_num.mean())

                map_dicts = {
                    "gender": {"Female": 0, "Male": 1, 0: 0, 1: 1},
                    "Partner": {"No": 0, "Yes": 1, 0: 0, 1: 1},
                    "Dependents": {"No": 0, "Yes": 1, 0: 0, 1: 1},
                    "PhoneService": {"No": 0, "Yes": 1, 0: 0, 1: 1},
                    "MultipleLines": {"No": 0, "Yes": 1, "No phone service": 2, 0: 0, 1: 1, 2: 2},
                    "Contract": {"Month-to-month": 1, "One year": 2, "Two year": 3, 1: 1, 2: 2, 3: 3}
                }
                for col, m in map_dicts.items():
                    if proc_df[col].dtype == object or proc_df[col].dtype.name == "category" or proc_df[col].dtype.name == "string":
                        proc_df[col] = proc_df[col].map(m).fillna(0)

                proc_df["SeniorCitizen"] = proc_df["SeniorCitizen"].astype(int)

                scaled_batch = scaler.transform(proc_df)
                probs = lr_model.predict_proba(scaled_batch)[:, 1]

                batch_results = batch_df.copy()
                batch_results["Churn_Probability"] = np.round(probs * 100, 2)
                batch_results["Predicted_Churn"] = np.where(probs >= decision_threshold, "CHURN ❌", "RETAIN ✅")
                batch_results["Risk_Tier"] = pd.cut(
                    probs,
                    bins=[-0.01, 0.35, 0.60, 1.0],
                    labels=["Low Risk", "Medium Risk", "High Risk"]
                )

                st.markdown("---")
                high_risk_count = (probs >= decision_threshold).sum()
                st.info(f"📊 Processed **{len(batch_results)} accounts**. Identified **{high_risk_count} at-risk customers** based on threshold `{decision_threshold}`.")

                st.dataframe(
                    batch_results[[
                        "gender", "tenure", "Contract", "TotalCharges",
                        "Churn_Probability", "Predicted_Churn", "Risk_Tier"
                    ]],
                    use_container_width=True
                )

                # Export CSV button
                csv_buffer = io.StringIO()
                batch_results.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Enriched Predictions (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="churn_predictions_export.csv",
                    mime="text/csv",
                    type="primary"
                )
