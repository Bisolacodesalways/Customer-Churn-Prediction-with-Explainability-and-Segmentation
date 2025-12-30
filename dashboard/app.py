import streamlit as st
from pathlib import Path
import pandas as pd
import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt


# Feature Name Mapping

FEATURE_NAME_MAP = {
    "failed_payment_sum": "Failed Payments (Last 3 Months)",
    "amount_paid_mean": "Average Monthly Spend",
    "heavy_feature_usage_mean": "Advanced Feature Usage",
    "ticket_count": "Support Tickets Raised",
    "price_per_month": "Monthly Price",
    "total_usage_hours_last": "Recent Usage (Hours)",
    "login_count_std": "Login Frequency Variability",
    "num_sessions_web_mean": "Average Web Sessions",
    "active_days_mean": "Active Days per Month",
    "days_late_max": "Maximum Payment Delay (Days)",
    "avg_satisfaction_score": "Customer Satisfaction Score",
    "contract_type_Monthly": "Monthly Contract",
    "contract_type_Annual": "Annual Contract",
    "plan_type_Standard": "Standard Plan",
    "plan_type_Basic": "Basic Plan",
    "segment_Individual": "Individual Customer",
    "segment_Small Business": "Small Business Customer",
}

def prettify_feature_names(feature_names):
    return [
        FEATURE_NAME_MAP.get(name, name.replace("_", " ").title())
        for name in feature_names
    ]



# Paths & Page Config

BASE_DIR = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📉 Customer Churn Prediction Dashboard")
st.markdown(
    "This dashboard predicts customer churn risk and explains **why** a customer is likely to churn using SHAP."
)

# Load Model & Data

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "models" / "log_reg_pipeline.pkl")

@st.cache_resource
def load_explainer():
    explainer_path = BASE_DIR / "models" / "shap_explainer.pkl"
    if explainer_path.exists():
        return joblib.load(explainer_path)
    else:
        st.warning("SHAP explainer not found. Explainability disabled.")
        return None


@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "data" / "processed" / "model_dataset.csv")

model = load_model()
explainer = load_explainer()
df = load_data()

def get_all_churn_probabilities(_model, _df):
    X_all = _df.drop(columns=["churn"])
    return _model.predict_proba(X_all)[:, 1]

all_probs = get_all_churn_probabilities(model, df)
st.success("Model and data loaded successfully!")

# Get feature names
def get_feature_names(model):
    preprocessor = model.named_steps["preprocessor"]

    num_features = preprocessor.transformers_[0][2]
    cat_transformer = preprocessor.transformers_[1][1]
    cat_features = preprocessor.transformers_[1][2]

    cat_feature_names = cat_transformer.get_feature_names_out(cat_features)

    return list(num_features) + list(cat_feature_names)

feature_names = get_feature_names(model)
pretty_feature_names = prettify_feature_names(feature_names)

# Sidebar – Segmentation

st.sidebar.header("📊 Segmentation")

segment_option = st.sidebar.selectbox(
    "Select customer segment",
    ["All", "Individual", "Small Business"]
)

filtered_df = df.copy()

if segment_option == "Individual":
    filtered_df = df[df["segment"] == "Individual"]

elif segment_option == "Small Business":
    filtered_df = df[df["segment"] == "Small Business"]

if filtered_df.empty:
    st.warning("No customers found for this segment.")
    st.stop()



# Customer Selector
st.subheader(" Why this customer is predicted to churn?")
customer_idx = st.selectbox(
    "Select a customer index",
    options=filtered_df.index.tolist()
)


# Prediction
customer_data = df.loc[[customer_idx]].drop(columns=["churn"])
churn_proba = model.predict_proba(customer_data)[0, 1]

# Get transformed features
# Transform customer data using the pipeline preprocessor
X_transformed = model.named_steps["preprocessor"].transform(customer_data)


# SHAP cannot handle sparse matrices cleanly
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

shap_values_customer = explainer(X_transformed)[0]


# Display Predictions
st.metric(
    label="Predicted Churn Probability",
    value=f"{churn_proba:.2%}"
)

# Percentile-based risk thresholds
high_risk_threshold = np.quantile(all_probs, 0.95)   # Top 5%
medium_risk_threshold = np.quantile(all_probs, 0.80) # Top 20%

if churn_proba >= high_risk_threshold:
    st.error("🔴 High risk of churn (Top 5%)")
elif churn_proba >= medium_risk_threshold:
    st.warning("🟠 Medium risk of churn (Top 20%)")
else:
    st.success("🟢 Low risk of churn")

base_rate = df["churn"].mean()
lift = churn_proba / base_rate

st.caption(
    f"📈 Relative risk vs average customer: **{lift:.1f}×**"
)



# =========================
# SHAP Waterfall (Customer)
# =========================
X_customer = model.named_steps["preprocessor"].transform(customer_data)
if hasattr(X_customer, "toarray"):
    X_customer = X_customer.toarray()

shap_values_customer = explainer(X_customer)[0]
pretty_feature_names = prettify_feature_names(feature_names)
shap_values_customer.feature_names = pretty_feature_names


fig, ax = plt.subplots(figsize=(10, 5))
shap.plots.waterfall(shap_values_customer, show=False)
st.pyplot(fig)

# Segmented Customer Insights
st.subheader(f"📈 Key Churn Drivers for {segment_option} Customers")

sample_df = filtered_df.sample(
    min(300, len(filtered_df)),
    random_state=42
).drop(columns=["churn"])

X_segment = model.named_steps["preprocessor"].transform(sample_df)

if hasattr(X_segment, "toarray"):
    X_segment = X_segment.toarray()

shap_values_segment = explainer(X_segment)

# Adding more improvements
pretty_feature_names = prettify_feature_names(feature_names)
shap_values_segment.feature_names = pretty_feature_names


fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.beeswarm(shap_values_segment, show=False)
st.pyplot(fig)
