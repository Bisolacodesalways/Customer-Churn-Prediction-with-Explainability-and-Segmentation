# Streamlit Imports
import streamlit as st
# import pandas as pd
# import numpy as np
# import shap
# import joblib
# import matplotlib.pyplot as plt

# # Load data and model artifacts with caching
# @st.cache_resource
# def load_model():
#     return joblib.load("models/log_reg_pipeline.pkl")


# @st.cache_resource
# def load_explainer():
#     return joblib.load("models/shap_explainer_logreg.pkl")


# @st.cache_data
# def load_data():
#     return pd.read_csv("data/processed/model_dataset.csv")

# model = load_model()
# explainer = load_explainer()
# df = load_data()


# App layout
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📉 Customer Churn Prediction Dashboard")
st.markdown(
    """
    This dashboard predicts customer churn risk and explains **why** a customer is likely to churn
    using SHAP explainability.
    """
)


# # Customer Selection
# customer_idx = st.slider(
#     "Select Customer Index",
#     min_value=0,
#     max_value=len(df) - 1,
#     value=10
# )

# customer = df.iloc[[customer_idx]]

# # Prediction
# churn_proba = model.predict_proba(customer)[0, 1]
# segment = (
#     "Small Business" if customer["segment"].values[0] == "Small Business"
#     else "Individual"
# )
# col1, col2, col3 = st.columns(3)

# col1.metric("Churn Probability", f"{churn_proba:.2%}")
# col2.metric("Customer Segment", segment)
# col3.metric("Risk Level", "High" if churn_proba > 0.5 else "Low")

# # SHAP Explanation
# X_transformed = model.named_steps["preprocessor"].transform(customer)
# shap_values = explainer(X_transformed)

# # Build Explanation Object
# shap_exp = shap.Explanation(
#     values=shap_values[0],
#     base_values=explainer.expected_value,
#     data=X_transformed.toarray()[0],
#     feature_names= explainer.feature_names
# )

# # Display SHAP waterfall plot
# st.subheader(" Why this customer is predicted to churn")

# fig, ax = plt.subplots(figsize=(8, 5))
# shap.plots.waterfall(shap_exp, max_display=10, show=False)
# st.pyplot(fig)

# # Raw Customer info
# with st.expander("View Customer Details"):
#     st.dataframe(customer.T)