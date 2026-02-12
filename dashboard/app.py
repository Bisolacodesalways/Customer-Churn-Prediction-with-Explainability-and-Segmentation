import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================================
# CONFIGURATION & PATHS
# ============================================================================

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
    """Convert technical feature names to human-readable format."""
    return [
        FEATURE_NAME_MAP.get(name, name.replace("_", " ").title())
        for name in feature_names
    ]


# ============================================================================
# PATH RESOLUTION
# ============================================================================

def get_project_root():
    """Get the project root directory."""
    try:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        return project_root
    except NameError:
        pass
    
    cwd = Path.cwd()
    if (cwd / "models").exists():
        return cwd
    elif (cwd.parent / "models").exists():
        return cwd.parent
    else:
        return cwd


BASE_DIR = get_project_root()
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# RESOURCE LOADING
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained model pipeline."""
    import joblib
    model_path = MODELS_DIR / "log_reg_pipeline.pkl"
    
    if not model_path.exists():
        st.error(f"❌ Model file not found at: {model_path}")
        st.stop()
    
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


@st.cache_data
def load_data():
    """Load the processed dataset."""
    data_path = DATA_DIR / "model_dataset.csv"
    
    if not data_path.exists():
        st.error(f"❌ Data file not found at: {data_path}")
        st.stop()
    
    try:
        return pd.read_csv(data_path)
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()


# ============================================================================
# SHAP EXPLAINER - Robust version with auto-detection
# ============================================================================

def get_model_final_estimator(pipeline):
    """
    Extract the final estimator from a sklearn pipeline.
    Handles various pipeline structures.
    """
    # Check pipeline steps
    if hasattr(pipeline, 'steps'):
        # Get the last step (usually the classifier/regressor)
        last_step_name, last_step = pipeline.steps[-1]
        return last_step
    else:
        return pipeline


@st.cache_resource
def create_shap_explainer(_model, _sample_data):
    """
    Create SHAP explainer with multiple fallback strategies.
    """
    try:
        import shap
        
        # Get the final estimator from pipeline
        estimator = get_model_final_estimator(_model)
        
        # Transform sample data
        X_sample = _model.named_steps["preprocessor"].transform(_sample_data)
        
        # Convert sparse to dense if needed
        if hasattr(X_sample, "toarray"):
            X_sample = X_sample.toarray()
        
        # Strategy 1: Try LinearExplainer (best for logistic regression)
        try:
            explainer = shap.LinearExplainer(
                estimator,
                X_sample,
                feature_perturbation="interventional"
            )
            st.success("✅ Created LinearExplainer for fast, accurate explanations")
            return explainer, X_sample
        except Exception as e1:
            # Strategy 2: Try KernelExplainer (works for any model, slower)
            try:
                def model_predict(X):
                    """Wrapper for model prediction."""
                    proba = estimator.predict_proba(X)
                    # Return probability of positive class
                    return proba[:, 1] if len(proba.shape) > 1 and proba.shape[1] > 1 else proba
                
                explainer = shap.KernelExplainer(
                    model_predict,
                    X_sample[:50]  # Use smaller background for speed
                )
                st.success("✅ Created KernelExplainer (slower but works for all models)")
                return explainer, X_sample
            except Exception as e2:
                # Strategy 3: Try Explainer (auto-detects best method)
                try:
                    explainer = shap.Explainer(
                        estimator.predict_proba,
                        X_sample[:50]
                    )
                    st.success("✅ Created auto-detected explainer")
                    return explainer, X_sample
                except Exception as e3:
                    st.warning(f"⚠️ Could not create SHAP explainer. Errors: {e1}, {e2}, {e3}")
                    return None, None
        
    except Exception as e:
        st.warning(f"⚠️ SHAP initialization failed: {str(e)}")
        st.info("Dashboard will work without SHAP visualizations.")
        return None, None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_feature_names(model):
    """Extract feature names from the model pipeline."""
    preprocessor = model.named_steps["preprocessor"]
    
    # Numerical features
    num_features = preprocessor.transformers_[0][2]
    
    # Categorical features
    cat_transformer = preprocessor.transformers_[1][1]
    cat_features = preprocessor.transformers_[1][2]
    cat_feature_names = cat_transformer.get_feature_names_out(cat_features)
    
    return list(num_features) + list(cat_feature_names)


@st.cache_data
def get_all_churn_probabilities(_model, _df):
    """Calculate churn probabilities for all customers."""
    X_all = _df.drop(columns=["churn"])
    return _model.predict_proba(X_all)[:, 1]


# ============================================================================
# LOAD RESOURCES
# ============================================================================

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown(
    "This dashboard predicts customer churn risk and explains **why** "
    "a customer is likely to churn using SHAP."
)

# Load model and data
with st.spinner("Loading model and data..."):
    model = load_model()
    df = load_data()

# Get feature names
feature_names = get_feature_names(model)
pretty_feature_names = prettify_feature_names(feature_names)

# Calculate all probabilities
all_probs = get_all_churn_probabilities(model, df)

# Create SHAP explainer
with st.spinner("Initializing explainability engine..."):
    sample_size = min(100, len(df))
    sample_df = df.sample(sample_size, random_state=42).drop(columns=["churn"])
    explainer, background_data = create_shap_explainer(model, sample_df)

if explainer is None:
    st.info("ℹ️ SHAP visualizations unavailable. Predictions will still work perfectly.")
else:
    st.success("✅ Model, data, and explainer loaded successfully!")


# ============================================================================
# SIDEBAR - SEGMENTATION
# ============================================================================

st.sidebar.header("🎯 Segmentation")

segment_option = st.sidebar.selectbox(
    "Select customer segment",
    ["All", "Individual", "Small Business"]
)

# Filter data based on segment
filtered_df = df.copy()

if segment_option == "Individual":
    filtered_df = df[df["segment"] == "Individual"]
elif segment_option == "Small Business":
    filtered_df = df[df["segment"] == "Small Business"]

if filtered_df.empty:
    st.warning("⚠️ No customers found for this segment.")
    st.stop()

st.sidebar.metric("Customers in Segment", len(filtered_df))
st.sidebar.metric("Average Churn Rate", f"{filtered_df['churn'].mean():.1%}")


# ============================================================================
# MAIN CONTENT - CUSTOMER SELECTION & PREDICTION
# ============================================================================

st.subheader("🔍 Why this customer is predicted to churn?")

customer_idx = st.selectbox(
    "Select a customer index",
    options=filtered_df.index.tolist(),
    help="Choose a customer to analyze their churn risk"
)

# Get customer data and make prediction
customer_data = df.loc[[customer_idx]].drop(columns=["churn"])
churn_proba = model.predict_proba(customer_data)[0, 1]

# Calculate risk thresholds
high_risk_threshold = np.quantile(all_probs, 0.95)   # Top 5%
medium_risk_threshold = np.quantile(all_probs, 0.80) # Top 20%

# Display prediction
col1, col2, col3 = st.columns([2, 2, 3])

with col1:
    st.metric(
        label="Predicted Churn Probability",
        value=f"{churn_proba:.2%}"
    )

with col2:
    base_rate = df["churn"].mean()
    lift = churn_proba / base_rate
    st.metric(
        label="Risk vs Average",
        value=f"{lift:.1f}×"
    )

with col3:
    if churn_proba >= high_risk_threshold:
        st.error("🔴 **High Risk** (Top 5%)")
    elif churn_proba >= medium_risk_threshold:
        st.warning("🟠 **Medium Risk** (Top 20%)")
    else:
        st.success("🟢 **Low Risk**")


# ============================================================================
# SHAP WATERFALL PLOT - INDIVIDUAL CUSTOMER
# ============================================================================

if explainer is not None:
    st.subheader("💡 What's driving this prediction?")
    
    try:
        import shap
        
        # Transform customer data
        X_customer = model.named_steps["preprocessor"].transform(customer_data)
        
        # Convert sparse matrix to dense if needed
        if hasattr(X_customer, "toarray"):
            X_customer = X_customer.toarray()
        
        # Calculate SHAP values - try different methods
        try:
            # Method 1: Try shap_values (for LinearExplainer/KernelExplainer)
            shap_values = explainer.shap_values(X_customer)
            
            # Handle different return formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Binary classification, get positive class
            
            # Get expected value
            if isinstance(explainer.expected_value, (list, np.ndarray)):
                expected_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
            else:
                expected_value = explainer.expected_value
            
            # Create explanation object
            explanation = shap.Explanation(
                values=shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                base_values=expected_value,
                data=X_customer[0],
                feature_names=pretty_feature_names
            )
            
        except AttributeError:
            # Method 2: Modern API (for newer Explainer)
            shap_result = explainer(X_customer)
            explanation = shap_result[0]
            explanation.feature_names = pretty_feature_names
        
        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig)
        plt.close()
        
        st.caption(
            "📊 This waterfall plot shows how each feature pushes the prediction "
            "higher (red) or lower (blue) from the baseline."
        )
        
    except Exception as e:
        st.error(f"❌ Error generating SHAP waterfall plot: {str(e)}")
        st.info("The prediction is still valid. Try selecting a different customer.")
else:
    st.info("ℹ️ SHAP visualizations are not available, but predictions are working correctly.")


# ============================================================================
# SEGMENT-LEVEL INSIGHTS
# ============================================================================

st.subheader(f"📈 Key Churn Drivers for {segment_option} Customers")

if explainer is not None:
    try:
        import shap
        
        # Sample customers from segment
        sample_size = min(200, len(filtered_df))
        sample_df = filtered_df.sample(sample_size, random_state=42).drop(columns=["churn"])
        
        # Transform data
        X_segment = model.named_steps["preprocessor"].transform(sample_df)
        
        # Convert sparse matrix to dense if needed
        if hasattr(X_segment, "toarray"):
            X_segment = X_segment.toarray()
        
        # Calculate SHAP values with progress indicator
        with st.spinner(f"Analyzing {sample_size} customers... (this may take 30-60 seconds)"):
            try:
                shap_values = explainer.shap_values(X_segment)
                
                # Handle different return formats
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Binary classification
                
            except AttributeError:
                # Modern API
                shap_result = explainer(X_segment)
                shap_values = shap_result.values
        
        # Create summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_segment,
            feature_names=pretty_feature_names,
            show=False,
            max_display=15
        )
        st.pyplot(fig)
        plt.close()
        
        st.caption(
            f"📊 This plot shows the most important features for predicting churn "
            f"across {sample_size} {segment_option.lower()} customers. "
            f"Red indicates high feature values, blue indicates low values."
        )
        
    except Exception as e:
        st.error(f"❌ Error generating segment insights: {str(e)}")
        st.info("Try reducing the sample size or selecting a different segment.")
else:
    st.info("ℹ️ SHAP segment analysis not available.")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "**Note:** This dashboard uses machine learning to predict churn probability. "
    "Predictions should be used as one input among many for business decisions."
)
