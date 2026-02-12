import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


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
# PATH RESOLUTION - Works both locally and on Streamlit Cloud
# ============================================================================

def get_project_root():
    """
    Get the project root directory.
    Works both locally and on Streamlit Cloud.
    """
    # Method 1: Try to get from __file__
    try:
        current_file = Path(__file__).resolve()
        # Go up one level from dashboard/ to project root
        project_root = current_file.parent.parent
        return project_root
    except NameError:
        # __file__ not available (e.g., in some environments)
        pass
    
    # Method 2: Use current working directory
    cwd = Path.cwd()
    
    # Check if we're in a subdirectory (like dashboard/)
    if (cwd / "models").exists():
        return cwd
    elif (cwd.parent / "models").exists():
        return cwd.parent
    else:
        return cwd


# Get base directory
BASE_DIR = get_project_root()

# Define paths
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
# RESOURCE LOADING WITH ROBUST ERROR HANDLING
# ============================================================================

def safe_pickle_load(filepath):
    """
    Safely load a pickle file with multiple fallback methods.
    Handles version compatibility issues.
    """
    try:
        # Method 1: Try with joblib (recommended for sklearn objects)
        import joblib
        return joblib.load(filepath)
    except Exception as e1:
        try:
            # Method 2: Try with standard pickle
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e2:
            try:
                # Method 3: Try with encoding parameter
                import joblib
                return joblib.load(filepath, encoding='latin1')
            except Exception as e3:
                # All methods failed
                raise Exception(f"Failed to load {filepath}. Tried joblib, pickle, and latin1 encoding. Errors: {e1}, {e2}, {e3}")


@st.cache_resource
def load_model():
    """Load the trained model pipeline."""
    model_path = MODELS_DIR / "log_reg_pipeline.pkl"
    
    if not model_path.exists():
        st.error(f"❌ Model file not found at: {model_path}")
        st.error("Please ensure model files are in the repository.")
        st.stop()
    
    try:
        return safe_pickle_load(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


@st.cache_resource
def load_explainer():
    """Load the SHAP explainer with fallback options."""
    explainer_path = MODELS_DIR / "shap_explainer.pkl"
    
    if not explainer_path.exists():
        st.warning("⚠️ SHAP explainer not found. Will generate on-the-fly (slower).")
        return None
    
    try:
        explainer = safe_pickle_load(explainer_path)
        return explainer
    except Exception as e:
        st.warning(f"⚠️ Could not load SHAP explainer: {str(e)}")
        st.info("💡 This is likely due to version incompatibility. Will generate explainer on-the-fly.")
        return None


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
# SHAP EXPLAINER GENERATION (Fallback when pickle fails)
# ============================================================================

@st.cache_resource
def generate_explainer_on_the_fly(_model, _sample_data):
    """
    Generate a SHAP explainer on-the-fly if the pickled version fails to load.
    This is slower but works across different environments.
    """
    try:
        import shap
        
        # Get preprocessed sample data
        X_sample = _model.named_steps["preprocessor"].transform(_sample_data)
        
        # Convert sparse to dense if needed
        if hasattr(X_sample, "toarray"):
            X_sample = X_sample.toarray()
        
        # Create explainer using the classifier
        classifier = _model.named_steps["classifier"]
        explainer = shap.Explainer(
            classifier.predict_proba,
            X_sample,
            feature_names=get_feature_names(_model)
        )
        
        st.success("✅ Generated SHAP explainer on-the-fly!")
        return explainer
        
    except Exception as e:
        st.error(f"❌ Could not generate SHAP explainer: {str(e)}")
        return None


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

# Load all resources
with st.spinner("Loading model and data..."):
    model = load_model()
    df = load_data()
    
    # Try to load pickled explainer first
    explainer = load_explainer()
    
    # If pickled explainer failed, generate on-the-fly
    if explainer is None:
        with st.spinner("Generating SHAP explainer... (this may take a minute)"):
            # Use a sample of the data to create explainer
            sample_size = min(100, len(df))
            sample_df = df.sample(sample_size, random_state=42).drop(columns=["churn"])
            explainer = generate_explainer_on_the_fly(model, sample_df)

# Calculate probabilities and get feature names
all_probs = get_all_churn_probabilities(model, df)
feature_names = get_feature_names(model)
pretty_feature_names = prettify_feature_names(feature_names)

st.success("✅ Model and data loaded successfully!")


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
        
        # Calculate SHAP values
        shap_values_customer = explainer(X_customer)
        
        # Handle both old and new SHAP API formats
        if hasattr(shap_values_customer, '__getitem__'):
            shap_values_customer = shap_values_customer[0]
        
        # Set feature names
        shap_values_customer.feature_names = pretty_feature_names
        
        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values_customer, show=False)
        st.pyplot(fig)
        plt.close()
        
        st.caption(
            "📊 This waterfall plot shows how each feature pushes the prediction "
            "higher (red) or lower (blue) from the baseline."
        )
        
    except Exception as e:
        st.error(f"❌ Error generating SHAP waterfall plot: {str(e)}")
        st.info("The prediction is still valid, but explainability visualization failed.")
else:
    st.info("ℹ️ SHAP explainer not available. Predictions work, but detailed explanations are limited.")


# ============================================================================
# SEGMENT-LEVEL INSIGHTS
# ============================================================================

st.subheader(f"📈 Key Churn Drivers for {segment_option} Customers")

if explainer is not None:
    try:
        import shap
        
        # Sample customers from segment
        sample_size = min(300, len(filtered_df))
        sample_df = filtered_df.sample(sample_size, random_state=42).drop(columns=["churn"])
        
        # Transform data
        X_segment = model.named_steps["preprocessor"].transform(sample_df)
        
        # Convert sparse matrix to dense if needed
        if hasattr(X_segment, "toarray"):
            X_segment = X_segment.toarray()
        
        # Calculate SHAP values
        shap_values_segment = explainer(X_segment)
        shap_values_segment.feature_names = pretty_feature_names
        
        # Create beeswarm plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.beeswarm(shap_values_segment, show=False, max_display=15)
        st.pyplot(fig)
        plt.close()
        
        st.caption(
            f"📊 This plot shows the most important features for predicting churn "
            f"across {sample_size} {segment_option.lower()} customers."
        )
        
    except Exception as e:
        st.error(f"❌ Error generating segment insights: {str(e)}")
        st.info("This can happen due to package version mismatches. Try regenerating the SHAP explainer locally.")
else:
    st.info("ℹ️ SHAP explainer not available for segment-level analysis.")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "**Note:** This dashboard uses machine learning to predict churn probability. "
    "Predictions should be used as one input among many for business decisions."
)
