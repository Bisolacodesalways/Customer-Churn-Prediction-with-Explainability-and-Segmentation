import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
# SHAP EXPLAINER
# ============================================================================

def get_model_final_estimator(pipeline):
    """Extract the final estimator from a sklearn pipeline."""
    if hasattr(pipeline, 'steps'):
        last_step_name, last_step = pipeline.steps[-1]
        return last_step
    else:
        return pipeline


@st.cache_resource
def create_shap_explainer(_model, _sample_data):
    """Create SHAP explainer with multiple fallback strategies."""
    try:
        import shap
        
        estimator = get_model_final_estimator(_model)
        X_sample = _model.named_steps["preprocessor"].transform(_sample_data)
        
        if hasattr(X_sample, "toarray"):
            X_sample = X_sample.toarray()
        
        try:
            explainer = shap.LinearExplainer(
                estimator,
                X_sample,
                feature_perturbation="interventional"
            )
            st.success("✅ Created LinearExplainer for fast, accurate explanations")
            return explainer, X_sample
        except Exception as e1:
            try:
                def model_predict(X):
                    proba = estimator.predict_proba(X)
                    return proba[:, 1] if len(proba.shape) > 1 and proba.shape[1] > 1 else proba
                
                explainer = shap.KernelExplainer(
                    model_predict,
                    X_sample[:50]
                )
                st.success("✅ Created KernelExplainer")
                return explainer, X_sample
            except Exception as e2:
                try:
                    explainer = shap.Explainer(
                        estimator.predict_proba,
                        X_sample[:50]
                    )
                    st.success("✅ Created auto-detected explainer")
                    return explainer, X_sample
                except Exception as e3:
                    st.warning(f"⚠️ Could not create SHAP explainer")
                    return None, None
        
    except Exception as e:
        st.warning(f"⚠️ SHAP initialization failed: {str(e)}")
        return None, None


# ============================================================================
# PLOTLY VISUALIZATION FUNCTIONS
# ============================================================================

def create_waterfall_plot(shap_values, feature_values, feature_names, base_value):
    """
    Create an interactive Plotly waterfall plot for SHAP values.
    """
    # Get top features by absolute SHAP value
    top_n = 15
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[-top_n:][::-1]
    
    # Prepare data
    features = [feature_names[i] for i in top_indices]
    values = [shap_values[i] for i in top_indices]
    
    # Create waterfall data
    cumsum = [base_value]
    for val in values:
        cumsum.append(cumsum[-1] + val)
    
    # Create figure
    fig = go.Figure()
    
    # Add bars
    colors = ['#FF4B4B' if v > 0 else '#4B8BFF' for v in values]
    
    fig.add_trace(go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=["relative"] * len(values) + ["total"],
        x=features + ["Prediction"],
        y=values + [cumsum[-1] - base_value],
        text=[f"{v:+.3f}" for v in values] + [f"{cumsum[-1]:.3f}"],
        textposition="outside",
        connector={"line": {"color": "rgb(200, 200, 200)"}},
        increasing={"marker": {"color": "#FF4B4B"}},
        decreasing={"marker": {"color": "#4B8BFF"}},
        totals={"marker": {"color": "#9467BD"}}
    ))
    
    fig.update_layout(
        title="Feature Impact on Churn Prediction (Waterfall)",
        xaxis_title="Features",
        yaxis_title="SHAP Value (impact on prediction)",
        height=600,
        showlegend=False,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.update_xaxes(tickangle=-45)
    
    return fig


def create_feature_importance_plot(shap_values_array, feature_names):
    """
    Create an interactive Plotly feature importance plot.
    """
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.mean(np.abs(shap_values_array), axis=0)
    
    # Sort by importance
    sorted_indices = np.argsort(mean_abs_shap)[-15:]  # Top 15
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_values = mean_abs_shap[sorted_indices]
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sorted_values,
        y=sorted_features,
        orientation='h',
        marker=dict(
            color=sorted_values,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Mean |SHAP|")
        ),
        text=[f"{v:.3f}" for v in sorted_values],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Top 15 Features by Importance (Mean Absolute SHAP)",
        xaxis_title="Mean Absolute SHAP Value",
        yaxis_title="Features",
        height=600,
        showlegend=False,
        hovermode='y unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_beeswarm_plot(shap_values_array, feature_values_array, feature_names):
    """
    Create an interactive scatter/beeswarm plot showing SHAP value distribution.
    """
    # Get top features
    mean_abs_shap = np.mean(np.abs(shap_values_array), axis=0)
    top_indices = np.argsort(mean_abs_shap)[-15:][::-1]
    
    # Create subplots for each feature
    fig = go.Figure()
    
    for idx, feat_idx in enumerate(top_indices):
        feature_name = feature_names[feat_idx]
        shap_vals = shap_values_array[:, feat_idx]
        feat_vals = feature_values_array[:, feat_idx]
        
        # Add jitter for visualization
        y_jitter = np.random.normal(0, 0.04, size=len(shap_vals))
        
        fig.add_trace(go.Scatter(
            x=shap_vals,
            y=[idx] * len(shap_vals) + y_jitter,
            mode='markers',
            marker=dict(
                color=feat_vals,
                colorscale='RdBu_r',
                size=4,
                opacity=0.6,
                showscale=idx == 0,
                colorbar=dict(
                    title="Feature<br>Value",
                    x=1.1
                )
            ),
            name=feature_name,
            hovertemplate=f'<b>{feature_name}</b><br>SHAP: %{{x:.4f}}<br>Value: %{{marker.color:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Feature Impact Distribution (Beeswarm Plot)",
        xaxis_title="SHAP Value (impact on model output)",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(top_indices))),
            ticktext=[feature_names[i] for i in top_indices]
        ),
        height=600,
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_risk_gauge(churn_proba):
    """Create an interactive gauge chart for churn risk."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=churn_proba * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Risk", 'font': {'size': 24}},
        delta={'reference': 50, 'suffix': "%"},
        gauge={
            'axis': {'range': [None, 100], 'ticksuffix': "%"},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "#90EE90"},
                {'range': [33, 66], 'color': "#FFD700"},
                {'range': [66, 100], 'color': "#FF6B6B"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_feature_names(model):
    """Extract feature names from the model pipeline."""
    preprocessor = model.named_steps["preprocessor"]
    
    num_features = preprocessor.transformers_[0][2]
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

st.title(" Customer Churn Prediction Dashboard")
st.markdown(
    "This dashboard predicts customer churn risk and explains **why** "
    "a customer is likely to churn using SHAP and interactive visualizations."
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

st.sidebar.header(" Segmentation")

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

st.sidebar.metric("Customers in Segment", f"{len(filtered_df):,}")
st.sidebar.metric("Average Churn Rate", f"{filtered_df['churn'].mean():.1%}")


# ============================================================================
# MAIN CONTENT - CUSTOMER SELECTION & PREDICTION
# ============================================================================

st.subheader("🔍 Individual Customer Analysis")

customer_idx = st.selectbox(
    "Select a customer index",
    options=filtered_df.index.tolist(),
    help="Choose a customer to analyze their churn risk"
)

# Get customer data and make prediction
customer_data = df.loc[[customer_idx]].drop(columns=["churn"])
churn_proba = model.predict_proba(customer_data)[0, 1]

# Calculate risk thresholds
high_risk_threshold = np.quantile(all_probs, 0.95)
medium_risk_threshold = np.quantile(all_probs, 0.80)

# Display prediction with interactive gauge
col1, col2 = st.columns([1, 2])

with col1:
    fig_gauge = create_risk_gauge(churn_proba)
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    st.markdown("### Prediction Details")
    
    col2_1, col2_2, col2_3 = st.columns(3)
    
    with col2_1:
        st.metric(
            label="Churn Probability",
            value=f"{churn_proba:.1%}"
        )
    
    with col2_2:
        base_rate = df["churn"].mean()
        lift = churn_proba / base_rate
        st.metric(
            label="Risk vs Average",
            value=f"{lift:.1f}×"
        )
    
    with col2_3:
        if churn_proba >= high_risk_threshold:
            st.error("🔴 **High Risk**")
            st.caption("Top 5%")
        elif churn_proba >= medium_risk_threshold:
            st.warning("🟠 **Medium Risk**")
            st.caption("Top 20%")
        else:
            st.success("🟢 **Low Risk**")
            st.caption("Below average")


# ============================================================================
# SHAP VISUALIZATIONS - INDIVIDUAL CUSTOMER
# ============================================================================

if explainer is not None:
    st.subheader(" What's driving this prediction?")
    
    try:
        import shap
        
        # Transform customer data
        X_customer = model.named_steps["preprocessor"].transform(customer_data)
        
        if hasattr(X_customer, "toarray"):
            X_customer = X_customer.toarray()
        
        # Calculate SHAP values
        try:
            shap_values = explainer.shap_values(X_customer)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            if isinstance(explainer.expected_value, (list, np.ndarray)):
                expected_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
            else:
                expected_value = explainer.expected_value
            
            shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            
        except AttributeError:
            shap_result = explainer(X_customer)
            shap_vals = shap_result[0].values
            expected_value = shap_result[0].base_values
        
        # Create interactive waterfall plot
        fig_waterfall = create_waterfall_plot(
            shap_vals,
            X_customer[0],
            pretty_feature_names,
            expected_value
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        st.caption(
            " **Interactive Waterfall Plot** - Hover over bars to see exact values. "
            "Red bars push prediction higher (toward churn), blue bars push it lower."
        )
        
    except Exception as e:
        st.error(f"❌ Error generating SHAP visualization: {str(e)}")
else:
    st.info("ℹ️ SHAP visualizations are not available.")


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
        
        if hasattr(X_segment, "toarray"):
            X_segment = X_segment.toarray()
        
        # Calculate SHAP values
        with st.spinner(f"Analyzing {sample_size} customers..."):
            try:
                shap_values = explainer.shap_values(X_segment)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
            except AttributeError:
                shap_result = explainer(X_segment)
                shap_values = shap_result.values
        
        # Create tabs for different views
        tab1, tab2 = st.tabs([" Beeswarm Plot", " Feature Importance"])
        
        with tab1:
            fig_beeswarm = create_beeswarm_plot(
                shap_values,
                X_segment,
                pretty_feature_names
            )
            st.plotly_chart(fig_beeswarm, use_container_width=True)
            st.caption(
                f" **Interactive Beeswarm Plot** - Shows distribution of feature impacts across {sample_size} customers. "
                "Color indicates feature value (red = high, blue = low). Hover to see details!"
            )
        
        with tab2:
            fig_importance = create_feature_importance_plot(
                shap_values,
                pretty_feature_names
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            st.caption(
                " **Feature Importance** - Average absolute impact of each feature on predictions. "
                "Higher values indicate stronger influence on churn prediction."
            )
        
    except Exception as e:
        st.error(f"❌ Error generating segment insights: {str(e)}")
else:
    st.info("ℹ️ SHAP segment analysis not available.")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "**Note:** This dashboard uses machine learning to predict churn probability. "
    "Predictions should be used as one input among many for business decisions. "
    "All visualizations are interactive - hover, zoom, and pan to explore!"
)
