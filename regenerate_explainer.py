"""
Script to regenerate SHAP explainer with compatible pickle protocol.
Run this locally to create a new shap_explainer.pkl file that works on Streamlit Cloud.
"""

import joblib
import pandas as pd
from pathlib import Path
import shap
import pickle

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

print("Loading model...")
model = joblib.load(MODELS_DIR / "log_reg_pipeline.pkl")

print("Loading data...")
df = pd.read_csv(DATA_DIR / "model_dataset.csv")

# Sample data for explainer
print("Creating sample data...")
sample_size = 100
sample_df = df.sample(sample_size, random_state=42).drop(columns=["churn"])

# Transform data
print("Transforming data...")
X_sample = model.named_steps["preprocessor"].transform(sample_df)

if hasattr(X_sample, "toarray"):
    X_sample = X_sample.toarray()

# Create SHAP explainer
print("Creating SHAP explainer...")
classifier = model.named_steps["classifier"]

# Use Explainer with predict_proba
explainer = shap.Explainer(
    classifier.predict_proba,
    X_sample
)

# Save with compatible protocol
print("Saving explainer with pickle protocol 4...")
output_path = MODELS_DIR / "shap_explainer.pkl"

# Save with protocol 4 (compatible with Python 3.7+)
with open(output_path, 'wb') as f:
    pickle.dump(explainer, f, protocol=4)

print(f"✅ SHAP explainer saved successfully to: {output_path}")
print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

# Verify it can be loaded
print("\nVerifying the saved explainer...")
try:
    loaded_explainer = joblib.load(output_path)
    print("✅ Explainer loads successfully!")
except Exception as e:
    print(f"❌ Error loading explainer: {e}")
