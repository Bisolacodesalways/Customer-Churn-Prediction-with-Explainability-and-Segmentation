# Customer Churn Prediction with Explainability and Segmentation

## Overview
#### Customer churn is a critical challenge for subscription-based businesses, utilities, and service providers. This project demonstrates an end-to-end applied machine learning solution to predict customer churn, explain model decisions, and translate predictions into business-actionable insights using an interactive dashboard.

#### The focus of this project is not just prediction accuracy, but interpretability, decision support, and real-world usability.

## Business Problem
#### Organizations often struggle to identify which customers are at risk of churning early enough to intervene. This problem is particularly challenging because:

#### Churn events are rare and highly imbalanced

####  Absolute probabilities are often small but relative risk matters

#### Business teams need to understand why a customer is at risk, not just the prediction

#### This project addresses these challenges by combining predictive modeling, explainable AI, and business-aligned risk scoring.

## Data & Feature Engineering

#### The dataset represents customer behavior, billing, engagement, and contract attributes commonly found in subscription and utility environments.

### Key feature groups include:

#### Usage behavior (activity levels, session patterns)

#### Billing & payment history

#### Customer engagement & support interactions

#### Contract and plan characteristics

#### Customer segments (e.g. individual vs small business)

#### Features were engineered and validated to ensure consistency, interpretability, and suitability for explainable modeling.

## Exploratory Data Analysis (EDA)
### Target Variable Analysis
#### Churn was found to be a rare event, with a very low churn rate relative to the total customer base.

#### This confirmed that the problem is highly imbalanced, making accuracy an inappropriate primary metric.

#### As a result, evaluation focused on ROC-AUC, precision/recall, and ranking-based risk assessment rather than raw classification accuracy.

#### Behavioral & Usage Patterns

### Customers who eventually churned showed:

#### Lower and more volatile product usage

#### Reduced engagement frequency over time

#### Higher variance in session activity compared to retained customers

#### Usage-based features demonstrated clear separation between churned and retained customers when analyzed in deciles and distributions.

### Billing & Payment Insights

#### Payment irregularities (e.g. late payments, failed transactions) were strongly associated with churn risk.

#### Customers with repeated payment issues exhibited significantly higher churn propensity than those with consistent billing behavior.

#### These findings motivated the inclusion of aggregated payment-history features in the model.

### Customer Support & Engagement

####  Churned customers tended to:

####  Raise more frequent support tickets

####  Show declining satisfaction and engagement prior to churn

####  This supported the hypothesis that churn is often preceded by friction signals, not just usage decline.

### Segmentation Analysis

####  Clear behavioral differences were observed between individual customers and small business customers:

#### Small business customers generally had higher baseline usage and longer tenure

####  Individual customers showed sharper drops in engagement prior to churn

#### Segment-specific analysis justified performing segment-level explainability using SHAP.

### Correlation & Feature Relationships

####  Strong correlations were observed between:

####  Usage intensity and retention

#### Payment stability and churn likelihood

####  Multicollinearity was assessed and managed through feature selection and regularization, reinforcing the suitability of logistic regression as a baseline model.

## Modelling Approach
####  Problem Framing

####  Binary classification: Churn (1) vs Non-Churn (0)

####   Highly imbalanced target variable

####  usiness objective: rank customers by risk, not just classify

### Models Evaluated

#### Logistic Regression (baseline & final model)

#### Tree-based models (for comparison)

### Model Selection

#### Logistic Regression was selected as the final model because:

#### Strong performance on ROC-AUC

#### table behavior on imbalanced data

#### Superior interpretability for stakeholders

#### Better alignment with explainability requirements

### Evaluation Metrics

#### ROC-AUC

#### Precision / Recall

#### Confusion Matrix

#### Business-aligned risk thresholds

## Explainability (SHAP)

### Explainability is a core component of this project.

### Using SHAP (SHapley Additive exPlanations):

#### Individual-level explanations show why a specific customer is at risk

#### Segment-level analysis highlights systemic churn drivers

#### Feature names are translated into business-friendly language for non-technical stakeholders

#### This enables transparent, auditable, and trustworthy model outputs.

## Risk Scoring Logic (Business-Aligned)

#### Instead of using a naïve 0.5 probability threshold, churn risk is defined using percentile-based scoring:

#### High risk: Top 5% of predicted churn probabilities

#### Medium risk: Top 20%

#### Low risk: Remaining customers

#### This approach reflects real-world operational constraints where teams can only intervene on a limited subset of customers.

## Interactive Dashboard (Streamlit)

#### An interactive Streamlit dashboard was built to operationalize the model.

#### Dashboard Capabilities:

#### Customer-level churn probability

#### Business-friendly risk classification

#### SHAP waterfall plots for individual explanations

#### SHAP beeswarm plots for segment-level insights

#### Dynamic customer and segment selection

#### The dashboard is designed for both technical and non-technical users, enabling data-driven retention decisions.

## How to Run the App Locally
#### pip install -r requirements.txt
#### streamlit run dashboard/app.py

#### Then open:
http://localhost:8501

## Key Insights
#### Churn is best treated as a ranking and prioritization problem

#### Explainability is essential for trust and adoption

#### Percentile-based risk scoring aligns ML outputs with business capacity

#### Simple, interpretable models can outperform complex models in real-world settings

#### Dashboards bridge the gap between ML models and decision-makers

## Future Improvements
#### Probability calibration for improved interpretability

#### Time-aware churn modeling

#### Automated retraining pipelines

#### Integration with CRM or customer success tools

