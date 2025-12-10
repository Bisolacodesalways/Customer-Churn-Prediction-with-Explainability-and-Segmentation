import pandas as pd
import numpy as np
import os

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


def load_data():
    customers = pd.read_csv(f"{RAW_DATA_DIR}/customers.csv")
    subs = pd.read_csv(f"{RAW_DATA_DIR}/subscriptions.csv")
    usage = pd.read_csv(f"{RAW_DATA_DIR}/usage_monthly.csv")
    billing = pd.read_csv(f"{RAW_DATA_DIR}/billing.csv")
    tickets = pd.read_csv(f"{RAW_DATA_DIR}/support_tickets.csv")

    usage["month"] = pd.to_datetime(usage["month"])
    billing["invoice_date"] = pd.to_datetime(billing["invoice_date"])
    tickets["ticket_date"] = pd.to_datetime(tickets["ticket_date"])

    return customers, subs, usage, billing, tickets


def engineer_usage_features(usage):
    """
    Aggregate 24 months of usage into meaningful customer-level features
    """
    usage_agg = usage.groupby("customer_id").agg({
        "login_count": ["mean", "std", "last"],
        "active_days": ["mean", "std"],
        "total_usage_hours": ["mean", "std", "last"],
        "num_features_used": ["mean"],
        "heavy_feature_usage": ["mean"],
        "num_sessions_mobile": ["mean"],
        "num_sessions_web": ["mean"]
    })

    usage_agg.columns = ["_".join(col).strip() for col in usage_agg.columns]
    usage_agg.reset_index(inplace=True)
    return usage_agg


def engineer_billing_features(billing):
    """
    Payment behavior features (strong drivers of churn)
    """
    billing["failed_payment"] = billing["was_failed_payment"].astype(int)

    billing_agg = billing.groupby("customer_id").agg({
        "days_late": ["mean", "max"],
        "failed_payment": ["sum"],
        "amount_paid": ["mean"],
    })

    billing_agg.columns = ["_".join(col) for col in billing_agg.columns]
    billing_agg.reset_index(inplace=True)
    return billing_agg


def engineer_support_features(tickets):
    """
    Support interaction features
    """
    tickets["ticket_count"] = 1

    support_agg = tickets.groupby("customer_id").agg({
        "ticket_count": "sum",
        "satisfaction_score": "mean",
        "resolution_time_hours": "mean",
    })

    support_agg.columns = [
        "ticket_count",
        "avg_satisfaction_score",
        "avg_resolution_time_hours"
    ]

    support_agg.reset_index(inplace=True)
    return support_agg


def create_churn_label(subs, usage, billing):
    """
    Churn definition:
    - Subscription cancelled
    - OR no usage for last 2 months
    - OR 2+ failed payments in last 3 months
    """
    churn_df = pd.DataFrame({"customer_id": subs["customer_id"]})

    # 1. Subscription status churn
    churn_df["cancelled"] = subs["status"].apply(lambda x: 1 if x == "Cancelled" else 0)

    # 2. No usage in last 2 months
    last_two = usage.groupby("customer_id").tail(2).groupby("customer_id")["login_count"].sum()
    churn_df["no_usage_2mo"] = (last_two == 0).astype(int).reindex(churn_df["customer_id"], fill_value=0).values

    # 3. Failed payments
    last_three = billing.groupby("customer_id").tail(3).groupby("customer_id")["was_failed_payment"].sum()
    churn_df["failed_payments_3mo"] = (last_three >= 2).astype(int).reindex(churn_df["customer_id"], fill_value=0).values

    churn_df["churn"] = churn_df[["cancelled", "no_usage_2mo", "failed_payments_3mo"]].max(axis=1)

    return churn_df


def build_dataset():
    customers, subs, usage, billing, tickets = load_data()

    print("Engineering usage features...")
    usage_features = engineer_usage_features(usage)

    print("Engineering billing features...")
    billing_features = engineer_billing_features(billing)

    print("Engineering support features...")
    support_features = engineer_support_features(tickets)

    print("Creating churn label...")
    churn_label = create_churn_label(subs, usage, billing)

    print("Merging all features...")
    df = (customers
          .merge(subs, on="customer_id", how="left")
          .merge(usage_features, on="customer_id", how="left")
          .merge(billing_features, on="customer_id", how="left")
          .merge(support_features, on="customer_id", how="left")
          .merge(churn_label, on="customer_id", how="left")
          )

    print("Saving dataset to data/processed/model_dataset.csv")
    df.to_csv(f"{PROCESSED_DATA_DIR}/model_dataset.csv", index=False)

    print("Done! Your modeling dataset is ready.")
    return df


if __name__ == "__main__":
    build_dataset()
