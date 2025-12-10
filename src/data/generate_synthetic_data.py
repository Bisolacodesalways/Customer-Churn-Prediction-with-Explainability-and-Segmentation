import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import random
import os

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# ------------------------------
# CONFIGURATION
# ------------------------------
N_CUSTOMERS = 10000
MONTHS = 24
START_DATE = datetime(2022, 1, 1)

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)


# ------------------------------
# 1. Generate Customer Table
# ------------------------------
def generate_customers(n_customers):
    provinces = ["ON", "BC", "AB", "QC", "MB", "SK", "NS", "NB"]
    acquisition_channels = ["Organic", "Referral", "Paid Ads", "Sales Team"]
    segments = ["Individual", "Small Business", "Enterprise"]

    customers = []

    for i in range(n_customers):
        signup_date = fake.date_between_dates(
            date_start=datetime(2020, 1, 1),
            date_end=datetime(2022, 12, 31)
        )

        customers.append({
            "customer_id": i + 1,
            "signup_date": signup_date,
            "age": np.random.randint(18, 75),
            "province": random.choice(provinces),
            "segment": random.choice(segments),
            "acquisition_channel": random.choice(acquisition_channels)
        })

    df = pd.DataFrame(customers)
    df.to_csv(f"{DATA_DIR}/customers.csv", index=False)
    return df


# ------------------------------
# 2. Generate Subscription Table
# ------------------------------
def generate_subscriptions(customers):
    plan_types = ["Basic", "Standard", "Premium"]
    contract_types = ["Monthly", "Annual"]

    subs = []
    for _, row in customers.iterrows():

        plan = random.choice(plan_types)
        contract = random.choice(contract_types)
        price_map = {"Basic": 20, "Standard": 40, "Premium": 70}

        subs.append({
            "customer_id": row.customer_id,
            "plan_type": plan,
            "contract_type": contract,
            "price_per_month": price_map[plan],
            "status": "Active",
            "churn_date": None
        })

    df = pd.DataFrame(subs)
    df.to_csv(f"{DATA_DIR}/subscriptions.csv", index=False)
    return df


# ------------------------------
# 3. Generate Usage Table (24 Months)
# ------------------------------
def generate_usage(customers):
    usage_rows = []

    for _, row in customers.iterrows():
        for m in range(MONTHS):
            date = START_DATE + pd.DateOffset(months=m)

            # Usage declines naturally over time for some users
            base_usage = np.random.normal(30, 10)
            trend = np.random.normal(0, 1) * (m / MONTHS)

            login_count = max(0, int(base_usage + trend + np.random.normal(0, 3)))
            active_days = max(0, int(np.random.normal(10, 4)))
            usage_hours = max(0, np.random.normal(20, 8))
            features_used = max(1, int(np.random.normal(4, 2)))

            heavy_feature_usage = np.clip(np.random.beta(2, 5), 0, 1)

            usage_rows.append({
                "customer_id": row.customer_id,
                "month": date.strftime("%Y-%m"),
                "login_count": login_count,
                "active_days": active_days,
                "total_usage_hours": usage_hours,
                "num_features_used": features_used,
                "heavy_feature_usage": heavy_feature_usage,
                "num_sessions_mobile": max(0, int(np.random.normal(8, 3))),
                "num_sessions_web": max(0, int(np.random.normal(15, 5))),
            })

    df = pd.DataFrame(usage_rows)
    df.to_csv(f"{DATA_DIR}/usage_monthly.csv", index=False)
    return df


# ------------------------------
# 4. Generate Billing Table
# ------------------------------
def generate_billing(customers, subscriptions):
    billing_rows = []

    for _, c in customers.iterrows():
        plan_price = subscriptions.loc[
            subscriptions.customer_id == c.customer_id, "price_per_month"
        ].values[0]

        for m in range(MONTHS):
            invoice_date = START_DATE + pd.DateOffset(months=m)

            # Payment behaviour affects churn
            days_late = max(0, int(np.random.normal(2, 3)))
            was_failed_payment = np.random.rand() < 0.05  # 5% fail rate

            amount_paid = plan_price if not was_failed_payment else 0

            billing_rows.append({
                "customer_id": c.customer_id,
                "invoice_date": invoice_date.strftime("%Y-%m-%d"),
                "amount_due": plan_price,
                "amount_paid": amount_paid,
                "days_late": days_late,
                "payment_method": random.choice(["Card", "Bank", "PayPal"]),
                "was_failed_payment": was_failed_payment,
            })

    df = pd.DataFrame(billing_rows)
    df.to_csv(f"{DATA_DIR}/billing.csv", index=False)
    return df


# ------------------------------
# 5. Generate Support Tickets
# ------------------------------
def generate_support_tickets(customers):
    tickets = []
    issue_types = ["Billing", "Technical", "Onboarding", "Cancellation Inquiry"]

    for _, c in customers.iterrows():
        # Some customers will open many tickets, others none
        num_tickets = np.random.poisson(1.2)

        for _ in range(num_tickets):
            date = fake.date_between_dates(
                date_start=START_DATE,
                date_end=START_DATE + pd.DateOffset(months=MONTHS)
            )

            tickets.append({
                "ticket_id": fake.uuid4(),
                "customer_id": c.customer_id,
                "ticket_date": date,
                "issue_type": random.choice(issue_types),
                "priority": random.choice(["Low", "Medium", "High"]),
                "resolution_time_hours": np.random.exponential(24),
                "satisfaction_score": np.random.randint(1, 6),
            })

    df = pd.DataFrame(tickets)
    df.to_csv(f"{DATA_DIR}/support_tickets.csv", index=False)
    return df


# ------------------------------
# RUN ALL GENERATORS
# ------------------------------
if __name__ == "__main__":
    print("Generating customers...")
    customers = generate_customers(N_CUSTOMERS)

    print("Generating subscriptions...")
    subscriptions = generate_subscriptions(customers)

    print("Generating usage data...")
    usage = generate_usage(customers)

    print("Generating billing data...")
    billing = generate_billing(customers, subscriptions)

    print("Generating support tickets...")
    tickets = generate_support_tickets(customers)

    print("Done! Raw data saved to data/raw/")
