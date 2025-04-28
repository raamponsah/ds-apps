import pandas as pd
import random
import numpy as np

# Helper functions
def generate_active_customer():
    return {
        "age": random.randint(18, 65),
        "gender": random.choice(["Male", "Female"]),
        "region": random.choice(["North", "South", "East", "West"]),
        "marital_status": random.choice(["Single", "Married"]),
        "occupation": random.choice(["Engineer", "Doctor", "Teacher", "Entrepreneur", "Technician"]),
        "education": random.choice(["High School", "Bachelors", "Masters", "PhD"]),
        "dependents": random.randint(0, 4),

        "account_type": random.choice(["Savings", "Current"]),
        "account_age_months": random.randint(24, 120),
        "num_products": random.randint(1, 5),
        "avg_balance": random.uniform(1000, 50000),
        "is_dormant": False,

        "mobile_banking_active": random.choices([True, False], weights=[90,10])[0],
        "monthly_mobile_logins": random.randint(20, 50),
        "ussd_usage": random.choices([True, False], weights=[80,20])[0],
        "internet_banking_active": random.choices([True, False], weights=[80,20])[0],
        "atm_txns_per_month": random.randint(5, 20),
        "account_linkage_active": random.choice([True, False]),

        "monthly_deposits": random.uniform(500, 50000),
        "monthly_withdrawals": random.uniform(100, 10000),
        "monthly_transfers": random.uniform(100, 10000),
        "loan_repayment_history": random.uniform(80, 100),

        "complaints_count": random.randint(0, 1),
        "days_since_last_complaint": random.randint(30, 365),
        "satisfaction_rating": random.uniform(4.0, 5.0),

        "has_rel_manager": random.choices([True, False], weights=[60,40])[0],
        "sector": random.choice(["Agriculture", "Manufacturing", "Services", "Healthcare"]),
        "monthly_fees": random.uniform(10, 100),
    }

def generate_unhealthy_customer():
    return {
        "age": random.randint(18, 65),
        "gender": random.choice(["Male", "Female"]),
        "region": random.choice(["North", "South", "East", "West"]),
        "marital_status": random.choice(["Single", "Married"]),
        "occupation": random.choice(["Unemployed", "Technician", "Temporary Worker"]),
        "education": random.choice(["None", "High School"]),
        "dependents": random.randint(0, 6),

        "account_type": random.choice(["Savings", "Current"]),
        "account_age_months": random.randint(0, 24),
        "num_products": random.randint(0, 2),
        "avg_balance": random.uniform(0, 1000),
        "is_dormant": True,

        "mobile_banking_active": random.choices([True, False], weights=[30,70])[0],
        "monthly_mobile_logins": random.randint(0, 10),
        "ussd_usage": random.choices([True, False], weights=[40,60])[0],
        "internet_banking_active": random.choices([True, False], weights=[30,70])[0],
        "atm_txns_per_month": random.randint(0, 5),
        "account_linkage_active": random.choice([True, False]),

        "monthly_deposits": random.uniform(0, 500),
        "monthly_withdrawals": random.uniform(0, 1000),
        "monthly_transfers": random.uniform(0, 1000),
        "loan_repayment_history": random.uniform(0, 60),

        "complaints_count": random.randint(2, 10),
        "days_since_last_complaint": random.randint(0, 60),
        "satisfaction_rating": random.uniform(0.0, 3.0),

        "has_rel_manager": random.choices([True, False], weights=[20,80])[0],
        "sector": random.choice(["Unemployed", "Retail", "Unknown"]),
        "monthly_fees": random.uniform(100, 500),
    }

# Generate data
data = []

for _ in range(20000):
    if random.random() < 0.7:
        data.append(generate_active_customer())
    else:
        data.append(generate_unhealthy_customer())

# Convert to DataFrame
df = pd.DataFrame(data)

# Save
df.to_csv('testing_customers_realistic.csv', index=False)

print("✅ 20000 realistic testing customers generated!")
