import pandas as pd
import numpy as np
from faker import Faker
from sklearn.datasets import make_classification
from datetime import date, timedelta
import random

# Initialize faker for realistic fake data
fake = Faker()


def generate_customer_data(num_samples=10000, churn_rate=0.2, random_state=42):
    """Generate synthetic customer data with realistic churn patterns"""
    np.random.seed(random_state)
    random.seed(random_state)

    # Base customer characteristics
    data = {
        'customer_id': [fake.uuid4() for _ in range(num_samples)],
        'age': np.random.randint(18, 80, size=num_samples),
        'gender': np.random.choice(['Male', 'Female', 'Other'], size=num_samples, p=[0.48, 0.5, 0.02]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], size=num_samples),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'],
                                           size=num_samples, p=[0.3, 0.5, 0.15, 0.05]),
        'occupation': np.random.choice([
            'Employed', 'Self-employed', 'Student', 'Retired', 'Unemployed'
        ], size=num_samples, p=[0.6, 0.15, 0.1, 0.1, 0.05]),
        'education': np.random.choice([
            'High School', 'College', 'Bachelor', 'Master', 'PhD'
        ], size=num_samples, p=[0.2, 0.3, 0.3, 0.15, 0.05]),
        'dependents': np.random.poisson(1.2, size=num_samples),
        'account_type': np.random.choice(['Savings', 'Checking', 'Premium'],
                                         size=num_samples, p=[0.6, 0.35, 0.05]),
        'account_age_months': np.random.randint(1, 120, size=num_samples),
        'num_products': np.random.randint(1, 5, size=num_samples),
        'avg_balance': np.round(np.abs(np.random.normal(5000, 3000, size=num_samples)), 2),
        'is_dormant': np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1]),
        'mobile_banking_active': np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7]),
        'monthly_mobile_logins': np.random.poisson(8, size=num_samples),
        'ussd_usage': np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6]),
        'internet_banking_active': np.random.choice([0, 1], size=num_samples, p=[0.5, 0.5]),
        'atm_txns_per_month': np.random.poisson(3, size=num_samples),
        'account_linkage_active': np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3]),
        'monthly_deposits': np.round(np.abs(np.random.normal(1500, 800, size=num_samples)), 2),
        'monthly_withdrawals': np.round(np.abs(np.random.normal(1200, 600, size=num_samples)), 2),
        'monthly_transfers': np.round(np.abs(np.random.normal(800, 400, size=num_samples)), 2),
        'loan_repayment_history': np.random.uniform(0, 1, size=num_samples),
        'complaints_count': np.random.poisson(0.5, size=num_samples),
        'days_since_last_complaint': np.random.choice(
            [np.random.randint(1, 365) for _ in range(num_samples)] + [365 * 2],
            size=num_samples, p=[0.3 / num_samples] * num_samples + [0.7]),
        'satisfaction_rating': np.random.randint(1, 6, size=num_samples),
        'has_rel_manager': np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2]),
        'sector': np.random.choice([
            'Retail', 'Technology', 'Healthcare', 'Education', 'Manufacturing', 'Other'
        ], size=num_samples),
        'monthly_fees': np.round(np.abs(np.random.normal(10, 5, size=num_samples)), 2)
    }

    df = pd.DataFrame(data)

    # Generate realistic churn labels with meaningful patterns
    churn_factors = (
            0.3 * (df['satisfaction_rating'] == 1) +
            0.2 * (df['complaints_count'] > 2) +
            0.15 * (df['is_dormant'] == 1) +
            0.1 * (df['mobile_banking_active'] == 0) +
            0.05 * (df['avg_balance'] < 1000) +
            0.05 * (df['days_since_last_complaint'] < 30) +
            0.15 * np.random.uniform(0, 1, size=num_samples)
    )

    # Normalize and apply churn rate
    churn_prob = churn_factors / churn_factors.max() * churn_rate * 1.5
    df['churned'] = np.random.binomial(1, churn_prob)

    # Adjust actual churn rate to match target
    current_rate = df['churned'].mean()
    if current_rate > 0:
        adjustment = churn_rate / current_rate
        churn_prob = np.minimum(churn_prob * adjustment, 0.95)
        df['churned'] = np.random.binomial(1, churn_prob)

    return df


def create_train_test_datasets():
    """Generate and save training and test datasets"""
    # Generate full dataset
    full_data = generate_customer_data(num_samples=15000, churn_rate=0.25)

    # Split into train and test (80/20)
    train_data = full_data.sample(frac=0.8, random_state=42)
    test_data = full_data.drop(train_data.index)

    # Save to CSV files
    train_data.to_csv('churn_train_dataset.csv', index=False)
    test_data.to_csv('churn_test_dataset.csv', index=False)

    print(f"Training dataset created with {len(train_data)} samples")
    print(f"Test dataset created with {len(test_data)} samples")
    print(f"Training churn rate: {train_data['churned'].mean():.2%}")
    print(f"Test churn rate: {test_data['churned'].mean():.2%}")


if __name__ == "__main__":
    create_train_test_datasets()