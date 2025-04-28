import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define constants
TRAIN_SIZE = 10000
TEST_SIZE = 2000
CHURN_RATE = 0.15  # ~15% churn rate
MISSING_RATE = 0.05  # 5% missing values

# Define categorical value options
gender_options = ['Male', 'Female']
region_options = ['North', 'South', 'East', 'West', 'Central']
marital_status_options = ['Single', 'Married', 'Divorced', 'Widowed']
occupation_options = ['Engineer', 'Teacher', 'Doctor', 'Business', 'Student', 'Retired', 'Other']
education_options = ['High School', 'Bachelor', 'Master', 'PhD']
account_type_options = ['Savings', 'Checking', 'Business', 'Investment']
sector_options = ['Technology', 'Finance', 'Healthcare', 'Education', 'Retail', 'Other']


# Function to generate synthetic data
def generate_data(n_samples, churn_rate):
    data = {}

    # Numerical features
    data['age'] = np.random.normal(40, 10, n_samples).clip(18, 80).astype(int)
    data['dependents'] = np.random.poisson(1.5, n_samples).clip(0, 5).astype(int)
    data['account_age_months'] = np.random.exponential(36, n_samples).clip(1, 120).astype(int)
    data['num_products'] = np.random.randint(1, 6, n_samples)
    data['avg_balance'] = np.random.lognormal(mean=8, sigma=1, size=n_samples).clip(100, 100000)
    data['monthly_mobile_logins'] = np.random.poisson(10, n_samples).clip(0, 50).astype(int)
    data['atm_txns_per_month'] = np.random.poisson(5, n_samples).clip(0, 20).astype(int)
    data['monthly_deposits'] = np.random.lognormal(mean=7, sigma=1, size=n_samples).clip(0, 50000)
    data['monthly_withdrawals'] = np.random.lognormal(mean=6, sigma=1, size=n_samples).clip(0, 40000)
    data['monthly_transfers'] = np.random.lognormal(mean=5, sigma=1, size=n_samples).clip(0, 30000)
    data['loan_repayment_history'] = np.random.beta(8, 2, n_samples).clip(0, 1)  # Skewed toward good repayment
    data['complaints_count'] = np.random.poisson(0.5, n_samples).clip(0, 10).astype(int)
    data['days_since_last_complaint'] = np.random.exponential(180, n_samples).clip(0, 365).astype(int)
    data['satisfaction_rating'] = np.random.normal(3.5, 1, n_samples).clip(1, 5)
    data['monthly_fees'] = np.random.lognormal(mean=2, sigma=1, size=n_samples).clip(0, 100)

    # Categorical features
    data['gender'] = np.random.choice(gender_options, n_samples, p=[0.5, 0.5])
    data['region'] = np.random.choice(region_options, n_samples, p=[0.2, 0.2, 0.2, 0.2, 0.2])
    data['marital_status'] = np.random.choice(marital_status_options, n_samples, p=[0.4, 0.4, 0.1, 0.1])
    data['occupation'] = np.random.choice(occupation_options, n_samples, p=[0.2, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1])
    data['education'] = np.random.choice(education_options, n_samples, p=[0.3, 0.4, 0.2, 0.1])
    data['account_type'] = np.random.choice(account_type_options, n_samples, p=[0.4, 0.3, 0.2, 0.1])
    data['sector'] = np.random.choice(sector_options, n_samples, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

    # Boolean features
    data['is_dormant'] = np.random.choice([True, False], n_samples, p=[0.1, 0.9])
    data['mobile_banking_active'] = np.random.choice([True, False], n_samples, p=[0.7, 0.3])
    data['ussd_usage'] = np.random.choice([True, False], n_samples, p=[0.4, 0.6])
    data['internet_banking_active'] = np.random.choice([True, False], n_samples, p=[0.6, 0.4])
    data['account_linkage_active'] = np.random.choice([True, False], n_samples, p=[0.3, 0.7])
    data['has_rel_manager'] = np.random.choice([True, False], n_samples, p=[0.2, 0.8])

    # Generate churned (target) with some logic based on features
    # Higher churn probability for: low satisfaction, high complaints, dormant accounts, low balance
    churn_prob = (
            0.3 * (data['satisfaction_rating'] < 2.5) +
            0.2 * (data['complaints_count'] > 2) +
            0.2 * data['is_dormant'] +
            0.2 * (data['avg_balance'] < 1000) +
            0.1 * (data['account_age_months'] < 12)
    ).clip(0, 1)
    churn_prob = churn_prob / churn_prob.max() * churn_rate  # Scale to desired churn rate
    data['churned'] = np.random.binomial(1, churn_prob, n_samples)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Introduce missing values
    for col in df.columns:
        if col != 'churned':  # Don't add missing values to target
            mask = np.random.random(n_samples) < MISSING_RATE
            df.loc[mask, col] = np.nan

    return df


# Generate training and testing data
train_df = generate_data(TRAIN_SIZE, CHURN_RATE)
test_df = generate_data(TEST_SIZE, CHURN_RATE)

# Save to CSV
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

print(f"Generated training data: {train_df.shape}, Churn distribution: {train_df['churned'].value_counts().to_dict()}")
print(f"Generated testing data: {test_df.shape}, Churn distribution: {test_df['churned'].value_counts().to_dict()}")