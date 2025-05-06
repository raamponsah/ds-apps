from django.core.paginator import Paginator
from django.db.models import Q
from django.shortcuts import render
from django.views.generic import TemplateView

from churn_app.models import TestingCustomerData
import random
import json


# Create your views here.
class CustomerChurnView(TemplateView):
    template_name = "churn_app/dashboard.html"


def interactive_churned_view(request):
    keys = [
        "age", "gender", "region", "marital_status", "occupation", "education",
        "dependents", "account_type", "account_age_months", "num_products", "avg_balance",
        "is_dormant", "mobile_banking_active", "monthly_mobile_logins", "ussd_usage",
        "internet_banking_active", "atm_txns_per_month", "account_linkage_active",
        "monthly_deposits", "monthly_withdrawals", "monthly_transfers", "loan_repayment_history",
        "complaints_count", "days_since_last_complaint", "satisfaction_rating",
        "has_rel_manager", "sector", "monthly_fees", "churned",
    ]

    region_filter = request.GET.get('region')
    gender_filter = request.GET.get('gender')
    records = TestingCustomerData.objects.all()
    search_query = request.GET.get('search', '').strip()

    if region_filter:
        records = records.filter(region=str(region_filter))

    if gender_filter:
        records = records.filter(gender=str(gender_filter))

    if search_query:
        records = records.filter(
            Q(gender__iexact=gender_filter) |
            Q(marital_status__icontains=search_query) |
            Q(occupation__iexact=search_query) |
            Q(education__icontains=search_query) |
            Q(account_type__icontains=search_query) |
            Q(sector__icontains=search_query)
        )

    print("Search query:", str(search_query))
    print("Region filter:", str(region_filter))
    print("Final SQL Query:", records.query)

    print(f"Search query: {search_query}")
    regions = TestingCustomerData.objects.values_list('region', flat=True).distinct()
    genders = TestingCustomerData.objects.values_list('gender', flat=True).distinct()
    sample_data = {key: i * random.randint(10, 50) for i, key in enumerate(keys)}

    paginator = Paginator(records, 50)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, "churn_app/dashboard.html", {
        "keys": json.dumps(keys),
        "data": json.dumps(sample_data),
        "page_obj": page_obj,
        "regions": regions,
        "genders": genders,
        "selected_region": str(region_filter),
        "selected_gender": str(gender_filter),
        "search_query": str(search_query),
    })



# Define possible categories for categorical fields
genders = ["Male", "Female"]
regions = ["North", "South", "East", "West"]
marital_statuses = ["Single", "Married", "Divorced", "Widowed"]
occupations = ["Engineer", "Teacher", "Doctor", "Trader", "Banker", "Unemployed"]
education_levels = ["None", "Primary", "Secondary", "Tertiary"]
account_types = ["Savings", "Current", "Fixed Deposit"]
sectors = ["Agriculture", "Finance", "Health", "Education", "Technology"]
bool_values = [0, 1]

# Generate a random record
record = {
    "age": random.randint(18, 75),
    "gender": random.choice(genders),
    "region": random.choice(regions),
    "marital_status": random.choice(marital_statuses),
    "occupation": random.choice(occupations),
    "education": random.choice(education_levels),
    "dependents": random.randint(0, 5),
    "account_type": random.choice(account_types),
    "account_age_months": random.randint(1, 240),
    "num_products": random.randint(1, 5),
    "avg_balance": round(random.uniform(100.0, 100000.0), 2),
    "is_dormant": random.choice(bool_values),
    "mobile_banking_active": random.choice(bool_values),
    "monthly_mobile_logins": random.randint(0, 100),
    "ussd_usage": random.randint(0, 50),
    "internet_banking_active": random.choice(bool_values),
    "atm_txns_per_month": random.randint(0, 30),
    "account_linkage_active": random.choice(bool_values),
    "monthly_deposits": round(random.uniform(0, 50000.0), 2),
    "monthly_withdrawals": round(random.uniform(0, 50000.0), 2),
    "monthly_transfers": round(random.uniform(0, 30000.0), 2),
    "loan_repayment_history": round(random.uniform(0, 1), 2),
    "complaints_count": random.randint(0, 10),
    "days_since_last_complaint": random.randint(0, 365),
    "satisfaction_rating": round(random.uniform(1.0, 5.0), 1),
    "has_rel_manager": random.choice(bool_values),
    "sector": random.choice(sectors),
    "monthly_fees": round(random.uniform(0, 1000.0), 2),
    "churned": 1  # Since we're generating for closed accounts
}


# Function to generate multiple randomized records
def generate_records(n=10):
    records = []
    for _ in range(n):
        record = {
            "age": random.randint(18, 75),
            "gender": random.choice(genders),
            "region": random.choice(regions),
            "marital_status": random.choice(marital_statuses),
            "occupation": random.choice(occupations),
            "education": random.choice(education_levels),
            "dependents": random.randint(0, 5),
            "account_type": random.choice(account_types),
            "account_age_months": random.randint(1, 240),
            "num_products": random.randint(1, 5),
            "avg_balance": round(random.uniform(100.0, 100000.0), 2),
            "is_dormant": random.choice(bool_values),
            "mobile_banking_active": random.choice(bool_values),
            "monthly_mobile_logins": random.randint(0, 100),
            "ussd_usage": random.randint(0, 50),
            "internet_banking_active": random.choice(bool_values),
            "atm_txns_per_month": random.randint(0, 30),
            "account_linkage_active": random.choice(bool_values),
            "monthly_deposits": round(random.uniform(0, 50000.0), 2),
            "monthly_withdrawals": round(random.uniform(0, 50000.0), 2),
            "monthly_transfers": round(random.uniform(0, 30000.0), 2),
            "loan_repayment_history": round(random.uniform(0, 1), 2),
            "complaints_count": random.randint(0, 10),
            "days_since_last_complaint": random.randint(0, 365),
            "satisfaction_rating": round(random.uniform(1.0, 5.0), 1),
            "has_rel_manager": random.choice(bool_values),
            "sector": random.choice(sectors),
            "monthly_fees": round(random.uniform(0, 1000.0), 2),
            "churned": 1  # Closed account
        }
        records.append(record)
    return records

# Generate 10 sample records
