from django.db import models

from ml_models.churning.churn_prediction_ml import get_verdict_from_f1, train_churn_model_from_file


class TrainingCustomerData(models.Model):
    # Demographics
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    region = models.CharField(max_length=100)
    marital_status = models.CharField(max_length=20)
    occupation = models.CharField(max_length=100)
    education = models.CharField(max_length=100)
    dependents = models.IntegerField()

    # Account Info
    account_type = models.CharField(max_length=50)
    account_age_months = models.IntegerField()
    num_products = models.IntegerField()
    avg_balance = models.FloatField()
    is_dormant = models.BooleanField()

    # Digital Engagement
    mobile_banking_active = models.BooleanField()
    monthly_mobile_logins = models.IntegerField()
    ussd_usage = models.BooleanField()
    internet_banking_active = models.BooleanField()
    atm_txns_per_month = models.IntegerField()
    account_linkage_active = models.BooleanField()

    # Transactions
    monthly_deposits = models.FloatField()
    monthly_withdrawals = models.FloatField()
    monthly_transfers = models.FloatField()
    loan_repayment_history = models.FloatField()  # e.g., repayment rate %


    # Service Interaction
    complaints_count = models.IntegerField()
    days_since_last_complaint = models.IntegerField()
    satisfaction_rating = models.FloatField()
    # branch_visits = models.IntegerField()

    # Bank Info
    has_rel_manager = models.BooleanField()
    sector = models.CharField(max_length=50)
    monthly_fees = models.FloatField()
    # cross_sell_success = models.BooleanField()

    # Target
    churned = models.BooleanField()  # 1 = churned, 0 = not churned

    def __str__(self):
        return f"{self.id} - {self.account_type} ({'Churned' if self.churned else 'Active'})"

class TestingCustomerData(models.Model):
    # Demographics
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    region = models.CharField(max_length=100)
    marital_status = models.CharField(max_length=20)
    occupation = models.CharField(max_length=100)
    education = models.CharField(max_length=100)
    dependents = models.IntegerField()

    # Account Info
    account_type = models.CharField(max_length=50)
    account_age_months = models.IntegerField()
    num_products = models.IntegerField()
    avg_balance = models.FloatField()
    is_dormant = models.BooleanField()

    # Digital Engagement
    mobile_banking_active = models.BooleanField()
    monthly_mobile_logins = models.IntegerField()
    ussd_usage = models.BooleanField()
    internet_banking_active = models.BooleanField()
    atm_txns_per_month = models.IntegerField()
    account_linkage_active = models.BooleanField()

    # Transactions
    monthly_deposits = models.FloatField()
    monthly_withdrawals = models.FloatField()
    monthly_transfers = models.FloatField()
    loan_repayment_history = models.FloatField()  # e.g., repayment rate %


    # Service Interaction
    complaints_count = models.IntegerField()
    days_since_last_complaint = models.IntegerField()
    satisfaction_rating = models.FloatField()
    # branch_visits = models.IntegerField()

    # Bank Info
    has_rel_manager = models.BooleanField()
    sector = models.CharField(max_length=50)
    monthly_fees = models.FloatField()
    # cross_sell_success = models.BooleanField()

    # Target
    churned = models.BooleanField(default=None)  # 1 = churned, 0 = not churned
    churn_probability = models.FloatField(default=0.0)  # Add this field

    def __str__(self):
        return f"{self.id} - {self.account_type} ({'Churned' if self.churned else 'Active'})"

class ChurnModelTrainer(models.Model):
    dataset = models.FileField(upload_to='datasets/', help_text="Upload your training dataset here.")
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    trained_on = models.DateTimeField(auto_now_add=True)
    verdict = models.CharField(max_length=20, null=True, blank=True)  # New field


    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)  # Save first to get the file
        file_path = self.dataset.path
        metrics = train_churn_model_from_file(file_path)

        self.accuracy = metrics['accuracy']
        self.precision = metrics['precision']
        self.recall = metrics['recall']
        self.f1_score = metrics['f1_score']
        self.verdict = get_verdict_from_f1(self.f1_score)
        super().save(update_fields=["accuracy", "precision", "recall", "f1_score","verdict"])

    def __str__(self):
        return f"{self.id} - {self.verdict} - trained on: {self.trained_on}"



