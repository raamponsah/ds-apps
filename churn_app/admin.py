from io import TextIOWrapper

import pandas as pd
from django.contrib import admin, messages
from django.shortcuts import render, redirect
from django.urls import path

from churn_app.models import TrainingCustomerData, ChurnModelTrainer, TestingCustomerData
from ml_models.churning.churn_prediction_ml import run_churn_test
from spool_app.admin import CsvImportForm


# Replace with your actual import

@admin.register(ChurnModelTrainer)
class ChurnModelTrainerAdmin(admin.ModelAdmin):
    list_display = ("trained_on", "accuracy", "precision", "recall", "f1_score", "verdict")

@admin.register(TrainingCustomerData)
class BankCustomerAdmin(admin.ModelAdmin):
    ...






@admin.register(TestingCustomerData)
class TestingCustomerAdmin(admin.ModelAdmin):
    change_list_template = "admin/spool_changelist.html"
    list_filter = ["churned"]
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path("upload-csv/", self.admin_site.admin_view(self.upload_csv), name="upload_testing_csv"),
        ]
        return custom_urls + urls

    def upload_csv(self, request):
        if request.method == "POST":
            form = CsvImportForm(request.POST, request.FILES)
            if form.is_valid():
                try:
                    file = TextIOWrapper(request.FILES['csv_file'].file, encoding='utf-8-sig')
                    df = pd.read_csv(file)

                    churned_customers = run_churn_test(df, TestingCustomerData)

                    self.message_user(request, f"{len(churned_customers)} likely churned customers imported.", messages.SUCCESS)
                    return redirect("..")
                except Exception as e:
                    self.message_user(request, f"Error: {e}", messages.ERROR)
        else:
            form = CsvImportForm()

        return render(request, "admin/churn_testing.html", {"form": form})

