import csv
from io import TextIOWrapper
from django import forms
from django.contrib import admin, messages
from django.urls import path
from django.shortcuts import render, redirect

from .models import Spool


class CsvImportForm(forms.Form):
    csv_file = forms.FileField()


@admin.register(Spool)
class SpoolAdmin(admin.ModelAdmin):
    list_display = ["name", "report_code","stored_procedure", "procedure_parameter_1", "procedure_parameter_2","access_list"]
    change_list_template = "admin/spool_changelist.html"

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path("upload-csv/", self.admin_site.admin_view(self.upload_csv), name="upload_spool_csv"),
        ]
        return custom_urls + urls

    def upload_csv(self, request):
        if request.method == "POST":
            form = CsvImportForm(request.POST, request.FILES)
            if form.is_valid():
                try:
                    # csv_file = TextIOWrapper(request.FILES['csv_file'].file, encoding='utf-8')
                    csv_file = TextIOWrapper(request.FILES['csv_file'].file, encoding='utf-8-sig')
                    reader = csv.DictReader(csv_file)

                    for row in reader:
                        Spool.objects.update_or_create(
                            report_code=row['report_code'],  # prevent duplicate report_code
                            defaults={
                                'name': row.get('name'),
                                'stored_procedure': row.get('stored_procedure'),
                                'procedure_parameter_1': row.get('procedure_parameter_1') or None,
                                'procedure_parameter_2': row.get('procedure_parameter_2') or None,
                                'access_list': row.get('access_list'),
                            }
                        )
                    self.message_user(request, "CSV uploaded successfully!", messages.SUCCESS)
                    return redirect("..")
                except Exception as e:
                    self.message_user(request, f"Error: {e}", messages.ERROR)
        else:
            form = CsvImportForm()

        return render(request, "admin/csv_form.html", {"form": form})


# myapp/admin.py
from django.contrib.admin import AdminSite

class MyAdminSite(AdminSite):
    class Media:
        css = {
            'all': ('spool_app/css/admin_custom.css',)
        }

admin_site = MyAdminSite()
