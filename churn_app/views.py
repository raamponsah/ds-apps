from django.shortcuts import render
from django.views.generic import TemplateView


# Create your views here.
class CustomerChurnView(TemplateView):
    template_name = "churn_app/dashboard.html"