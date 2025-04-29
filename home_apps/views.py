from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import render
from django.views import View
from django.views.generic import ListView

from home_apps.models import AppModel


# Create your views here.
class HomeView(LoginRequiredMixin,ListView):
    template_name = 'home_apps/home.html'
    Model = AppModel
    context_object_name = 'apps'
    login_url = '/login/'

    def get_queryset(self):
        return AppModel.objects.all()
