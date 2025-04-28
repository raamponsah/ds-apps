from django.urls import path

from home_apps.views import HomeView

urlpatterns = [
    path("", HomeView.as_view(), name="home"),
]