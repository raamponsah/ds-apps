from django.urls import path

from churn_app.views import CustomerChurnView
from home_apps.urls import urlpatterns

urlpatterns = [
    path("", CustomerChurnView.as_view()),
]