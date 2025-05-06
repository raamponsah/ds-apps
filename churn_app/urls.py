from django.urls import path

from churn_app.views import CustomerChurnView, interactive_churned_view
from home_apps.urls import urlpatterns

urlpatterns = [
    # path("", CustomerChurnView.as_view()),
    path("",interactive_churned_view, name="interactive_churned_view"),
]