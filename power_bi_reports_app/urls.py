from django.urls import path

from power_bi_reports_app.views import PowerBiReportsView, PowerBiReportDetailView

urlpatterns = [
    path("", PowerBiReportsView.as_view(), name="power_bi_reports"),
    path("<int:pk>", PowerBiReportDetailView.as_view(), name="power_bi_report"),

]