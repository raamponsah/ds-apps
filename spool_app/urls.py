from django.contrib import admin
from django.urls import path
from spool_app.views import SpoolListView, generate_report


admin.site.site_title = "CBG Spooling Reports"
admin.site.site_header = "CBG Spool Adminstration"
admin.site.index_title = "Site administration"
urlpatterns = [
    path("", SpoolListView.as_view(), name="spool_list"),
    path("gen-report/<str:report_code>", generate_report, name="report_gen"),
]