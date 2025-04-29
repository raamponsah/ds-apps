
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin

from django.contrib.auth.views import LogoutView
# from django.contrib import admin
from django.urls import path, include

from spool_app.views import SpoolListView, LoginSpoolView

admin.site.site_title = "CBG Data Science Apps"
admin.site.site_header = "CBG DSA Administration"
admin.site.index_title = "Site administration"

urlpatterns = [
       path("", include("home_apps.urls")),
       path("spooled-reports/", include("spool_app.urls")),
       path("churn-dashboard/", include("churn_app.urls")),
       path('login/', LoginSpoolView.as_view(), name='login'),
       path('logout/', LogoutView.as_view(next_page='login'), name='logout'),
       path('admin/', admin.site.urls),
]


if not settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)