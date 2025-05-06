from django.urls import path

from ai_sql_app.views import chatbot_view

urlpatterns = [
    path("", chatbot_view, name="chatbot"),
]