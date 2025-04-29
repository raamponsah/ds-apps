from django.contrib import admin

from home_apps.models import AppModel


# Register your models here.
@admin.register(AppModel)
class AppModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'link', 'image_url', 'status')
    list_filter = ('status',)
    search_fields = ('name',)
    ordering = ('name',)

