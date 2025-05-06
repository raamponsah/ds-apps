from django.db import models

# Create your models here.
class AppModel(models.Model):
    name = models.CharField(max_length=100)
    link = models.URLField(null=True, blank=True)
    image_url = models.URLField(null=True, blank=True)
    status = models.CharField(max_length=100, choices=[
        ("alpha", "Alpha"),
        ("beta", "Beta"),
        ("active-development", "Active Development"),
        ("maintenance", "Maintenance"),
        ("active", "Active"),
        ("inactive", "Inactive"),
        ("deleted", "Deleted"),
    ], default="maintenance")
    color = models.CharField(max_length=10, default="blue", null=True, blank=True, choices=[
        ("black", "Black"),
        ("yellow", "Yellow"),
        ("red", "Red"),
        ("orange", "Orange"),
        ("blue", "Blue"),
        ("purple", "Purple"),
        ("cyan", "Cyan"),
        ("gray", "Gray"),
        ("white", "White"),
        ("green", "Green"),
        ("teal", "Teal"),
        ("default", "Default"),
    ])

    description = models.TextField(null=True, blank=True, max_length=150)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['name']
        verbose_name_plural = 'apps'
        verbose_name = 'app'

        ordering = ('status',)

