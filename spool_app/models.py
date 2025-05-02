import random
import string

from django.contrib.auth import get_user_model
from django.db import models

User = get_user_model()


def generate_report_code():
    """Generate a unique 6-character alphanumeric report_code."""
    code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    # Ensure uniqueness by checking the database
    while Spool.objects.filter(report_code=code).exists():
        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6)).upper()
    return code


def get_spools_by_user(user_email):
    return Spool.objects.filter(access_list__contains=user_email)


class Spool(models.Model):
    name = models.CharField(max_length=100, blank=False)
    report_code = models.CharField(max_length=10, unique=True, blank=True, null=True, default=generate_report_code)
    stored_procedure = models.TextField()
    procedure_parameter_1 = models.CharField(null=True, blank=True, choices={
        ("integer", "Integer"),
        ("float", "Float"),
        ("text", "Text"),
        ("date", "Date"),
    })
    procedure_parameter_2 = models.CharField(null=True, blank=True, choices={
        ("integer", "Integer"),
        ("float", "Float"),
        ("text", "Text"),
        ("date", "Date"),
    })
    access_list = models.TextField(null=True, blank=True)

    @property
    def all_access_list(self):
        return self.access_list.split(',') if self.access_list else []



    def save(self, *args, **kwargs):
        """Override the save method to automatically generate a report_code if not provided."""
        if not self.report_code:
            self.report_code = generate_report_code()
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ["report_code"]



class UserDownloadHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    spool_report_downloaded = models.ForeignKey(Spool, related_name='spool_report_downloaded', on_delete=models.CASCADE)
    spool_report_password = models.CharField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["timestamp"]
        verbose_name = "User Download History"
        verbose_name_plural = "User Download Histories"

    def __str__(self):
        return f"{self.user.username} downloaded {self.spool_report_downloaded} at {self.timestamp}"


    def save(self, *args, **kwargs):
        if not self.pk:  # Only for new entries

            user_entries = self.__class__.objects.filter(user=self.user)
            current_count = user_entries.count()
            if current_count >= 5:
                excess_count = current_count - 4
                oldest_entries = user_entries.order_by('timestamp')[:excess_count]
                for entry in oldest_entries:
                    entry.delete()
        super().save(*args, **kwargs)

