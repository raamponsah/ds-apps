import random
import string

from django.db import models


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

    class Meta:
        ordering = ["report_code"]




    def __str__(self):
        return self.report_code



