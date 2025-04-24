from django import template

register = template.Library()

@register.filter
def username_to_fullname(value):
    try:
        parts = value.split(".")
        return " ".join(part.capitalize() for part in parts)
    except ValueError:
        return value
