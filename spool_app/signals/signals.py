# from django.contrib.auth.models import User
# from django.db.models.signals import post_migrate
#
# # Add a `shortname` property dynamically to User model
# def add_shortname_property(sender, **kwargs):
#     # This method ensures that the property is added after migrations.
#     @property
#     def shortname(self):
#         return self.username.split("@")[0]  # Extract shortname part
#
#     User.add_to_class('shortname', shortname)
#
# # Connect the signal
# post_migrate.connect(add_shortname_property, sender=User)
