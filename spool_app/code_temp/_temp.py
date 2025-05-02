import datetime
import os
import secrets
import string

import pyAesCrypt
from django.core.mail import send_mail

buffer_size = 64 *1024

def generate_password():
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(8))

with open("main.txt", 'w+') as file:
    file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

pyAesCrypt.encryptFile("main.txt","main.txt.aes", generate_password(), buffer_size)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "wide_spool.settings")

# send_mail(
#     "Subject here",
#     "Here is the message.",
#     "raphaelkofiamponsah@gmail.com",
#     ["k8amponsah@gmail.com"],
#     fail_silently=False,
# )

