import logging
from ms_active_directory import ADDomain
from django.contrib.auth.models import User
from django.contrib.auth.backends import BaseBackend

logger = logging.getLogger(__name__)

class MSADBackend(BaseBackend):
    def authenticate(self, request, username=None, password=None):
        domain = ADDomain('cbg.com')
        try:
            logger.info(f"Trying to authenticate {username}")
            session = domain.create_session_as_user(username, password)
            logger.info(f"Session created for {username}")

            short_username = username.split("@")[0]  # Extract sAMAccountName
            user_info = session.find_user_by_sam_name(
                short_username, ['givenName', 'sn', 'mail', 'memberOf']
            )

            if not user_info:
                logger.warning(f"User info not found for {short_username}")
                return None

            user, created = User.objects.get_or_create(username=short_username)
            user.first_name = user_info.get('givenName', '')
            user.last_name = user_info.get('sn', '')
            user.email = user_info.get('mail', '')

            groups = user_info.get('memberOf', [])
            logger.info(f"{short_username}'s AD groups: {groups}")

            # Adjust these based on actual AD group naming
            in_django_users = any("CN=Django_Users" in g for g in groups)
            in_django_admins = any("CN=Django_Admins" in g for g in groups)

            # # TEMP FIX: Force is_staff=True so admin works
            # user.is_staff = True  # <- Remove or conditionally set later
            # user.is_superuser = True

            user.save()
            logger.info(
                f"User {short_username} {'created' if created else 'updated'} "
                f"successfully | Staff: {user.is_staff}, Superuser: {user.is_superuser}"
            )
            return user

        except Exception as e:
            logger.error(f"Authentication failed for {username}: {e}")
            return None

    def get_user(self, user_id):
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None
