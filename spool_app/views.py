import re
from django.shortcuts import render, get_object_or_404
from django.core.exceptions import ValidationError
from django.http import HttpResponse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from openpyxl.utils import get_column_letter
import pandas as pd
import datetime
import os
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import LoginView, LogoutView
from django.views.generic import ListView
from dotenv import load_dotenv
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from spool_app.models import Spool, get_spools_by_user

load_dotenv()
DOMAIN_NAME = "cbg.com.gh"



class SpoolListView(LoginRequiredMixin,ListView):
    model = Spool
    context_object_name = 'spools'
    template_name = "spool_app/index.html"
    login_url = '/login/'
    paginate_by = 10

    def get_queryset(self):
        user = self.request.user
        user_email = f"{user}@{DOMAIN_NAME}"
        if user.is_authenticated:
            return get_spools_by_user(user_email)
        return Spool.objects.none()




class LoginSpoolView(LoginView):
    template_name = "spool_app/login.html"
    redirect_authenticated_user = True

    def form_valid(self, form):
        user = form.get_user()
        messages.success(self.request, f"Welcome, {" ".join(user.username.split(".")).title()}!")
        return super().form_valid(form)

    def get_success_url(self):
        return self.request.GET.get('next') or '/'


class LogoutSpoolView(LogoutView):
    template_name = "spool_app/login.html"




def sanitize_column_name(name: str) -> str:
    """Convert invalid column names to valid ones."""
    if not name:
        return "unnamed_column"
    # Remove invalid characters, replace with underscore
    name = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
    # Ensure it starts with a letter
    if not name[0].isalpha():
        name = f"col_{name}"
    # Truncate to reasonable length
    return name[:31]  # Excel column name length limit





def generate_report(request, report_code: str):
    if not report_code or not isinstance(report_code, str):
        logger.error("Invalid report_code provided")
        return render(request, "400.html", {"message": "Invalid report code"}, status=400)

    try:
        spool = get_object_or_404(Spool, report_code=report_code)

        if not spool.stored_procedure:
            raise ValidationError("No stored procedure defined for this report")

        engine = create_engine(
            os.getenv('DB_CONNECTION_STRING'),
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600
        )
        Session = sessionmaker(bind=engine)

        param1 = request.POST.get("param1")
        param2 = request.POST.get("param2")
        params = {"param1": param1, "param2": param2}

        logger.info(f"Executing stored procedure for report: {report_code}")

        with Session() as session:
            if param1 and param2:
                stmt = text(f"EXEC {spool.stored_procedure} :param1, :param2")
                result = session.execute(stmt, {"param1": param1, "param2": param2})
            elif param1:
                stmt = text(f"EXEC {spool.stored_procedure} :param1")
                result = session.execute(stmt, {"param1": param1})
            else:
                stmt = text(f"EXEC {spool.stored_procedure}")
                result = session.execute(stmt)

            raw_columns = list(result.keys())
            columns = [sanitize_column_name(col) for col in raw_columns]
            data = result.fetchall()

            if not data:
                logger.warning(f"No data returned for report: {report_code}")
                return render(request, "204.html", {"message": "No data available for this report"}, status=204)

            df = pd.DataFrame(data, columns=columns)

            response = HttpResponse(
                content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{spool.report_code}_report_{timestamp}.xlsx"
            response['Content-Disposition'] = f'attachment; filename="{filename}"'

            with pd.ExcelWriter(response, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Report', index=False)
                worksheet = writer.sheets['Report']
                for idx, col in enumerate(df.columns):
                    try:
                        max_length = max(df[col].astype(str).map(len).max(), len(str(col)))
                        column_letter = get_column_letter(idx + 1)
                        worksheet.column_dimensions[column_letter].width = min(max_length + 2, 50)
                    except Exception as e:
                        logger.warning(f"Failed to adjust width for column {col}: {str(e)}")

        logger.info(f"Successfully generated report: {report_code}")
        return response

    except ValidationError as ve:
        logger.error(f"Validation error for report {report_code}: {str(ve)}")
        return render(request, "400.html", {"message": f"Validation error: {str(ve)}"}, status=400)

    except SQLAlchemyError as se:
        logger.error(f"Database error for report {report_code}: {str(se)}")
        return render(request, "500.html", {"message": "Database error occurred"}, status=500)

    except Exception as e:
        logger.error(f"Unexpected error for report {report_code}: {str(e)}")
        return render(request, "500.html", {"message": "An unexpected error occurred"}, status=500)
