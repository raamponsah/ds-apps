from django.core.management.base import BaseCommand
from spool_app.models import Spool

class Command(BaseCommand):
    help = 'Seeds the database with initial Spool data'

    def handle(self, *args, **kwargs):
        spools = spools = [
    {
        "name": "Daily Sales Report",
        "report_code": "DSR001",
        "stored_procedure": "EXEC generate_daily_sales"
    },
    {
        "name": "Monthly Revenue Summary",
        "report_code": "MRS002",
        "stored_procedure": "EXEC generate_monthly_revenue"
    },
    {
        "name": "Customer Activity Log",
        "report_code": "CAL003",
        "stored_procedure": "EXEC get_customer_logs"
    },
    {
        "name": "Inventory Stock Snapshot",
        "report_code": "ISS004",
        "stored_procedure": "EXEC get_inventory_snapshot"
    },
    {
        "name": "Employee Attendance Summary",
        "report_code": "EAS005",
        "stored_procedure": "EXEC hr_attendance_summary"
    },
    {
        "name": "Vendor Payments Ledger",
        "report_code": "VPL006",
        "stored_procedure": "EXEC finance_vendor_ledger"
    },
    {
        "name": "Product Performance Tracker",
        "report_code": "PPT007",
        "stored_procedure": "EXEC track_product_performance"
    },
    {
        "name": "System Error Audit Trail",
        "report_code": "SEAT008",
        "stored_procedure": "EXEC audit_system_errors"
    },
    {
        "name": "User Login History",
        "report_code": "ULH009",
        "stored_procedure": "EXEC get_login_history"
    },
    {
        "name": "Profit & Loss Overview",
        "report_code": "PLO010",
        "stored_procedure": "EXEC financial_pl_summary"
    },
    {
        "name": "Top Selling Products",
        "report_code": "TSP011",
        "stored_procedure": "EXEC get_top_sellers"
    },
    {
        "name": "Pending Orders Breakdown",
        "report_code": "POB012",
        "stored_procedure": "EXEC get_pending_orders"
    },
    {
        "name": "Refund Transactions Report",
        "report_code": "RTR013",
        "stored_procedure": "EXEC generate_refund_transactions"
    },
    {
        "name": "Quarterly Growth Analysis",
        "report_code": "QGA014",
        "stored_procedure": "EXEC analyze_growth_quarterly"
    },
    {
        "name": "Departmental Budget Allocation",
        "report_code": "DBA015",
        "stored_procedure": "EXEC fetch_budget_allocations"
    },
]


        for spool in spools:
            obj, created = Spool.objects.get_or_create(
                report_code=spool["report_code"],
                defaults={
                    "name": spool["name"],
                    "stored_procedure": spool["stored_procedure"],
                }
            )
            if created:
                self.stdout.write(self.style.SUCCESS(f"Created Spool: {obj.report_code}"))
            else:
                self.stdout.write(f"Spool already exists: {obj.report_code}")
