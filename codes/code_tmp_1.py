
#
# def generate_report(request, report_code: str):
#     if not report_code or not isinstance(report_code, str):
#         logger.error("Invalid report_code provided")
#         return HttpResponse(status=400, content="Invalid report code")
#
#     try:
#         spool = get_object_or_404(Spool, report_code=report_code)
#
#         if not spool.stored_procedure:
#             raise ValidationError("No stored procedure defined for this report")
#
#         # Create engine with connection pooling settings
#         engine = create_engine(
#             os.getenv('DB_CONNECTION_STRING'),
#             pool_size=5,  # Adjust this depending on your needs
#             max_overflow=10,
#             pool_timeout=30,  # Timeout in seconds
#             pool_recycle=3600  # Recycle connections every hour
#         )
#
#         # Session maker
#         Session = sessionmaker(bind=engine)
#
#         param1 = request.POST.get("param1")
#         param2 = request.POST.get("param2")
#
#         print(param1, param2)
#
#         params = {"param1": param1, "param2": param2}
#
#         logger.info(f"Executing stored procedure for report: {report_code}")
#
#         # Using context manager for session to ensure proper cleanup
#         with Session() as session:
#             if param1 and param2:
#                 logger.info(f"Applying params: {params}")
#                 stmt = text(f"EXEC {spool.stored_procedure} :param1, :param2")
#                 result = session.execute(stmt, {"param1": param1, "param2": param2})
#
#             elif param1:
#                 logger.info(f"Applying params: {params}")
#                 stmt = text(f"EXEC {spool.stored_procedure} :param1")
#                 result = session.execute(stmt, {"param1": param1})
#
#             else:
#                 stmt = text(f"EXEC {spool.stored_procedure}")
#                 result = session.execute(stmt)
#
#             raw_columns = list(result.keys())
#             columns = [sanitize_column_name(col) for col in raw_columns]
#             data = result.fetchall()
#
#             if not data:
#                 logger.warning(f"No data returned for report: {report_code}")
#                 return HttpResponse(status=204, content="No data available for this report")
#
#             df = pd.DataFrame(data, columns=columns)
#
#             response = HttpResponse(
#                 content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
#             )
#
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename = f"{spool.report_code}_report_{timestamp}.xlsx"
#             response['Content-Disposition'] = f'attachment; filename="{filename}"'
#
#             with pd.ExcelWriter(response, engine='openpyxl') as writer:
#                 df.to_excel(writer, sheet_name='Report', index=False)
#
#                 worksheet = writer.sheets['Report']
#                 for idx, col in enumerate(df.columns):
#                     try:
#                         max_length = max(
#                             df[col].astype(str).map(len).max(),
#                             len(str(col))
#                         )
#                         column_letter = get_column_letter(idx + 1)  # Fixed here
#                         worksheet.column_dimensions[column_letter].width = min(max_length + 2, 50)
#                     except Exception as e:
#                         logger.warning(f"Failed to adjust width for column {col}: {str(e)}")
#
#         logger.info(f"Successfully generated report: {report_code}")
#         return response
#
#     except ValidationError as ve:
#         logger.error(f"Validation error for report {report_code}: {str(ve)}")
#         return HttpResponse(status=400, content=f"Validation error: {str(ve)}")
#
#     except SQLAlchemyError as se:
#         logger.error(f"Database error for report {report_code}: {str(se)}")
#         return HttpResponse(status=500, content="Database error occurred")
#
#     except Exception as e:
#         logger.error(f"Unexpected error for report {report_code}: {str(e)}")
#         return HttpResponse(status=500, content="An unexpected error occurred")
#
#