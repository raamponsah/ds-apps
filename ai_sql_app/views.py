import os
import re
import uuid
import pandas as pd
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=16384,
    timeout=360,
    max_retries=60,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Connect to MSSQL database
mssql_uri = os.getenv("DB_CONNECTION_STRING")
engine = create_engine(mssql_uri)

# Prompt template for T-SQL generation
prompt_template = """
You are an expert SQL assistant for Microsoft SQL Server, using T-SQL syntax. The user will provide a natural language query about the following tables: {tables}. Below are the schemas for these tables:

{schemas}

Your job is to:

1. Generate a precise T-SQL query to answer the user's question.
2. Use `SET ROWCOUNT 0` to ensure all rows are returned, unless the user specifies a limit (e.g., 'top 5').
3. Only include `TOP` when the user explicitly requests a limited number of rows.
4. Assume tables are in the `dbo` schema unless specified otherwise.
5. If the query targets a specific table (e.g., 'genres' for Genre), prioritize that table unless a join is clearly needed.
6. For multi-table queries, use joins or subqueries as appropriate, ensuring column compatibility.
7. Return ONLY the T-SQL query as plain text. Do NOT include markdown (e.g., ```sql), code blocks, comments, or explanations.

User query: {query}

T-SQL query:
"""

def format_schemas(selected_tables, table_schemas):
    schema_text = ""
    for table in selected_tables:
        schema_text += f"Table: {table}\n"
        for col in table_schemas[table]:
            schema_text += f"- {col['COLUMN_NAME']} ({col['DATA_TYPE']})\n"
        schema_text += "\n"
    return schema_text

def clean_sql_query(query):
    # Remove markdown, code blocks, comments, and extra whitespace
    query = re.sub(r'```sql|```|/\*.*?\*/|--.*?\n', '', query, flags=re.DOTALL)
    query = re.sub(r'\s+', ' ', query).strip()
    # Remove duplicate SET ROWCOUNT 0
    if query.count("SET ROWCOUNT 0") > 1:
        parts = query.split(";", 1)
        cleaned = parts[1].replace("SET ROWCOUNT 0", "").strip()
        query = f"SET ROWCOUNT 0; {cleaned}"
    elif not query.startswith("SET ROWCOUNT 0") and not any(word in query.lower() for word in ["top", "first", "limit"]):
        query = f"SET ROWCOUNT 0; {query}"
    return query

def identify_target_table(query, selected_tables):
    query_lower = query.lower()
    for table in selected_tables:
        table_lower = table.lower()
        if table_lower in query_lower or table_lower + "s" in query_lower or table_lower[:-1] in query_lower:
            return table
    return selected_tables[0] if selected_tables else None

def estimate_row_count(sql_query, selected_tables, target_table, engine):
    if len(selected_tables) == 1 or target_table:
        table = target_table or selected_tables[0]
        return pd.read_sql(f"SELECT COUNT(*) FROM {table}", engine).iloc[0, 0]
    try:
        count_query = f"SELECT COUNT(*) AS cnt FROM ({sql_query.replace('SET ROWCOUNT 0;', '')}) AS subquery"
        return pd.read_sql(count_query, engine).iloc[0, 0]
    except:
        return max(pd.read_sql(f"SELECT COUNT(*) FROM {table}", engine).iloc[0, 0] for table in selected_tables)
@login_required(login_url="/login")
def chatbot_view(request):
    if request.method == "POST":
        user_input = request.POST.get("message", "")
        # Retrieve table schemas
        with engine.connect() as conn:
            tables = pd.read_sql("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'", conn)
            table_schemas = {}
            for table in tables["TABLE_NAME"]:
                columns = pd.read_sql(f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}'", conn)
                table_schemas[table] = columns[["COLUMN_NAME", "DATA_TYPE"]].to_dict(orient="records")
        selected_tables = tables["TABLE_NAME"].tolist()
        target_table = identify_target_table(user_input, selected_tables)
        schemas = format_schemas(selected_tables, table_schemas)
        formatted_prompt = prompt_template.format(tables=", ".join(selected_tables), schemas=schemas, query=user_input)
        response = llm.invoke(formatted_prompt)
        sql_query = clean_sql_query(response.content.strip())
        try:
            full_df = pd.read_sql(sql_query, engine)
            output_content = full_df.to_html(classes="table table-striped")
        except Exception as e:
            output_content = f"Error processing query: {e}"
        return render(request, "ai_sql_app/chatbot.html", {"response": output_content, "user_input": user_input})
    return render(request, "ai_sql_app/chatbot.html")

