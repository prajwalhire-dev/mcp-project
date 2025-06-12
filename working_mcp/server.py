import os
import json
import sqlite3
import pandas as pd
from mcp.server.fastmcp import FastMCP
from anthropic import Anthropic
from dotenv import load_dotenv
from typing import Dict

# Load environment variables from .env file
load_dotenv()

# Initialise MCP server
mcp = FastMCP(
    name="SQLQueryAgent",
)

# Initialize the Anthropic client
anthropic_client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),)

# --- Absolute Paths for Data Files ---
# This ensures the server can find the files regardless of how it's started.
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "data", "electric_vehicle_population.db")
DATA_DICT_PATH = os.path.join(BASE_DIR, "data", "data_dictionary.csv")

# --- Helper Function ---
def get_data_dictionary_description():
    """Reads the data dictionary CSV and formats it into a string for the AI."""
    try:
        df = pd.read_csv(DATA_DICT_PATH)
        description = "This is the data dictionary. It explains the columns in the database tables:\n"
        for _, row in df.iterrows():
            description += f"- Column '{row['Column Header']}' (also called '{row['Business Header']}'): {row['Definition']}. Example: {row['Example']}\n"
        return description
    except FileNotFoundError:
        return "Data dictionary file not found. I will proceed without it."
    except Exception as e:
        return f"Error reading data dictionary: {e}"

# --- Tool 1: NER Generator (LLM) ---
@mcp.tool()
def ner_generator_dynamic(question: str) -> Dict:
    """
    Analyzes a question to extract key entities and returns them as a Python dictionary.
    """
    data_dictionary = get_data_dictionary_description()
    prompt = f"""
    You are a data analyst. Your job is to extract key entities from a user's question.
    Use the provided data dictionary to understand the columns. Your output MUST be a single, raw JSON object.

    Data Dictionary:
    {data_dictionary}

    User Question: "{question}"

    Extract the necessary components to answer the question. Your output MUST be a single JSON object with keys: "table", "columns_to_select", and "filters".
    """
    try:
        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        json_str = response.content[0].text
        # Parse the LLM's string output into a clean dictionary before returning
        return json.loads(json_str[json_str.find('{') : json_str.rfind('}') + 1])
    except Exception as e:
        return {"error": f"Error in ner_generator_dynamic: {e}"}

# --- Tool 2: Create SQL (LLM) ---
@mcp.tool()
def create_sql(question: str, ner_dict: Dict) -> str:
    """
    Creates a full SQLite query as a JSON string.
    """
    ner_json = json.dumps(ner_dict, indent=2)
    prompt = f"""
    You are an expert SQLite developer. Create a single, valid SQLite query to answer the user's question.
    Use the extracted JSON entities as a guide. Your output MUST be a single JSON object with one key: "sql_query".

    Original Question: "{question}"
    Extracted Entities: {ner_json}
    """
    try:
        response = anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        # Return the raw text, which should be a JSON string
        return response.content[0].text
    except Exception as e:
        return json.dumps({"error": f"LLM Error in create_sql: {e}"})

# --- Tool 3: Run SQLite Query (No LLM) ---
@mcp.tool()
def run_sqlite_query(sql_json: str) -> Dict:
    """Executes a SQL query and returns the data as a Python dictionary."""
    try:
        json_str = sql_json[sql_json.find('{') : sql_json.rfind('}') + 1]
        data = json.loads(json_str)
        sql_query = data.get("sql_query")

        if data.get("error"):
            return {"error": f"Cannot execute due to previous error: {data['error']}", "data": []}
        if not sql_query:
            return {"error": "No SQL query provided.", "data": []}

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        conn.close()

        formatted_results = [dict(zip(column_names, row)) for row in results]
        return {"data": formatted_results}
    except Exception as e:
        return {"error": f"Database query failed: {e}", "data": []}

# --- Tool 4: Generate Final Answer (LLM) ---
@mcp.tool()
def generate_final_answer(question: str, query_result_dict: Dict) -> str:
    """Takes the database results and generates a human-readable answer."""
    query_result_json = json.dumps(query_result_dict, indent=2)
    prompt = f"""
    You are a helpful assistant. Answer the user's question based on the provided data.
    If the data contains an error, explain it simply. If the data is empty, say so.

    Original Question: "{question}"
    Data from Database: {query_result_json}
    """
    try:
        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        return f"Error formulating final answer: {e}"

# --- Run Server ---
if __name__ == "__main__":
    print("MCP server with multi-step AI pipeline is starting...")
    mcp.run(transport="stdio")