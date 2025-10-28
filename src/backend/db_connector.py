import sqlite3
import logging

logging.basicConfig(level=logging.INFO)

DB_PATH = "D:\\Unified patient agent\\src\\mock_data\\healthcare.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    logging.info(f"[DB] Connected to {DB_PATH}")
    return conn

def fetch_table(table_name: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    conn.close()
    logging.info(f"[DB] Fetched {len(rows)} rows from {table_name}")
    return [dict(zip(columns, row)) for row in rows]
