import os
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]


def get_conn():
    """Open a new database connection with pgvector support registered."""
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn
