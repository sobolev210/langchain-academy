from langgraph.store.postgres import PostgresStore
from psycopg import Connection

conn_string = "postgresql://postgres:mysecret@localhost:5432/langchain-db"

conn = Connection.connect(conn_string)
conn.autocommit = True  # <-- key line: no transaction block

try:
    store = PostgresStore(conn)
    store.setup()  # migrations (includes CREATE INDEX CONCURRENTLY)
finally:
    conn.close()
