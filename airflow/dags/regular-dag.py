from airflow import DAG
from airflow.operators.postgres_operator import PostgresOperator
from datetime import datetime

# Define the SQL query
sql_query = """
SELECT * FROM real_estate re 
LIMIT 10
"""
# Define the DAG
dag = DAG(
    'my_sql_dag',
    description='A DAG to execute SQL query',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags = ['cian']
)

# Define the PostgresOperator
execute_sql_task = PostgresOperator(
    task_id='execute_sql',
    postgres_conn_id='estate_data',  # Connection ID configured in Airflow
    sql=sql_query,
    dag=dag,
)

