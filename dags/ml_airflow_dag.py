from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.etl.etl import run_etl
from src.models.train import train_model

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'titanic_ml_pipeline',
    default_args=default_args,
    description='A simple ML pipeline for Titanic dataset',
    schedule_interval=timedelta(days=1)
)

etl_task = PythonOperator(
    task_id='run_etl',
    python_callable=run_etl,
    dag=dag
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

etl_task >> train_task
