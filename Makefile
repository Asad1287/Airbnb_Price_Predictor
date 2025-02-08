.PHONY: train test inference etl airflow streamlit api

train:
	@echo "Starting training..."
	python src/models/train.py

test:
	@echo "Running tests..."
	pytest tests/

inference:
	@echo "Running inference..."
	python src/models/inference.py

etl:
	@echo "Running ETL pipeline..."
	python src/etl/etl.py

airflow:
	@echo "Starting Airflow webserver..."
	airflow webserver -p 8080

streamlit:
	@echo "Starting Streamlit app..."
	streamlit run app/streamlit_app.py

api:
	@echo "Starting API server..."
	uvicorn api/main.py:app --reload
