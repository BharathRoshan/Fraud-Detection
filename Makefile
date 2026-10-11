.PHONY: install train track api monitor test dashboard

install:
	python -m pip install -r requirements.txt

train:
	python src/train.py --data creditcard.csv --output-dir artifacts

track:
	python src/train_tracked.py --data creditcard.csv --output-dir artifacts

api:
	uvicorn src.api:app --reload --host 127.0.0.1 --port 8000

monitor:
	python src/monitoring.py --reference data/reference.csv --current data/current.csv --output-dir monitoring_reports

test:
	pytest -q

dashboard:
	streamlit run app/dashboard.py
