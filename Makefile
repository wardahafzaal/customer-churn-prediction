# Windows users without `make`: run the commands manually (see README).
INPUT ?= data/raw/online_retail_II.xlsx

data:      ; python src/load_clean.py --input $(INPUT)
synthetic: ; python src/make_synthetic_data.py && python src/load_clean.py --input data/raw/synthetic_retail.csv
train:     ; python src/train.py
score:     ; python src/score.py
test:      ; python -m pytest -q
api:       ; uvicorn app.api:app --reload
dashboard: ; streamlit run app/dashboard.py
mlflow:    ; mlflow ui --backend-store-uri mlruns
docker:    ; docker build -t churn-api . && docker run -p 8000:8000 churn-api
