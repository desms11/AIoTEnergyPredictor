AIoT Energy Predictor — Interview Talking Points
================================================

Use these concise talking points when discussing the project in interviews.

1) Problem & Impact
- Predict hourly energy consumption from simulated IoT sensors to reduce energy waste and improve operational planning.

2) Data Generation
- Created realistic 1-year hourly dataset (8,737 rows) with seasonality, occupancy cycles, and controlled noise.
- Purposefully injected 2% missing values to exercise robust imputation.

3) Storage & Ingestion
- Ingested CSV into SQLite via SQLAlchemy ORM to show production-like schema and integrity checks.

4) Feature Engineering
- Time-series aware preprocessing: forward-fill missing values, MinMax scaling, 24-hour lookback windows → supervised sequences.
- Ensured no data leakage by using time-based splits (train/val/test).

5) Model Choice
- LSTM selected for temporal dependencies (stacked LSTM architecture with dropout).
- Training with Adam optimizer, MSE loss, and early stopping to prevent overfitting.

6) Results
- Test MAE ≈ 113 kWh, RMSE ≈ 152 kWh, R² ≈ 0.86 — explain how this meets business goals (relative error, operational relevance).

7) Scalability
- Implemented aggregation patterns (daily/hourly/day-of-week/occupancy) using Pandas; designed to swap to PySpark for cluster-scale processing.
- Discussed trade-offs: local development speed vs. production scalability.

8) Engineering Practices
- Modular code, config via `src/utils.py`, results saved to `results/`, large artifacts excluded via `.gitignore`.
- Reproducibility: seed control, virtualenv, `requirements.txt`.

9) Challenges & Trade-offs
- Spark initialization failed due to macOS Java compatibility; used Pandas fallback and documented the fix path.
- Discuss alternatives: Dockerized Spark, AWS EMR, or using Databricks for production pipelines.

10) Next Steps
- Add unit tests, CI, and GitHub Actions
- Swap Pandas to working PySpark implementation and run on EMR
- Add explainability (SHAP) and probabilistic forecasting

Quick demo script
-----------------
Run these commands during the interview to demo the pipeline:

```bash
source venv/bin/activate
python -m src.data_simulator
python -m src.database_setup
python -m src.data_processor
python -m src.model
python -m src.evaluation
python -m src.spark_pipeline
```

Keep answers crisp and relate technical choices to business impact (cost savings, capacity planning, anomaly detection).