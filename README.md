AIoT Energy Predictor
=====================

Overview
--------
This repository is a self-contained portfolio project that demonstrates an end-to-end AIoT energy prediction pipeline. It simulates IoT sensor data, stores it in SQLite, performs feature engineering, trains an LSTM model to predict hourly energy consumption, evaluates results, and provides scalable aggregation patterns (Pandas implementation that mirrors Spark patterns).

Highlights
---------
- Synthetic dataset: 8,737 hourly records (2023-01-01 → 2023-12-31)
- Model: Stacked LSTM (64 → 32 units) with dropout, trained with early stopping
- Performance: MAE ≈ 113 kWh, RMSE ≈ 152 kWh, R² ≈ 0.86 on test set
- Aggregations: Daily, hourly, day-of-week, occupancy-level (Pandas fallback for Spark)

Quickstart
----------
1. Create and activate virtualenv:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run pipeline (generate data → ingest → process → train → evaluate):

```bash
# simulate data
python -m src.data_simulator
# ingest into SQLite
python -m src.database_setup
# process and create sequences
python -m src.data_processor
# train model
python -m src.model
# evaluate
python -m src.evaluation
# run aggregation pipeline (Pandas-based)
python -m src.spark_pipeline
```

Demo
----
Use this short flow when presenting the project live:

1. Show the problem statement and pipeline overview in the README.
2. Run the aggregation step to highlight hourly, daily, and occupancy patterns.
3. Open `results/predictions_vs_actual.png` to discuss model quality.
4. Point to the metrics in `results/model_metrics.txt` and explain MAE, RMSE, and R².
5. Close with the business angle: forecasting energy demand for smarter operations and cost control.

Files of interest
-----------------
- [src/data_simulator.py](src/data_simulator.py#L1) — generates synthetic IoT data
- [src/database_setup.py](src/database_setup.py#L1) — SQLAlchemy ingestion to `data/sensor_data.db`
- [src/data_processor.py](src/data_processor.py#L1) — scaling, lag features, splits, saved `data/processed_data.pkl`
- [src/model.py](src/model.py#L1) — builds, trains, saves `models/lstm_model.h5`
- [src/evaluation.py](src/evaluation.py#L1) — metrics and plots saved to `results/`
- [src/spark_pipeline.py](src/spark_pipeline.py#L1) — Pandas-based aggregation (Spark patterns)

Results
-------
- Model file: `models/lstm_model.h5`
- Processed data: `data/processed_data.pkl`
- SQLite DB: `data/sensor_data.db`
- Aggregation CSVs: `results/spark_daily_aggregation.csv`, `results/spark_hourly_pattern.csv`, `results/spark_day_of_week_pattern.csv`, `results/spark_occupancy_energy.csv`

Next steps
----------
- Push repository to GitHub (remove large data/model files from push via `.gitignore`)
- Add CI to run linting and unit tests
- Replace Pandas aggregation with a working Spark job if a compatible Java/Spark environment is available

License
-------
Personal portfolio project. Contact owner for reuse.