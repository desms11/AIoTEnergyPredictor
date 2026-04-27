AIoT Energy Predictor — Architecture
=====================================

Overview
--------
This document describes the modular architecture and data flow of the AIoT Energy Predictor project.

High-level Modules
------------------
- `src/data_simulator.py` — Synthetic IoT sensor generator (temperature, humidity, occupancy, energy)
- `src/database_setup.py` — SQLAlchemy models and ingestion into `data/sensor_data.db`
- `src/data_processor.py` — Time-series transforms, MinMax scaling, lookback window → sequences
- `src/model.py` — LSTM model building, training, checkpointing, serialization
- `src/evaluation.py` — Predictions, metrics (MAE, RMSE, R²), and plots
- `src/spark_pipeline.py` — Aggregation patterns (Pandas implementation mirroring Spark)
- `src/utils.py` — Path helpers, config, logging

Data Flow
---------
1. Simulation: `data_simulator` writes `data/raw_sensor_data.csv` (hourly rows)
2. Ingestion: `database_setup` reads CSV and stores rows in `data/sensor_data.db`
3. Processing: `data_processor` reads DB, scales features, creates lagged sequences (24-hour lookback) and saves `data/processed_data.pkl`
4. Modeling: `model` loads processed data, trains LSTM, and saves `models/lstm_model.h5`
5. Evaluation: `evaluation` loads model + processed data, computes metrics, saves plots in `results/`
6. Aggregation: `spark_pipeline` produces aggregated CSV reports in `results/` (daily/hourly/day-of-week/occupancy)

Model Design
------------
- Input shape: 24 timesteps × 4 features (temperature, humidity, occupancy, energy)
- Architecture: LSTM(64, return_sequences=True) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16, relu) → Dense(1)
- Loss/opt: MSE loss, Adam optimizer (lr=0.001)
- Regularization: Dropout + early stopping (patience=5)

Scalability and Production Notes
--------------------------------
- Storage: Move from SQLite → Postgres / TimescaleDB for retention and query performance
- Feature store: Persist scaled features and engineered features to a feature store (Feast or in-house)
- Model serving: Use TensorFlow Serving or a lightweight FastAPI wrapper for real-time predictions
- Batch/streaming: Use Spark (PySpark) for large-scale aggregations and Kafka for streaming ingestion
- Orchestration: Use Airflow or Prefect to schedule simulation/ingestion/process/train/evaluate jobs

Why Pandas fallback for Spark?
-----------------------------
Local macOS environment exhibited Java/Spark compatibility issues. For the portfolio, Pandas demonstrates identical aggregation logic and produces CSV outputs suitable for visualizations and documentation. The code documents how to switch to Spark in production.

Files and responsibilities
--------------------------
- `data/` — raw CSV, processed pickle, SQLite DB (ignored by Git)
- `models/` — saved model artifacts (ignored by Git)
- `results/` — evaluation plots & aggregation CSVs (ignored by Git)
- `src/` — source modules described above

Deployment checklist
--------------------
- Configure `JAVA_HOME` and compatible Java (if running Spark locally)
- Install required Python dependencies from `requirements.txt`
- Create virtualenv and run the pipeline using `python -m src.<module>` commands

Contact / Notes
----------------
For interview or demo requests, show metrics, plots in `results/` and run the aggregation to produce CSVs used in slides.