AIoT Energy Predictor — Project Notes
=====================================

This file captures the core ideas behind the project, the main implementation choices, and the sequence used to run the pipeline end-to-end.

1) Problem Statement
- Predict hourly energy consumption from simulated IoT sensor readings to support energy efficiency and operational planning.

2) Data Design
- Built a 1-year hourly dataset (8,737 rows) with seasonality, occupancy cycles, and controlled noise.
- Added a small amount of missing data so the preprocessing step has to handle realistic gaps.

3) Storage Layer
- Loaded the CSV into SQLite through SQLAlchemy ORM to keep the schema structured and easy to validate.

4) Time-Series Preparation
- Used forward-fill imputation, MinMax scaling, and 24-hour lookback windows to turn raw readings into supervised sequences.
- Kept the train/validation/test split time-based to avoid leakage.

5) Model Choice
- Chose an LSTM because it is a strong fit for sequential patterns and repeating demand cycles.
- Trained with Adam, MSE loss, dropout, and early stopping for regularization.

6) Observed Results
- Test MAE is about 113 kWh, RMSE about 152 kWh, and R² about 0.86.
- These values are useful to explain performance in practical terms: average error, error spread, and explained variance.

7) Scale-Up Notes
- Aggregation patterns were implemented for daily, hourly, day-of-week, and occupancy-level views.
- The pipeline is structured so the same logic can be moved to PySpark for larger deployments.

8) Implementation Notes
- The code is modular, uses configuration helpers in `src/utils.py`, and stores generated outputs in `results/`.
- Reproducibility comes from fixed seeds, a virtual environment, and pinned dependencies.

9) Trade-offs
- Spark was not reliable in the local macOS environment because of Java compatibility, so the repository uses a Pandas fallback for the aggregation step.
- In a production setup, Spark would be a better fit on a managed cluster or cloud platform.

10) Next Iterations
- Add unit tests, CI, and GitHub Actions.
- Replace the fallback aggregation with a fully working PySpark setup in a compatible environment.
- Add explainability and probabilistic forecasting if the project grows further.

Run Sequence
------------
```bash
source venv/bin/activate
python -m src.data_simulator
python -m src.database_setup
python -m src.data_processor
python -m src.model
python -m src.evaluation
python -m src.spark_pipeline
```

Keep the discussion focused on how the technical choices support energy prediction, data quality, and scaling.