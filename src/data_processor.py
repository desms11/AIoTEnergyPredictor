"""Data processing and feature engineering for LSTM training."""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .database_setup import SensorReading, Base
from .utils import setup_logging, get_data_dir, get_project_root, Config

logger = setup_logging(__name__)


class DataProcessor:
    """
    Processes and engineers features from raw sensor data.
    
    Pipeline:
    1. Load data from SQL database
    2. Handle missing values (forward fill)
    3. Normalize data to [0, 1]
    4. Create lag features (past 24 hours)
    5. Split into train/validation/test sets
    6. Save to pickle for efficient loading
    """
    
    def __init__(self, db_path='data/sensor_data.db', lookback=Config.LOOKBACK_WINDOW):
        """
        Initialize data processor.
        
        Args:
            db_path: Path to SQLite database
            lookback: Number of previous timesteps to use as input (default 24 hours)
        """
        if isinstance(db_path, str) and not db_path.startswith('/'):
            db_path = get_project_root() / db_path
        
        self.db_path = Path(db_path)
        self.db_url = f'sqlite:///{self.db_path}'
        self.lookback = lookback
        
        # Initialize scaler (will be fitted during processing)
        self.scaler = None
        
        logger.info(f"Initialized DataProcessor: lookback={lookback} hours")
    
    def load_data_from_db(self):
        """
        Query sensor data from database and return as DataFrame.
        
        Returns:
            pandas.DataFrame: DataFrame with columns [Timestamp, Temperature, Humidity, Occupancy, Energy_Consumption]
        """
        logger.info("Loading data from database...")
        
        try:
            engine = create_engine(self.db_url, echo=False)
            Session = sessionmaker(bind=engine)
            session = Session()
            
            # Query all sensor readings ordered by timestamp
            query_result = session.query(SensorReading).order_by(SensorReading.timestamp.asc()).all()
            
            # Convert to DataFrame
            data = []
            for reading in query_result:
                data.append({
                    'Timestamp': reading.timestamp,
                    'Temperature': reading.temperature,
                    'Humidity': reading.humidity,
                    'Occupancy': reading.occupancy,
                    'Energy_Consumption': reading.energy_consumption
                })
            
            df = pd.DataFrame(data)
            logger.info(f"✓ Loaded {len(df)} records from database")
            
            session.close()
            engine.dispose()
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            raise
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset.
        
        For time-series data, forward fill is appropriate
        (propagate last known value forward).
        
        Args:
            df: Input DataFrame
        
        Returns:
            pandas.DataFrame: DataFrame with missing values handled
        """
        logger.info("Handling missing values...")
        
        # Check initial null count
        initial_nulls = df.isnull().sum().sum()
        if initial_nulls > 0:
            logger.info(f"  Found {initial_nulls} missing values")
            # Forward fill (propagate last known value)
            df = df.fillna(method='ffill')
            # Backward fill for any remaining NaNs at the start
            df = df.fillna(method='bfill')
            logger.info(f"  ✓ Missing values filled")
        else:
            logger.info("  ✓ No missing values found")
        
        return df
    
    def scale_data(self, df):
        """
        Normalize numeric columns to [0, 1] range.
        
        LSTM networks train better with normalized data.
        Uses MinMaxScaler to fit on entire dataset.
        
        Args:
            df: Input DataFrame
        
        Returns:
            pandas.DataFrame: DataFrame with scaled columns
        """
        logger.info("Scaling data to [0, 1]...")
        
        # Create copy to avoid modifying original
        df_scaled = df.copy()
        
        # Columns to scale (exclude Timestamp)
        columns_to_scale = ['Temperature', 'Humidity', 'Occupancy', 'Energy_Consumption']
        
        # Initialize and fit scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])
        
        logger.info(f"✓ Data scaled using MinMaxScaler")
        logger.info(f"  Scaler fitted on {len(df)} samples")
        
        return df_scaled
    
    def create_lag_features(self, df, lookback=None):
        """
        Create lag features for time-series supervised learning.
        
        Transforms data from:
            [t] Temperature, Humidity, Occupancy, Energy_Consumption → Flat
        To:
            [t-lookback, ..., t-1] (4 features × lookback) → Energy_Consumption[t]
        
        This allows LSTM to learn: "Given past 24 hours, predict next hour's energy"
        
        Args:
            df: Input DataFrame (must be sorted by time)
            lookback: Number of previous timesteps to use (default: self.lookback)
        
        Returns:
            tuple: (X, y) where X has shape (n_samples, lookback*n_features) and y has shape (n_samples,)
        """
        if lookback is None:
            lookback = self.lookback
        
        logger.info(f"Creating lag features (lookback={lookback} hours)...")
        
        # Extract feature columns (exclude Timestamp)
        feature_cols = ['Temperature', 'Humidity', 'Occupancy', 'Energy_Consumption']
        data = df[feature_cols].values  # Shape: (n_samples, n_features)
        
        X, y = [], []
        
        # Create sequences
        for i in range(len(data) - lookback):
            # Input: past 'lookback' hours of all features
            X.append(data[i:i+lookback, :].flatten())  # Flatten to 1D
            # Output: next hour's energy consumption
            y.append(data[i+lookback, 3])  # Index 3 is Energy_Consumption
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"✓ Created lag features:")
        logger.info(f"  X shape: {X.shape} (samples, lookback*features)")
        logger.info(f"  y shape: {y.shape} (samples,)")
        logger.info(f"  Each sample uses {lookback} hours × 4 features = {X.shape[1]} values")
        
        return X, y
    
    def split_data(self, X, y, train_ratio=Config.TRAIN_RATIO, val_ratio=Config.VAL_RATIO):
        """
        Split data into training, validation, and test sets.
        
        Important: Time-series data should NOT be randomly shuffled
        (maintains temporal order for validation).
        
        Args:
            X: Feature matrix
            y: Target vector
            train_ratio: Fraction for training (default 0.7)
            val_ratio: Fraction for validation (default 0.15)
                       Test ratio = 1 - train_ratio - val_ratio
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info(f"Splitting data: train={train_ratio}, val={val_ratio}, test={1-train_ratio-val_ratio}...")
        
        n_samples = len(X)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        logger.info(f"✓ Data split successfully:")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Val:   {len(X_val)} samples")
        logger.info(f"  Test:  {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def process(self, output_path=None):
        """
        Execute complete data processing pipeline.
        
        Pipeline:
        1. Load data from database
        2. Handle missing values
        3. Scale to [0, 1]
        4. Create lag features
        5. Split into train/val/test
        6. Save to pickle file
        
        Args:
            output_path: Path to save pickle file. If None, uses default.
        
        Returns:
            dict: Dictionary containing all processed datasets
        """
        if output_path is None:
            output_path = get_data_dir() / 'processed_data.pkl'
        else:
            output_path = Path(output_path)
        
        logger.info("\n" + "="*60)
        logger.info("Data Processing Pipeline")
        logger.info("="*60 + "\n")
        
        # Step 1: Load
        df = self.load_data_from_db()
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Scale
        df_scaled = self.scale_data(df)
        
        # Step 4: Create lag features
        X, y = self.create_lag_features(df_scaled)
        
        # Step 5: Split
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Step 6: Save to pickle
        logger.info(f"\nSaving processed data to {output_path}...")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data_dict = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': self.scaler,  # Save scaler for inverse transform later
            'lookback': self.lookback
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data_dict, f)
        
        logger.info(f"✓ Processed data saved to {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        logger.info("\n" + "="*60)
        logger.info("✓ Data processing complete!")
        logger.info("="*60 + "\n")
        
        return data_dict


def main():
    """Main function to run data processing pipeline."""
    logger.info("\n" + "="*60)
    logger.info("AIoT Energy Predictor - Data Processing")
    logger.info("="*60)
    
    # Create processor
    processor = DataProcessor()
    
    # Run processing pipeline
    data = processor.process()
    
    logger.info("Data processing pipeline complete!")


if __name__ == '__main__':
    main()
