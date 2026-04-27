"""LSTM deep learning model for energy consumption prediction."""

import pickle
import logging
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from .utils import setup_logging, get_data_dir, get_models_dir, Config

logger = setup_logging(__name__)


class EnergyPredictionModel:
    """
    LSTM-based deep learning model for time-series energy consumption prediction.
    
    Architecture:
    - Input layer: (lookback, n_features) = (24, 4)
    - LSTM layer 1: 64 units with dropout
    - LSTM layer 2: 32 units with dropout
    - Dense layer: 16 units
    - Output layer: 1 unit (next hour's energy)
    
    The model learns to predict the next hour's energy consumption
    given the past 24 hours of sensor data.
    """
    
    def __init__(self, input_shape=(24, 4), config=None):
        """
        Initialize the LSTM model.
        
        Args:
            input_shape: Tuple (lookback_window, n_features). Default (24, 4).
            config: Configuration object with model parameters.
                   If None, uses Config defaults.
        """
        self.input_shape = input_shape
        self.config = config or Config
        self.model = None
        self.history = None
        
        logger.info(f"Initializing EnergyPredictionModel with input shape {input_shape}")
        self._build_model()
    
    def _build_model(self):
        """
        Build the Sequential LSTM model.
        
        Architecture:
        - LSTM(64) + Dropout(0.2) - captures long-term patterns
        - LSTM(32) + Dropout(0.2) - refines patterns
        - Dense(16) - feature combination
        - Dense(1) - output (next hour's energy)
        """
        logger.info("Building LSTM model architecture...")
        
        self.model = Sequential([
            # First LSTM layer: 64 units
            # return_sequences=True because we feed to another LSTM layer
            LSTM(
                units=self.config.LSTM_UNITS_1,
                activation='relu',
                input_shape=self.input_shape,
                return_sequences=True,
                name='lstm_1'
            ),
            # Dropout to prevent overfitting
            Dropout(rate=self.config.DROPOUT_RATE, name='dropout_1'),
            
            # Second LSTM layer: 32 units
            # return_sequences=False because we want single output value
            LSTM(
                units=self.config.LSTM_UNITS_2,
                activation='relu',
                name='lstm_2'
            ),
            # Dropout to prevent overfitting
            Dropout(rate=self.config.DROPOUT_RATE, name='dropout_2'),
            
            # Dense layer for feature combination
            Dense(
                units=16,
                activation='relu',
                name='dense_1'
            ),
            
            # Output layer: single value (energy consumption)
            Dense(
                units=1,
                name='output'
            )
        ])
        
        # Compile the model
        logger.info("Compiling model...")
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='mse',  # Mean Squared Error
            metrics=['mae']  # Mean Absolute Error
        )
        
        # Print model summary
        logger.info("Model architecture:")
        self.model.summary()
        logger.info("✓ Model built and compiled successfully")
    
    def train(self, X_train, y_train, X_val, y_val, epochs=None, batch_size=None, verbose=1):
        """
        Train the LSTM model.
        
        Uses early stopping to prevent overfitting:
        - Monitors validation loss
        - Stops if validation loss doesn't improve for 'patience' epochs
        - Restores best weights
        
        Args:
            X_train: Training feature data (n_samples, lookback, n_features)
            y_train: Training target data (n_samples,)
            X_val: Validation feature data
            y_val: Validation target data
            epochs: Number of epochs to train (default: Config.EPOCHS)
            batch_size: Batch size (default: Config.BATCH_SIZE)
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
        
        Returns:
            dict: Training history with keys 'loss', 'val_loss', 'mae', 'val_mae'
        """
        if epochs is None:
            epochs = self.config.EPOCHS
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
        
        logger.info("\n" + "="*60)
        logger.info("Training LSTM Model")
        logger.info("="*60 + "\n")
        
        logger.info(f"Training configuration:")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {self.config.LEARNING_RATE}")
        logger.info(f"  Early stopping patience: {self.config.EARLY_STOPPING_PATIENCE}")
        logger.info(f"  Dropout rate: {self.config.DROPOUT_RATE}")
        
        logger.info(f"\nData shapes:")
        logger.info(f"  X_train: {X_train.shape}")
        logger.info(f"  y_train: {y_train.shape}")
        logger.info(f"  X_val: {X_val.shape}")
        logger.info(f"  y_val: {y_val.shape}\n")
        
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=self.config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train the model
        logger.info("Starting training...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        logger.info("\n✓ Training complete!")
        logger.info(f"  Final training loss: {self.history.history['loss'][-1]:.6f}")
        logger.info(f"  Final validation loss: {self.history.history['val_loss'][-1]:.6f}")
        logger.info(f"  Final training MAE: {self.history.history['mae'][-1]:.6f}")
        logger.info(f"  Final validation MAE: {self.history.history['val_mae'][-1]:.6f}")
        
        return self.history.history
    
    def predict(self, X):
        """
        Generate predictions on new data.
        
        Args:
            X: Feature data (n_samples, lookback, n_features)
        
        Returns:
            numpy.ndarray: Predictions (n_samples,)
        """
        predictions = self.model.predict(X)
        return predictions.flatten()
    
    def save(self, model_path=None):
        """
        Save the trained model to disk.
        
        Saves in HDF5 format (.h5).
        
        Args:
            model_path: Path to save model. If None, uses default models/ directory.
        
        Returns:
            str: Path where model was saved
        """
        if model_path is None:
            model_path = get_models_dir() / 'lstm_model.h5'
        else:
            model_path = Path(model_path)
        
        # Ensure directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {model_path}...")
        self.model.save(model_path)
        logger.info(f"✓ Model saved successfully ({model_path.stat().st_size / 1024 / 1024:.2f} MB)")
        
        return str(model_path)
    
    @staticmethod
    def load(model_path):
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to saved model file (.h5)
        
        Returns:
            tensorflow.keras.models.Sequential: Loaded model
        """
        logger.info(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        logger.info(f"✓ Model loaded successfully")
        return model


def main():
    """Main function to train the LSTM model."""
    logger.info("\n" + "="*60)
    logger.info("AIoT Energy Predictor - LSTM Model Training")
    logger.info("="*60 + "\n")
    
    # Load processed data
    logger.info("Loading processed data...")
    data_path = get_data_dir() / 'processed_data.pkl'
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    lookback = data['lookback']
    
    logger.info(f"✓ Loaded processed data with lookback={lookback}")
    
    # Reshape for LSTM: (samples, timesteps, features)
    # Input shape must be (samples, 24, 4)
    logger.info("\nReshaping data for LSTM...")
    X_train_reshaped = X_train.reshape((X_train.shape[0], lookback, 4))
    X_val_reshaped = X_val.reshape((X_val.shape[0], lookback, 4))
    X_test_reshaped = X_test.reshape((X_test.shape[0], lookback, 4))
    
    logger.info(f"✓ Reshaped training data: {X_train_reshaped.shape}")
    logger.info(f"✓ Reshaped validation data: {X_val_reshaped.shape}")
    logger.info(f"✓ Reshaped test data: {X_test_reshaped.shape}")
    
    # Build and train model
    model = EnergyPredictionModel(input_shape=(lookback, 4))
    
    # Train
    history = model.train(
        X_train_reshaped, y_train,
        X_val_reshaped, y_val,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        verbose=1
    )
    
    # Save model
    model.save()
    
    logger.info("\n" + "="*60)
    logger.info("✓ Model training complete!")
    logger.info("="*60 + "\n")


if __name__ == '__main__':
    main()
