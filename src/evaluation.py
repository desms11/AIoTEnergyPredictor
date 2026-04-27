"""Model evaluation and visualization of predictions."""

import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .model import EnergyPredictionModel
from .utils import setup_logging, get_data_dir, get_models_dir, get_results_dir

logger = setup_logging(__name__)

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 5)


class ModelEvaluator:
    """
    Evaluates trained LSTM model on test data.
    
    Includes:
    - Prediction generation
    - Metric calculation (MAE, RMSE, R²)
    - Inverse transformation to original scale
    - Visualization of results
    """
    
    def __init__(self, model_path=None, data_path=None):
        """
        Initialize evaluator with model and data.
        
        Args:
            model_path: Path to saved model (.h5). If None, uses default.
            data_path: Path to processed data pickle. If None, uses default.
        """
        if model_path is None:
            model_path = get_models_dir() / 'lstm_model.h5'
        if data_path is None:
            data_path = get_data_dir() / 'processed_data.pkl'
        
        logger.info("Loading model and data...")
        
        # Load model
        self.model = EnergyPredictionModel.load(str(model_path))
        logger.info(f"✓ Model loaded from {model_path}")
        
        # Load processed data
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.scaler = self.data['scaler']
        self.lookback = self.data['lookback']
        logger.info(f"✓ Data loaded (lookback={self.lookback})")
    
    def evaluate_on_test_set(self):
        """
        Generate predictions on test set and calculate metrics.
        
        Returns:
            tuple: (y_test_original, y_pred_original, metrics_dict)
                where metrics_dict contains 'MAE', 'RMSE', 'R2'
        """
        logger.info("\nGenerating predictions on test set...")
        
        # Get test data
        X_test = self.data['X_test']
        y_test = self.data['y_test']
        
        # Reshape for LSTM
        X_test_reshaped = X_test.reshape((X_test.shape[0], self.lookback, 4))
        
        # Generate predictions (on normalized scale [0,1])
        y_pred = self.model.predict(X_test_reshaped).flatten()
        
        logger.info(f"✓ Generated {len(y_pred)} predictions")
        
        # Inverse transform to original scale
        logger.info("Inverse transforming predictions to original scale...")
        
        # The scaler was fit on 4 features: [Temp, Humidity, Occupancy, Energy]
        # We need to inverse transform only the energy values (index 3)
        
        # Create dummy arrays with full feature set for inverse transform
        n_samples = len(y_test)
        
        # Create array with dummy values for features 0-2, actual values for feature 3
        y_test_full = np.zeros((n_samples, 4))
        y_test_full[:, 3] = y_test
        
        y_pred_full = np.zeros((n_samples, 4))
        y_pred_full[:, 3] = y_pred
        
        # Inverse transform
        y_test_original = self.scaler.inverse_transform(y_test_full)[:, 3]
        y_pred_original = self.scaler.inverse_transform(y_pred_full)[:, 3]
        
        logger.info(f"✓ Inverse transformed to original scale (kWh)")
        
        # Calculate metrics
        logger.info("\nCalculating metrics...")
        
        mae = mean_absolute_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        r2 = r2_score(y_test_original, y_pred_original)
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        logger.info(f"✓ Metrics calculated:")
        logger.info(f"  MAE: {mae:.4f} kWh")
        logger.info(f"  RMSE: {rmse:.4f} kWh")
        logger.info(f"  R² Score: {r2:.4f}")
        
        return y_test_original, y_pred_original, metrics
    
    def plot_predictions_vs_actual(self, y_actual, y_pred, n_points=500):
        """
        Plot actual vs predicted energy consumption over time.
        
        Args:
            y_actual: Actual values
            y_pred: Predicted values
            n_points: Number of points to plot (default 500)
        """
        logger.info(f"\nGenerating predictions vs actual plot ({n_points} points)...")
        
        plt.figure(figsize=(14, 5))
        
        # Plot only first n_points to avoid crowding
        points = min(n_points, len(y_actual))
        
        plt.plot(range(points), y_actual[:points], label='Actual', linewidth=1.5, color='#2E86AB')
        plt.plot(range(points), y_pred[:points], label='Predicted', linewidth=1.5, 
                linestyle='--', color='#A23B72')
        
        plt.xlabel('Time (hours)', fontsize=11, fontweight='bold')
        plt.ylabel('Energy Consumption (kWh)', fontsize=11, fontweight='bold')
        plt.title('LSTM Energy Prediction: Actual vs Predicted', fontsize=13, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = get_results_dir() / 'predictions_vs_actual.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved to {output_path}")
    
    def plot_error_distribution(self, y_actual, y_pred):
        """
        Plot histogram of prediction errors.
        
        Args:
            y_actual: Actual values
            y_pred: Predicted values
        """
        logger.info("Generating error distribution plot...")
        
        errors = y_actual - y_pred
        
        plt.figure(figsize=(10, 5))
        plt.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='#F18F01')
        
        plt.xlabel('Prediction Error (kWh)', fontsize=11, fontweight='bold')
        plt.ylabel('Frequency', fontsize=11, fontweight='bold')
        plt.title('Distribution of Prediction Errors', fontsize=13, fontweight='bold')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        plt.axvline(x=errors.mean(), color='green', linestyle='--', linewidth=2, 
                   label=f'Mean Error: {errors.mean():.2f} kWh')
        
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        output_path = get_results_dir() / 'error_distribution.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved to {output_path}")
    
    def plot_residuals(self, y_actual, y_pred, n_points=500):
        """
        Plot residuals over time to detect patterns or bias.
        
        Args:
            y_actual: Actual values
            y_pred: Predicted values
            n_points: Number of points to plot (default 500)
        """
        logger.info(f"Generating residuals plot ({n_points} points)...")
        
        residuals = y_actual - y_pred
        points = min(n_points, len(residuals))
        
        plt.figure(figsize=(14, 5))
        plt.scatter(range(points), residuals[:points], alpha=0.5, s=15, color='#0F3460')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.axhline(y=residuals.mean(), color='green', linestyle='--', linewidth=2, 
                   label=f'Mean: {residuals.mean():.2f} kWh')
        
        plt.xlabel('Time (hours)', fontsize=11, fontweight='bold')
        plt.ylabel('Residual (kWh)', fontsize=11, fontweight='bold')
        plt.title('Residuals Over Time (First 500 Hours)', fontsize=13, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = get_results_dir() / 'residuals.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved to {output_path}")
    
    def save_metrics(self, metrics):
        """
        Save evaluation metrics to text file.
        
        Args:
            metrics: Dictionary with keys 'MAE', 'RMSE', 'R2'
        """
        output_path = get_results_dir() / 'model_metrics.txt'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving metrics to {output_path}...")
        
        with open(output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("LSTM Energy Prediction Model - Test Set Metrics\n")
            f.write("="*60 + "\n\n")
            f.write("Model Architecture:\n")
            f.write("  - LSTM Layer 1: 64 units + Dropout(0.2)\n")
            f.write("  - LSTM Layer 2: 32 units + Dropout(0.2)\n")
            f.write("  - Dense Layer: 16 units\n")
            f.write("  - Output Layer: 1 unit (Energy Consumption)\n\n")
            f.write("Input Features:\n")
            f.write("  - Lookback window: 24 hours\n")
            f.write("  - Features per timestep: 4 (Temperature, Humidity, Occupancy, Energy)\n")
            f.write("  - Total input dimensions: 24 × 4 = 96 values\n\n")
            f.write("="*60 + "\n")
            f.write("METRICS:\n")
            f.write("="*60 + "\n")
            for key, value in metrics.items():
                if key == 'R2':
                    f.write(f"{key}: {value:.6f}\n")
                else:
                    f.write(f"{key}: {value:.4f} kWh\n")
            f.write("\n")
            f.write("Interpretation:\n")
            f.write(f"  - Average prediction error (MAE): {metrics['MAE']:.2f} kWh\n")
            f.write(f"  - Model explains {metrics['R2']*100:.1f}% of variance (R²)\n")
            f.write("="*60 + "\n")
        
        logger.info(f"✓ Metrics saved to {output_path}")


def main():
    """Main function to evaluate model and generate visualizations."""
    logger.info("\n" + "="*60)
    logger.info("AIoT Energy Predictor - Model Evaluation")
    logger.info("="*60 + "\n")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate on test set
    y_actual, y_pred, metrics = evaluator.evaluate_on_test_set()
    
    # Generate visualizations
    logger.info("\n" + "-"*60)
    logger.info("Generating Visualizations")
    logger.info("-"*60)
    
    evaluator.plot_predictions_vs_actual(y_actual, y_pred)
    evaluator.plot_error_distribution(y_actual, y_pred)
    evaluator.plot_residuals(y_actual, y_pred)
    
    # Save metrics
    evaluator.save_metrics(metrics)
    
    logger.info("\n" + "="*60)
    logger.info("✓ Evaluation complete!")
    logger.info("="*60 + "\n")


if __name__ == '__main__':
    main()
