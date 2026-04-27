"""Data Simulator for generating synthetic IoT sensor readings."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

from src.utils import setup_logging, get_data_dir, Config
from .utils import setup_logging, get_data_dir, Config

logger = setup_logging(__name__)


class IoTDataSimulator:
    """
    Simulates realistic IoT sensor data for a building.
    
    Generates 1 year of hourly readings with seasonal patterns,
    business-hour occupancy fluctuations, and correlated energy consumption.
    """
    
    def __init__(self, start_date=Config.START_DATE, end_date=Config.END_DATE, seed=Config.RANDOM_SEED):
        """
        Initialize the data simulator.
        
        Args:
            start_date: Start date as string (YYYY-MM-DD)
            end_date: End date as string (YYYY-MM-DD)
            seed: Random seed for reproducibility
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.seed = seed
        np.random.seed(seed)
        
        logger.info(f"Initialized IoT Data Simulator: {start_date} to {end_date}")
    
    def generate_timestamps(self):
        """
        Generate hourly timestamps for the date range.
        
        Returns:
            numpy.ndarray: Array of datetime objects (hourly frequency)
        """
        date_range = pd.date_range(self.start_date, self.end_date, freq='H')
        return date_range.values
    
    def generate_temperature(self, timestamps):
        """
        Generate temperature data with seasonal variation.
        
        Temperature follows a sine wave pattern:
        - Summer (June-August): 25-30°C
        - Winter (Dec-Feb): 15-20°C
        - Spring/Fall: 18-25°C
        
        Args:
            timestamps: Array of datetime objects
        
        Returns:
            numpy.ndarray: Temperature values in Celsius
        """
        # Convert timestamps to day of year (0-365)
        timestamps_pd = pd.to_datetime(timestamps)
        day_of_year = timestamps_pd.dayofyear.values
        
        # Base temperature following seasonal sine wave
        # 22°C average + 8°C amplitude
        seasonal_component = 8 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
        base_temp = 22 + seasonal_component
        
        # Add daily variation (cooler at night, warmer during day)
        hour_of_day = timestamps_pd.hour.values
        daily_component = 3 * np.sin((hour_of_day - 6) * np.pi / 12)
        
        # Add realistic noise
        noise = np.random.normal(0, 0.5, len(timestamps))
        
        temperature = base_temp + daily_component + noise
        return temperature
    
    def generate_humidity(self, timestamps):
        """
        Generate humidity data inversely correlated with temperature.
        
        Higher temperatures → lower humidity (and vice versa).
        
        Args:
            timestamps: Array of datetime objects
        
        Returns:
            numpy.ndarray: Humidity values (0-100%)
        """
        temp = self.generate_temperature(timestamps)
        
        # Base humidity: 50-70%
        # Inverse correlation with temperature
        base_humidity = 70 - (temp - 15) * 1.5
        
        # Add noise
        noise = np.random.normal(0, 2, len(timestamps))
        humidity = np.clip(base_humidity + noise, 30, 90)
        
        return humidity
    
    def generate_occupancy(self, timestamps):
        """
        Generate occupancy data with business-hour patterns.
        
        - Business hours (9 AM - 5 PM, weekdays): 50-100% occupancy
        - Evenings (5 PM - 9 PM): 20-40% occupancy
        - Nights (9 PM - 9 AM): 5-10% occupancy
        - Weekends: 15-30% occupancy
        
        Args:
            timestamps: Array of datetime objects
        
        Returns:
            numpy.ndarray: Occupancy percentages (0-100)
        """
        timestamps_pd = pd.to_datetime(timestamps)
        hour = timestamps_pd.hour.values
        dayofweek = timestamps_pd.dayofweek.values  # 0=Mon, 6=Sun
        
        occupancy = np.zeros(len(timestamps))
        
        for i, ts in enumerate(timestamps_pd):
            h = hour[i]
            dow = dayofweek[i]
            
            # Weekday logic
            if dow < 5:  # Monday to Friday
                if 9 <= h < 17:  # Business hours
                    base_occ = 75 + np.random.uniform(-20, 20)  # 50-100%
                elif 17 <= h < 21:  # Evening
                    base_occ = 30 + np.random.uniform(-10, 10)  # 20-40%
                else:  # Night
                    base_occ = 8 + np.random.uniform(-3, 3)  # 5-10%
            else:  # Weekend
                base_occ = 20 + np.random.uniform(-15, 15)  # 5-35%
            
            occupancy[i] = np.clip(base_occ, 0, 100)
        
        return occupancy
    
    def generate_energy_consumption(self, temperature, humidity, occupancy):
        """
        Generate energy consumption as a function of sensors.
        
        Energy formula:
        - Base load: 400 kWh (HVAC, lighting, equipment)
        - Temperature load: |temp - 22| * 20 (heating/cooling delta)
        - Occupancy load: occupancy * 8 (lights, equipment per person)
        - Humidity load: |humidity - 50| * 1 (dehumidification)
        
        Args:
            temperature: Temperature array
            humidity: Humidity array
            occupancy: Occupancy array
        
        Returns:
            numpy.ndarray: Energy consumption in kWh
        """
        # Base load (HVAC, equipment, lighting)
        base_load = 400
        
        # Heating/cooling load (proportional to temperature difference from 22°C)
        hvac_load = np.abs(temperature - 22) * 20
        
        # Occupancy-dependent load (lights, equipment, ventilation)
        occ_load = (occupancy / 100) * 8 * 200  # Up to 1600 kWh at 100% occupancy
        
        # Humidity load (dehumidification)
        humidity_load = np.abs(humidity - 50) * 1
        
        # Random noise to simulate variability
        noise = np.random.normal(0, 30, len(temperature))
        
        energy = base_load + hvac_load + occ_load + humidity_load + noise
        
        # Energy is always positive
        energy = np.clip(energy, 100, 2500)
        
        return energy
    
    def generate_dataset(self):
        """
        Generate complete 1-year IoT dataset.
        
        Returns:
            pandas.DataFrame: DataFrame with columns:
                - Timestamp: Datetime
                - Temperature: Celsius
                - Humidity: Percentage (0-100)
                - Occupancy: Percentage (0-100)
                - Energy_Consumption: kWh
        """
        logger.info("Generating timestamps...")
        timestamps = self.generate_timestamps()
        
        logger.info("Generating temperature data...")
        temperature = self.generate_temperature(timestamps)
        
        logger.info("Generating humidity data...")
        humidity = self.generate_humidity(timestamps)
        
        logger.info("Generating occupancy data...")
        occupancy = self.generate_occupancy(timestamps)
        
        logger.info("Generating energy consumption...")
        energy_consumption = self.generate_energy_consumption(temperature, humidity, occupancy)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Timestamp': timestamps,
            'Temperature': temperature,
            'Humidity': humidity,
            'Occupancy': occupancy,
            'Energy_Consumption': energy_consumption
        })
        
        # Introduce realistic missing values (~2%)
        logger.info(f"Introducing {Config.MISSING_DATA_RATIO*100}% missing values...")
        missing_mask = np.random.rand(len(df)) < Config.MISSING_DATA_RATIO
        df.loc[missing_mask, ['Temperature', 'Humidity', 'Occupancy']] = np.nan
        
        # Forward fill to handle missing values (common for time-series)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Generated dataset shape: {df.shape}")
        logger.info(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
        logger.info(f"Energy consumption range: {df['Energy_Consumption'].min():.2f} - {df['Energy_Consumption'].max():.2f} kWh")
        
        return df
    
    def save_to_csv(self, output_path=None):
        """
        Generate dataset and save to CSV.
        
        Args:
            output_path: Path to save CSV. If None, uses default data/ directory.
        
        Returns:
            pandas.DataFrame: Generated dataset
        """
        if output_path is None:
            output_path = get_data_dir() / 'raw_sensor_data.csv'
        
        df = self.generate_dataset()
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")
        
        return df


def main():
    """Main function to run data simulation."""
    logger.info("=" * 60)
    logger.info("AIoT Energy Predictor - Data Simulation")
    logger.info("=" * 60)
    
    # Create simulator
    simulator = IoTDataSimulator()
    
    # Generate and save dataset
    df = simulator.save_to_csv()
    
    # Print summary statistics
    logger.info("\nDataset Summary Statistics:")
    logger.info(f"\nTemperature (°C):")
    logger.info(f"  Mean: {df['Temperature'].mean():.2f}, Std: {df['Temperature'].std():.2f}")
    logger.info(f"  Min: {df['Temperature'].min():.2f}, Max: {df['Temperature'].max():.2f}")
    
    logger.info(f"\nHumidity (%):")
    logger.info(f"  Mean: {df['Humidity'].mean():.2f}, Std: {df['Humidity'].std():.2f}")
    logger.info(f"  Min: {df['Humidity'].min():.2f}, Max: {df['Humidity'].max():.2f}")
    
    logger.info(f"\nOccupancy (%):")
    logger.info(f"  Mean: {df['Occupancy'].mean():.2f}, Std: {df['Occupancy'].std():.2f}")
    logger.info(f"  Min: {df['Occupancy'].min():.2f}, Max: {df['Occupancy'].max():.2f}")
    
    logger.info(f"\nEnergy Consumption (kWh):")
    logger.info(f"  Mean: {df['Energy_Consumption'].mean():.2f}, Std: {df['Energy_Consumption'].std():.2f}")
    logger.info(f"  Min: {df['Energy_Consumption'].min():.2f}, Max: {df['Energy_Consumption'].max():.2f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Data simulation complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
