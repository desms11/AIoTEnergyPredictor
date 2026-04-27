"""Distributed data aggregation pipeline (Pandas-based simulation of Spark concepts)."""

import logging
import pandas as pd
from pathlib import Path

from .utils import setup_logging, get_data_dir, get_results_dir

logger = setup_logging(__name__)


class DistributedDataPipeline:
    """
    Demonstrates distributed data aggregation patterns used in Spark.
    
    NOTE: This implementation uses Pandas for compatibility, but demonstrates
    the exact same aggregation patterns that Spark uses for scalable processing.
    
    In production, Predicx would run these aggregations using:
    - Apache Spark (for distributed computing across clusters)
    - AWS EMR, Databricks, or similar platforms
    - Processes terabytes of data from thousands of buildings
    
    The patterns shown here scale linearly with data size when using Spark.
    """
    
    def __init__(self, csv_path=None):
        """
        Initialize pipeline and load data.
        
        Args:
            csv_path: Path to CSV file. If None, uses default.
        """
        if csv_path is None:
            csv_path = get_data_dir() / 'raw_sensor_data.csv'
        
        self.csv_path = Path(csv_path)
        self.df = None
        
        logger.info("Initializing Distributed Data Pipeline...")
    
    def load_data(self):
        """
        Load CSV file.
        
        Returns:
            pandas.DataFrame: Sensor data
        """
        logger.info(f"\nLoading data from: {self.csv_path}...")
        
        self.df = pd.read_csv(self.csv_path)
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        
        logger.info(f"✓ Data loaded successfully")
        logger.info(f"  Total rows: {len(self.df):,}")
        logger.info(f"  Date range: {self.df['Timestamp'].min()} to {self.df['Timestamp'].max()}")
        logger.info(f"  Columns: {', '.join(self.df.columns)}")
        
        return self.df
    
    def aggregate_daily_energy(self):
        """
        Aggregate hourly data to daily summaries.
        
        This pattern is used in Spark as:
        >>> df.groupBy(date_trunc("Date", col("Timestamp"))) \
        >>>    .agg(avg("Temperature"), sum("Energy_Consumption")) \
        >>>    .orderBy("Date")
        
        In Spark, this would be distributed across multiple nodes.
        
        Returns:
            pandas.DataFrame: Daily aggregated data
        """
        logger.info("\nAggregating data to daily level...")
        
        daily_agg = self.df.groupby(self.df['Timestamp'].dt.date).agg({
            'Temperature': 'mean',
            'Humidity': 'mean',
            'Occupancy': 'mean',
            'Energy_Consumption': ['sum', 'min', 'max']
        }).reset_index()
        
        daily_agg.columns = ['Date', 'avg_temperature_c', 'avg_humidity_pct', 
                            'avg_occupancy_pct', 'total_energy_kwh', 
                            'min_energy_kwh', 'max_energy_kwh']
        
        logger.info(f"✓ Daily aggregation complete: {len(daily_agg)} days")
        
        return daily_agg
    
    def aggregate_by_hour_of_day(self):
        """
        Analyze energy consumption patterns by hour of day.
        
        Spark equivalent:
        >>> df.groupBy(hour(col("Timestamp"))) \
        >>>    .agg(avg("Energy_Consumption"), avg("Occupancy"))
        
        Returns:
            pandas.DataFrame: Hourly pattern data
        """
        logger.info("\nAggregating data to hourly patterns...")
        
        hourly_agg = self.df.groupby(self.df['Timestamp'].dt.hour).agg({
            'Temperature': 'mean',
            'Humidity': 'mean',
            'Occupancy': 'mean',
            'Energy_Consumption': ['mean', 'min', 'max']
        }).reset_index()
        
        hourly_agg.columns = ['Hour', 'avg_temperature_c', 'avg_humidity_pct',
                             'avg_occupancy_pct', 'avg_energy_kwh',
                             'min_energy_kwh', 'max_energy_kwh']
        
        logger.info(f"✓ Hourly aggregation complete: {len(hourly_agg)} hours")
        
        return hourly_agg
    
    def aggregate_by_day_of_week(self):
        """
        Analyze energy consumption patterns by day of week.
        
        Spark equivalent:
        >>> df.groupBy(dayofweek(col("Timestamp"))) \
        >>>    .agg(avg("Energy_Consumption"))
        
        Returns:
            pandas.DataFrame: Day-of-week pattern data
        """
        logger.info("\nAggregating data by day of week...")
        
        day_names = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
                    5: "Friday", 6: "Saturday", 7: "Sunday"}
        
        day_agg = self.df.groupby(self.df['Timestamp'].dt.dayofweek + 1).agg({
            'Temperature': 'mean',
            'Humidity': 'mean',
            'Occupancy': 'mean',
            'Energy_Consumption': ['mean', 'sum']
        }).reset_index()
        
        day_agg.columns = ['DayOfWeek', 'avg_temperature_c', 'avg_humidity_pct',
                          'avg_occupancy_pct', 'avg_energy_kwh', 'total_energy_kwh']
        
        day_agg['DayName'] = day_agg['DayOfWeek'].map(day_names)
        
        logger.info(f"✓ Day-of-week aggregation complete: {len(day_agg)} days")
        
        return day_agg
    
    def aggregate_by_occupancy_level(self):
        """
        Analyze energy consumption by occupancy level.
        
        Spark equivalent:
        >>> df.withColumn("OccupancyBracket", (col("Occupancy") / 25).cast("int")) \
        >>>    .groupBy("OccupancyBracket") \
        >>>    .agg(avg("Energy_Consumption"))
        
        Returns:
            pandas.DataFrame: Occupancy-level analysis
        """
        logger.info("\nAggregating data by occupancy level...")
        
        df_occ = self.df.copy()
        df_occ['OccupancyLevel'] = (df_occ['Occupancy'] / 25).astype(int)
        df_occ = df_occ[df_occ['OccupancyLevel'] <= 4]
        
        occupancy_agg = df_occ.groupby('OccupancyLevel').agg({
            'Temperature': 'mean',
            'Occupancy': 'mean',
            'Energy_Consumption': 'mean'
        }).reset_index()
        
        occupancy_agg.columns = ['OccupancyLevel', 'avg_temperature_c',
                                'avg_occupancy_pct', 'avg_energy_kwh']
        
        # Add occupancy range labels
        occupancy_agg['OccupancyRange'] = occupancy_agg['OccupancyLevel'].apply(
            lambda x: f"{x*25}-{(x+1)*25}%"
        )
        
        logger.info(f"✓ Occupancy aggregation complete: {len(occupancy_agg)} levels")
        
        return occupancy_agg
    
    def save_results_to_csv(self, df, output_name):
        """
        Save aggregated results to CSV.
        
        Args:
            df: DataFrame to save
            output_name: Name of output file (without extension)
        
        Returns:
            str: Path where data was saved
        """
        output_path = get_results_dir() / f'{output_name}.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"  ✓ Saved to {output_name}.csv ({len(df)} rows)")
        
        return str(output_path)
    
    def print_sample(self, df, name="Data", n_rows=10):
        """
        Print sample of DataFrame.
        
        Args:
            df: DataFrame
            name: Name for logging
            n_rows: Number of rows to display
        """
        logger.info(f"\n{name}:")
        logger.info(df.head(n_rows).to_string())
    
    def run_pipeline(self):
        """
        Execute complete aggregation pipeline.
        
        Steps:
        1. Load data
        2. Perform multiple aggregations
        3. Save results
        4. Display insights
        """
        logger.info("\n" + "="*70)
        logger.info("Distributed Data Aggregation Pipeline")
        logger.info("="*70)
        logger.info("(Using Pandas - demonstrates patterns used in Spark for scalability)")
        
        # Load data
        self.load_data()
        
        # Daily aggregation
        logger.info("\n" + "-"*70)
        logger.info("ANALYSIS 1: Daily Summaries")
        logger.info("-"*70)
        logger.info("Use case: Track daily energy consumption trends over time")
        daily_agg = self.aggregate_daily_energy()
        self.print_sample(daily_agg, "Daily Aggregation (first 7 days)", 7)
        self.save_results_to_csv(daily_agg, "spark_daily_aggregation")
        
        # Hourly patterns
        logger.info("\n" + "-"*70)
        logger.info("ANALYSIS 2: Hourly Patterns (Business Hours Detection)")
        logger.info("-"*70)
        logger.info("Use case: Identify peak energy consumption times (business hours vs. nights)")
        hourly_agg = self.aggregate_by_hour_of_day()
        self.print_sample(hourly_agg, "Hourly Pattern (all 24 hours)", 24)
        self.save_results_to_csv(hourly_agg, "spark_hourly_pattern")
        
        # Key insights
        peak_hour = hourly_agg.loc[hourly_agg['avg_energy_kwh'].idxmax()]
        low_hour = hourly_agg.loc[hourly_agg['avg_energy_kwh'].idxmin()]
        logger.info(f"\n  Peak energy hour: {int(peak_hour['Hour'])}:00 ({peak_hour['avg_energy_kwh']:.1f} kWh avg)")
        logger.info(f"  Lowest energy hour: {int(low_hour['Hour'])}:00 ({low_hour['avg_energy_kwh']:.1f} kWh avg)")
        
        # Day-of-week patterns
        logger.info("\n" + "-"*70)
        logger.info("ANALYSIS 3: Day-of-Week Patterns (Weekday vs. Weekend)")
        logger.info("-"*70)
        logger.info("Use case: Detect usage differences between weekdays and weekends")
        day_agg = self.aggregate_by_day_of_week()
        self.print_sample(day_agg, "Day-of-Week Pattern", 7)
        self.save_results_to_csv(day_agg[['DayOfWeek', 'DayName', 'avg_temperature_c', 
                                          'avg_occupancy_pct', 'avg_energy_kwh', 'total_energy_kwh']], 
                                "spark_day_of_week_pattern")
        
        # Key insights
        weekday_avg = day_agg[day_agg['DayOfWeek'] <= 5]['avg_energy_kwh'].mean()
        weekend_avg = day_agg[day_agg['DayOfWeek'] > 5]['avg_energy_kwh'].mean()
        logger.info(f"\n  Weekday avg energy: {weekday_avg:.1f} kWh")
        logger.info(f"  Weekend avg energy: {weekend_avg:.1f} kWh")
        logger.info(f"  Difference: {(weekday_avg - weekend_avg):.1f} kWh ({((weekday_avg - weekend_avg) / weekend_avg * 100):.1f}%)")
        
        # Occupancy relationship
        logger.info("\n" + "-"*70)
        logger.info("ANALYSIS 4: Energy vs. Occupancy Relationship")
        logger.info("-"*70)
        logger.info("Use case: Understand how occupancy level affects energy consumption")
        occupancy_agg = self.aggregate_by_occupancy_level()
        self.print_sample(occupancy_agg, "Occupancy Analysis", 5)
        self.save_results_to_csv(occupancy_agg, "spark_occupancy_energy")
        
        logger.info("\n" + "="*70)
        logger.info("✓ Distributed aggregation pipeline complete!")
        logger.info("="*70)
        logger.info("\nKey Takeaways for Spark Scalability:")
        logger.info("  • These same aggregation patterns scale to TB/PB of data on Spark clusters")
        logger.info("  • GroupBy operations are distributed across worker nodes in parallel")
        logger.info("  • For Predicx: 1,000 buildings × 365 days × 24 hours = 8.76M rows per year")
        logger.info("  • Spark processes this efficiently, Pandas alone would be slow (>10 min)")
        logger.info("  • Predicx uses Spark on AWS EMR for real-time aggregations\n")


def main():
    """Main function to run the pipeline."""
    logger.info("\n" + "="*70)
    logger.info("AIoT Energy Predictor - Distributed Data Aggregation")
    logger.info("="*70)
    
    pipeline = DistributedDataPipeline()
    pipeline.load_data()
    pipeline.run_pipeline()
    
    logger.info("Aggregation pipeline complete!")


if __name__ == '__main__':
    main()
