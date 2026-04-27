"""Database setup and data ingestion for IoT sensor readings."""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

from sqlalchemy import create_engine, Column, DateTime, Float, Integer, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError

from .utils import setup_logging, get_data_dir, get_project_root

logger = setup_logging(__name__)

# SQLAlchemy declarative base for ORM models
Base = declarative_base()


class SensorReading(Base):
    """
    ORM model representing a single IoT sensor reading.
    
    Attributes:
        id: Primary key
        timestamp: DateTime of the reading (UTC)
        temperature: Temperature in Celsius
        humidity: Humidity as percentage (0-100)
        occupancy: Occupancy percentage (0-100)
        energy_consumption: Energy consumption in kWh
    """
    __tablename__ = 'sensor_readings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, unique=True, index=True)
    temperature = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    occupancy = Column(Float, nullable=False)
    energy_consumption = Column(Float, nullable=False)
    
    # Create index on timestamp for faster queries
    __table_args__ = (
        Index('idx_timestamp', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<SensorReading(timestamp={self.timestamp}, temp={self.temperature}°C, energy={self.energy_consumption}kWh)>"


class DatabaseManager:
    """
    Manages database operations for IoT sensor data.
    
    Handles:
    - Database connection and table creation
    - CSV data ingestion into database
    - Data verification and integrity checks
    """
    
    def __init__(self, db_path='data/sensor_data.db'):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file (relative to project root)
        """
        # Convert relative path to absolute path
        if isinstance(db_path, str) and not db_path.startswith('/'):
            db_path = get_project_root() / db_path
        
        self.db_path = Path(db_path)
        self.db_url = f'sqlite:///{self.db_path}'
        
        # Create engine and session factory
        self.engine = create_engine(
            self.db_url,
            echo=False,  # Set to True for SQL logging
            connect_args={'timeout': 30}  # 30 second timeout
        )
        
        # Create session factory
        Session = scoped_session(sessionmaker(bind=self.engine))
        self.Session = Session
        
        logger.info(f"Initialized DatabaseManager: {self.db_path}")
    
    def create_tables(self):
        """
        Create database tables from ORM models.
        
        Creates all tables defined in Base.metadata.
        
        Returns:
            bool: True if successful
        """
        try:
            logger.info("Creating database tables...")
            Base.metadata.create_all(self.engine)
            logger.info("✓ Database tables created successfully")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Error creating tables: {e}")
            return False
    
    def ingest_csv(self, csv_path=None):
        """
        Load CSV data into database.
        
        Reads CSV file and creates SensorReading records in database.
        Uses bulk insert for efficiency.
        
        Args:
            csv_path: Path to CSV file. If None, uses default data/raw_sensor_data.csv
        
        Returns:
            int: Number of records inserted
        """
        if csv_path is None:
            csv_path = get_data_dir() / 'raw_sensor_data.csv'
        else:
            csv_path = Path(csv_path)
        
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return 0
        
        try:
            logger.info(f"Loading CSV file: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Data validation
            required_columns = ['Timestamp', 'Temperature', 'Humidity', 'Occupancy', 'Energy_Consumption']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return 0
            
            # Convert timestamp to datetime
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            logger.info(f"CSV loaded: {len(df)} rows")
            
            # Create SensorReading objects
            session = self.Session()
            try:
                # Clear existing data (optional, comment out to append)
                # session.query(SensorReading).delete()
                
                logger.info("Creating SensorReading objects...")
                readings = [
                    SensorReading(
                        timestamp=row['Timestamp'],
                        temperature=float(row['Temperature']),
                        humidity=float(row['Humidity']),
                        occupancy=float(row['Occupancy']),
                        energy_consumption=float(row['Energy_Consumption'])
                    )
                    for _, row in df.iterrows()
                ]
                
                # Bulk insert
                logger.info(f"Inserting {len(readings)} records into database...")
                session.add_all(readings)
                session.commit()
                
                logger.info(f"✓ Successfully ingested {len(readings)} records")
                return len(readings)
            
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error during ingestion: {e}")
                return 0
            finally:
                session.close()
        
        except Exception as e:
            logger.error(f"Error reading CSV or ingesting data: {e}")
            return 0
    
    def query_all_data(self):
        """
        Query all sensor readings from database.
        
        Returns:
            pandas.DataFrame: DataFrame with all sensor data
        """
        session = self.Session()
        try:
            query_result = session.query(SensorReading).all()
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
            return df
        finally:
            session.close()
    
    def verify_integrity(self):
        """
        Verify data integrity in database.
        
        Checks:
        - Total number of records
        - Date range
        - Data statistics
        - Null values
        """
        session = self.Session()
        try:
            logger.info("\n" + "="*60)
            logger.info("Database Integrity Check")
            logger.info("="*60)
            
            # Count records
            count = session.query(SensorReading).count()
            logger.info(f"Total records in database: {count}")
            
            if count == 0:
                logger.warning("⚠ No records found in database")
                return False
            
            # Get date range
            oldest = session.query(SensorReading.timestamp).order_by(SensorReading.timestamp.asc()).first()
            newest = session.query(SensorReading.timestamp).order_by(SensorReading.timestamp.desc()).first()
            
            logger.info(f"Date range: {oldest[0]} to {newest[0]}")
            
            # Query all data for statistics
            df = self.query_all_data()
            
            # Check for nulls
            null_counts = df.isnull().sum()
            if null_counts.any():
                logger.warning(f"⚠ Null values found:\n{null_counts}")
            else:
                logger.info("✓ No null values found")
            
            # Print statistics
            logger.info("\nData Statistics:")
            logger.info(f"\nTemperature (°C):")
            logger.info(f"  Mean: {df['Temperature'].mean():.2f}°C")
            logger.info(f"  Min: {df['Temperature'].min():.2f}°C, Max: {df['Temperature'].max():.2f}°C")
            
            logger.info(f"\nHumidity (%):")
            logger.info(f"  Mean: {df['Humidity'].mean():.2f}%")
            logger.info(f"  Min: {df['Humidity'].min():.2f}%, Max: {df['Humidity'].max():.2f}%")
            
            logger.info(f"\nOccupancy (%):")
            logger.info(f"  Mean: {df['Occupancy'].mean():.2f}%")
            logger.info(f"  Min: {df['Occupancy'].min():.2f}%, Max: {df['Occupancy'].max():.2f}%")
            
            logger.info(f"\nEnergy Consumption (kWh):")
            logger.info(f"  Mean: {df['Energy_Consumption'].mean():.2f}kWh")
            logger.info(f"  Min: {df['Energy_Consumption'].min():.2f}kWh, Max: {df['Energy_Consumption'].max():.2f}kWh")
            
            logger.info("\n" + "="*60)
            logger.info("✓ Database integrity check complete")
            logger.info("="*60 + "\n")
            
            return True
        
        finally:
            session.close()
    
    def close(self):
        """Close database connection."""
        self.Session.remove()
        self.engine.dispose()
        logger.info("Database connection closed")


def main():
    """Main function to set up database and ingest data."""
    logger.info("="*60)
    logger.info("AIoT Energy Predictor - Database Setup")
    logger.info("="*60 + "\n")
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Create tables
    db_manager.create_tables()
    
    # Ingest data from CSV
    records_inserted = db_manager.ingest_csv()
    
    if records_inserted > 0:
        # Verify integrity
        db_manager.verify_integrity()
    else:
        logger.error("No records were inserted. Database setup failed.")
    
    # Close connection
    db_manager.close()
    
    logger.info("Database setup complete!")


if __name__ == '__main__':
    main()
