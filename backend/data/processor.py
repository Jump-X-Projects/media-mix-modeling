import pandas as pd
import numpy as np
from typing import List, Tuple

class DataProcessor:
    """Process and validate data for model training"""
    
    def __init__(self):
        self.data = None
        self.media_columns = None
        self.target_column = None
        
    def setup(self, data: pd.DataFrame, media_columns: List[str], target_column: str):
        """
        Set up the processor with data and column mappings
        
        Args:
            data: Input DataFrame
            media_columns: List of media spend columns
            target_column: Name of target (revenue) column
        """
        # Validate required columns
        required_cols = ['date'] + media_columns + [target_column]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Validate numeric columns first
        numeric_cols = media_columns + [target_column]
        for col in numeric_cols:
            # Check for missing values first
            if data[col].isnull().any():
                raise ValueError(f"Column {col} contains missing values")
            
            # Convert to numeric to ensure proper validation
            try:
                data[col] = pd.to_numeric(data[col])
            except:
                raise ValueError(f"Column {col} contains non-numeric values")
            
            # Check for negative values
            if (data[col] < 0).any():
                raise ValueError(f"Column {col} contains negative values")
        
        # Validate date column
        try:
            data['date'] = pd.to_datetime(data['date'])
        except Exception as e:
            raise ValueError(f"Invalid date format in date column: {str(e)}")
        
        # Check for duplicate dates
        if data['date'].duplicated().any():
            raise ValueError("Duplicate dates found in data")
        
        # Sort by date and reset index
        data = data.sort_values('date').reset_index(drop=True)
        
        # Ensure dates are continuous
        date_range = pd.date_range(start=data['date'].min(), end=data['date'].max(), freq='D')
        if len(date_range) != len(data):
            raise ValueError("Missing dates in data")
            
        # Compare dates using string representation to avoid timezone issues
        data_dates = data['date'].dt.strftime('%Y-%m-%d')
        range_dates = date_range.strftime('%Y-%m-%d')
        if not data_dates.equals(pd.Series(range_dates)):
            raise ValueError("Dates are not continuous")
        
        self.data = data
        self.media_columns = media_columns
        self.target_column = target_column
        
    def process(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Process data for model training
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if self.data is None:
            raise ValueError("Processor not set up. Call setup() first.")
            
        # Extract features and target
        X = data[self.media_columns].copy()
        y = data[self.target_column].copy()
        
        return X, y 