import pandas as pd
from typing import Tuple, List
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        self.media_columns = None
        self.target_column = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def setup(self, media_columns: List[str], target_column: str):
        """Configure the processor with column information"""
        self.media_columns = media_columns
        self.target_column = target_column
    
    def read_file(self, file_content: bytes, file_type: str) -> pd.DataFrame:
        """Read data from uploaded file content"""
        try:
            if file_type.lower() in ['.xlsx', '.xls']:
                return pd.read_excel(file_content)
            elif file_type.lower() == '.csv':
                return pd.read_csv(file_content)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")
    
    def validate_columns(self, data: pd.DataFrame) -> bool:
        """Validate that required columns exist in the data"""
        missing_cols = []
        if self.media_columns:
            missing_cols.extend([col for col in self.media_columns if col not in data.columns])
        if self.target_column and self.target_column not in data.columns:
            missing_cols.append(self.target_column)
            
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        return True
    
    def process(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Process the data for modeling"""
        # Validate columns
        self.validate_columns(data)
        
        # Handle missing values
        data = data.copy()
        data[self.media_columns] = data[self.media_columns].fillna(0)
        data[self.target_column] = data[self.target_column].fillna(data[self.target_column].mean())
        
        # Extract features and target
        X = data[self.media_columns]
        y = data[self.target_column]
        
        return X, y 
    
    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features using StandardScaler"""
        if not self.is_fitted:
            self.scaler.fit(X)
            self.is_fitted = True
            
        scaled_data = self.scaler.transform(X)
        return pd.DataFrame(scaled_data, columns=X.columns, index=X.index)
        
    def inverse_scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled features"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted yet")
            
        original_data = self.scaler.inverse_transform(X)
        return pd.DataFrame(original_data, columns=X.columns, index=X.index) 