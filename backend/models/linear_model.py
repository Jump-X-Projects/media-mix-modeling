import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from typing import Optional, Dict

from .base_model import BaseModel

class LinearMediaMixModel(BaseModel):
    """Linear regression implementation of Media Mix Model"""
    def __init__(self):
        super().__init__(name="Linear MMM")
        self.model = LinearRegression()
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model"""
        self.feature_names = X.columns.tolist()
        self.target_name = y.name
        self.model.fit(X, y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance based on coefficients"""
        if self.model is None or not self.feature_names:
            return None
            
        importance = pd.Series(
            np.abs(self.model.coef_),
            index=self.feature_names
        ).sort_values(ascending=False)
        
        return importance
    
    def save(self, path: str) -> None:
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save")
            
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'target_name': self.target_name
        }
        joblib.dump(model_data, path)
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.target_name = model_data['target_name'] 