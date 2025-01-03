import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from typing import Optional, Dict

from .base_model import BaseModel

class LightGBMMediaMixModel(BaseModel):
    """LightGBM implementation of Media Mix Model"""
    def __init__(self, num_leaves=31, learning_rate=0.1, n_estimators=100, **kwargs):
        super().__init__(name="LightGBM MMM")
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'verbose': -1,
            **kwargs
        }
        self.model = lgb.LGBMRegressor(**self.params)
        
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
        """Get feature importance"""
        if self.model is None or not self.feature_names:
            return None
            
        importance = pd.Series(
            self.model.feature_importances_,
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
            'target_name': self.target_name,
            'params': self.params
        }
        joblib.dump(model_data, path)
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.target_name = model_data['target_name']
        self.params = model_data['params'] 