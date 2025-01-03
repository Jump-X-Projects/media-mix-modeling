from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional

class BaseModel(ABC):
    """Base class for all Media Mix Models"""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.model = None
        self.feature_names: List[str] = []
        self.target_name: str = ""
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(X)
        return {
            "r2_score": r2_score(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "mae": mean_absolute_error(y, predictions)
        }
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance if available"""
        return None
    
    def save(self, path: str) -> None:
        """Save model to disk"""
        pass
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        pass 
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 3) -> Dict[str, float]:
        """Perform cross-validation"""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = {
            'r2': [],
            'rmse': [],
            'mae': []
        }
        
        for train_idx, val_idx in kf.split(X):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            
            # Train model
            self.fit(X_train, y_train)
            
            # Evaluate
            metrics = self.evaluate(X_val, y_val)
            scores['r2'].append(metrics['r2_score'])
            scores['rmse'].append(metrics['rmse'])
            scores['mae'].append(metrics['mae'])
        
        # Calculate mean scores
        return {
            'r2_mean': np.mean(scores['r2']),
            'rmse_mean': np.mean(scores['rmse']),
            'mae_mean': np.mean(scores['mae']),
            'r2_std': np.std(scores['r2']),
            'rmse_std': np.std(scores['rmse']),
            'mae_std': np.std(scores['mae'])
        } 