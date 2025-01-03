from typing import Dict, Type, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, KFold

from .base_model import BaseModel
from .lightgbm_model import LightGBMMediaMixModel
from .xgboost_model import XGBoostMediaMixModel
from .bayesian_model import BayesianMediaMixModel
from .linear_model import LinearMediaMixModel

class ModelFactory:
    """Factory class for creating Media Mix Models"""
    
    _models: Dict[str, Type[BaseModel]] = {
        'linear': LinearMediaMixModel,
        'lightgbm': LightGBMMediaMixModel,
        'xgboost': XGBoostMediaMixModel,
        'bayesian': BayesianMediaMixModel
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseModel:
        """Create a model instance of the specified type"""
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model_class = cls._models[model_type]
        
        # Filter kwargs based on model type
        if model_type == 'linear':
            # Linear model doesn't accept any parameters
            filtered_kwargs = {}
        elif model_type == 'bayesian':
            # Bayesian model accepts num_epochs and learning_rate
            filtered_kwargs = {
                'num_epochs': kwargs.get('num_epochs', 1000),
                'learning_rate': kwargs.get('learning_rate', 0.01)
            }
        else:
            # Other models accept all parameters
            filtered_kwargs = kwargs.copy()  # Create a copy to avoid shared state
            
        return model_class(**filtered_kwargs)
    
    @classmethod
    def preprocess_data(cls, X):
        """Preprocess data using standardization"""
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    
    @classmethod
    def cross_validate(cls, model: BaseModel, X, y, cv=5, time_based=False):
        """Perform cross-validation with option for time-based splitting"""
        if time_based:
            cv_splitter = TimeSeriesSplit(n_splits=cv)
        else:
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
            
        scores = {
            'test_r2': [],
            'test_rmse': [],
            'test_mae': []
        }
        
        for train_idx, test_idx in cv_splitter.split(X):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            
            # Create a new instance of the model for each fold
            fold_model = cls.create_model(model.__class__.__name__.lower().replace('mediamixmodel', ''))
            fold_model.fit(X_train, y_train)
            metrics = fold_model.evaluate(X_test, y_test)
            
            scores['test_r2'].append(metrics['r2_score'])
            scores['test_rmse'].append(metrics['rmse'])
            scores['test_mae'].append(metrics['mae'])
            
        return scores
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """Get a dictionary of available models and their descriptions"""
        return {
            'linear': 'Linear regression model',
            'lightgbm': 'LightGBM gradient boosting model',
            'xgboost': 'XGBoost gradient boosting model',
            'bayesian': 'Bayesian Media Mix Model'
        } 