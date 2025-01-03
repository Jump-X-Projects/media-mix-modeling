import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, List, Optional, Tuple
import optuna
from copy import deepcopy

from .model_factory import ModelFactory
from .base_model import BaseModel

class ModelTuner:
    """Handles model tuning and cross-validation"""
    
    def __init__(
        self,
        model_type: str,
        n_splits: int = 5,
        n_trials: int = 50,
        random_state: int = 42
    ):
        self.model_type = model_type
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.random_state = random_state
        self.best_params = None
        self.cv_results = None
        
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Optional[Dict] = None
    ) -> Dict[str, List[float]]:
        """Perform time series cross-validation"""
        # Create model
        model = ModelFactory.create_model(self.model_type, params)
        
        # Setup cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # Define scoring metrics
        scoring = {
            'r2': make_scorer(r2_score),
            'rmse': make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred))),
            'mae': make_scorer(mean_absolute_error)
        }
        
        # Create wrapper class for sklearn compatibility
        model_wrapper = ModelWrapper(model)
        
        # Perform cross-validation
        cv_results = cross_validate(
            model_wrapper,
            X,
            y,
            cv=tscv,
            scoring=scoring,
            return_train_score=True
        )
        
        self.cv_results = cv_results
        return cv_results
    
    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict:
        """Tune model hyperparameters using Optuna"""
        
        def objective(trial):
            # Get hyperparameter suggestions based on model type
            params = self._suggest_params(trial)
            
            # Perform cross-validation with suggested parameters
            cv_results = self.cross_validate(X, y, params)
            
            # Return mean validation RMSE
            return np.mean(cv_results['test_rmse'])
        
        # Create and run Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        return study.best_params
    
    def _suggest_params(self, trial: optuna.Trial) -> Dict:
        """Suggest hyperparameters based on model type"""
        if self.model_type == 'lightgbm':
            return {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300)
            }
        elif self.model_type == 'xgboost':
            return {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.1, 1.0),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300)
            }
        elif self.model_type == 'bayesian':
            return {
                'num_epochs': trial.suggest_int('num_epochs', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
                'prior_scale': trial.suggest_float('prior_scale', 0.1, 2.0)
            }
        else:  # linear
            return {}

class ModelWrapper:
    """Wrapper class to make our models compatible with sklearn's cross_validation"""
    
    def __init__(self, model: BaseModel):
        self.model = model
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the model"""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def get_params(self, deep: bool = True) -> Dict:
        """Get model parameters"""
        return self.model.params if hasattr(self.model, 'params') else {} 