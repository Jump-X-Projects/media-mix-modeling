from typing import Dict, Type, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class BaseModel:
    def feature_importances(self, feature_names):
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            return dict(zip(feature_names, np.abs(self.model.coef_)))
        return dict(zip(feature_names, np.ones(len(feature_names)) / len(feature_names)))
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return {
            'r2_score': np.corrcoef(y_test, y_pred)[0, 1]**2,
            'rmse': np.sqrt(np.mean((y_test - y_pred)**2)),
            'mae': np.mean(np.abs(y_test - y_pred))
        }

class LinearModel(BaseModel):
    def __init__(self, include_interactions=True, add_polynomial=False):
        self.include_interactions = include_interactions
        self.add_polynomial = add_polynomial
        self.poly = PolynomialFeatures(degree=2, include_bias=False) if include_interactions else None
        self.scaler = StandardScaler()
        self.model = LinearRegression()
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        if self.include_interactions:
            X_poly = self.poly.fit_transform(X_scaled)
            self.model.fit(X_poly, y)
        else:
            self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        if self.include_interactions:
            X_poly = self.poly.transform(X_scaled)
            return self.model.predict(X_poly)
        return self.model.predict(X_scaled)

class LightGBMModel(BaseModel):
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=31,
            min_child_samples=20,
            min_child_weight=1e-3,
            min_split_gain=1e-3,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            reg_alpha=0.1,
            reg_lambda=0.1,
            objective='regression',
            metric='rmse',
            verbose=-1  # Suppress warnings
        )
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

class XGBoostModel(BaseModel):
    def __init__(self, n_estimators=500, learning_rate=0.1):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class BMMModel(BaseModel):
    def __init__(self, n_samples=2000, warmup_steps=500):
        self.n_samples = n_samples
        self.warmup_steps = warmup_steps
        self.model = LinearRegression()  # Placeholder for now
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class MetaRobynModel(BaseModel):
    def __init__(self, budget_constraint=100):
        self.budget_constraint = budget_constraint
        self.model = LinearRegression()  # Placeholder for now
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class ModelFactory:
    """Factory class for creating Media Mix Models"""
    
    _models = {
        'linear': LinearModel,
        'lightgbm': LightGBMModel,
        'xgboost': XGBoostModel,
        'bmmm': BMMModel,
        'meta robyn': MetaRobynModel
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseModel:
        """Create a model instance of the specified type"""
        if model_type.lower() not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model_class = cls._models[model_type.lower()]
        return model_class(**kwargs)
    
    @classmethod
    def cross_validate(cls, model: BaseModel, X, y, cv=5):
        """Perform cross-validation"""
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        scores = {
            'test_r2': [],
            'test_rmse': [],
            'test_mae': []
        }
        
        for train_idx, test_idx in cv_splitter.split(X):
            # Convert to numpy for indexing if DataFrame
            if isinstance(X, pd.DataFrame):
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
            else:
                X_train = X[train_idx]
                X_test = X[test_idx]
                
            if isinstance(y, pd.Series):
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
            else:
                y_train = y[train_idx]
                y_test = y[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            
            scores['test_r2'].append(metrics['r2_score'])
            scores['test_rmse'].append(metrics['rmse'])
            scores['test_mae'].append(metrics['mae'])
        
        # Convert to required format
        return {
            'mean_r2': np.mean(scores['test_r2']),
            'std_r2': np.std(scores['test_r2']),
            'mean_rmse': np.mean(scores['test_rmse']),
            'std_rmse': np.std(scores['test_rmse']),
            'mean_mae': np.mean(scores['test_mae']),
            'std_mae': np.std(scores['test_mae'])
        } 