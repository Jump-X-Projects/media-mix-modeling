from typing import Dict, Type, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import logging
from typing import Union, List
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Base exception for model-related errors"""
    pass

class ModelCreationError(ModelError):
    """Raised when model creation fails"""
    pass

class ModelTrainingError(ModelError):
    """Raised when model training fails"""
    pass

class ModelPredictionError(ModelError):
    """Raised when model prediction fails"""
    pass

class InvalidInputError(ModelError):
    """Raised when input data is invalid"""
    pass

class BaseModel:
    def __init__(self):
        self.is_fitted = False
        
    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series] = None) -> None:
        """Validate input data"""
        try:
            if X is None:
                raise InvalidInputError("Input features (X) cannot be None")
            
            if isinstance(X, (pd.DataFrame, pd.Series)):
                if X.isna().any().any():
                    raise InvalidInputError("Input features contain missing values")
            elif isinstance(X, np.ndarray):
                if np.isnan(X).any():
                    raise InvalidInputError("Input features contain missing values")
                    
            if y is not None:
                if isinstance(y, (pd.Series, pd.DataFrame)):
                    if y.isna().any().any():
                        raise InvalidInputError("Target variable contains missing values")
                elif isinstance(y, np.ndarray):
                    if np.isnan(y).any():
                        raise InvalidInputError("Target variable contains missing values")
                        
        except Exception as e:
            logger.error(f"Input validation failed: {str(e)}")
            raise InvalidInputError(f"Input validation failed: {str(e)}")
    
    def feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importances with error handling"""
        try:
            if not self.is_fitted:
                raise ModelError("Model must be fitted before getting feature importances")
                
            if hasattr(self.model, 'feature_importances_'):
                return dict(zip(feature_names, self.model.feature_importances_))
            elif hasattr(self.model, 'coef_'):
                return dict(zip(feature_names, np.abs(self.model.coef_)))
            return dict(zip(feature_names, np.ones(len(feature_names)) / len(feature_names)))
            
        except Exception as e:
            logger.error(f"Error getting feature importances: {str(e)}")
            raise ModelError(f"Failed to get feature importances: {str(e)}")
    
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """Evaluate model with error handling"""
        try:
            if not self.is_fitted:
                raise ModelError("Model must be fitted before evaluation")
                
            self._validate_input(X_test, y_test)
            y_pred = self.predict(X_test)
            
            return {
                'r2_score': np.corrcoef(y_test, y_pred)[0, 1]**2,
                'rmse': np.sqrt(np.mean((y_test - y_pred)**2)),
                'mae': np.mean(np.abs(y_test - y_pred))
            }
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise ModelError(f"Model evaluation failed: {str(e)}")

class LinearModel(BaseModel):
    def __init__(self, include_interactions=True, add_polynomial=False):
        super().__init__()
        try:
            self.include_interactions = include_interactions
            self.add_polynomial = add_polynomial
            self.poly = PolynomialFeatures(degree=2, include_bias=False) if include_interactions else None
            self.scaler = StandardScaler()
            self.model = LinearRegression()
        except Exception as e:
            logger.error(f"Error initializing LinearModel: {str(e)}")
            raise ModelCreationError(f"Failed to initialize LinearModel: {str(e)}")
    
    def fit(self, X, y):
        """Fit the model with error handling"""
        try:
            self._validate_input(X, y)
            X_scaled = self.scaler.fit_transform(X)
            
            if self.include_interactions:
                X_poly = self.poly.fit_transform(X_scaled)
                self.model.fit(X_poly, y)
            else:
                self.model.fit(X_scaled, y)
                
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Error fitting LinearModel: {str(e)}\n{traceback.format_exc()}")
            if isinstance(e, InvalidInputError):
                raise
            raise ModelTrainingError(f"Failed to train LinearModel: {str(e)}")
    
    def predict(self, X):
        try:
            if not self.is_fitted:
                raise ModelError("Model must be fitted before making predictions")
                
            self._validate_input(X)
            X_scaled = self.scaler.transform(X)
            
            if self.include_interactions:
                X_poly = self.poly.transform(X_scaled)
                return self.model.predict(X_poly)
            return self.model.predict(X_scaled)
            
        except Exception as e:
            logger.error(f"Error during LinearModel prediction: {str(e)}")
            raise ModelPredictionError(f"Failed to make predictions: {str(e)}")

class LightGBMModel(BaseModel):
    def __init__(self, n_estimators=100, learning_rate=0.1):
        super().__init__()
        try:
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
                verbose=-1
            )
            self.scaler = StandardScaler()
        except Exception as e:
            logger.error(f"Error initializing LightGBMModel: {str(e)}")
            raise ModelCreationError(f"Failed to initialize LightGBMModel: {str(e)}")
    
    def fit(self, X, y):
        try:
            self._validate_input(X, y)
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Error fitting LightGBMModel: {str(e)}\n{traceback.format_exc()}")
            raise ModelTrainingError(f"Failed to train LightGBMModel: {str(e)}")
    
    def predict(self, X):
        try:
            if not self.is_fitted:
                raise ModelError("Model must be fitted before making predictions")
                
            self._validate_input(X)
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
            
        except Exception as e:
            logger.error(f"Error during LightGBMModel prediction: {str(e)}")
            raise ModelPredictionError(f"Failed to make predictions: {str(e)}")

class XGBoostModel(BaseModel):
    def __init__(self, n_estimators=500, learning_rate=0.1):
        super().__init__()
        try:
            self.model = XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate
            )
        except Exception as e:
            logger.error(f"Error initializing XGBoostModel: {str(e)}")
            raise ModelCreationError(f"Failed to initialize XGBoostModel: {str(e)}")
    
    def fit(self, X, y):
        try:
            self._validate_input(X, y)
            self.model.fit(X, y)
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Error fitting XGBoostModel: {str(e)}\n{traceback.format_exc()}")
            raise ModelTrainingError(f"Failed to train XGBoostModel: {str(e)}")
    
    def predict(self, X):
        try:
            if not self.is_fitted:
                raise ModelError("Model must be fitted before making predictions")
                
            self._validate_input(X)
            return self.model.predict(X)
            
        except Exception as e:
            logger.error(f"Error during XGBoostModel prediction: {str(e)}")
            raise ModelPredictionError(f"Failed to make predictions: {str(e)}")

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
        """Create a model instance with error handling"""
        try:
            if not isinstance(model_type, str):
                raise ModelCreationError("Model type must be a string")
                
            model_type = model_type.lower()
            if model_type not in cls._models:
                raise ModelCreationError(f"Unknown model type: {model_type}")
                
            model_class = cls._models[model_type]
            
            # Filter kwargs to only include valid parameters for the model
            valid_params = model_class.__init__.__code__.co_varnames[1:]  # Exclude 'self'
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            
            # Validate parameters
            if model_type == 'lightgbm' and 'n_estimators' in filtered_kwargs:
                if filtered_kwargs['n_estimators'] <= 0:
                    raise ModelCreationError("n_estimators must be positive")
            
            if set(kwargs.keys()) - set(filtered_kwargs.keys()):
                logger.warning(f"Ignored invalid parameters: {set(kwargs.keys()) - set(filtered_kwargs.keys())}")
            
            return model_class(**filtered_kwargs)
            
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}\n{traceback.format_exc()}")
            if isinstance(e, ModelCreationError):
                raise
            raise ModelCreationError(f"Failed to create model: {str(e)}")
    
    @classmethod
    def cross_validate(cls, model: BaseModel, X, y, cv=5) -> Dict[str, float]:
        """Perform cross-validation with error handling"""
        try:
            if not isinstance(model, BaseModel):
                raise InvalidInputError("Model must be an instance of BaseModel")
                
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
            
            scores = {
                'test_r2': [],
                'test_rmse': [],
                'test_mae': []
            }
            
            for train_idx, test_idx in cv_splitter.split(X):
                try:
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
                    
                    # Train and evaluate
                    model.fit(X_train, y_train)
                    metrics = model.evaluate(X_test, y_test)
                    
                    scores['test_r2'].append(metrics['r2_score'])
                    scores['test_rmse'].append(metrics['rmse'])
                    scores['test_mae'].append(metrics['mae'])
                    
                except Exception as e:
                    logger.error(f"Error in cross-validation fold: {str(e)}")
                    continue
            
            if not scores['test_r2']:
                raise ModelError("All cross-validation folds failed")
            
            return {
                'mean_r2': np.mean(scores['test_r2']),
                'std_r2': np.std(scores['test_r2']),
                'mean_rmse': np.mean(scores['test_rmse']),
                'std_rmse': np.std(scores['test_rmse']),
                'mean_mae': np.mean(scores['test_mae']),
                'std_mae': np.std(scores['test_mae'])
            }
            
        except Exception as e:
            logger.error(f"Error during cross-validation: {str(e)}\n{traceback.format_exc()}")
            raise ModelError(f"Cross-validation failed: {str(e)}") 