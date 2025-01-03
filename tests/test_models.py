import pytest
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from backend.models.model_factory import ModelFactory
import pandas as pd

@pytest.fixture
def sample_data():
    """Create sample data for testing with non-linear relationships"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data
    X = {
        'TV_Spend': np.random.uniform(1000, 5000, n_samples),
        'Radio_Spend': np.random.uniform(500, 2000, n_samples),
        'Social_Spend': np.random.uniform(300, 1500, n_samples)
    }
    X = pd.DataFrame(X)
    
    # Add non-linear relationships and interactions
    y = (
        2.5 * X['TV_Spend'] +  # Linear effect
        0.3 * (X['TV_Spend'] ** 2) / 1000 +  # Quadratic effect
        1.8 * X['Radio_Spend'] +
        -0.2 * (X['Radio_Spend'] ** 2) / 100 +  # Diminishing returns
        1.2 * X['Social_Spend'] +
        0.5 * np.sqrt(X['Social_Spend']) * X['TV_Spend'] / 100 +  # Interaction effect
        np.random.normal(0, 1000, n_samples)  # Increased noise
    )
    
    return X, pd.Series(y, name='Revenue')

def calculate_metrics(y_true, y_pred):
    """Calculate and return model quality metrics"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {
        'R-squared': r2,
        'RMSE': rmse,
        'MAE': mae
    }

def test_linear_model(sample_data):
    """Test Linear Regression model"""
    X, y = sample_data
    model = ModelFactory.create_model('linear')
    model.fit(X, y)
    y_pred = model.predict(X)
    
    metrics = calculate_metrics(y, y_pred)
    print("\nLinear Regression Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    assert metrics['R-squared'] > 0.7  # Expect reasonable fit

def test_lightgbm_model(sample_data):
    """Test LightGBM model"""
    X, y = sample_data
    model = ModelFactory.create_model('lightgbm')
    model.fit(X, y)
    y_pred = model.predict(X)
    
    metrics = calculate_metrics(y, y_pred)
    print("\nLightGBM Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    assert metrics['R-squared'] > 0.7

def test_xgboost_model(sample_data):
    """Test XGBoost model"""
    X, y = sample_data
    model = ModelFactory.create_model('xgboost')
    model.fit(X, y)
    y_pred = model.predict(X)
    
    metrics = calculate_metrics(y, y_pred)
    print("\nXGBoost Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    assert metrics['R-squared'] > 0.7

def test_bayesian_model(sample_data):
    """Test Bayesian MMM including uncertainty estimates"""
    X, y = sample_data
    model = ModelFactory.create_model('bayesian')
    model.fit(X, y)
    y_pred = model.predict(X)
    
    metrics = calculate_metrics(y, y_pred)
    print("\nBayesian MMM Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Additional Bayesian-specific tests
    assert hasattr(model, 'get_uncertainty')
    uncertainty = model.get_uncertainty(X)
    assert uncertainty.shape == (len(X),)
    assert metrics['R-squared'] > 0.7

def test_model_factory():
    """Test model creation and parameter handling"""
    models = ['linear', 'lightgbm', 'xgboost', 'bayesian']
    for model_type in models:
        model = ModelFactory.create_model(model_type)
        assert model is not None
        
def test_data_processor(sample_data):
    """Test data preprocessing functionality"""
    X, y = sample_data
    
    # Test scaling
    X_scaled = ModelFactory.preprocess_data(X)
    assert X_scaled.shape == X.shape
    assert np.all(np.abs(X_scaled.mean()) < 0.1)  # Check if mean is close to 0
    assert np.all(np.abs(X_scaled.std() - 1) < 0.1)  # Check if std is close to 1

def test_cross_validation(sample_data):
    """Test CV with different splitting strategies"""
    X, y = sample_data
    model = ModelFactory.create_model('linear')
    
    # Time-based split
    cv_scores = ModelFactory.cross_validate(
        model, X, y,
        cv=5,
        time_based=True
    )
    
    print("\nCross-Validation Scores:")
    print(f"Mean R-squared: {np.mean(cv_scores['test_r2']):.4f}")
    print(f"Std R-squared: {np.std(cv_scores['test_r2']):.4f}")
    
    assert len(cv_scores['test_r2']) == 5
    assert np.mean(cv_scores['test_r2']) > 0.6 