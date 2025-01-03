import numpy as np
import pandas as pd
import pytest
from backend.models.model_factory import ModelFactory

@pytest.fixture
def sample_data():
    """Generate synthetic data with known relationships"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate spend data with realistic patterns
    tv_spend = np.random.gamma(10, 1000, n_samples)  # TV typically has larger budgets
    radio_spend = np.random.gamma(5, 500, n_samples)
    social_spend = np.random.gamma(8, 800, n_samples)
    
    # Add seasonality
    time = np.arange(n_samples)
    seasonality = 1000 * np.sin(2 * np.pi * time / 365) + 500
    
    # Create non-linear effects
    tv_effect = 0.5 * tv_spend + 0.1 * tv_spend**2
    radio_effect = 300 * np.log(radio_spend + 1)
    social_effect = 0.3 * social_spend - 0.05 * social_spend**2
    
    # Add interaction effects
    tv_social_interaction = 0.1 * tv_spend * social_spend / 10000
    
    # Generate revenue with noise
    revenue = (tv_effect + radio_effect + social_effect + 
              tv_social_interaction + seasonality + 
              np.random.normal(0, 1000, n_samples))
    
    # Create DataFrame
    data = pd.DataFrame({
        'TV_Spend': tv_spend,
        'Radio_Spend': radio_spend,
        'Social_Spend': social_spend,
        'Revenue': revenue
    })
    
    return data

def test_linear_model(sample_data):
    """Test Linear model with interactions"""
    X = sample_data.drop('Revenue', axis=1)
    y = sample_data['Revenue']
    
    model = ModelFactory.create_model('linear', include_interactions=True)
    model.fit(X, y)
    metrics = model.evaluate(X, y)
    
    print("\nLinear Model Metrics:")
    print(f"R² Score: {metrics['r2_score']:.3f}")
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"MAE: {metrics['mae']:.3f}")
    
    assert metrics['r2_score'] > 0.6, "Linear model R² should be reasonable"

def test_lightgbm_model(sample_data):
    """Test LightGBM model"""
    X = sample_data.drop('Revenue', axis=1)
    y = sample_data['Revenue']
    
    model = ModelFactory.create_model('lightgbm', n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    metrics = model.evaluate(X, y)
    
    print("\nLightGBM Model Metrics:")
    print(f"R² Score: {metrics['r2_score']:.3f}")
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"MAE: {metrics['mae']:.3f}")
    
    assert metrics['r2_score'] > 0.7, "LightGBM model R² should be good"

def test_xgboost_model(sample_data):
    """Test XGBoost model"""
    X = sample_data.drop('Revenue', axis=1)
    y = sample_data['Revenue']
    
    model = ModelFactory.create_model('xgboost', n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    metrics = model.evaluate(X, y)
    
    print("\nXGBoost Model Metrics:")
    print(f"R² Score: {metrics['r2_score']:.3f}")
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"MAE: {metrics['mae']:.3f}")
    
    assert metrics['r2_score'] > 0.7, "XGBoost model R² should be good"

def test_model_cross_validation(sample_data):
    """Test cross-validation for all models"""
    X = sample_data.drop('Revenue', axis=1)
    y = sample_data['Revenue']
    
    for model_type in ['linear', 'lightgbm', 'xgboost']:
        model = ModelFactory.create_model(model_type)
        cv_results = ModelFactory.cross_validate(model, X, y, cv=5)
        
        print(f"\n{model_type.upper()} Cross-Validation Results:")
        print(f"R² Score: {cv_results['mean_r2']:.3f} ± {cv_results['std_r2']:.3f}")
        print(f"RMSE: {cv_results['mean_rmse']:.3f} ± {cv_results['std_rmse']:.3f}")
        print(f"MAE: {cv_results['mean_mae']:.3f} ± {cv_results['std_mae']:.3f}")
        
        assert cv_results['mean_r2'] > 0.6, f"{model_type} CV R² should be reasonable" 