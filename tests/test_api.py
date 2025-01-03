import pytest
from fastapi.testclient import TestClient
from io import BytesIO
import pandas as pd
import numpy as np
from api import app

client = TestClient(app)

@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features with non-linear relationships
    tv_spend = np.random.normal(3000, 500, n_samples)
    radio_spend = np.random.normal(800, 200, n_samples)
    social_spend = np.random.normal(1000, 300, n_samples)
    
    # Generate target with non-linear effects and interactions
    revenue = (
        2.5 * tv_spend +  # Linear effect
        0.0001 * tv_spend**2 +  # Quadratic effect
        3.0 * np.log(radio_spend + 1) +  # Diminishing returns
        1.5 * social_spend +
        0.0005 * tv_spend * social_spend +  # Interaction effect
        np.random.normal(0, 500, n_samples)  # Noise
    )
    
    data = pd.DataFrame({
        'TV_Spend': tv_spend,
        'Radio_Spend': radio_spend,
        'Social_Spend': social_spend,
        'Revenue': revenue
    })
    
    return data

def test_health_check():
    """Test API health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_list_models():
    """Test listing available models"""
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, dict)
    assert "linear" in models
    assert "lightgbm" in models
    assert "xgboost" in models
    assert "bayesian" in models

def test_train_model(sample_data):
    """Test model training endpoint"""
    # Prepare data
    csv_data = sample_data.to_csv(index=False).encode('utf-8')
    csv_buffer = BytesIO(csv_data)
    
    files = {
        "file": ("data.csv", csv_buffer, "text/csv")
    }
    
    data = {
        "model_type": "lightgbm",
        "target_column": "Revenue",
        "feature_columns": ["TV_Spend", "Radio_Spend", "Social_Spend"]
    }
    
    response = client.post("/train", files=files, data=data)
    assert response.status_code == 200
    assert "model_id" in response.json()
    assert "metrics" in response.json()

def test_predict(sample_data):
    """Test prediction endpoint"""
    # First train a model
    csv_data = sample_data.to_csv(index=False).encode('utf-8')
    csv_buffer = BytesIO(csv_data)
    
    # Train model
    files = {
        "file": ("data.csv", csv_buffer, "text/csv")
    }
    data = {
        "model_type": "lightgbm",
        "target_column": "Revenue",
        "feature_columns": ["TV_Spend", "Radio_Spend", "Social_Spend"]
    }
    train_response = client.post("/train", files=files, data=data)
    assert train_response.status_code == 200
    model_id = train_response.json()["model_id"]
    
    # Make predictions
    predict_data = {
        "TV_Spend": [3000.0, 3500.0],
        "Radio_Spend": [800.0, 900.0],
        "Social_Spend": [1000.0, 1200.0]
    }
    
    response = client.post(f"/predict/{model_id}", json=predict_data)
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert len(response.json()["predictions"]) == 2

def test_evaluate_model(sample_data):
    """Test model evaluation endpoint"""
    # Prepare data
    csv_data = sample_data.to_csv(index=False).encode('utf-8')
    csv_buffer = BytesIO(csv_data)
    
    files = {
        "file": ("data.csv", csv_buffer, "text/csv")
    }
    data = {
        "model_type": "lightgbm",
        "target_column": "Revenue",
        "feature_columns": ["TV_Spend", "Radio_Spend", "Social_Spend"],
        "cv_folds": 3
    }
    
    response = client.post("/evaluate", files=files, data=data)
    assert response.status_code == 200
    assert "cv_scores" in response.json()
    cv_scores = response.json()["cv_scores"]
    assert "mean_r2" in cv_scores
    assert "mean_rmse" in cv_scores
    assert "mean_mae" in cv_scores

def test_error_handling():
    """Test API error responses"""
    # Test invalid model type
    csv_data = b"TV_Spend,Radio_Spend,Social_Spend,Revenue\n3000,800,1000,10000"
    csv_buffer = BytesIO(csv_data)
    
    files = {
        "file": ("data.csv", csv_buffer, "text/csv")
    }
    data = {
        "model_type": "invalid_model",
        "target_column": "Revenue",
        "feature_columns": ["TV_Spend", "Radio_Spend", "Social_Spend"]
    }
    
    response = client.post("/train", files=files, data=data)
    assert response.status_code == 500
    
    # Test invalid model ID for predictions
    predict_data = {
        "TV_Spend": [3000.0],
        "Radio_Spend": [800.0],
        "Social_Spend": [1000.0]
    }
    response = client.post("/predict/invalid_id", json=predict_data)
    assert response.status_code == 404

def test_bayesian_model_uncertainty(sample_data):
    """Test Bayesian model uncertainty estimates"""
    # Prepare data
    csv_data = sample_data.to_csv(index=False).encode('utf-8')
    csv_buffer = BytesIO(csv_data)
    
    # Train Bayesian model
    files = {
        "file": ("data.csv", csv_buffer, "text/csv")
    }
    data = {
        "model_type": "bayesian",
        "target_column": "Revenue",
        "feature_columns": ["TV_Spend", "Radio_Spend", "Social_Spend"]
    }
    train_response = client.post("/train", files=files, data=data)
    assert train_response.status_code == 200
    model_id = train_response.json()["model_id"]
    
    # Make predictions with uncertainty
    predict_data = {
        "TV_Spend": [3000.0, 3500.0],
        "Radio_Spend": [800.0, 900.0],
        "Social_Spend": [1000.0, 1200.0]
    }
    
    response = client.post(f"/predict/{model_id}", json=predict_data, params={"include_uncertainty": True})
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert "uncertainty" in response.json()
    assert len(response.json()["predictions"]) == 2
    assert len(response.json()["uncertainty"]) == 2 