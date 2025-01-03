import pytest
import numpy as np
import pandas as pd
from backend.models.model_factory import (
    ModelFactory, ModelError, ModelCreationError,
    ModelTrainingError, ModelPredictionError, InvalidInputError
)

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    X = pd.DataFrame({
        'TV_Spend': np.random.normal(100, 20, 100),
        'Radio_Spend': np.random.normal(30, 10, 100)
    })
    y = 2 * X['TV_Spend'] + 3 * X['Radio_Spend'] + np.random.normal(0, 10, 100)
    return X, y

@pytest.fixture
def invalid_data():
    """Create invalid data for testing"""
    X = pd.DataFrame({
        'TV_Spend': [1, 2, np.nan, 4],
        'Radio_Spend': [10, 20, 30, 40]
    })
    y = pd.Series([100, 200, 300, 400])
    return X, y

class TestModelCreationErrors:
    def test_invalid_model_type(self):
        """Test creating model with invalid type"""
        with pytest.raises(ModelCreationError) as exc:
            ModelFactory.create_model("invalid_model")
        assert "Unknown model type" in str(exc.value)
    
    def test_invalid_parameters(self):
        """Test creating model with invalid parameters"""
        with pytest.raises(ModelCreationError) as exc:
            # LightGBM will raise error for negative n_estimators
            ModelFactory.create_model("lightgbm", n_estimators=-1)
        assert "n_estimators must be positive" in str(exc.value)

class TestInputValidationErrors:
    def test_missing_values(self, invalid_data):
        """Test training with missing values"""
        X, y = invalid_data
        model = ModelFactory.create_model("linear")
        with pytest.raises(InvalidInputError) as exc:
            model.fit(X, y)
        assert "missing values" in str(exc.value)
    
    def test_none_input(self):
        """Test prediction with None input"""
        model = ModelFactory.create_model("linear")
        with pytest.raises(InvalidInputError) as exc:
            model._validate_input(None)  # Test validation directly
        assert "cannot be None" in str(exc.value)
    
    def test_wrong_shape(self, sample_data):
        """Test prediction with wrong input shape"""
        X, y = sample_data
        model = ModelFactory.create_model("linear")
        model.fit(X, y)
        
        # Try predicting with wrong number of features
        wrong_X = pd.DataFrame({'TV_Spend': [100]})  # Missing Radio_Spend
        with pytest.raises(ModelPredictionError) as exc:
            model.predict(wrong_X)
        assert "feature names" in str(exc.value).lower()

class TestTrainingErrors:
    def test_predict_before_fit(self, sample_data):
        """Test prediction before fitting"""
        X, _ = sample_data
        model = ModelFactory.create_model("linear")
        with pytest.raises(ModelError) as exc:
            model.predict(X)
        assert "must be fitted" in str(exc.value)
    
    def test_feature_importance_before_fit(self):
        """Test getting feature importance before fitting"""
        model = ModelFactory.create_model("linear")
        with pytest.raises(ModelError) as exc:
            model.feature_importances(['feature1'])
        assert "must be fitted" in str(exc.value)

@pytest.mark.asyncio
class TestEndToEndErrors:
    async def test_training_endpoint_errors(self, client):
        """Test error responses from training endpoint"""
        # Test invalid model type
        response = client.post(
            "/api/train",
            files={
                'file': ('data.csv', b'date,TV_Spend,Radio_Spend,Revenue\n2023-01-01,100,50,1000'),
            },
            data={
                'model_type': 'invalid_model',
                'media_columns': 'TV_Spend,Radio_Spend',
                'target_column': 'Revenue'
            }
        )
        assert response.status_code == 400
        data = response.json()
        assert "Unknown model type" in data['details']
        
        # Test missing columns
        response = client.post(
            "/api/train",
            files={
                'file': ('data.csv', b'date,TV_Spend,Revenue\n2023-01-01,100,1000'),
            },
            data={
                'model_type': 'linear',
                'media_columns': 'TV_Spend,NonExistentColumn',
                'target_column': 'Revenue'
            }
        )
        assert response.status_code == 400
        data = response.json()
        assert "Missing required columns" in data['details']
    
    async def test_prediction_endpoint_errors(self, client):
        """Test error responses from prediction endpoint"""
        # First create a model
        response = client.post(
            "/api/train",
            files={
                'file': ('data.csv', b'date,TV_Spend,Radio_Spend,Revenue\n2023-01-01,100,50,1000'),
            },
            data={
                'model_type': 'linear',
                'media_columns': 'TV_Spend,Radio_Spend',
                'target_column': 'Revenue'
            }
        )
        assert response.status_code == 200
        model_id = response.json()['model_id']
        
        # Test prediction with missing features
        response = client.post(
            f"/api/models/{model_id}/predict",
            json={
                'TV_Spend': [100]
                # Missing Radio_Spend
            }
        )
        assert response.status_code == 400
        data = response.json()
        assert "Missing features" in data['details']
        
        # Test prediction with invalid model ID
        response = client.post(
            "/api/models/invalid-id/predict",
            json={
                'TV_Spend': [100],
                'Radio_Spend': [50]
            }
        )
        assert response.status_code == 404
        data = response.json()
        assert "Model not found" in data['details'] 