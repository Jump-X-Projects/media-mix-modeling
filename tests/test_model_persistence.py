import pytest
import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path

from backend.models.model_persistence import ModelPersistence
from backend.models.model_factory import ModelFactory

@pytest.fixture
def test_storage_dir():
    """Create a temporary storage directory for testing"""
    test_dir = Path("test_model_storage")
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    # Cleanup after tests
    shutil.rmtree(test_dir)

@pytest.fixture
def persistence(test_storage_dir):
    """Create a ModelPersistence instance for testing"""
    return ModelPersistence(storage_dir=str(test_storage_dir))

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    X = pd.DataFrame({
        'TV_Spend': np.random.normal(100, 20, 100),
        'Radio_Spend': np.random.normal(50, 10, 100)
    })
    y = 2 * X['TV_Spend'] + 1.5 * X['Radio_Spend'] + np.random.normal(0, 10, 100)
    return X, y

def test_save_and_load_model(persistence, sample_data):
    """Test basic save and load functionality"""
    X, y = sample_data
    
    # Create and train a model
    model = ModelFactory.create_model("linear")
    model.fit(X, y)
    
    # Save the model
    model_id = "test_model_1"
    metadata = {
        'media_columns': X.columns.tolist(),
        'model_type': 'linear',
        'num_features': len(X.columns)
    }
    persistence.save_model(model_id, model, metadata)
    
    # Load the model
    loaded = persistence.load_model(model_id)
    
    # Verify model and metadata
    assert loaded is not None
    assert 'model' in loaded
    assert loaded['model_type'] == 'linear'
    assert loaded['num_features'] == len(X.columns)
    
    # Verify predictions match
    original_preds = model.predict(X)
    loaded_preds = loaded['model'].predict(X)
    np.testing.assert_array_almost_equal(original_preds, loaded_preds)

def test_model_persistence_across_instances(test_storage_dir, sample_data):
    """Test model persistence when creating new ModelPersistence instances"""
    X, y = sample_data
    
    # First instance - save model
    persistence1 = ModelPersistence(str(test_storage_dir))
    model = ModelFactory.create_model("linear")
    model.fit(X, y)
    
    model_id = "test_model_2"
    metadata = {'model_type': 'linear'}
    persistence1.save_model(model_id, model, metadata)
    
    # Second instance - load model
    persistence2 = ModelPersistence(str(test_storage_dir))
    loaded = persistence2.load_model(model_id)
    
    assert loaded is not None
    assert loaded['model_type'] == 'linear'

def test_delete_model(persistence, sample_data):
    """Test model deletion"""
    X, y = sample_data
    
    # Save a model
    model = ModelFactory.create_model("linear")
    model.fit(X, y)
    
    model_id = "test_model_3"
    persistence.save_model(model_id, model, {'model_type': 'linear'})
    
    # Verify model exists
    assert persistence.load_model(model_id) is not None
    
    # Delete model
    success = persistence.delete_model(model_id)
    assert success
    
    # Verify model is gone
    assert persistence.load_model(model_id) is None
    
    # Verify model file is deleted
    model_path = persistence.storage_dir / f"{model_id}.joblib"
    assert not model_path.exists()

def test_list_models(persistence, sample_data):
    """Test listing saved models"""
    X, y = sample_data
    
    # Save multiple models
    models_data = [
        ("model1", "linear"),
        ("model2", "lightgbm"),
        ("model3", "xgboost")
    ]
    
    for model_id, model_type in models_data:
        model = ModelFactory.create_model(model_type)
        model.fit(X, y)
        persistence.save_model(model_id, model, {'model_type': model_type})
    
    # List models
    models = persistence.list_models()
    
    # Verify
    assert len(models) == len(models_data)
    for model_id, model_type in models_data:
        assert model_id in models
        assert models[model_id]['model_type'] == model_type

def test_invalid_model_handling(persistence):
    """Test handling of invalid model IDs and corrupted files"""
    # Test loading non-existent model
    assert persistence.load_model("nonexistent_model") is None
    
    # Test deleting non-existent model
    assert not persistence.delete_model("nonexistent_model")
    
    # Create corrupted model file
    model_id = "corrupted_model"
    model_path = persistence.storage_dir / f"{model_id}.joblib"
    model_path.write_text("corrupted data")
    
    persistence.metadata[model_id] = {
        "model_path": str(model_path),
        "model_type": "linear"
    }
    persistence._save_metadata()
    
    # Test loading corrupted model
    assert persistence.load_model(model_id) is None

def test_metadata_persistence(test_storage_dir):
    """Test metadata persistence across instances"""
    # First instance - save metadata
    persistence1 = ModelPersistence(str(test_storage_dir))
    model_id = "test_model_4"
    metadata = {"test_key": "test_value"}
    
    # Create dummy model file
    model_path = persistence1.storage_dir / f"{model_id}.joblib"
    model_path.write_text("dummy model")
    
    persistence1.metadata[model_id] = {
        "model_path": str(model_path),
        **metadata
    }
    persistence1._save_metadata()
    
    # Second instance - verify metadata
    persistence2 = ModelPersistence(str(test_storage_dir))
    assert model_id in persistence2.metadata
    assert persistence2.metadata[model_id]["test_key"] == "test_value" 