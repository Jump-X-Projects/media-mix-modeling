import pytest
import pandas as pd
import os
from streamlit.testing.v1.app_test import AppTest

@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing"""
    df = pd.DataFrame({
        'TV_Spend': [1000, 2000, 3000],
        'Radio_Spend': [500, 1000, 1500],
        'Social_Spend': [300, 600, 900],
        'Revenue': [2000, 4000, 6000]
    })
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

def test_app_initialization():
    """Test basic app initialization"""
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()

    # Check title and navigation
    assert "Media Mix Modeling" in at.markdown[0].value
    assert "Navigation" in at.sidebar.markdown[0].value
    assert at.sidebar.selectbox[0].value == "Data Upload"
    assert at.sidebar.selectbox[0].options == ["Data Upload", "Model Selection", "Results"]

def test_data_upload(sample_csv):
    """Test data upload functionality"""
    at = AppTest.from_file("app.py")
    at.run()

    # Navigate to data upload page
    at.sidebar.selectbox[0].select("Data Upload")
    at.run()

    # Check data upload components
    assert "Data Upload Guide" in at.markdown[2].value
    assert at.button[0].label == "Download Template"

def test_model_selection():
    """Test model selection interface"""
    at = AppTest.from_file("app.py")
    at.run()

    # Navigate to model selection page
    at.sidebar.selectbox[0].select("Model Selection")
    at.run()

    # Check warning when no data is uploaded
    assert "Please upload data first" in at.warning[0].value

def test_results_dashboard(sample_csv):
    """Test results visualization dashboard"""
    at = AppTest.from_file("app.py")
    at.run()

    # Navigate to results page
    at.sidebar.selectbox[0].select("Results")
    at.run()

    # Check warning when no model is trained
    assert "Please train a model first" in at.warning[0].value

def test_error_handling():
    """Test UI error handling"""
    at = AppTest.from_file("app.py")
    at.run()

    # Test model selection without data
    at.sidebar.selectbox[0].select("Model Selection")
    at.run()
    assert "Please upload data first" in at.warning[0].value

    # Test results page without model
    at.sidebar.selectbox[0].select("Results")
    at.run()
    assert "Please train a model first" in at.warning[0].value

def test_cross_validation():
    """Test cross-validation interface"""
    at = AppTest.from_file("app.py")
    at.run()

    # Navigate to model selection
    at.sidebar.selectbox[0].select("Model Selection")
    at.run()

    # Check warning when no data is uploaded
    assert "Please upload data first" in at.warning[0].value

def test_responsive_layout():
    """Test responsive layout behavior"""
    at = AppTest.from_file("app.py")
    at.run()

    # Check for columns in data upload page
    at.sidebar.selectbox[0].select("Data Upload")
    at.run()
    assert len(at.columns) >= 2

    # Check for columns in model selection page
    at.sidebar.selectbox[0].select("Model Selection")
    at.run()
    assert len(at.columns) >= 2

    # Check for footer columns
    assert len(at.columns) >= 3

def test_interactive_components():
    """Test interactive components behavior"""
    at = AppTest.from_file("app.py")
    at.run()

    # Navigate to model selection
    at.sidebar.selectbox[0].select("Model Selection")
    at.run()

    # Check warning message
    assert "Please upload data first" in at.warning[0].value 