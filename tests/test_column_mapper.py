"""Tests for column mapping functionality"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backend.utils.column_mapper import ColumnMapper

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    dates = pd.date_range('2023-01-01', periods=10)
    return pd.DataFrame({
        'Date': dates,
        'TV_Spend': np.random.uniform(100, 1000, 10),
        'Radio_Cost': np.random.uniform(50, 500, 10),
        'Social_Investment': np.random.uniform(200, 800, 10),
        'Daily_Revenue': np.random.uniform(1000, 5000, 10)
    })

@pytest.fixture
def invalid_data():
    """Create invalid data for testing"""
    return pd.DataFrame({
        'Date': ['invalid_date'] * 5,
        'TV_Spend': [-100] * 5,  # Negative values
        'Radio_Cost': [np.nan] * 5,  # Missing values
        'Revenue': [1000] * 5  # Constant values
    })

def test_column_identification(sample_data):
    """Test column identification based on patterns"""
    mapper = ColumnMapper()
    
    # Test spend column identification
    spend_cols = mapper.identify_columns(sample_data, 'spend')
    assert len(spend_cols) == 3
    assert 'TV_Spend' in spend_cols
    assert 'Radio_Cost' in spend_cols
    assert 'Social_Investment' in spend_cols
    
    # Test revenue column identification
    revenue_cols = mapper.identify_columns(sample_data, 'revenue')
    assert len(revenue_cols) == 1
    assert 'Daily_Revenue' in revenue_cols
    
    # Test date column identification
    date_cols = mapper.identify_columns(sample_data, 'date')
    assert len(date_cols) == 1
    assert 'Date' in date_cols

def test_column_type_validation(sample_data):
    """Test column type validation"""
    mapper = ColumnMapper()
    
    # Test valid mappings
    valid, error = mapper.validate_column_types(
        sample_data,
        {
            'date': 'Date',
            'spend': 'TV_Spend',
            'revenue': 'Daily_Revenue'
        }
    )
    assert valid
    assert error == ""
    
    # Test invalid date
    invalid_data = sample_data.copy()
    invalid_data['Date'] = 'invalid'
    valid, error = mapper.validate_column_types(
        invalid_data,
        {'date': 'Date'}
    )
    assert not valid
    assert "must be a valid date" in error

def test_numeric_validation(sample_data, invalid_data):
    """Test numeric column validation"""
    mapper = ColumnMapper()
    
    # Test valid numeric columns
    valid, error = mapper.validate_numeric_columns(
        sample_data,
        ['TV_Spend', 'Radio_Cost', 'Daily_Revenue']
    )
    assert valid
    assert error == ""
    
    # Test negative values
    valid, error = mapper.validate_numeric_columns(
        invalid_data,
        ['TV_Spend']
    )
    assert not valid
    assert "negative values" in error
    
    # Test missing values
    valid, error = mapper.validate_numeric_columns(
        invalid_data,
        ['Radio_Cost']
    )
    assert not valid
    assert "missing values" in error

def test_date_validation(sample_data):
    """Test date column validation"""
    mapper = ColumnMapper()
    
    # Test valid date column
    valid, error, dates = mapper.validate_date_column(
        sample_data,
        'Date'
    )
    assert valid
    assert error == ""
    assert isinstance(dates, pd.Series)
    
    # Test invalid date format
    invalid_data = sample_data.copy()
    invalid_data['Date'] = 'invalid'
    valid, error, dates = mapper.validate_date_column(
        invalid_data,
        'Date'
    )
    assert not valid
    assert "Could not parse dates" in error
    assert dates is None

def test_column_suggestions(sample_data):
    """Test column mapping suggestions"""
    mapper = ColumnMapper()
    
    suggestions = mapper.suggest_column_mappings(sample_data)
    
    assert 'spend' in suggestions
    assert 'revenue' in suggestions
    assert 'date' in suggestions
    
    assert len(suggestions['spend']) == 3
    assert len(suggestions['revenue']) == 1
    assert len(suggestions['date']) == 1

def test_required_columns_validation():
    """Test validation of required columns"""
    mapper = ColumnMapper()
    
    # Test valid mappings
    valid, error = mapper.validate_required_columns({
        'date': ['Date'],
        'spend': ['TV_Spend'],
        'revenue': ['Revenue']
    })
    assert valid
    assert error == ""
    
    # Test missing required column
    valid, error = mapper.validate_required_columns({
        'date': ['Date'],
        'spend': ['TV_Spend']
    })
    assert not valid
    assert "Missing required column type" in error 