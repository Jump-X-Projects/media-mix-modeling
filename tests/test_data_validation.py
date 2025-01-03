import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backend.data.processor import DataProcessor

@pytest.fixture
def processor():
    return DataProcessor()

@pytest.fixture
def valid_data():
    dates = pd.date_range('2023-01-01', '2023-01-10')
    return pd.DataFrame({
        'date': dates,
        'google_spend': np.random.uniform(100, 1000, 10),
        'facebook_spend': np.random.uniform(100, 1000, 10),
        'revenue': np.random.uniform(1000, 5000, 10)
    })

def test_valid_data_setup(processor, valid_data):
    """Test setup with valid data"""
    processor.setup(
        valid_data,
        media_columns=['google_spend', 'facebook_spend'],
        target_column='revenue'
    )
    assert processor.data is not None
    assert processor.media_columns == ['google_spend', 'facebook_spend']
    assert processor.target_column == 'revenue'

def test_missing_columns(processor, valid_data):
    """Test validation of missing columns"""
    data = valid_data.drop('google_spend', axis=1)
    with pytest.raises(ValueError, match="Missing required columns"):
        processor.setup(
            data,
            media_columns=['google_spend', 'facebook_spend'],
            target_column='revenue'
        )

def test_non_numeric_values(processor, valid_data):
    """Test validation of non-numeric values"""
    data = valid_data.copy()
    data.loc[0, 'google_spend'] = 'invalid'
    with pytest.raises(ValueError, match="contains non-numeric values"):
        processor.setup(
            data,
            media_columns=['google_spend', 'facebook_spend'],
            target_column='revenue'
        )

def test_negative_values(processor, valid_data):
    """Test validation of negative values"""
    data = valid_data.copy()
    data.loc[0, 'google_spend'] = -100
    with pytest.raises(ValueError, match="contains negative values"):
        processor.setup(
            data,
            media_columns=['google_spend', 'facebook_spend'],
            target_column='revenue'
        )

def test_missing_values(processor, valid_data):
    """Test validation of missing values"""
    data = valid_data.copy()
    data.loc[0, 'google_spend'] = np.nan
    with pytest.raises(ValueError, match="contains missing values"):
        processor.setup(
            data,
            media_columns=['google_spend', 'facebook_spend'],
            target_column='revenue'
        )

def test_invalid_date_format(processor, valid_data):
    """Test validation of date format"""
    data = valid_data.copy()
    data.loc[0, 'date'] = 'invalid_date'
    with pytest.raises(ValueError, match="Invalid date format"):
        processor.setup(
            data,
            media_columns=['google_spend', 'facebook_spend'],
            target_column='revenue'
        )

def test_alternative_column_names(processor):
    """Test handling of alternative column names"""
    dates = pd.date_range('2023-01-01', '2023-01-10')
    data = pd.DataFrame({
        'date': dates,
        'Google Ads Cost': np.random.uniform(100, 1000, 10),
        'FB Marketing Expense': np.random.uniform(100, 1000, 10),
        'Total Sales': np.random.uniform(1000, 5000, 10)
    })
    
    processor.setup(
        data,
        media_columns=['Google Ads Cost', 'FB Marketing Expense'],
        target_column='Total Sales'
    )
    assert processor.data is not None
    assert processor.media_columns == ['Google Ads Cost', 'FB Marketing Expense']
    assert processor.target_column == 'Total Sales'

def test_data_processing(processor, valid_data):
    """Test data processing functionality"""
    processor.setup(
        valid_data,
        media_columns=['google_spend', 'facebook_spend'],
        target_column='revenue'
    )
    
    X, y = processor.process(valid_data)
    assert X.shape == (10, 2)
    assert y.shape == (10,)
    assert list(X.columns) == ['google_spend', 'facebook_spend']

def test_date_sorting(processor, valid_data):
    """Test data is sorted by date"""
    shuffled_data = valid_data.sample(frac=1)  # Randomly shuffle
    processor.setup(
        shuffled_data,
        media_columns=['google_spend', 'facebook_spend'],
        target_column='revenue'
    )
    
    expected_dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    pd.testing.assert_series_equal(
        processor.data['date'].dt.normalize(),
        pd.Series(expected_dates),
        check_names=False
    )

def test_duplicate_dates(processor, valid_data):
    """Test handling of duplicate dates"""
    data = pd.concat([valid_data, valid_data.iloc[0:1]])  # Duplicate first row
    with pytest.raises(ValueError, match="Duplicate dates found"):
        processor.setup(
            data,
            media_columns=['google_spend', 'facebook_spend'],
            target_column='revenue'
        ) 