import pytest
import numpy as np
import pandas as pd
from backend.models.spend_optimizer import SpendOptimizer, SpendOptimizerError
from sklearn.linear_model import LinearRegression

@pytest.fixture
def sample_model():
    """Create a simple linear model for testing"""
    model = LinearRegression()
    X = np.array([[100, 200], [200, 400], [300, 600]])
    y = np.array([1000, 2000, 3000])
    model.fit(X, y)
    return model

@pytest.fixture
def sample_historical_data():
    """Create sample historical data"""
    return pd.DataFrame({
        'TV_Spend': [1000, 1500, 2000, 2500, 3000],
        'Social_Spend': [500, 750, 1000, 1250, 1500]
    })

@pytest.fixture
def optimizer(sample_model, sample_historical_data):
    """Create SpendOptimizer instance"""
    return SpendOptimizer(
        model=sample_model,
        feature_names=['TV_Spend', 'Social_Spend'],
        historical_data=sample_historical_data
    )

class TestSpendOptimizerValidation:
    """Test spend optimization validation"""
    
    def test_valid_inputs(self, optimizer):
        """Test optimization with valid inputs"""
        current_spend = np.array([1500.0, 750.0])
        total_budget = 3000.0
        
        result = optimizer.optimize(current_spend, total_budget)
        
        assert isinstance(result, dict)
        assert len(result) == 2
        assert all(spend >= 0 for spend in result.values())
        assert np.isclose(sum(result.values()), total_budget)
    
    def test_negative_budget(self, optimizer):
        """Test validation of negative total budget"""
        current_spend = np.array([1500.0, 750.0])
        
        with pytest.raises(SpendOptimizerError) as exc:
            optimizer.optimize(current_spend, -1000.0)
        assert "must be positive" in str(exc.value)
    
    def test_budget_too_low(self, optimizer):
        """Test validation of budget below minimum threshold"""
        current_spend = np.array([1500.0, 750.0])
        total_budget = 500.0  # Less than 50% of current spend
        
        with pytest.raises(SpendOptimizerError) as exc:
            optimizer.optimize(current_spend, total_budget)
        assert "cannot be less than 50% of current spend" in str(exc.value)
    
    def test_budget_too_high(self, optimizer):
        """Test validation of budget above maximum threshold"""
        current_spend = np.array([1500.0, 750.0])
        total_budget = 10000.0  # More than 200% of current spend
        
        with pytest.raises(SpendOptimizerError) as exc:
            optimizer.optimize(current_spend, total_budget)
        assert "cannot be more than 200% of current spend" in str(exc.value)
    
    def test_invalid_historical_data(self, sample_model):
        """Test validation of invalid historical data"""
        invalid_data = pd.DataFrame({
            'TV_Spend': [1000, np.nan, 2000],  # Contains missing values
            'Social_Spend': [500, 'invalid', 1500]  # Contains non-numeric values
        })
        
        with pytest.raises(SpendOptimizerError) as exc:
            SpendOptimizer(
                model=sample_model,
                feature_names=['TV_Spend', 'Social_Spend'],
                historical_data=invalid_data
            )
        assert "contains missing values" in str(exc.value).lower()
    
    def test_missing_channels(self, sample_model, sample_historical_data):
        """Test validation of missing channels in historical data"""
        with pytest.raises(SpendOptimizerError) as exc:
            SpendOptimizer(
                model=sample_model,
                feature_names=['TV_Spend', 'Radio_Spend'],  # Radio_Spend not in data
                historical_data=sample_historical_data
            )
        assert "missing channels in historical data" in str(exc.value).lower()
    
    def test_zero_values(self, optimizer):
        """Test handling of zero values in current spend"""
        current_spend = np.array([1500.0, 0.0])
        total_budget = 2000.0
        
        result = optimizer.optimize(current_spend, total_budget)
        assert all(spend > 0 for spend in result.values())
    
    def test_allocation_bounds(self, optimizer):
        """Test validation of allocation bounds"""
        current_spend = np.array([1500.0, 750.0])
        total_budget = 4000.0  # Within global bounds but may exceed channel bounds
        
        result = optimizer.optimize(current_spend, total_budget)
        
        # Check channel-specific bounds
        for channel, spend in result.items():
            bounds = optimizer.channel_bounds[channel]
            assert bounds[0] <= spend <= bounds[1]
    
    def test_non_numeric_current_spend(self, optimizer):
        """Test validation of non-numeric current spend"""
        current_spend = np.array(['1500', 'invalid'])
        
        with pytest.raises(SpendOptimizerError) as exc:
            optimizer.optimize(current_spend, 2000.0)
        assert "must be numeric" in str(exc.value).lower()
    
    def test_optimization_failure(self, optimizer):
        """Test handling of optimization failure"""
        # Create a situation where optimization might fail
        current_spend = np.array([1.0, 1.0])  # Very small values
        total_budget = 1000000.0  # Very large budget
        
        with pytest.raises(SpendOptimizerError) as exc:
            optimizer.optimize(current_spend, total_budget)
        assert "cannot be more than 200% of current spend" in str(exc.value).lower()

class TestSpendOptimizerResults:
    """Test spend optimization results"""
    
    def test_result_format(self, optimizer):
        """Test format of optimization results"""
        current_spend = np.array([1500.0, 750.0])
        total_budget = 3000.0
        
        result = optimizer.optimize(current_spend, total_budget)
        
        assert isinstance(result, dict)
        assert set(result.keys()) == {'TV_Spend', 'Social_Spend'}
        assert all(isinstance(v, float) for v in result.values())
    
    def test_budget_constraint(self, optimizer):
        """Test that results satisfy budget constraint"""
        current_spend = np.array([1500.0, 750.0])
        total_budget = 3000.0
        
        result = optimizer.optimize(current_spend, total_budget)
        
        assert np.isclose(sum(result.values()), total_budget, rtol=1e-5)
    
    def test_channel_constraints(self, optimizer):
        """Test that results satisfy channel constraints"""
        current_spend = np.array([1500.0, 750.0])
        total_budget = 3000.0
        
        result = optimizer.optimize(current_spend, total_budget)
        
        for channel, spend in result.items():
            min_spend, max_spend = optimizer.channel_bounds[channel]
            assert min_spend <= spend <= max_spend 