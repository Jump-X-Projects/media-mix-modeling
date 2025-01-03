import pytest
import numpy as np
import pandas as pd
from backend.models.spend_optimizer import SpendOptimizer
from backend.models.model_factory import ModelFactory
from backend.data.processor import DataProcessor

@pytest.fixture
def sample_data():
    """Create realistic sample data for integration testing"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100)
    data = pd.DataFrame({
        'Date': dates,
        'TV_Spend': np.random.uniform(1000, 5000, 100),
        'Social_Spend': np.random.uniform(500, 2500, 100),
        'Search_Spend': np.random.uniform(300, 1500, 100),
        'Revenue': np.random.uniform(5000, 15000, 100)
    })
    return data

@pytest.fixture
def processor(sample_data):
    """Create data processor"""
    processor = DataProcessor()
    processor.setup(
        data=sample_data,
        media_columns=['TV_Spend', 'Social_Spend', 'Search_Spend'],
        target_column='Revenue'
    )
    return processor

@pytest.fixture
def trained_model(processor, sample_data):
    """Create and train a model"""
    factory = ModelFactory()
    model = factory.create_model('lightgbm')
    X, y = processor.process(sample_data)
    model.fit(X, y)
    return model

class TestSpendOptimizationIntegration:
    """Integration tests for spend optimization"""
    
    def test_end_to_end_optimization(self, trained_model, processor, sample_data):
        """Test complete optimization flow"""
        # Get current spend
        spend_cols = processor.media_columns
        current_spend = sample_data[spend_cols].mean()
        current_total = current_spend.sum()
        
        # Create optimizer
        optimizer = SpendOptimizer(
            model=trained_model,
            feature_names=spend_cols,
            historical_data=sample_data
        )
        
        # Run optimization
        result = optimizer.optimize(
            current_spend=current_spend.values,
            total_budget=current_total * 1.2  # 20% increase
        )
        
        # Verify results
        assert isinstance(result, dict)
        assert set(result.keys()) == set(spend_cols)
        assert np.isclose(sum(result.values()), current_total * 1.2)
        
        # Verify improved performance
        current_X = processor.process(pd.DataFrame([current_spend]))[0]
        optimized_X = processor.process(pd.DataFrame([result]))[0]
        
        current_revenue = trained_model.predict(current_X)[0]
        optimized_revenue = trained_model.predict(optimized_X)[0]
        
        assert optimized_revenue >= current_revenue
    
    def test_optimization_with_constraints(self, trained_model, processor, sample_data):
        """Test optimization respects all constraints"""
        spend_cols = processor.media_columns
        current_spend = sample_data[spend_cols].mean()
        
        optimizer = SpendOptimizer(
            model=trained_model,
            feature_names=spend_cols,
            historical_data=sample_data
        )
        
        # Test different budget scenarios
        budgets = [
            current_spend.sum() * 0.8,  # 20% decrease
            current_spend.sum(),        # Same budget
            current_spend.sum() * 1.2   # 20% increase
        ]
        
        for budget in budgets:
            result = optimizer.optimize(
                current_spend=current_spend.values,
                total_budget=budget
            )
            
            # Verify budget constraint
            assert np.isclose(sum(result.values()), budget)
            
            # Verify channel bounds
            for channel, spend in result.items():
                bounds = optimizer.channel_bounds[channel]
                assert bounds[0] <= spend <= bounds[1]
    
    def test_optimization_with_different_models(self, processor, sample_data):
        """Test optimization works with different model types"""
        spend_cols = processor.media_columns
        current_spend = sample_data[spend_cols].mean()
        X, y = processor.process(sample_data)
        
        model_types = ['linear', 'lightgbm', 'xgboost']
        
        for model_type in model_types:
            # Train model
            factory = ModelFactory()
            model = factory.create_model(model_type)
            model.fit(X, y)
            
            # Create optimizer
            optimizer = SpendOptimizer(
                model=model,
                feature_names=spend_cols,
                historical_data=sample_data
            )
            
            # Run optimization
            result = optimizer.optimize(
                current_spend=current_spend.values,
                total_budget=current_spend.sum()
            )
            
            # Verify basic properties
            assert isinstance(result, dict)
            assert set(result.keys()) == set(spend_cols)
            assert np.isclose(sum(result.values()), current_spend.sum())
    
    def test_data_processor_integration(self, trained_model, processor, sample_data):
        """Test integration with data processor"""
        spend_cols = processor.media_columns
        current_spend = sample_data[spend_cols].mean()
        
        optimizer = SpendOptimizer(
            model=trained_model,
            feature_names=spend_cols,
            historical_data=sample_data
        )
        
        # Run optimization
        result = optimizer.optimize(
            current_spend=current_spend.values,
            total_budget=current_spend.sum()
        )
        
        # Verify we can process the results
        result_df = pd.DataFrame([result])
        processed_result, _ = processor.process(result_df)
        
        # Verify we can get predictions
        prediction = trained_model.predict(processed_result)
        assert isinstance(prediction, np.ndarray)
        assert len(prediction) == 1 