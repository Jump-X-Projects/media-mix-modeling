import numpy as np
from scipy.optimize import minimize
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd

class SpendOptimizerError(Exception):
    """Custom exception for spend optimization errors"""
    pass

class SpendOptimizer:
    """Optimizer for media spend allocation"""
    
    def __init__(self, model: Any, feature_names: List[str], historical_data: Optional[pd.DataFrame] = None):
        """
        Initialize the spend optimizer
        
        Args:
            model: Trained model instance
            feature_names: List of feature names (media channels)
            historical_data: Optional historical data for setting realistic bounds
        """
        self.model = model
        self.feature_names = feature_names
        self.historical_data = historical_data
        
        # Validate historical data first
        if historical_data is not None:
            self._validate_historical_data()
            
        self.channel_bounds = self._compute_channel_bounds() if historical_data is not None else None
    
    def _validate_historical_data(self) -> None:
        """Validate historical data for completeness and correctness"""
        # Check for missing channels
        missing_channels = [col for col in self.feature_names if col not in self.historical_data.columns]
        if missing_channels:
            raise SpendOptimizerError(
                f"Missing channels in historical data: {', '.join(missing_channels)}"
            )
        
        # Check for invalid values in each channel
        for channel in self.feature_names:
            channel_data = self.historical_data[channel]
            
            # Check for non-numeric values
            if not np.issubdtype(channel_data.dtype, np.number):
                raise SpendOptimizerError(
                    f"Channel {channel} contains non-numeric values"
                )
            
            # Check for missing values
            if channel_data.isnull().any():
                raise SpendOptimizerError(
                    f"Channel {channel} contains missing values"
                )
            
            # Check for negative values
            if (channel_data < 0).any():
                raise SpendOptimizerError(
                    f"Channel {channel} contains negative values"
                )
    
    def _compute_channel_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Compute realistic bounds for each channel based on historical data"""
        bounds = {}
        for channel in self.feature_names:
            channel_data = self.historical_data[channel]
            min_spend = max(0, channel_data.min())  # Ensure non-negative
            max_spend = channel_data.max() * 1.5  # Allow 50% increase from historical max
            bounds[channel] = (min_spend, max_spend)
        return bounds
    
    def validate_budget(self, total_budget: float, current_spend: np.ndarray) -> None:
        """Validate the total budget"""
        # Validate current spend is numeric
        if not np.issubdtype(current_spend.dtype, np.number):
            raise SpendOptimizerError("Current spend values must be numeric")
            
        if total_budget <= 0:
            raise SpendOptimizerError("Total budget must be positive")
            
        current_total = np.sum(current_spend)
        if total_budget < current_total * 0.5:
            raise SpendOptimizerError(
                f"Total budget (${total_budget:,.2f}) cannot be less than 50% of current spend (${current_total:,.2f})"
            )
        if total_budget > current_total * 2:
            raise SpendOptimizerError(
                f"Total budget (${total_budget:,.2f}) cannot be more than 200% of current spend (${current_total:,.2f})"
            )
    
    def validate_channel_allocation(self, allocation: Dict[str, float]) -> None:
        """Validate individual channel allocations"""
        if not self.channel_bounds:
            return
            
        for channel, spend in allocation.items():
            if spend < 0:
                raise SpendOptimizerError(f"Spend for {channel} cannot be negative")
                
            min_spend, max_spend = self.channel_bounds[channel]
            if spend < min_spend:
                raise SpendOptimizerError(
                    f"Spend for {channel} (${spend:,.2f}) is below minimum historical spend (${min_spend:,.2f})"
                )
            if spend > max_spend:
                raise SpendOptimizerError(
                    f"Spend for {channel} (${spend:,.2f}) is above maximum allowed spend (${max_spend:,.2f})"
                )
    
    def optimize(self, current_spend: np.ndarray, total_budget: Optional[float] = None) -> Dict[str, float]:
        """
        Optimize spend allocation to maximize predicted revenue
        
        Args:
            current_spend: Current spend levels for each channel
            total_budget: Optional total budget constraint. If None, uses current total spend.
            
        Returns:
            Dictionary mapping channel names to optimized spend values
        """
        # Set total budget if not provided
        if total_budget is None:
            total_budget = np.sum(current_spend)
            
        # Validate inputs
        self.validate_budget(total_budget, current_spend)
        
        def objective(x):
            try:
                # Reshape spend values for prediction
                spend = x.reshape(1, -1)
                # Predict revenue (negative since we want to maximize)
                return -self.model.predict(spend)[0]
            except Exception as e:
                raise SpendOptimizerError(f"Optimization failed: {str(e)}")
            
        def constraint(x):
            # Total spend should equal budget
            return total_budget - np.sum(x)
            
        # Initial guess is proportional allocation of new budget
        x0 = current_spend * (total_budget / np.sum(current_spend))
        
        # Set bounds based on historical data or defaults
        if self.channel_bounds:
            bounds = [self.channel_bounds[name] for name in self.feature_names]
        else:
            bounds = [(0, total_budget) for _ in range(len(self.feature_names))]
        
        try:
            # Optimize
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=[{'type': 'eq', 'fun': constraint}]
            )
            
            if not result.success:
                raise SpendOptimizerError(f"Optimization failed: {result.message}")
            
            # Create allocation dictionary
            allocation = dict(zip(self.feature_names, result.x))
            
            # Validate final allocation
            self.validate_channel_allocation(allocation)
            
            return allocation
            
        except Exception as e:
            if isinstance(e, SpendOptimizerError):
                raise
            raise SpendOptimizerError(f"Optimization failed: {str(e)}") 