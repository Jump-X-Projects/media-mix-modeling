import numpy as np
from scipy.optimize import minimize
from typing import List, Optional, Dict, Any

class SpendOptimizer:
    """Optimizer for media spend allocation"""
    
    def __init__(self, model: Any, feature_names: List[str], budget_constraints: Optional[Dict] = None):
        """
        Initialize the spend optimizer
        
        Args:
            model: Trained model instance
            feature_names: List of feature names (media channels)
            budget_constraints: Optional dictionary of budget constraints per channel
        """
        self.model = model
        self.feature_names = feature_names
        self.budget_constraints = budget_constraints or {}
        
    def optimize(self, current_spend: np.ndarray) -> Dict[str, float]:
        """
        Optimize spend allocation to maximize predicted revenue
        
        Args:
            current_spend: Current spend levels for each channel
            
        Returns:
            Dictionary mapping channel names to optimized spend values
        """
        def objective(x):
            # Reshape spend values for prediction
            spend = x.reshape(1, -1)
            # Predict revenue (negative since we want to maximize)
            return -self.model.predict(spend)[0]
            
        def constraint(x):
            # Total spend should not exceed current total
            return np.sum(current_spend) - np.sum(x)
            
        # Initial guess is current spend
        x0 = current_spend.flatten()
        
        # Bounds - each channel spend must be non-negative
        bounds = [(0, None) for _ in range(len(self.feature_names))]
        
        # Add channel-specific constraints if provided
        constraints = [{'type': 'ineq', 'fun': constraint}]
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Return optimized spend as dictionary
        return dict(zip(self.feature_names, result.x)) 