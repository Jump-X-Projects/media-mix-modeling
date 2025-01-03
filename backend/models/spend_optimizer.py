import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
import pandas as pd

class SpendOptimizer:
    def __init__(self, model, feature_names: List[str]):
        """Initialize spend optimizer with trained model"""
        self.model = model
        self.feature_names = feature_names
        
    def _objective(self, x: np.ndarray, target_revenue: float) -> float:
        """Objective function for optimization"""
        # Reshape input for prediction
        X = pd.DataFrame([x], columns=self.feature_names)
        
        # Predict revenue
        pred_revenue = self.model.predict(X)[0]
        
        # Minimize difference between predicted and target revenue
        return abs(pred_revenue - target_revenue)
    
    def _constraint_total_budget(self, x: np.ndarray, total_budget: float) -> float:
        """Constraint: total spend equals budget"""
        return np.sum(x) - total_budget
    
    def _constraint_channel_limits(self, x: np.ndarray, min_spend: np.ndarray, max_spend: np.ndarray) -> List[float]:
        """Constraints: channel spend within limits"""
        constraints = []
        for i in range(len(x)):
            # Min spend constraint
            constraints.append(x[i] - min_spend[i])
            # Max spend constraint
            constraints.append(max_spend[i] - x[i])
        return constraints
    
    def optimize(
        self,
        target_revenue: float,
        total_budget: float,
        min_spend: Optional[Dict[str, float]] = None,
        max_spend: Optional[Dict[str, float]] = None,
        current_spend: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Optimize spend allocation for target revenue
        
        Args:
            target_revenue: Target revenue to achieve
            total_budget: Total budget constraint
            min_spend: Minimum spend per channel
            max_spend: Maximum spend per channel
            current_spend: Current spend allocation (used as initial guess)
            
        Returns:
            Tuple of (optimal allocation, optimization details)
        """
        # Set default spend limits if not provided
        if min_spend is None:
            min_spend = {feature: 0.0 for feature in self.feature_names}
        if max_spend is None:
            max_spend = {feature: total_budget for feature in self.feature_names}
            
        # Convert to arrays
        min_spend_array = np.array([min_spend.get(f, 0.0) for f in self.feature_names])
        max_spend_array = np.array([max_spend.get(f, total_budget) for f in self.feature_names])
        
        # Set initial guess
        if current_spend is None:
            x0 = np.full(len(self.feature_names), total_budget / len(self.feature_names))
        else:
            x0 = np.array([current_spend.get(f, 0.0) for f in self.feature_names])
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: self._constraint_total_budget(x, total_budget)},
            {'type': 'ineq', 'fun': lambda x: self._constraint_channel_limits(x, min_spend_array, max_spend_array)}
        ]
        
        # Run optimization
        result = minimize(
            self._objective,
            x0,
            args=(target_revenue,),
            method='SLSQP',
            constraints=constraints,
            bounds=[(min_spend_array[i], max_spend_array[i]) for i in range(len(self.feature_names))]
        )
        
        # Prepare results
        optimal_allocation = dict(zip(self.feature_names, result.x))
        
        # Calculate predicted revenue with optimal allocation
        X_opt = pd.DataFrame([result.x], columns=self.feature_names)
        predicted_revenue = self.model.predict(X_opt)[0]
        
        details = {
            'success': result.success,
            'predicted_revenue': predicted_revenue,
            'revenue_gap': abs(predicted_revenue - target_revenue),
            'optimization_message': result.message,
            'iterations': result.nit
        }
        
        return optimal_allocation, details
    
    def get_roi_estimates(self, spend_allocation: Dict[str, float]) -> Dict[str, float]:
        """Calculate ROI estimates for each channel"""
        base_revenue = self.model.predict(pd.DataFrame([spend_allocation]))[0]
        roi_estimates = {}
        
        # Calculate marginal ROI for each channel
        for channel in self.feature_names:
            # Increase spend by 1% for the channel
            test_allocation = spend_allocation.copy()
            delta = test_allocation[channel] * 0.01
            test_allocation[channel] += delta
            
            # Calculate revenue difference
            test_revenue = self.model.predict(pd.DataFrame([test_allocation]))[0]
            marginal_revenue = test_revenue - base_revenue
            
            # Calculate ROI
            roi_estimates[channel] = (marginal_revenue / delta) if delta > 0 else 0
            
        return roi_estimates 