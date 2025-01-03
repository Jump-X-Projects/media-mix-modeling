import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.distributions import constraints
from typing import Dict, Optional, Tuple
import pandas as pd

from .base_model import BaseModel

class BayesianMediaMixModel(BaseModel):
    """Bayesian implementation of Media Mix Model"""
    def __init__(self, num_epochs=1000, learning_rate=0.01, device='cpu'):
        super().__init__(name="Bayesian MMM")
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        self.trained_params = None
        self.weight_samples = None
        self.bias_samples = None
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the Bayesian MMM"""
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X.values).to(self.device)
        y_tensor = torch.FloatTensor(y.values).to(self.device)
        
        # Standardize data
        X_mean = X_tensor.mean(0, keepdim=True)
        X_std = X_tensor.std(0, keepdim=True) + 1e-6
        y_mean = y_tensor.mean()
        y_std = y_tensor.std() + 1e-6
        
        X_tensor = (X_tensor - X_mean) / X_std
        y_tensor = (y_tensor - y_mean) / y_std
        
        # Initialize parameters with better priors
        num_features = X.shape[1]
        
        # Initialize weights using linear regression
        linear_model = torch.nn.Linear(num_features, 1)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(linear_model.parameters(), lr=0.1)
        
        # Train linear model for initialization
        for _ in range(100):
            optimizer.zero_grad()
            output = linear_model(X_tensor)
            loss = criterion(output.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()
        
        # Use linear model weights as initialization
        weight_loc = linear_model.weight.data.squeeze().clone().requires_grad_(True)
        bias_loc = linear_model.bias.data.clone().requires_grad_(True)
        
        # Initialize scales with small values
        weight_scale = torch.ones(num_features).mul(0.1).requires_grad_(True)
        bias_scale = torch.tensor(0.1).requires_grad_(True)
        noise_scale = torch.tensor(0.1).requires_grad_(True)
        
        # Create optimizer with lower learning rate
        params = [weight_loc, weight_scale, bias_loc, bias_scale, noise_scale]
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            
            # Sample parameters with reparameterization trick
            eps_w = torch.randn_like(weight_loc)
            eps_b = torch.randn_like(bias_loc)
            
            weights = weight_loc + weight_scale.abs() * eps_w
            bias = bias_loc + bias_scale.abs() * eps_b
            
            # Forward pass
            mean = torch.matmul(X_tensor, weights) + bias
            
            # Negative ELBO loss
            likelihood = -0.5 * ((y_tensor - mean) ** 2).mean() / noise_scale.abs() - torch.log(noise_scale.abs())
            kl_weights = -0.5 * (1 + 2 * torch.log(weight_scale.abs()) - weight_loc ** 2 - weight_scale.abs() ** 2).mean()
            kl_bias = -0.5 * (1 + 2 * torch.log(bias_scale.abs()) - bias_loc ** 2 - bias_scale.abs() ** 2).mean()
            
            loss = -(likelihood - 0.1 * (kl_weights + kl_bias))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                
                # Store best parameters
                best_params = {
                    'weight_loc': weight_loc.detach().clone(),
                    'weight_scale': weight_scale.detach().clone(),
                    'bias_loc': bias_loc.detach().clone(),
                    'bias_scale': bias_scale.detach().clone(),
                    'noise_scale': noise_scale.detach().clone()
                }
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item()}")
        
        # Use best parameters
        self.trained_params = {
            'weight_loc': best_params['weight_loc'].numpy(),
            'weight_scale': best_params['weight_scale'].abs().numpy(),
            'bias_loc': best_params['bias_loc'].numpy(),
            'bias_scale': best_params['bias_scale'].abs().numpy(),
            'noise_scale': best_params['noise_scale'].abs().numpy(),
            'X_mean': X_mean.numpy(),
            'X_std': X_std.numpy(),
            'y_mean': y_mean.item(),
            'y_std': y_std.item()
        }
        
        # Generate samples for uncertainty estimation
        num_samples = 1000
        weight_samples = []
        bias_samples = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                weights = (
                    torch.tensor(self.trained_params['weight_loc']) +
                    torch.tensor(self.trained_params['weight_scale']) * torch.randn(num_features)
                ).numpy()
                bias = (
                    torch.tensor(self.trained_params['bias_loc']) +
                    torch.tensor(self.trained_params['bias_scale']) * torch.randn(1)
                ).numpy()
                weight_samples.append(weights)
                bias_samples.append(bias)
        
        self.weight_samples = np.array(weight_samples)
        self.bias_samples = np.array(bias_samples)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the mean of the posterior"""
        if self.trained_params is None:
            raise ValueError("Model not trained yet")
            
        # Standardize input
        X_std = (X.values - self.trained_params['X_mean']) / self.trained_params['X_std']
        
        # Use mean of posterior for predictions
        weights = self.trained_params['weight_loc']
        bias = self.trained_params['bias_loc']
        
        # Make predictions and unstandardize
        y_std = np.dot(X_std, weights) + bias
        return y_std * self.trained_params['y_std'] + self.trained_params['y_mean']
    
    def get_uncertainty_estimates(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Get prediction intervals using posterior samples"""
        if self.weight_samples is None or self.bias_samples is None:
            raise ValueError("Model not trained with uncertainty estimation")
            
        # Standardize input
        X_std = (X.values - self.trained_params['X_mean']) / self.trained_params['X_std']
        
        predictions = []
        for w, b in zip(self.weight_samples, self.bias_samples):
            # Make predictions and unstandardize
            y_std = np.dot(X_std, w) + b
            y = y_std * self.trained_params['y_std'] + self.trained_params['y_mean']
            predictions.append(y)
        
        predictions = np.array(predictions)
        lower = np.percentile(predictions, 2.5, axis=0)
        upper = np.percentile(predictions, 97.5, axis=0)
        
        return lower, upper
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Calculate feature importance based on posterior weight distributions"""
        if self.weight_samples is None or not self.feature_names:
            return None
        
        # Calculate mean absolute weight values
        importance = np.mean(np.abs(self.weight_samples), axis=0)
        return pd.Series(importance, index=self.feature_names)
    
    def save(self, path: str) -> None:
        """Save model parameters"""
        if self.trained_params is None:
            raise ValueError("No model to save")
            
        model_data = {
            'trained_params': self.trained_params,
            'weight_samples': self.weight_samples,
            'bias_samples': self.bias_samples,
            'feature_names': self.feature_names
        }
        np.savez(path, **model_data)
    
    def load(self, path: str) -> None:
        """Load model parameters"""
        data = np.load(path, allow_pickle=True)
        self.trained_params = data['trained_params'].item()
        self.weight_samples = data['weight_samples']
        self.bias_samples = data['bias_samples']
        self.feature_names = data['feature_names'].tolist() 
    
    def get_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """Get uncertainty estimates for predictions"""
        if self.trained_params is None:
            raise ValueError("Model not trained yet")
            
        # Standardize input
        X_std = (X.values - self.trained_params['X_mean']) / self.trained_params['X_std']
        
        # Generate multiple predictions using sampled weights
        predictions = []
        for w, b in zip(self.weight_samples, self.bias_samples):
            y_std = np.dot(X_std, w) + b
            y = y_std * self.trained_params['y_std'] + self.trained_params['y_mean']
            predictions.append(y)
        
        # Calculate standard deviation across predictions
        return np.std(predictions, axis=0) 