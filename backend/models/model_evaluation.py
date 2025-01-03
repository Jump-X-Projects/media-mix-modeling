import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, List, Optional, Tuple, Union, Callable
import plotly.graph_objects as go
from scipy import stats

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_significance(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate statistical significance metrics"""
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Perform t-test
    t_stat, p_value = stats.ttest_1samp(residuals, 0)
    
    # Calculate confidence intervals (95%)
    ci_lower = np.mean(y_pred) - 1.96 * np.std(residuals)
    ci_upper = np.mean(y_pred) + 1.96 * np.std(residuals)
    
    # Calculate R-squared significance
    r2 = r2_score(y_true, y_pred)
    f_stat = r2 / (1 - r2) * (len(y_true) - 2)
    r2_p_value = 1 - stats.f.cdf(f_stat, 1, len(y_true) - 2)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'r2_significance': r2_p_value
    }

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Evaluate model predictions with multiple metrics"""
    metrics = {
        'r2_score': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred)
    }
    
    # Add significance metrics
    metrics.update(calculate_significance(y_true, y_pred))
    
    return metrics

def cross_validate(
    model_factory: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    time_series: bool = False,
    n_splits: int = 5
) -> Dict:
    """Perform cross-validation with multiple metrics"""
    # Choose CV strategy
    if time_series:
        cv = TimeSeriesSplit(n_splits=n_splits)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize metric lists
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    mape_scores = []
    significance_metrics = []
    
    # Perform cross-validation
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Create and train model
        model = model_factory()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = evaluate_predictions(y_test, y_pred)
        
        # Store results
        r2_scores.append(metrics['r2_score'])
        rmse_scores.append(metrics['rmse'])
        mae_scores.append(metrics['mae'])
        mape_scores.append(metrics['mape'])
        significance_metrics.append({
            't_statistic': metrics['t_statistic'],
            'p_value': metrics['p_value'],
            'r2_significance': metrics['r2_significance']
        })
    
    # Calculate aggregate metrics
    return {
        'r2_scores': r2_scores,
        'rmse_scores': rmse_scores,
        'mae_scores': mae_scores,
        'mape_scores': mape_scores,
        'significance': significance_metrics,
        'aggregate': {
            'mean_r2': np.mean(r2_scores),
            'std_r2': np.std(r2_scores),
            'mean_rmse': np.mean(rmse_scores),
            'std_rmse': np.std(rmse_scores),
            'mean_mae': np.mean(mae_scores),
            'std_mae': np.std(mae_scores),
            'mean_mape': np.mean(mape_scores),
            'std_mape': np.std(mape_scores)
        }
    }

def plot_cv_results(cv_results: Dict) -> Tuple[go.Figure, go.Figure]:
    """
    Create visualizations for cross-validation results.
    
    Args:
        cv_results: Dictionary containing cross-validation results
    
    Returns:
        Tuple of (metrics_plot, predictions_plot)
    """
    # Metrics distribution plot
    metrics_data = {
        'R²': cv_results['r2_scores'],
        'RMSE': cv_results['rmse_scores'],
        'MAE': cv_results['mae_scores']
    }
    
    metrics_plot = go.Figure()
    for metric_name, values in metrics_data.items():
        metrics_plot.add_trace(go.Box(
            y=values,
            name=metric_name,
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
    
    metrics_plot.update_layout(
        title="Cross-Validation Metrics Distribution",
        yaxis_title="Score",
        showlegend=False
    )
    
    # Predictions vs Actual plot
    predictions_plot = go.Figure()
    
    # Combine all folds
    all_actual = np.concatenate(cv_results['fold_actual'])
    all_predictions = np.concatenate(cv_results['fold_predictions'])
    
    predictions_plot.add_trace(go.Scatter(
        x=all_actual,
        y=all_predictions,
        mode='markers',
        name='Predictions',
        marker=dict(
            size=8,
            opacity=0.6
        )
    ))
    
    # Add diagonal line
    min_val = min(all_actual.min(), all_predictions.min())
    max_val = max(all_actual.max(), all_predictions.max())
    predictions_plot.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(
            color='red',
            dash='dash'
        )
    ))
    
    predictions_plot.update_layout(
        title="Cross-Validation: Predicted vs Actual",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values"
    )
    
    return metrics_plot, predictions_plot

def compare_models(
    models: Dict[str, object],
    X: np.ndarray,
    y: np.ndarray,
    time_series: bool = False
) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Compare multiple models using cross-validation.
    
    Args:
        models: Dictionary of model names and their classes
        X: Feature matrix
        y: Target vector
        time_series: Whether the data is time series
    
    Returns:
        Tuple of (comparison_df, comparison_plot)
    """
    results = {}
    
    # Perform cross-validation for each model
    for model_name, model_class in models.items():
        cv_results = cross_validate(
            model_class,
            X,
            y,
            time_series=time_series
        )
        results[model_name] = cv_results['aggregate']
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, metrics in results.items():
        row = {
            'Model': model_name,
            'R² (mean)': metrics['mean_r2'],
            'R² (std)': metrics['std_r2'],
            'RMSE (mean)': metrics['mean_rmse'],
            'RMSE (std)': metrics['std_rmse'],
            'MAE (mean)': metrics['mean_mae'],
            'MAE (std)': metrics['std_mae']
        }
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create comparison plot
    comparison_plot = go.Figure()
    
    # Add bars for each metric
    metrics = ['R²', 'RMSE', 'MAE']
    for metric in metrics:
        comparison_plot.add_trace(go.Bar(
            name=metric,
            x=comparison_df['Model'],
            y=comparison_df[f'{metric} (mean)'],
            error_y=dict(
                type='data',
                array=comparison_df[f'{metric} (std)'],
                visible=True
            )
        ))
    
    comparison_plot.update_layout(
        title="Model Comparison",
        barmode='group',
        xaxis_title="Model",
        yaxis_title="Score"
    )
    
    return comparison_df, comparison_plot 