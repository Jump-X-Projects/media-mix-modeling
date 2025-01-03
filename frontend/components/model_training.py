import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import Dict, Optional

from backend.models import ModelFactory, DataProcessor
from backend.models.model_tuning import ModelTuner
from backend.utils.sample_data import generate_sample_data

def render_model_training(data: Optional[pd.DataFrame] = None):
    """Render the model training interface"""
    st.header("Model Training")
    
    # Option to use sample data
    use_sample = st.checkbox(
        "Use sample data",
        help="Generate synthetic data for testing"
    )
    
    if use_sample:
        data = generate_sample_data()
        st.success("Generated sample data!")
        st.dataframe(data.head())
    elif data is None:
        st.warning("Please upload data first in the Data Upload section.")
        return
    
    # Model selection
    available_models = ModelFactory.get_available_models()
    model_type = st.selectbox(
        "Select Model Type",
        list(available_models.keys()),
        format_func=lambda x: available_models[x]
    )
    
    # Column selection
    st.subheader("Feature Selection")
    
    # Select target variable
    target_col = st.selectbox(
        "Select Target Variable (Revenue)",
        data.columns
    )
    
    # Select media spend columns
    media_cols = st.multiselect(
        "Select Media Spend Columns",
        [col for col in data.columns if col != target_col],
        help="Select columns containing media spend data"
    )
    
    # Select control variables
    control_cols = st.multiselect(
        "Select Control Variables (Optional)",
        [col for col in data.columns if col not in media_cols + [target_col]],
        help="Select additional control variables (e.g., seasonality)"
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        do_cv = st.checkbox("Perform Cross-Validation", value=True)
        do_tuning = st.checkbox("Tune Hyperparameters", value=False)
        
        if do_cv:
            n_splits = st.slider(
                "Number of CV Splits",
                min_value=2,
                max_value=10,
                value=5
            )
        
        if do_tuning:
            n_trials = st.slider(
                "Number of Tuning Trials",
                min_value=10,
                max_value=100,
                value=50
            )
    
    # Model parameters
    st.subheader("Model Parameters")
    params = get_model_params(model_type) if not do_tuning else None
    
    # Train button
    if st.button("Train Model"):
        if not media_cols:
            st.error("Please select at least one media spend column.")
            return
            
        with st.spinner("Training model..."):
            try:
                # Prepare data
                processor = DataProcessor()
                processor.setup(
                    data=data,
                    media_columns=media_cols,
                    target_column=target_col,
                    control_columns=control_cols
                )
                X, y = processor.process(data)
                
                if do_tuning:
                    # Tune hyperparameters
                    st.write("Tuning hyperparameters...")
                    tuner = ModelTuner(
                        model_type,
                        n_splits=n_splits if do_cv else 5,
                        n_trials=n_trials
                    )
                    best_params = tuner.tune_hyperparameters(X, y)
                    st.success("Hyperparameter tuning complete!")
                    st.write("Best parameters:", best_params)
                    params = best_params
                
                if do_cv:
                    # Perform cross-validation
                    st.write("Performing cross-validation...")
                    tuner = ModelTuner(model_type, n_splits=n_splits)
                    cv_results = tuner.cross_validate(X, y, params)
                    show_cv_results(cv_results)
                
                # Train final model
                model = ModelFactory.create_model(model_type, params)
                model.fit(X, y)
                
                # Store model and processor in session state
                st.session_state['model'] = model
                st.session_state['processor'] = processor
                
                # Show results
                show_training_results(model, X, y)
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")

def show_cv_results(cv_results: Dict):
    """Display cross-validation results"""
    st.subheader("Cross-Validation Results")
    
    # Calculate mean and std for each metric
    metrics = {
        'R² Score': ('r2', 'Higher is better'),
        'RMSE': ('rmse', 'Lower is better'),
        'MAE': ('mae', 'Lower is better')
    }
    
    cols = st.columns(len(metrics))
    for col, (metric_name, (metric_key, interpretation)) in zip(cols, metrics.items()):
        train_scores = cv_results[f'train_{metric_key}']
        test_scores = cv_results[f'test_{metric_key}']
        
        with col:
            st.metric(
                f"{metric_name} (Test)",
                f"{np.mean(test_scores):.3f} ± {np.std(test_scores):.3f}",
                help=interpretation
            )
            st.metric(
                f"{metric_name} (Train)",
                f"{np.mean(train_scores):.3f} ± {np.std(train_scores):.3f}"
            )
    
    # Plot train vs test scores
    fig = go.Figure()
    for metric_name, (metric_key, _) in metrics.items():
        fig.add_trace(go.Box(
            y=cv_results[f'train_{metric_key}'],
            name=f'{metric_name} (Train)',
            boxpoints='all'
        ))
        fig.add_trace(go.Box(
            y=cv_results[f'test_{metric_key}'],
            name=f'{metric_name} (Test)',
            boxpoints='all'
        ))
    
    fig.update_layout(
        title="Cross-Validation Scores Distribution",
        yaxis_title="Score",
        boxmode='group'
    )
    st.plotly_chart(fig)

def get_model_params(model_type: str) -> Dict:
    """Get model-specific parameters"""
    params = {}
    
    if model_type == 'linear':
        # Linear regression has no parameters to tune
        pass
        
    elif model_type == 'lightgbm':
        params['learning_rate'] = st.slider(
            "Learning Rate",
            min_value=0.01,
            max_value=0.3,
            value=0.05,
            step=0.01
        )
        params['num_leaves'] = st.slider(
            "Number of Leaves",
            min_value=2,
            max_value=256,
            value=31
        )
        params['feature_fraction'] = st.slider(
            "Feature Fraction",
            min_value=0.1,
            max_value=1.0,
            value=0.9
        )
        
    elif model_type == 'xgboost':
        params['learning_rate'] = st.slider(
            "Learning Rate",
            min_value=0.01,
            max_value=0.3,
            value=0.05,
            step=0.01
        )
        params['max_depth'] = st.slider(
            "Max Depth",
            min_value=1,
            max_value=10,
            value=6
        )
        params['subsample'] = st.slider(
            "Subsample Ratio",
            min_value=0.1,
            max_value=1.0,
            value=0.8
        )
        
    elif model_type == 'bayesian':
        params['num_epochs'] = st.slider(
            "Number of Epochs",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100
        )
        params['learning_rate'] = st.slider(
            "Learning Rate",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            step=0.001
        )
        
    return params

def show_training_results(model, X: pd.DataFrame, y: pd.Series):
    """Display model training results"""
    st.subheader("Training Results")
    
    # Model performance metrics
    metrics = model.evaluate(X, y)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", f"{metrics['r2_score']:.3f}")
    with col2:
        st.metric("RMSE", f"{metrics['rmse']:.3f}")
    with col3:
        st.metric("MAE", f"{metrics['mae']:.3f}")
    
    # Feature importance plot
    importance = model.get_feature_importance()
    if importance is not None:
        st.subheader("Feature Importance")
        fig = px.bar(
            importance,
            orientation='h',
            title="Feature Importance"
        )
        st.plotly_chart(fig)
    
    # Predictions vs Actual plot
    predictions = model.predict(X)
    fig = px.scatter(
        x=y,
        y=predictions,
        labels={'x': 'Actual', 'y': 'Predicted'},
        title="Predictions vs Actual"
    )
    fig.add_shape(
        type='line',
        x0=y.min(),
        y0=y.min(),
        x1=y.max(),
        y1=y.max(),
        line=dict(color='red', dash='dash')
    )
    st.plotly_chart(fig)
    
    # For Bayesian model, show uncertainty
    if hasattr(model, 'get_uncertainty_estimates'):
        st.subheader("Prediction Uncertainty")
        lower, upper = model.get_uncertainty_estimates(X)
        
        # Create DataFrame for uncertainty plot
        plot_df = pd.DataFrame({
            'Actual': y,
            'Predicted': predictions,
            'Lower': lower,
            'Upper': upper
        }).sort_values('Actual')
        
        fig = px.line(
            plot_df,
            x=plot_df.index,
            y=['Predicted', 'Lower', 'Upper'],
            title="Predictions with 95% Credible Intervals"
        )
        fig.add_scatter(
            x=plot_df.index,
            y=plot_df['Actual'],
            mode='markers',
            name='Actual'
        )
        st.plotly_chart(fig) 