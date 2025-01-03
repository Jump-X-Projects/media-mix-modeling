import streamlit as st
from backend.models.model_factory import ModelFactory
from typing import Dict, Any, Optional
import pandas as pd

def render_model_selection():
    """Render the model selection interface"""
    st.header("Model Selection")

    # Model type selection
    model_types = {
        "Linear Regression": "linear",
        "LightGBM": "lightgbm",
        "XGBoost": "xgboost",
        "Bayesian MMM": "bayesian"
    }
    
    selected_model = st.selectbox(
        "Choose a model type",
        list(model_types.keys()),
        help="Select the type of model to use for analysis"
    )
    
    # Model parameters
    params = get_model_parameters(model_types[selected_model])
    
    # Cross-validation settings
    st.checkbox("Time series data", help="Enable time-based cross-validation")
    st.slider("Number of folds", min_value=2, max_value=10, value=5)
    
    # Training button
    if st.button("Evaluate with Cross-Validation"):
        if st.session_state.data is not None:
            with st.spinner("Training model..."):
                try:
                    # Create and train model
                    model = ModelFactory.create_model(
                        model_type=model_types[selected_model],
                        params=params
                    )
                    
                    # Store model in session state
                    st.session_state.model = model
                    st.success("Model trained successfully!")
                    
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
        else:
            st.warning("Please upload data first")

def get_model_parameters(model_type: str) -> Dict[str, Any]:
    """Get parameters for the selected model type"""
    params = {}
    
    if model_type == "linear":
        params["fit_intercept"] = st.checkbox("Fit Intercept", value=True)
        
    elif model_type == "lightgbm":
        params.update({
            "num_leaves": st.slider("Number of leaves", 2, 256, 31),
            "learning_rate": st.slider("Learning rate", 0.001, 0.1, 0.05),
            "n_estimators": st.slider("Number of estimators", 10, 1000, 100)
        })
        
    elif model_type == "xgboost":
        params.update({
            "max_depth": st.slider("max_depth", 1, 10, 6),
            "learning_rate": st.slider("Learning rate", 0.001, 0.1, 0.05),
            "n_estimators": st.slider("Number of estimators", 10, 1000, 100)
        })
        
    elif model_type == "bayesian":
        params.update({
            "num_epochs": st.slider("Number of epochs", 100, 2000, 500),
            "learning_rate": st.slider("Learning rate", 0.001, 0.1, 0.01)
        })
    
    return params 