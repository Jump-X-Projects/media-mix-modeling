import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def render_results_dashboard():
    """Render the results dashboard with visualizations"""
    st.header("Results Dashboard")
    
    if st.session_state.model is None:
        st.warning("Please train a model first")
        return
    
    # Create tabs for different visualizations
    model_performance_tab, feature_analysis_tab, channel_attribution_tab, predictions_tab = st.tabs([
        "Model Performance",
        "Feature Analysis",
        "Channel Attribution",
        "Predictions"
    ])
    
    with model_performance_tab:
        show_model_performance()
    
    with feature_analysis_tab:
        show_feature_analysis()
    
    with channel_attribution_tab:
        show_channel_attribution()
    
    with predictions_tab:
        show_predictions()

def show_model_performance():
    """Show model performance metrics and visualizations"""
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        r2 = st.session_state.model.score(st.session_state.data, st.session_state.data[st.session_state.target_col])
        st.metric("R² Score", f"{r2:.3f}")
    
    with col2:
        y_true = st.session_state.data[st.session_state.target_col]
        y_pred = st.session_state.model.predict(st.session_state.data)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        st.metric("RMSE", f"{rmse:.2f}")
    
    with col3:
        mae = np.mean(np.abs(y_true - y_pred))
        st.metric("MAE", f"{mae:.2f}")

def show_feature_analysis():
    """Show feature importance and relationships"""
    if hasattr(st.session_state.model, 'feature_importances_'):
        importances = st.session_state.model.feature_importances_
        features = st.session_state.selected_features
        
        # Create bar plot
        fig = go.Figure(data=[
            go.Bar(x=features, y=importances)
        ])
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Features",
            yaxis_title="Importance"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_channel_attribution():
    """Show channel attribution analysis"""
    if hasattr(st.session_state.model, 'get_attribution'):
        attribution = st.session_state.model.get_attribution()
        features = st.session_state.selected_features
        
        # Create bar plot
        fig = go.Figure(data=[
            go.Bar(x=features, y=attribution)
        ])
        fig.update_layout(
            title="Channel Attribution",
            xaxis_title="Channels",
            yaxis_title="Attribution (%)"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_predictions():
    """Show predictions and forecasting"""
    y_true = st.session_state.data[st.session_state.target_col]
    y_pred = st.session_state.model.predict(st.session_state.data)
    
    # Actual vs Predicted plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions'
    ))
    fig.add_trace(go.Scatter(
        x=[min(y_true), max(y_true)],
        y=[min(y_true), max(y_true)],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash')
    ))
    fig.update_layout(
        title="Actual vs Predicted Values",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values"
    )
    st.plotly_chart(fig, use_container_width=True) 