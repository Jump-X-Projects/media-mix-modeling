import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Tuple

def render_predictions():
    """Render the predictions and analysis interface"""
    st.header("Predictions & Analysis")
    
    if 'model' not in st.session_state or 'processor' not in st.session_state:
        st.warning("Please train a model first in the Model Training section.")
        return
        
    model = st.session_state['model']
    processor = st.session_state['processor']
    
    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs([
        "What-If Analysis",
        "Spend Optimization",
        "ROI Analysis"
    ])
    
    with tab1:
        render_what_if_analysis(model, processor)
        
    with tab2:
        render_spend_optimization(model, processor)
        
    with tab3:
        render_roi_analysis(model, processor)

def render_what_if_analysis(model, processor):
    """Render what-if analysis interface"""
    st.subheader("What-If Analysis")
    st.write("Adjust media spend to see predicted impact on revenue")
    
    # Create input fields for each media channel
    adjustments = {}
    for col in processor.media_columns:
        adjustment = st.slider(
            f"Adjust {col} spend",
            min_value=-100,
            max_value=100,
            value=0,
            help="Percentage change in spend"
        )
        adjustments[col] = adjustment
    
    if st.button("Calculate Impact"):
        # Create base case
        base_spend = pd.DataFrame(
            {col: [100] for col in processor.media_columns}
        )
        
        # Create adjusted case
        adjusted_spend = pd.DataFrame(
            {col: [100 + adj] for col, adj in adjustments.items()}
        )
        
        # Add control variables if any
        if processor.control_columns:
            for col in processor.control_columns:
                base_spend[col] = 0
                adjusted_spend[col] = 0
        
        # Process data
        X_base, _ = processor.process(base_spend)
        X_adjusted, _ = processor.process(adjusted_spend)
        
        # Get predictions
        base_pred = model.predict(X_base)[0]
        adjusted_pred = model.predict(X_adjusted)[0]
        
        # Calculate impact
        impact = ((adjusted_pred - base_pred) / base_pred) * 100
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Predicted Impact",
                f"{impact:.1f}%",
                delta=f"{impact:.1f}%"
            )
        
        # For Bayesian model, show uncertainty
        if hasattr(model, 'get_uncertainty_estimates'):
            lower, upper = model.get_uncertainty_estimates(X_adjusted)
            with col2:
                st.metric(
                    "Prediction Interval",
                    f"({lower[0]:.1f}, {upper[0]:.1f})"
                )

def render_spend_optimization(model, processor):
    """Render spend optimization interface"""
    st.subheader("Spend Optimization")
    st.write("Find optimal spend allocation for maximum revenue")
    
    # Total budget input
    total_budget = st.number_input(
        "Total Budget",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )
    
    if st.button("Optimize Allocation"):
        # Simple grid search optimization
        best_allocation, best_revenue = optimize_spend(
            model,
            processor,
            total_budget,
            processor.media_columns
        )
        
        # Display results
        st.subheader("Optimal Allocation")
        
        # Create bar chart
        fig = px.bar(
            x=list(best_allocation.keys()),
            y=list(best_allocation.values()),
            title="Recommended Spend Allocation"
        )
        st.plotly_chart(fig)
        
        st.metric("Predicted Revenue", f"${best_revenue:,.2f}")

def render_roi_analysis(model, processor):
    """Render ROI analysis interface"""
    st.subheader("ROI Analysis")
    st.write("Analyze return on investment for each channel")
    
    # Calculate ROI for each channel
    rois = calculate_channel_rois(model, processor)
    
    # Create ROI chart
    fig = go.Figure()
    
    # Add bar chart for ROI
    fig.add_trace(
        go.Bar(
            x=list(rois.keys()),
            y=[roi['roi'] for roi in rois.values()],
            name="ROI"
        )
    )
    
    # Add error bars for Bayesian model
    if hasattr(model, 'get_uncertainty_estimates'):
        fig.add_trace(
            go.Scatter(
                x=list(rois.keys()),
                y=[roi['upper'] for roi in rois.values()],
                mode='lines',
                name="Upper CI",
                line=dict(dash='dash')
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(rois.keys()),
                y=[roi['lower'] for roi in rois.values()],
                mode='lines',
                name="Lower CI",
                line=dict(dash='dash'),
                fill='tonexty'
            )
        )
    
    fig.update_layout(
        title="Channel ROI Analysis",
        yaxis_title="ROI (Return per $ spent)"
    )
    st.plotly_chart(fig)

def optimize_spend(
    model,
    processor,
    total_budget: float,
    channels: List[str],
    n_points: int = 10
) -> Tuple[dict, float]:
    """Simple grid search optimization for spend allocation"""
    n_channels = len(channels)
    best_allocation = None
    best_revenue = float('-inf')
    
    # Generate random allocations
    for _ in range(n_points):
        # Generate random weights
        weights = np.random.dirichlet(np.ones(n_channels))
        allocation = {
            channel: weight * total_budget
            for channel, weight in zip(channels, weights)
        }
        
        # Create feature DataFrame
        X = pd.DataFrame([allocation])
        
        # Add control variables if any
        if processor.control_columns:
            for col in processor.control_columns:
                X[col] = 0
        
        # Process and predict
        X_processed, _ = processor.process(X)
        revenue = model.predict(X_processed)[0]
        
        if revenue > best_revenue:
            best_revenue = revenue
            best_allocation = allocation
    
    return best_allocation, best_revenue

def calculate_channel_rois(model, processor) -> dict:
    """Calculate ROI for each channel"""
    rois = {}
    base_spend = 100  # Base spend amount
    
    for channel in processor.media_columns:
        # Create base case
        base_case = pd.DataFrame(
            {col: [base_spend] for col in processor.media_columns}
        )
        
        # Create increased spend case
        increased_case = base_case.copy()
        increased_case[channel] *= 1.1  # 10% increase
        
        # Add control variables if any
        if processor.control_columns:
            for col in processor.control_columns:
                base_case[col] = 0
                increased_case[col] = 0
        
        # Process data
        X_base, _ = processor.process(base_case)
        X_increased, _ = processor.process(increased_case)
        
        # Get predictions
        base_pred = model.predict(X_base)[0]
        increased_pred = model.predict(X_increased)[0]
        
        # Calculate ROI
        spend_increase = (increased_case[channel].iloc[0] - base_case[channel].iloc[0])
        revenue_increase = increased_pred - base_pred
        roi = revenue_increase / spend_increase
        
        roi_data = {'roi': roi}
        
        # Add uncertainty estimates for Bayesian model
        if hasattr(model, 'get_uncertainty_estimates'):
            lower, upper = model.get_uncertainty_estimates(X_increased)
            roi_data.update({
                'lower': (lower[0] - base_pred) / spend_increase,
                'upper': (upper[0] - base_pred) / spend_increase
            })
        
        rois[channel] = roi_data
    
    return rois 