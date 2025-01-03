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
    
    try:
        # Get current spend values
        spend_cols = processor.media_columns
        current_spend = processor.data[spend_cols].mean()
        current_total = current_spend.sum()
        
        # Show current spend summary
        st.write("Current Spend Summary:")
        current_spend_df = pd.DataFrame({
            'Channel': spend_cols,
            'Current Spend': current_spend.values,
            'Percentage': (current_spend.values / current_total) * 100
        })
        current_spend_df['Current Spend'] = current_spend_df['Current Spend'].apply(lambda x: f"${x:,.2f}")
        current_spend_df['Percentage'] = current_spend_df['Percentage'].apply(lambda x: f"{x:.1f}%")
        st.table(current_spend_df)
        
        # Total budget input with validation
        st.write("Current total spend: ${:,.2f}".format(current_total))
        col1, col2 = st.columns([2, 1])
        with col1:
            total_budget = st.number_input(
                "Total Budget",
                min_value=float(current_total * 0.5),  # Minimum 50% of current
                max_value=float(current_total * 2),    # Maximum 200% of current
                value=float(current_total),
                step=current_total * 0.1,              # Step size 10% of current
                format="%.2f",
                help="Enter the total budget to optimize. Must be between 50% and 200% of current spend."
            )
        with col2:
            st.metric(
                "Budget Change",
                f"${total_budget:,.2f}",
                f"{((total_budget - current_total) / current_total) * 100:+.1f}%"
            )
        
        # Create optimizer with historical data
        optimizer = SpendOptimizer(
            model=model,
            feature_names=spend_cols,
            historical_data=processor.data
        )
        
        # Show optimization constraints before running
        with st.expander("View Optimization Constraints", expanded=False):
            st.markdown("""
            The optimization follows these constraints:
            - Total budget must be between 50% and 200% of current spend
            - Individual channel spend cannot be negative
            - Channel spend cannot exceed 150% of historical maximum
            - Channel spend cannot be below historical minimum
            - Total spend must equal specified budget
            """)
            
            if optimizer.channel_bounds:
                st.subheader("Channel-Specific Bounds")
                bounds_df = pd.DataFrame([
                    {
                        'Channel': channel,
                        'Minimum Spend': f"${bounds[0]:,.2f}",
                        'Maximum Spend': f"${bounds[1]:,.2f}",
                        'Current Spend': f"${current_spend[channel]:,.2f}"
                    }
                    for channel, bounds in optimizer.channel_bounds.items()
                ])
                st.table(bounds_df)
        
        if st.button("Optimize Allocation", help="Run optimization to find the best spend allocation"):
            with st.spinner("Optimizing spend allocation..."):
                try:
                    # Run optimization
                    best_allocation = optimizer.optimize(
                        current_spend=current_spend.values,
                        total_budget=total_budget
                    )
                    
                    # Display results
                    st.subheader("Optimal Allocation")
                    
                    # Create comparison dataframe
                    comparison_df = pd.DataFrame({
                        'Channel': list(best_allocation.keys()),
                        'Current Spend': current_spend.values,
                        'Optimized Spend': list(best_allocation.values()),
                        'Change (%)': [(new - old) / old * 100 
                                    for new, old in zip(best_allocation.values(), current_spend.values)]
                    })
                    
                    # Bar chart comparing current vs optimized
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Current',
                        x=comparison_df['Channel'],
                        y=comparison_df['Current Spend'],
                        marker_color='lightgray'
                    ))
                    fig.add_trace(go.Bar(
                        name='Optimized',
                        x=comparison_df['Channel'],
                        y=comparison_df['Optimized Spend'],
                        marker_color='rgb(26, 118, 255)'
                    ))
                    fig.update_layout(
                        title='Current vs Optimized Spend by Channel',
                        barmode='group',
                        yaxis_title='Spend ($)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations table
                    st.subheader("Channel Recommendations")
                    recommendations = []
                    for _, row in comparison_df.iterrows():
                        if row['Change (%)'] > 10:
                            action = "Increase"
                            rationale = "Opportunity for higher returns"
                            icon = "📈"
                        elif row['Change (%)'] < -10:
                            action = "Decrease"
                            rationale = "Potentially overinvested"
                            icon = "📉"
                        else:
                            action = "Maintain"
                            rationale = "Current spend is near optimal"
                            icon = "✅"
                        
                        recommendations.append({
                            'Channel': f"{icon} {row['Channel']}",
                            'Action': action,
                            'Current Spend': f"${row['Current Spend']:,.2f}",
                            'Recommended Spend': f"${row['Optimized Spend']:,.2f}",
                            'Change': f"{row['Change (%)']:+.1f}%",
                            'Rationale': rationale
                        })
                    
                    st.table(pd.DataFrame(recommendations))
                    
                    # Predicted impact
                    current_X = processor.process(pd.DataFrame([current_spend]))[0]
                    optimized_X = processor.process(pd.DataFrame([best_allocation]))[0]
                    
                    current_revenue = model.predict(current_X)[0]
                    optimized_revenue = model.predict(optimized_X)[0]
                    revenue_impact = (optimized_revenue - current_revenue) / current_revenue * 100
                    
                    st.subheader("Predicted Impact")
                    impact_cols = st.columns(3)
                    with impact_cols[0]:
                        st.metric(
                            "Current Revenue",
                            f"${current_revenue:,.2f}"
                        )
                    with impact_cols[1]:
                        st.metric(
                            "Optimized Revenue",
                            f"${optimized_revenue:,.2f}",
                            f"{revenue_impact:+.1f}%"
                        )
                    with impact_cols[2]:
                        roi_change = (optimized_revenue/total_budget - current_revenue/current_total) * 100
                        st.metric(
                            "ROI Change",
                            f"{roi_change:+.1f}%",
                            help="Change in return on investment"
                        )
                    
                except SpendOptimizerError as e:
                    st.error(f"🚫 Optimization Error: {str(e)}")
                    st.info("Try adjusting the total budget or reviewing the optimization constraints.")
                except Exception as e:
                    st.error(f"⚠️ An unexpected error occurred: {str(e)}")
                    st.warning("Please contact support if this error persists.")
    
    except Exception as e:
        st.error("Failed to initialize spend optimization:")
        st.error(str(e))
        st.info("Please ensure your data is properly loaded and contains valid spend values.")

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