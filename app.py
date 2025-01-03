import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from backend.models.model_factory import ModelFactory
from backend.models.spend_optimizer import SpendOptimizer

def init_session_state():
    """Initialize session state variables"""
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'show_results_button' not in st.session_state:
        st.session_state.show_results_button = False

def next_step():
    st.session_state.step += 1
    st.rerun()

def prev_step():
    st.session_state.step -= 1
    st.rerun()

def render_step_1_data_upload():
    """Render the data upload page"""
    st.title("Media Mix Modeling")
    st.subheader("Step 1: Data Upload")
    
    with st.expander("📋 Data Requirements", expanded=True):
        st.markdown("""
        Your data should include:
        - Daily media spend by channel (e.g., TV, Radio, Social, Print)
        - Daily revenue
        - Optional: Other variables (seasonality, promotions, etc.)
        
        **Data Format:**
        - CSV file
        - One row per day
        - Numeric values for spend and revenue
        """)
        
    uploaded_file = st.file_uploader("Upload your data file (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            
            st.write("Data Preview:")
            st.dataframe(data.head())
            
            st.write("Data Statistics:")
            st.dataframe(data.describe())
            
            # Data validation
            if data.isnull().any().any():
                st.warning("⚠️ Your data contains missing values. Please handle them before proceeding.")
            
            if (data.select_dtypes(include=[np.number]) < 0).any().any():
                st.warning("⚠️ Some numeric columns contain negative values. Please verify your data.")
            
            if st.button("Continue to Model Selection →"):
                next_step()
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def model_selection_page():
    """Render the model selection page"""
    st.header("Step 2: Model Selection")
    
    if 'data' not in st.session_state:
        st.error("Please upload data first")
        return
        
    # Model selection
    model_type = st.selectbox(
        "Select Model Type",
        ["Linear", "LightGBM", "XGBoost", "BMMM", "Meta Robyn"],
        help="Choose the type of model to train"
    )
    
    # Model-specific parameters
    model_params = {}
    if model_type in ["LightGBM", "XGBoost"]:
        n_trees = st.slider("Number of Trees", 100, 1000, 500)
        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
        model_params = {
            "n_estimators": n_trees,
            "learning_rate": learning_rate
        }
    elif model_type == "BMMM":
        n_samples = st.slider("Number of Samples", 1000, 5000, 2000)
        warmup_steps = st.slider("Warmup Steps", 100, 1000, 500)
        model_params = {
            "n_samples": n_samples,
            "warmup_steps": warmup_steps
        }
    elif model_type == "Meta Robyn":
        budget = st.number_input("Budget Constraint", 10000)
        model_params = {"budget_constraint": budget}
    
    # Train button
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            result = train_model(st.session_state.data, model_type.lower(), **model_params)
            
            if result:
                st.session_state['model_results'] = result
                st.success("Model trained successfully!")
                next_step()
            
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Data Upload"):
            prev_step()
    with col2:
        if 'model_results' in st.session_state:
            if st.button("View Results →"):
                next_step()

def render_step_3_results():
    """Render the results page"""
    st.title("Media Mix Modeling")
    st.subheader("Step 3: Results")
    
    if 'model_results' not in st.session_state:
        st.error("Please train a model first")
        if st.button("← Back to Model Selection"):
            prev_step()
        return
    
    results = st.session_state.model_results
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Performance", 
        "🔍 Feature Analysis", 
        "💰 Revenue Predictor",
        "⚡ Spend Optimizer"
    ])
    
    with tab1:
        # Display metrics
        st.subheader("Model Performance Metrics")
        metrics = results['cv_metrics']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{metrics['mean_r2']:.3f} ± {metrics['std_r2']:.3f}")
        with col2:
            st.metric("RMSE", f"{metrics['mean_rmse']:.3f} ± {metrics['std_rmse']:.3f}")
        with col3:
            st.metric("MAE", f"{metrics['mean_mae']:.3f} ± {metrics['std_mae']:.3f}")
            
        # Add residual plot
        model = results['model']
        data = st.session_state.data
        X = data.drop('Revenue', axis=1)
        y_pred = model.predict(X)
        y_true = data['Revenue']
        residuals = y_true - y_pred
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuals'
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title="Residual Plot",
            xaxis_title="Predicted Revenue",
            yaxis_title="Residuals"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature importance plot
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': list(results['feature_importance'].keys()),
                'Importance': list(results['feature_importance'].values())
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Correlation matrix
            st.subheader("Correlation Matrix")
            corr_matrix = data.corr()
            fig = px.imshow(
                corr_matrix,
                title='Correlation Matrix',
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Revenue Predictor")
        st.markdown("""
        Simulate potential revenue by adjusting media spend across channels.
        Use the sliders to modify spend levels and see the predicted revenue impact.
        """)
        
        # Get current spend values
        spend_cols = [col for col in data.columns if col != 'Revenue']
        current_spend = data[spend_cols].mean()
        
        # Create sliders for each channel
        new_spend = {}
        for col in spend_cols:
            current_val = float(current_spend[col])
            new_spend[col] = st.slider(
                f"{col}",
                min_value=current_val * 0.5,
                max_value=current_val * 1.5,
                value=current_val,
                format="$%.2f"
            )
        
        # Create prediction input
        X_pred = pd.DataFrame([new_spend])
            
        # Make prediction
        predicted_revenue = model.predict(X_pred)[0]
        current_revenue = data['Revenue'].mean()
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Predicted Revenue",
                f"${predicted_revenue:,.2f}",
                f"{((predicted_revenue - current_revenue) / current_revenue) * 100:.1f}%"
            )
        with col2:
            st.metric(
                "Current Revenue (avg)",
                f"${current_revenue:,.2f}"
            )
    
    with tab4:
        st.subheader("Spend Optimizer")
        st.markdown("""
        View optimized spend allocation across channels to maximize revenue.
        The optimizer considers current performance and channel interactions.
        """)
        
        # Get current spend
        current_spend = data[spend_cols].mean().values
        
        # Create optimizer
        optimizer = SpendOptimizer(
            model=model,
            feature_names=spend_cols,
            budget_constraints=None
        )
        
        # Get optimized spend
        optimized_spend = optimizer.optimize(current_spend)
        
        # Display current vs optimized spend
        spend_df = pd.DataFrame({
            'Channel': list(optimized_spend.keys()),
            'Current Spend': current_spend,
            'Optimized Spend': list(optimized_spend.values())
        })
        
        # Calculate and add ROI
        spend_df['ROI (%)'] = ((spend_df['Optimized Spend'] - spend_df['Current Spend']) 
                              / spend_df['Current Spend'] * 100)
        
        # Bar chart comparing spends
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Current',
            x=spend_df['Channel'],
            y=spend_df['Current Spend'],
            marker_color='lightgray'
        ))
        fig.add_trace(go.Bar(
            name='Optimized',
            x=spend_df['Channel'],
            y=spend_df['Optimized Spend'],
            marker_color='rgb(26, 118, 255)'
        ))
        fig.update_layout(
            title='Current vs Optimized Spend by Channel',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ROI Analysis
        fig = px.bar(
            spend_df,
            x='Channel',
            y='ROI (%)',
            title='Expected ROI by Channel',
            color='ROI (%)',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations table
        st.subheader("Spend Recommendations")
        recommendations = []
        for _, row in spend_df.iterrows():
            if row['ROI (%)'] > 10:
                action = "Increase"
            elif row['ROI (%)'] < -10:
                action = "Decrease"
            else:
                action = "Maintain"
            recommendations.append({
                'Channel': row['Channel'],
                'Action': action,
                'Current Spend': f"${row['Current Spend']:,.2f}",
                'Recommended Spend': f"${row['Optimized Spend']:,.2f}",
                'Expected ROI': f"{row['ROI (%)']:,.1f}%"
            })
        st.table(pd.DataFrame(recommendations))
    
    if st.button("← Back to Model Selection"):
        prev_step()

def train_model(data, model_type, **model_params):
    """Train the selected model with the uploaded data"""
    try:
        # Get feature names (all columns except 'Revenue')
        feature_names = [col for col in data.columns if col != 'Revenue']
        X = data[feature_names]
        y = data['Revenue']
        
        # Create and train model
        model = ModelFactory.create_model(model_type, **model_params)
        model.fit(X, y)
        
        # Calculate metrics
        metrics = model.evaluate(X, y)
        cv_metrics = ModelFactory.cross_validate(model, X, y)
        feature_importance = model.feature_importances(feature_names)
        
        # Store results in session state
        st.session_state['model_results'] = {
            'model': model,
            'metrics': metrics,
            'cv_metrics': cv_metrics,
            'feature_importance': feature_importance
        }
        
        return st.session_state['model_results']
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(
        page_title="Media Mix Modeling",
        page_icon="📊",
        layout="wide"
    )
    
    init_session_state()
    
    # Progress bar
    progress_text = ["1. Data Upload", "2. Model Selection", "3. Results"]
    st.progress((st.session_state.step - 1) / 2)
    st.markdown(f"**Current Step: {progress_text[st.session_state.step - 1]}**")
    
    # Render current step
    if st.session_state.step == 1:
        render_step_1_data_upload()
    elif st.session_state.step == 2:
        model_selection_page()
    else:
        render_step_3_results()
    
    st.markdown("---")
    st.markdown("Made with ❤️ by Your Team")

if __name__ == "__main__":
    main() 