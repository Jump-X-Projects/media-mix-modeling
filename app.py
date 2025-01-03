import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from backend.models.model_factory import ModelFactory

def main():
    st.set_page_config(
        page_title="Media Mix Modeling",
        page_icon="📊",
        layout="wide"
    )

    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Data Upload"

    # Title and description (using markdown for consistent rendering)
    st.markdown("# Media Mix Modeling")
    st.markdown("Analyze and optimize your media spend across channels")

    # Sidebar navigation (using markdown for consistent rendering)
    with st.sidebar:
        st.markdown("# Navigation")
        current_page = st.selectbox(
            "Go to",
            ["Data Upload", "Model Selection", "Results"]
        )
        st.session_state.current_page = current_page

    # Main content area with responsive layout
    if st.session_state.current_page == "Data Upload":
        render_data_upload()
    elif st.session_state.current_page == "Model Selection":
        render_model_selection()
    elif st.session_state.current_page == "Results":
        render_results_dashboard()

    # Footer with responsive layout
    st.markdown("---")
    footer_cols = st.columns(3)
    with footer_cols[1]:
        st.markdown("<div style='text-align: center'>Made with ❤️ by Your Team</div>", unsafe_allow_html=True)

def render_data_upload():
    """Render data upload section"""
    st.header("Data Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your data file (CSV or Excel)",
            type=["csv", "xlsx"]
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                st.session_state.data = data
                st.write("Data Preview:")
                st.dataframe(data.head())
                
                # Column selection
                st.subheader("Column Selection")
                target_col = st.selectbox(
                    "Select target variable (Revenue)",
                    data.columns
                )
                feature_cols = st.multiselect(
                    "Select feature variables (Media channels)",
                    [col for col in data.columns if col != target_col]
                )
                
                if target_col and feature_cols:
                    st.session_state.target_column = target_col
                    st.session_state.feature_columns = feature_cols
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with col2:
        st.markdown("### Data Upload Guide")
        st.markdown("""
        Your data should include:
        - Daily media spend by channel
        - Daily revenue
        - Optional: Other variables
        """)
        
        if st.button("Download Template"):
            # Create sample template
            template = pd.DataFrame({
                'Date': pd.date_range(start='2024-01-01', periods=10),
                'TV_Spend': np.random.uniform(1000, 5000, 10),
                'Radio_Spend': np.random.uniform(500, 2000, 10),
                'Social_Spend': np.random.uniform(300, 1500, 10),
                'Revenue': np.random.uniform(5000, 15000, 10)
            })
            
            # Convert to CSV
            csv = template.to_csv(index=False)
            st.download_button(
                "Download CSV Template",
                csv,
                "mmm_template.csv",
                "text/csv"
            )

def render_model_selection():
    """Render model selection section"""
    st.header("Model Selection")
    
    if st.session_state.data is None:
        st.warning("Please upload data first")
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        model_type = st.selectbox(
            "Select Model Type",
            ["Linear", "LightGBM", "XGBoost", "Bayesian"],
            help="Choose the type of model to use for analysis"
        )
        
        # Cross-validation settings
        st.subheader("Cross-Validation Settings")
        cv_folds = st.slider("Number of CV Folds", min_value=2, max_value=10, value=5)
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                try:
                    # Create and train model
                    model = ModelFactory.create_model(model_type.lower())
                    X = st.session_state.data[st.session_state.feature_columns]
                    y = st.session_state.data[st.session_state.target_column]
                    
                    # Perform cross-validation
                    cv_scores = ModelFactory.cross_validate(model, X, y, cv=cv_folds)
                    
                    # Train final model
                    model.fit(X, y)
                    st.session_state.model = model
                    
                    # Store results
                    st.session_state.results = {
                        'cv_scores': cv_scores,
                        'feature_importance': model.feature_importances(X.columns) if hasattr(model, 'feature_importances') else None
                    }
                    
                    st.success("Model trained successfully!")
                    
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
    
    with col2:
        st.markdown("### Model Information")
        if model_type == "Linear":
            st.markdown("""
            **Linear Regression**
            - Simple and interpretable
            - Assumes linear relationships
            - Fast training
            """)
        elif model_type == "LightGBM":
            st.markdown("""
            **LightGBM**
            - Gradient boosting framework
            - Handles non-linear relationships
            - Fast training speed
            """)
        elif model_type == "XGBoost":
            st.markdown("""
            **XGBoost**
            - Gradient boosting framework
            - High predictive accuracy
            - Handles missing values
            """)
        elif model_type == "Bayesian":
            st.markdown("""
            **Bayesian Model**
            - Provides uncertainty estimates
            - Handles small datasets well
            - More interpretable results
            """)

def render_results_dashboard():
    """Render results visualization dashboard"""
    st.header("Results Dashboard")
    
    if st.session_state.model is None:
        st.warning("Please train a model first")
        return
    
    # Display cross-validation results
    st.subheader("Model Performance")
    cv_scores = st.session_state.results['cv_scores']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "R² Score",
            f"{cv_scores['mean_r2']:.3f} ± {cv_scores['std_r2']:.3f}"
        )
    with col2:
        st.metric(
            "RMSE",
            f"{cv_scores['mean_rmse']:.3f} ± {cv_scores['std_rmse']:.3f}"
        )
    with col3:
        st.metric(
            "MAE",
            f"{cv_scores['mean_mae']:.3f} ± {cv_scores['std_mae']:.3f}"
        )
    
    # Feature importance plot
    if st.session_state.results['feature_importance'] is not None:
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': st.session_state.feature_columns,
            'Importance': st.session_state.results['feature_importance']
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Actual vs Predicted plot
    st.subheader("Actual vs Predicted")
    X = st.session_state.data[st.session_state.feature_columns]
    y = st.session_state.data[st.session_state.target_column]
    y_pred = st.session_state.model.predict(X)
    
    fig = px.scatter(
        x=y,
        y=y_pred,
        labels={'x': 'Actual', 'y': 'Predicted'},
        title='Actual vs Predicted Values'
    )
    fig.add_trace(
        go.Scatter(
            x=[y.min(), y.max()],
            y=[y.min(), y.max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash')
        )
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 