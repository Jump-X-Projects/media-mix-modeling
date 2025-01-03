import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple
import numpy as np

def render_data_upload() -> Optional[pd.DataFrame]:
    """Enhanced data upload interface with validation and preview features"""
    st.header("Data Upload")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx"],
        help="Upload your media spend and revenue data"
    )
    
    data = None
    if uploaded_file is not None:
        try:
            # Load and validate data
            data = load_and_validate_data(uploaded_file)
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(data.head(), use_container_width=True)
            
            # Column selection
            spend_cols = [col for col in data.columns if 'spend' in col.lower()]
            revenue_cols = [col for col in data.columns if 'revenue' in col.lower()]
            
            selected_features = st.multiselect(
                "Select Media Channels",
                spend_cols,
                default=spend_cols,
                help="Select the media channels to include in the analysis"
            )
            
            target_col = st.selectbox(
                "Select Target Variable",
                revenue_cols,
                help="Select the revenue column to predict"
            )
            
            if selected_features and target_col:
                # Store selected columns in session state
                st.session_state.selected_features = selected_features
                st.session_state.target_col = target_col
                st.session_state.data = data
                
                st.success("Data configured successfully!")
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            
    return data

def load_and_validate_data(file) -> pd.DataFrame:
    """Load and validate uploaded data"""
    # Load data based on file type
    if file.name.endswith('.csv'):
        data = pd.read_csv(file)
    else:
        data = pd.read_excel(file)
    
    # Basic validation
    required_cols = validate_columns(data)
    if not required_cols['valid']:
        st.warning(required_cols['message'])
        st.stop()
    
    # Convert date column if present
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
    
    return data

def validate_columns(data: pd.DataFrame) -> dict:
    """Validate required columns and data types"""
    # Check for date column
    if 'Date' not in data.columns:
        return {
            'valid': False,
            'message': "Missing 'Date' column"
        }
    
    # Check for at least one spend column
    spend_cols = [col for col in data.columns if 'spend' in col.lower()]
    if not spend_cols:
        return {
            'valid': False,
            'message': "No spend columns found. Column names should contain 'spend'"
        }
    
    # Check for revenue column
    revenue_cols = [col for col in data.columns if 'revenue' in col.lower()]
    if not revenue_cols:
        return {
            'valid': False,
            'message': "No revenue column found. Column name should contain 'revenue'"
        }
    
    return {'valid': True, 'message': ""}

def show_data_preview(data: pd.DataFrame):
    """Show interactive data preview with statistics"""
    st.subheader("Data Preview")
    
    # Data summary tabs
    tab1, tab2, tab3 = st.tabs(["Preview", "Statistics", "Time Series"])
    
    with tab1:
        st.dataframe(
            data.head(),
            use_container_width=True
        )
        st.caption(f"Total rows: {len(data)}")
    
    with tab2:
        show_data_statistics(data)
    
    with tab3:
        show_time_series_preview(data)

def show_data_statistics(data: pd.DataFrame):
    """Show key statistics about the data"""
    # Basic statistics
    st.markdown("#### Basic Statistics")
    stats_df = data.describe()
    st.dataframe(stats_df, use_container_width=True)
    
    # Missing values
    st.markdown("#### Missing Values")
    missing = data.isnull().sum()
    if missing.any():
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Values': missing.values,
            'Percentage': (missing.values / len(data)) * 100
        })
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("No missing values found!")
    
    # Correlation heatmap
    st.markdown("#### Correlation Matrix")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr = data[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        fig.update_layout(
            title="Correlation Heatmap",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

def show_time_series_preview(data: pd.DataFrame):
    """Show time series preview of spend and revenue"""
    if 'Date' not in data.columns:
        st.warning("No date column found for time series visualization")
        return
    
    # Identify spend and revenue columns
    spend_cols = [col for col in data.columns if 'spend' in col.lower()]
    revenue_cols = [col for col in data.columns if 'revenue' in col.lower()]
    
    # Plot spend over time
    if spend_cols:
        fig = go.Figure()
        for col in spend_cols:
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data[col],
                name=col,
                mode='lines'
            ))
        fig.update_layout(
            title="Channel Spend Over Time",
            xaxis_title="Date",
            yaxis_title="Spend",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Plot revenue over time
    if revenue_cols:
        fig = go.Figure()
        for col in revenue_cols:
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data[col],
                name=col,
                mode='lines'
            ))
        fig.update_layout(
            title="Revenue Over Time",
            xaxis_title="Date",
            yaxis_title="Revenue",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def generate_template() -> pd.DataFrame:
    """Generate a template DataFrame"""
    dates = pd.date_range(start='2023-01-01', periods=10)
    return pd.DataFrame({
        'Date': dates,
        'TV_Spend': [1000] * 10,
        'Radio_Spend': [500] * 10,
        'Social_Spend': [750] * 10,
        'Revenue': [5000] * 10
    }) 