import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, List
import numpy as np
from datetime import datetime
from backend.utils.column_mapper import ColumnMapper
from backend.config.column_config import VALIDATION_RULES, DATE_FORMATS

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

def render_data_upload() -> Optional[pd.DataFrame]:
    """Enhanced data upload interface with validation and preview features"""
    st.header("Data Upload")
    
    # Initialize column mapper
    column_mapper = ColumnMapper()
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx"],
        help="Upload your media spend and revenue data"
    )
    
    data = None
    if uploaded_file is not None:
        try:
            # Load data
            data = load_data(uploaded_file)
            
            # Get column suggestions
            suggestions = column_mapper.suggest_column_mappings(data)
            
            # Column mapping interface
            st.subheader("Column Mapping")
            with st.expander("Configure Column Mappings", expanded=True):
                # Date column selection
                date_cols = suggestions.get('date', [])
                if not date_cols:
                    date_cols = data.columns.tolist()
                date_col = st.selectbox(
                    "Select Date Column",
                    date_cols,
                    help="Column containing dates"
                )
                
                # Date format selection if needed
                if date_col:
                    date_format = st.selectbox(
                        "Date Format",
                        DATE_FORMATS,
                        help="Select the format that matches your dates"
                    )
                
                # Spend columns selection
                spend_cols = suggestions.get('spend', [])
                if not spend_cols:
                    spend_cols = [col for col in data.columns if col != date_col]
                selected_spend_cols = st.multiselect(
                    "Select Media Spend Columns",
                    spend_cols,
                    default=spend_cols,
                    help="Columns containing media spend data"
                )
                
                # Revenue column selection
                revenue_cols = suggestions.get('revenue', [])
                if not revenue_cols:
                    revenue_cols = [col for col in data.columns 
                                  if col not in [date_col] + selected_spend_cols]
                revenue_col = st.selectbox(
                    "Select Revenue Column",
                    revenue_cols,
                    help="Column containing revenue data"
                )
            
            # Validate mappings
            if date_col and selected_spend_cols and revenue_col:
                mappings = {
                    'date': date_col,
                    'spend': selected_spend_cols,
                    'revenue': [revenue_col]
                }
                
                # Validate column types
                valid_types, type_error = column_mapper.validate_column_types(data, {
                    'date': date_col,
                    'spend': selected_spend_cols[0],  # Validate first spend column
                    'revenue': revenue_col
                })
                
                if not valid_types:
                    st.error(f"Column Type Error: {type_error}")
                    return None
                
                # Validate date column
                valid_date, date_error, parsed_dates = column_mapper.validate_date_column(
                    data,
                    date_col
                )
                
                if not valid_date:
                    st.error(f"Date Error: {date_error}")
                    return None
                
                # Update date column with parsed dates
                data[date_col] = parsed_dates
                
                # Validate numeric columns
                valid_numeric, numeric_error = column_mapper.validate_numeric_columns(
                    data,
                    selected_spend_cols + [revenue_col]
                )
                
                if not valid_numeric:
                    st.error(f"Numeric Error: {numeric_error}")
                    return None
                
                # Store mappings in session state
                st.session_state.column_mappings = mappings
                
                # Show data preview
                show_data_preview(data)
                
                st.success("Data validation successful!")
                return data
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    return None

def load_data(file) -> pd.DataFrame:
    """Load data from uploaded file"""
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        raise DataValidationError(f"Failed to read file: {str(e)}")

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