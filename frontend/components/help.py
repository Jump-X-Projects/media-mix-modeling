import streamlit as st

def show_help_sidebar():
    """Show help information in the sidebar"""
    with st.sidebar.expander("Help & Documentation"):
        st.markdown("""
        ### Quick Start Guide
        
        1. **Data Upload**
           - Upload your CSV or Excel file
           - Select media spend columns
           - Choose revenue column
        
        2. **Model Selection**
           - Choose a model type
           - Adjust model parameters
           - Enable cross-validation if needed
        
        3. **Results Analysis**
           - View model performance metrics
           - Analyze feature importance
           - Check statistical significance
           - Get spend optimization suggestions
        
        ### Data Format Requirements
        
        Your data should include:
        - Daily/weekly media spend columns
        - Revenue column
        - All numeric values
        - No missing values (or they'll be handled automatically)
        
        ### Model Types
        
        1. **Linear Regression**
           - Simple, interpretable model
           - Good for understanding basic relationships
        
        2. **LightGBM**
           - Advanced tree-based model
           - Handles non-linear relationships
        
        3. **XGBoost**
           - Powerful gradient boosting
           - Good for complex patterns
        
        4. **Bayesian MMM**
           - Provides uncertainty estimates
           - Good for understanding confidence levels
        
        ### Metrics Explanation
        
        - **R²**: Explains variance in revenue (0-1)
        - **RMSE**: Average prediction error
        - **MAE**: Absolute prediction error
        - **MAPE**: Percentage error
        - **p-value**: Statistical significance
        
        ### Need Help?
        
        Contact support at: support@example.com
        """)

def get_tooltips() -> dict:
    """Return tooltips for various UI elements"""
    return {
        'data_upload': {
            'file_upload': "Upload a CSV or Excel file containing your media spend and revenue data",
            'media_columns': "Select columns that contain your daily/weekly media spend data",
            'revenue_column': "Select the column that contains your revenue data"
        },
        'model_selection': {
            'model_type': "Choose the type of model to analyze your data",
            'cross_validation': "Enable to get more robust performance estimates",
            'time_series': "Enable if your data has temporal dependencies",
            'parameters': "Adjust model parameters to optimize performance"
        },
        'results': {
            'metrics': "Key performance indicators for your model",
            'feature_importance': "Impact of each media channel on revenue",
            'significance': "Statistical confidence in the results",
            'optimization': "Suggestions for optimal budget allocation"
        }
    }

def show_metric_help(metric_name: str):
    """Show help text for specific metrics"""
    help_text = {
        'r2': """
        R² Score (Coefficient of Determination)
        - Measures how well the model explains variance in revenue
        - Range: 0 to 1 (higher is better)
        - Example: 0.75 means model explains 75% of revenue variance
        """,
        'rmse': """
        Root Mean Square Error (RMSE)
        - Average prediction error in revenue units
        - Lower values indicate better accuracy
        - Same unit as your revenue
        """,
        'mae': """
        Mean Absolute Error (MAE)
        - Average absolute prediction error
        - Lower values indicate better accuracy
        - Same unit as your revenue
        """,
        'mape': """
        Mean Absolute Percentage Error (MAPE)
        - Average percentage error in predictions
        - Lower values indicate better accuracy
        - Example: 15% means predictions are off by 15% on average
        """,
        'significance': """
        Statistical Significance (p-value)
        - Measures confidence in the results
        - Range: 0 to 1 (lower is better)
        - p < 0.05 is typically considered significant
        """
    }
    return help_text.get(metric_name, "No help available for this metric")

def show_feature_help(feature_type: str):
    """Show help text for feature analysis"""
    help_text = {
        'importance': """
        Feature Importance
        - Shows relative impact of each channel on revenue
        - Higher values indicate stronger influence
        - Based on model coefficients or feature importance scores
        """,
        'correlation': """
        Correlation Matrix
        - Shows relationships between channels
        - Range: -1 to 1
        - Higher absolute values indicate stronger relationships
        """,
        'roi': """
        Return on Investment (ROI)
        - Revenue generated per unit of spend
        - Higher values indicate better performance
        - Calculated using marginal effects
        """
    }
    return help_text.get(feature_type, "No help available for this feature") 