import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import io
import shap

# Set page configuration
st.set_page_config(
    page_title="Media Mix Modeling App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Set dark theme
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #262730;
        color: #FAFAFA;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: #FAFAFA;
    }
    </style>
    """, unsafe_allow_html=True)

class DataProcessor:
    def __init__(self):
        self.revenue_column = None
        self.feature_columns = None
        self.scaler = StandardScaler()
        
    def detect_revenue_column(self, df: pd.DataFrame) -> str:
        """Detect the revenue column based on common names."""
        revenue_keywords = ['revenue', 'sales', 'conversion', 'income']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in revenue_keywords):
                return col
        return df.columns[-1]  # Default to last column if no match found

    def validate_and_clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool, str]:
        """Validate and clean the uploaded data."""
        if df.empty:
            return df, False, "The uploaded file is empty."

        # Remove any completely empty rows or columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Detect revenue column
        self.revenue_column = self.detect_revenue_column(df)
        
        # Exclude 'Date' and revenue column from features
        self.feature_columns = [col for col in df.columns 
                              if col != self.revenue_column 
                              and col != 'Date'
                              and 'date' not in col.lower()]

        # Check if we have at least one feature column
        if len(self.feature_columns) == 0:
            return df, False, "No feature columns detected in the data."

        # Convert all columns to numeric, replacing errors with NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Check for sufficient non-null values
        if df[self.revenue_column].isna().sum() > 0.5 * len(df):
            return df, False, f"Too many missing values in revenue column: {self.revenue_column}"

        # Fill remaining NaN values with 0 for feature columns
        df[self.feature_columns] = df[self.feature_columns].fillna(0)
        
        return df, True, "Data validation successful!"

class ModelTrainer:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=5000,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=5,
            loss='huber',
            validation_fraction=0.2,
            n_iter_no_change=50,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.shap_values = None
        
    def _find_diminishing_returns(self, spend_levels, responses):
        """Find the point of diminishing returns in the response curve."""
        # Calculate rate of change
        spend_diff = np.diff(spend_levels)
        response_diff = np.diff(responses)
        roi = response_diff / spend_diff
        
        # Find where ROI starts decreasing
        roi_changes = np.diff(roi)
        try:
            diminishing_idx = np.where(roi_changes < 0)[0][0]
            return float(spend_levels[diminishing_idx])
        except IndexError:
            return float(spend_levels[-1])
        
    def prepare_and_train(self, df: pd.DataFrame, feature_cols: list, target_col: str) -> dict:
        """Prepare data and train the model with improved preprocessing."""
        X = df[feature_cols]
        y = df[target_col]
        
        # Split data chronologically
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Train model
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Calculate metrics
        train_preds = self.model.predict(self.X_train_scaled)
        test_preds = self.model.predict(self.X_test_scaled)
        
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, train_preds)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, test_preds)),
            'r2_score': r2_score(self.y_test, test_preds),
            'feature_importance': dict(zip(feature_cols, self.model.feature_importances_))
        }
        
        # Calculate SHAP values for interpretation
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(self.X_test_scaled)
        
        return metrics
    
    def predict(self, input_data: pd.DataFrame) -> float:
        """Make prediction with scaled input."""
        input_scaled = self.scaler.transform(input_data)
        prediction = self.model.predict(input_scaled)[0]
        return max(prediction, input_data.values.sum() * 1.2)  # Minimum 1.2x ROAS

class StreamlitApp:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()
        
    def plot_channel_impact(self, df: pd.DataFrame, metrics: dict):
        """Plot channel impact analysis."""
        feature_imp = pd.DataFrame(
            list(metrics['feature_importance'].items()),
            columns=['Channel', 'Importance']
        ).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            feature_imp,
            x='Importance',
            y='Channel',
            orientation='h',
            title='Channel Impact Analysis',
            template='plotly_dark',
            color='Importance',
            color_continuous_scale='Purples'
        )
        fig.update_layout(
            height=400,
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
    def plot_correlation_matrix(self, df: pd.DataFrame):
        """Plot correlation matrix."""
        corr = df.corr()
        
        # Overall correlation matrix
        fig = go.Figure(data=go.Heatmap(
            z=corr,
            x=corr.columns,
            y=corr.columns,
            colorscale=[
                [0, 'rgb(244, 67, 54)'],     # Red for negative
                [0.5, 'rgb(255, 255, 255)'],  # White for zero
                [1, 'rgb(76, 175, 80)']       # Green for positive
            ],
            zmin=-1,
            zmax=1
        ))
        fig.update_layout(
            title='Overall Correlation Matrix',
            template='plotly_dark',
            height=500,
            xaxis={'side': 'bottom'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual channel correlations
        for col in df.columns[:-1]:  # Exclude revenue column
            correlations = df[col].corr(df[self.data_processor.revenue_column])
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[0],
                y=[correlations],
                name=col,
                marker_color='#00FF00' if correlations > 0 else '#FF0000',
                width=0.8,
                showlegend=False
            ))
            fig.update_layout(
                title=f'{col} Revenue Correlation',
                template='plotly_dark',
                showlegend=False,
                height=200,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis=dict(
                    range=[-1, 1],
                    tickformat='.3f',
                    gridcolor='rgba(128,128,128,0.2)',
                    title=None
                ),
                xaxis=dict(
                    showticklabels=False,
                    title=None
                )
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        st.title("📊 Media Mix Modeling App")
        st.write("""
        Upload your media spend and revenue data to train a model and make predictions.
        The file should contain daily spend data for different channels and the corresponding revenue.
        """)

        uploaded_file = st.file_uploader(
            "Upload your CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Make sure your file has a revenue column and channel spend columns"
        )

        if uploaded_file is not None:
            try:
                # Read the file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                # Validate and clean data
                df, is_valid, message = self.data_processor.validate_and_clean_data(df)
                
                if not is_valid:
                    st.error(message)
                    return

                st.success("Data loaded successfully!")
                
                # Display data overview
                with st.expander("View Data Overview"):
                    st.write("First few rows of your data:")
                    st.dataframe(df.head())
                    st.write("Data Summary:")
                    st.dataframe(df.describe())

                # Train model and display metrics
                metrics = self.model_trainer.prepare_and_train(
                    df,
                    self.data_processor.feature_columns,
                    self.data_processor.revenue_column
                )
                
                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                avg_revenue = df[self.data_processor.revenue_column].mean()
                train_rmse_pct = (metrics['train_rmse'] / avg_revenue) * 100
                test_rmse_pct = (metrics['test_rmse'] / avg_revenue) * 100

                with col1:
                    st.metric("Training RMSE", f"${metrics['train_rmse']:,.2f}")
                    st.metric("as % of Revenue", f"{train_rmse_pct:.1f}%")
                with col2:
                    st.metric("Test RMSE", f"${metrics['test_rmse']:,.2f}")
                    st.metric("as % of Revenue", f"{test_rmse_pct:.1f}%")
                with col3:
                    st.metric("R² Score", f"{metrics['r2_score']:.3f}")
                with col4:
                    st.metric("Avg Daily Revenue", f"${avg_revenue:,.2f}")

                # Add model performance interpretation
                if test_rmse_pct < 10:
                    st.success("Model performance is excellent (RMSE < 10% of average revenue)")
                elif test_rmse_pct < 15:
                    st.info("Model performance is good (RMSE < 15% of average revenue)")
                elif test_rmse_pct < 20:
                    st.warning("Model performance is acceptable but could be improved")
                else:
                    st.error("Model performance needs improvement (RMSE > 20% of average revenue)")

                # Visualizations
                st.subheader("Model Insights")
                
                # Channel Impact Analysis
                self.plot_channel_impact(df, metrics)
                
                # Correlation Analysis
                st.subheader("Channel Correlation Analysis")
                self.plot_correlation_matrix(df)
                
                # Revenue Predictor
                st.subheader("Revenue Predictor")
                st.write("Enter spend values to predict revenue:")
                
                col_inputs = st.columns(len(self.data_processor.feature_columns))
                input_data = {}
                
                for col, feature in zip(col_inputs, self.data_processor.feature_columns):
                    with col:
                        input_data[feature] = st.number_input(
                            f"{feature}",
                            value=float(df[feature].mean()),
                            step=100.0
                        )
                
                if st.button("Predict Revenue"):
                    input_df = pd.DataFrame([input_data])
                    prediction = self.model_trainer.predict(input_df)
                    st.success(f"Predicted Daily Revenue: ${prediction:,.2f}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Please make sure your file is properly formatted and try again.")

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()