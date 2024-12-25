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
from quality_markers import ModelQualityMarkers

# Set page configuration
st.set_page_config(
    page_title="Media Mix Modeling App",
    page_icon="📊",
    layout="wide"
)

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
        # Use a more sophisticated model configuration
        self.model = GradientBoostingRegressor(
            n_estimators=5000,            # More trees for better learning
            learning_rate=0.01,           # Slightly higher learning rate
            max_depth=5,                  # Allow more complex relationships
            subsample=0.8,                # Prevent overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            loss='huber',                 # More robust to outliers
            validation_fraction=0.2,
            n_iter_no_change=50,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.shap_values = None
        
    def prepare_and_train(self, df: pd.DataFrame, feature_cols: list, target_col: str) -> dict:
        """Prepare data and train the model with improved preprocessing."""
        X = df[feature_cols]
        y = df[target_col]
        
        # Add feature engineering
        X = self._engineer_features(X)
        
        # Split data chronologically
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # Keep time series order
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
            'feature_importance': dict(zip(
                self.X_train.columns,  # Use engineered feature names
                self.model.feature_importances_
            ))
        }
        
        # Calculate SHAP values for interpretation
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(self.X_test_scaled)
        
        # Add interpretation metrics
        metrics['channel_contributions'] = self._calculate_channel_contributions()
        metrics['marginal_effects'] = self._calculate_marginal_effects()
        
        # Add quality markers
        quality_markers = ModelQualityMarkers(
            model=self.model,
            X_train=self.X_train_scaled,
            X_test=self.X_test_scaled,
            y_train=self.y_train,
            y_test=self.y_test,
            feature_names=self.X_train.columns
        )
        metrics['quality_markers'] = quality_markers.get_all_metrics()
        metrics['model_assessment'] = quality_markers.get_model_assessment()
        
        return metrics
    
    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to improve model performance."""
        X_new = X.copy()
        
        # Add interaction terms for related channels
        X_new['Social_Synergy'] = X_new['Facebook_Spend'] * X_new['Instagram_Spend']
        X_new['Video_Synergy'] = X_new['YouTube_Spend'] * X_new['TikTok_Spend']
        
        # Add polynomial terms for non-linear relationships
        for col in X.columns:
            X_new[f'{col}_Squared'] = X_new[col] ** 2
        
        # Add total spend feature
        X_new['Total_Spend'] = X_new[X.columns].sum(axis=1)
        
        return X_new
    
    def predict(self, input_data: pd.DataFrame) -> float:
        """Make prediction with engineered features."""
        # Add engineered features to input data
        input_engineered = self._engineer_features(input_data)
        
        # Scale the input
        input_scaled = self.scaler.transform(input_engineered)
        
        # Make prediction
        prediction = self.model.predict(input_scaled)[0]
        
        # Apply business logic constraints
        prediction = max(prediction, input_data.values.sum() * 1.2)  # Minimum 1.2x ROAS
        
        return prediction
    
    def _calculate_channel_contributions(self):
        """Calculate each channel's contribution to revenue"""
        contributions = {}
        for idx, feature in enumerate(self.X_train.columns):
            mean_impact = np.abs(self.shap_values[:, idx]).mean()
            contributions[feature] = {
                'absolute_contribution': mean_impact,
                'relative_contribution': mean_impact / np.abs(self.shap_values).mean(),
                'direction': 'Positive' if self.model.feature_importances_[idx] > 0 else 'Negative'
            }
        return contributions
        
    def _calculate_marginal_effects(self):
        """Calculate how changes in spend affect revenue"""
        effects = {}
        for feature in self.X_train.columns:
            if '_Spend' in feature:
                base_value = self.X_train[feature].mean()
                variations = np.linspace(base_value * 0.5, base_value * 1.5, 20)
                responses = []
                
                for var in variations:
                    test_data = self.X_train.copy()
                    test_data[feature] = var
                    pred = self.model.predict(self.scaler.transform(test_data))
                    responses.append(pred.mean())
                    
                effects[feature] = {
                    'spend_levels': variations.tolist(),
                    'revenue_impact': responses,
                    'diminishing_returns_point': self._find_diminishing_returns(variations, responses)
                }
        return effects

    def _find_diminishing_returns(self, spend_levels, responses):
        """Find the point where marginal returns start diminishing significantly"""
        # Calculate marginal returns
        marginal_returns = np.diff(responses) / np.diff(spend_levels)
        
        # Find where marginal returns drop below half of max
        max_return = marginal_returns[0]  # Assume highest return is at start
        threshold = max_return * 0.5
        
        # Find first point where returns drop below threshold
        diminishing_points = np.where(marginal_returns < threshold)[0]
        if len(diminishing_points) > 0:
            return spend_levels[diminishing_points[0]]
        else:
            return spend_levels[-1]  # Return max spend if no clear diminishing point

    def _generate_recommendations(self, metrics):
        """Generate optimization recommendations based on model insights"""
        recommendations = {}
        
        # Analyze channel contributions
        contributions = metrics['channel_contributions']
        for channel, stats in contributions.items():
            if '_Spend' in channel:
                base_recommendation = ""
                
                # Check contribution direction
                if stats['direction'] == 'Positive':
                    if stats['relative_contribution'] > 0.2:
                        base_recommendation = "High impact channel. "
                    elif stats['relative_contribution'] > 0.1:
                        base_recommendation = "Moderate impact channel. "
                    else:
                        base_recommendation = "Low impact channel. "
                else:
                    base_recommendation = "Negative impact detected. "
                
                # Add marginal effects insight if available
                if channel in metrics['marginal_effects']:
                    effects = metrics['marginal_effects'][channel]
                    dim_point = effects['diminishing_returns_point']
                    current_spend = np.mean(effects['spend_levels'])
                    
                    if dim_point < current_spend:
                        base_recommendation += "Consider reducing spend (diminishing returns)."
                    elif dim_point > current_spend * 1.2:
                        base_recommendation += "Room for spend increase before diminishing returns."
                    else:
                        base_recommendation += "Current spend level is near optimal."
                
                recommendations[channel] = base_recommendation
        
        return recommendations

class StreamlitApp:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()
        
    def run(self):
        st.title("📊 Media Mix Modeling App")
        st.write("""
        Upload your media spend and revenue data to train a model and make predictions.
        The file should contain daily spend data for different channels and the corresponding revenue.
        """)

        # File upload section
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

                # Display quality markers section
                st.header("Model Quality Assessment")
                
                # Display basic metrics
                st.subheader("Model Quality Metrics")
                
                # Style the metrics table
                def style_metric_row(row):
                    color_map = {
                        'Excellent': 'background-color: #90EE90',
                        'Good': 'background-color: #FFFFE0',
                        'Poor': 'background-color: #FFB6C1'
                    }
                    return [''] * len(row) if row.name != 'Assessment' else [color_map.get(x, '') for x in row]
                
                styled_metrics = metrics['quality_markers']['basic_metrics'].style.apply(style_metric_row, axis=1)
                st.dataframe(styled_metrics)
                
                # Display feature significance
                st.subheader("Feature Significance Analysis")
                
                def style_significance_row(row):
                    color_map = {
                        'Significant': 'background-color: #90EE90',
                        'Not Significant': 'background-color: #FFB6C1'
                    }
                    return [''] * len(row) if row.name != 'Assessment' else [color_map.get(x, '') for x in row]
                
                styled_significance = metrics['quality_markers']['feature_significance'].style.apply(style_significance_row, axis=1)
                st.dataframe(styled_significance)
                
                # Display overall assessment
                st.subheader("Overall Model Assessment")
                for line in metrics['model_assessment']['assessment']:
                    st.write(line)
                
                st.subheader("Recommendations")
                for rec in metrics['model_assessment']['recommendations']:
                    st.write(rec)
                
                # Add explanatory text
                with st.expander("📚 Understanding Model Quality Metrics"):
                    st.write("""
                    **R-squared (R²)**: Measures how well the model explains the variance in the target variable.
                    - \> 0.8: Excellent
                    - \> 0.7: Good
                    - < 0.7: Poor
                    
                    **Adjusted R-squared**: Similar to R², but penalizes adding predictors that don't help the model.
                    - \> 0.75: Excellent
                    - \> 0.65: Good
                    - < 0.65: Poor
                    
                    **RMSE (Root Mean Square Error)**: Average prediction error in the same units as revenue.
                    - < 10% of avg revenue: Excellent
                    - < 15% of avg revenue: Good
                    - \> 15% of avg revenue: Poor
                    
                    **MAPE (Mean Absolute Percentage Error)**: Average percentage error of predictions.
                    - < 10%: Excellent
                    - < 20%: Good
                    - \> 20%: Poor
                    
                    **Feature Significance (p-value)**:
                    - < 0.05: Feature has significant impact
                    - \> 0.05: Feature might not be useful
                    """)

                # Calculate average revenue and RMSE percentages
                avg_revenue = df[self.data_processor.revenue_column].mean()
                train_rmse_pct = (metrics['train_rmse'] / avg_revenue) * 100
                test_rmse_pct = (metrics['test_rmse'] / avg_revenue) * 100

                # Display metrics with context
                st.write(f"Average Daily Revenue: ${avg_revenue:,.2f}")
                col1, col2, col3, col4 = st.columns(4)
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

                # Add interpretation
                if test_rmse_pct < 10:
                    st.success("Model performance is excellent (RMSE < 10% of average revenue)")
                elif test_rmse_pct < 15:
                    st.info("Model performance is good (RMSE < 15% of average revenue)")
                elif test_rmse_pct < 20:
                    st.warning("Model performance is acceptable but could be improved")
                else:
                    st.error("Model performance needs improvement (RMSE > 20% of average revenue)")

                # Feature importance plot
                st.subheader("Channel Impact Analysis")
                fig = px.bar(
                    x=list(metrics['feature_importance'].keys()),
                    y=list(metrics['feature_importance'].values()),
                    title="Channel Importance",
                    labels={'x': 'Channel', 'y': 'Importance Score'}
                )
                st.plotly_chart(fig)

                # Correlation Analysis
                st.subheader("Channel Correlation Analysis")
                
                # Calculate correlations excluding Date column
                correlation_cols = self.data_processor.feature_columns + [self.data_processor.revenue_column]
                corr_matrix = df[correlation_cols].corr()
                
                # Create overall correlation heatmap using plotly
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=corr_matrix,
                    x=correlation_cols,
                    y=correlation_cols,
                    zmin=-1,
                    zmax=1,
                    colorscale=[
                        [0.0, 'rgb(244, 67, 54)'],    # Red for negative
                        [0.5, 'rgb(255, 255, 255)'],  # White for zero
                        [1.0, 'rgb(76, 175, 80)']     # Green for positive
                    ],
                    hoverongaps=False,
                    hovertemplate='%{x} → %{y}<br>Correlation: %{z:.3f}<extra></extra>'
                ))
                
                fig_heatmap.update_layout(
                    title=dict(
                        text="Overall Correlation Matrix",
                        font=dict(size=20),
                        x=0.5,
                        xanchor='center'
                    ),
                    width=700,
                    height=700,
                    xaxis=dict(tickangle=45),
                    yaxis=dict(tickangle=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                # First, create a container for the correlation matrix
                container = st.container()

                # Update the CSS for correlation matrix centering
                st.markdown(
                    """
                    <style>
                        [data-testid="stHorizontalBlock"] {
                            align-items: center;
                            justify-content: center;
                        }
                        .stPlotlyChart {
                            margin: 0 auto;
                        }
                        .plot-container {
                            display: flex;
                            justify-content: center;
                            width: 100%;
                        }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # Use columns to center the plot
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.plotly_chart(fig_heatmap, use_container_width=False)
                
                # Create individual channel correlation cards
                def create_correlation_card(channel, correlations):
                    # Drop self-correlation and handle Date column safely
                    correlations = correlations.copy()
                    if channel in correlations:
                        correlations = correlations.drop(channel)
                    if 'Date' in correlations:
                        correlations = correlations.drop('Date')
                    
                    # Define color scale
                    def get_correlation_color(value):
                        if value >= 0.7:
                            return "rgba(76, 175, 80, 0.9)"  # Strong positive (green)
                        elif value >= 0.4:
                            return "rgba(255, 193, 7, 0.9)"  # Moderate positive (amber)
                        elif value >= 0:
                            return "rgba(156, 39, 176, 0.7)"  # Weak positive (purple)
                        else:
                            return "rgba(244, 67, 54, 0.9)"  # Negative (red)
                    
                    # Define correlation strength text
                    def get_correlation_text(value):
                        if abs(value) >= 0.7:
                            return "Strong"
                        elif abs(value) >= 0.4:
                            return "Moderate"
                        else:
                            return "Weak"
                    
                    # Create card using plotly
                    fig = go.Figure()
                    
                    # Add bars
                    fig.add_trace(go.Bar(
                        x=correlations.index,
                        y=correlations.values,
                        marker_color=[get_correlation_color(v) for v in correlations.values],
                        hovertemplate="<br>".join([
                            "<b>%{x}</b>",
                            "Correlation: %{y:.2f}",
                            "Strength: " + "<br>".join([get_correlation_text(v) for v in correlations.values]),
                        ])
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=dict(
                            text=f"<b>{channel}</b> Correlations",
                            font=dict(size=16),
                            x=0.5,
                            xanchor='center'
                        ),
                        showlegend=False,
                        margin=dict(l=20, r=20, t=40, b=60),  # Increased bottom margin for labels
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        width=500,
                        height=400,  # Increased height
                        xaxis=dict(
                            showgrid=False,
                            tickangle=45,
                            title_text="",
                            tickfont=dict(size=12)
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(128,128,128,0.2)',
                            range=[-1, 1],
                            title_text="Correlation Strength",
                            tickformat='.2f',
                            tickfont=dict(size=12),
                            title_font=dict(size=14)
                        ),
                        hoverlabel=dict(
                            bgcolor="white",
                            font_size=14,
                            font_family="Arial"
                        )
                    )
                    
                    return fig
                
                # Add custom CSS
                st.markdown("""
                <style>
                    .correlation-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                        gap: 1rem;
                        padding: 1rem;
                    }
                    .correlation-card {
                        background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
                        border-radius: 10px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        transition: transform 0.2s, box-shadow 0.2s;
                        overflow: hidden;
                    }
                    .correlation-card:hover {
                        transform: translateY(-5px);
                        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
                    }
                </style>
                """, unsafe_allow_html=True)
                
                # Create grid layout
                st.markdown('<div class="correlation-grid">', unsafe_allow_html=True)
                
                # Create a card for each channel
                cols = st.columns(2)
                for idx, channel in enumerate(self.data_processor.feature_columns):
                    with cols[idx % 2]:
                        st.markdown('<div class="correlation-card">', unsafe_allow_html=True)
                        fig = create_correlation_card(channel, corr_matrix[channel])
                        st.plotly_chart(fig, use_container_width=False)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add correlation interpretation guide
                with st.expander("📊 Understanding Correlation Values"):
                    st.markdown("""
                    | Correlation Range | Strength | Interpretation |
                    |-------------------|-----------|----------------|
                    | 0.7 to 1.0 | Strong | Strong positive relationship |
                    | 0.4 to 0.7 | Moderate | Moderate positive relationship |
                    | 0.0 to 0.4 | Weak | Weak positive relationship |
                    | -1.0 to 0.0 | Negative | Negative relationship |
                    
                    - **Strong**: Changes in one variable strongly predict changes in another
                    - **Moderate**: There's a noticeable but not dominant relationship
                    - **Weak**: Variables have minimal influence on each other
                    - **Negative**: Variables move in opposite directions
                    """)
                
                # Revenue prediction section
                st.subheader("Revenue Predictor")
                st.write("Enter channel spend values to predict revenue:")
                
                # Create input fields for each channel
                input_values = {}
                cols = st.columns(3)
                for idx, channel in enumerate(self.data_processor.feature_columns):
                    with cols[idx % 3]:
                        input_values[channel] = st.number_input(
                            f"{channel} Spend ($)",
                            min_value=0.0,
                            value=float(df[channel].mean()),
                            step=100.0
                        )
                
                # Make prediction
                if st.button("Predict Revenue"):
                    input_df = pd.DataFrame([input_values])
                    predicted_revenue = self.model_trainer.predict(input_df)
                    st.success(f"Predicted Daily Revenue: ${predicted_revenue:,.2f}")
                
                # Display interpretability
                self.display_interpretability(metrics)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Please make sure your file is properly formatted and try again.")

    def display_interpretability(self, metrics):
        st.subheader("Model Interpretability")
        
        # Channel Contributions
        st.write("### Channel Contributions to Revenue")
        contributions = pd.DataFrame(metrics['channel_contributions']).T
        fig_contrib = px.bar(
            contributions,
            y='relative_contribution',
            title="Relative Channel Contributions",
            color='direction',
            labels={'relative_contribution': 'Contribution to Revenue (%)'}
        )
        st.plotly_chart(fig_contrib)
        
        # Marginal Effects
        st.write("### Spend Impact Analysis")
        for channel, effects in metrics['marginal_effects'].items():
            fig_effects = px.line(
                x=effects['spend_levels'],
                y=effects['revenue_impact'],
                title=f"{channel} Spend Impact on Revenue",
                labels={'x': 'Spend Level ($)', 'y': 'Predicted Revenue ($)'}
            )
            fig_effects.add_vline(
                x=effects['diminishing_returns_point'],
                line_dash="dash",
                annotation_text="Diminishing Returns Point"
            )
            st.plotly_chart(fig_effects)
            
        # Add recommendations
        st.write("### Optimization Recommendations")
        recommendations = self._generate_recommendations(metrics)
        for channel, rec in recommendations.items():
            st.info(f"**{channel}**: {rec}")

if __name__ == "__main__":
    app = StreamlitApp()
    app.run() 