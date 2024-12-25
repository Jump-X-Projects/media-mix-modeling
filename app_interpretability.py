import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from quality_markers import ModelQualityMarkers

# Define channel-specific decay rates
channel_decay_rates = {
    'Google_Ads_Spend': 0.3,
    'Facebook_Spend': 0.5,
    'Instagram_Spend': 0.5,
    'TikTok_Spend': 0.4,
    'YouTube_Spend': 0.6,
    'default': 0.5  # Default decay rate for undefined channels
}

def apply_adstock(x, decay_rate=None, max_lag=4):
    """
    Apply adstock transformation with channel-specific decay rates
    """
    channel_decay_rates = {
        'Google_Ads_Spend': 0.3,
        'Facebook_Spend': 0.5,
        'Instagram_Spend': 0.5,
        'TikTok_Spend': 0.4,
        'YouTube_Spend': 0.6
    }
    
    if decay_rate is None:
        decay_rate = 0.5
    
    x = np.array(x)
    x_decayed = np.zeros_like(x, dtype=float)
    
    for t in range(len(x)):
        for lag in range(min(t + 1, max_lag)):
            x_decayed[t] += x[t - lag] * (decay_rate ** lag)
    
    return x_decayed

def apply_diminishing_returns(x, alpha=None, channel=None):
    """
    Apply sophisticated S-curve transformation for better handling of extreme values
    """
    channel_params = {
        'Google_Ads_Spend': {'alpha': 0.7, 'inflection': 0.6, 'saturation': 2.5},
        'Facebook_Spend': {'alpha': 0.5, 'inflection': 0.5, 'saturation': 2.0},
        'Instagram_Spend': {'alpha': 0.5, 'inflection': 0.5, 'saturation': 2.0},
        'TikTok_Spend': {'alpha': 0.6, 'inflection': 0.4, 'saturation': 1.8},
        'YouTube_Spend': {'alpha': 0.4, 'inflection': 0.7, 'saturation': 3.0}
    }
    
    if channel and channel in channel_params:
        params = channel_params[channel]
    else:
        params = {'alpha': 0.5, 'inflection': 0.5, 'saturation': 2.0}
    
    def s_curve(x, alpha, inflection, saturation):
        # Normalized sigmoid with adjustable steepness and saturation
        x = np.array(x)
        # Handle negative or zero values
        x = np.maximum(x, 1e-10)
        
        # Normalize based on inflection point
        x_norm = x / (inflection * x.mean() if x.mean() > 0 else 1)
        
        # S-curve transformation with adjustable saturation
        return saturation * (1 / (1 + np.exp(-alpha * np.log(x_norm))))
    
    return s_curve(x, params['alpha'], params['inflection'], params['saturation'])

def get_channel_ranges(data, feature_cols):
    """
    Calculate channel-specific ranges and parameters
    """
    ranges = {}
    for col in feature_cols:
        min_val = data[col].min()
        max_val = data[col].max()
        mean_val = data[col].mean()
        std_val = data[col].std()
        
        # Dynamic range calculation
        lower_bound = max(0, mean_val - 3*std_val)
        upper_bound = mean_val + 3*std_val
        
        # Extrapolation factors
        below_min_factor = 0.5  # Steeper decline below min
        above_max_factor = 1.5  # More gradual increase above max
        
        ranges[col] = {
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'below_min_factor': below_min_factor,
            'above_max_factor': above_max_factor
        }
    return ranges

st.set_page_config(page_title="Media Mix Model Insights", layout="wide")

st.title("Media Mix Model Insights Dashboard")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        
        if 'Revenue' in data.columns:
            target_col = 'Revenue'
        else:
            st.warning("No 'Revenue' column found. Please select your target variable.")
            target_col = st.selectbox("Select the target variable", data.columns)
        
        feature_cols = [col for col in data.columns if col not in [target_col, 'Date']]
        
        if not feature_cols:
            st.error("No feature columns found in the dataset.")
            st.stop()
        
        X_transformed = pd.DataFrame()
        channel_ranges = get_channel_ranges(data, feature_cols)
        
        for col in feature_cols:
            decay_rate = channel_decay_rates.get(col, channel_decay_rates['default'])
            try:
                adstocked = apply_adstock(data[col].values, decay_rate=decay_rate)
                transformed = apply_diminishing_returns(adstocked, channel=col)
                X_transformed[col] = transformed
            except Exception as e:
                st.error(f"Error transforming {col}: {str(e)}")
                st.stop()
        
        X = X_transformed
        y = data[target_col]
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            with st.spinner("Training model... This may take a moment."):
                model = xgb.XGBRegressor(
                    n_estimators=5000,
                    learning_rate=0.05,
                    max_depth=5,
                    min_child_weight=2,
                    gamma=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    early_stopping_rounds=50
                )
                
                model.fit(
                    X_train_scaled, 
                    y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    verbose=False
                )
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")
            st.stop()
            
        # Use best iteration for predictions
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)
        test_pred = model.predict(X_test_scaled)
        
        # Model Performance
        st.header("Model Performance")
        
        train_score = model.score(X_train_scaled, y_train)
        val_score = model.score(X_val_scaled, y_val)
        test_score = model.score(X_test_scaled, y_test)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training R² Score", f"{train_score:.3f}")
        with col2:
            st.metric("Validation R² Score", f"{val_score:.3f}")
        with col3:
            st.metric("Testing R² Score", f"{test_score:.3f}")
        
        # Actual vs Predicted Plot
        st.subheader("Actual vs Predicted Revenue")
        fig_pred = go.Figure()
        
        fig_pred.add_trace(go.Scatter(
            x=y_train,
            y=train_pred,
            mode='markers',
            name='Training Data',
            marker=dict(color='blue', size=8, opacity=0.6)
        ))
        
        fig_pred.add_trace(go.Scatter(
            x=y_test,
            y=test_pred,
            mode='markers',
            name='Testing Data',
            marker=dict(color='red', size=8, opacity=0.6)
        ))
        
        max_val = max(max(y_train), max(y_test))
        min_val = min(min(y_train), min(y_test))
        fig_pred.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='black', dash='dash')
        ))
        
        fig_pred.update_layout(
            title="Actual vs Predicted Revenue",
            xaxis_title="Actual Revenue",
            yaxis_title="Predicted Revenue",
            height=500
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Feature Importance with Confidence Intervals
        st.header("Feature Importance Analysis")
        
        n_iterations = 100
        n_samples = len(X_train)
        importance_scores = []
        
        for _ in range(n_iterations):
            indices = np.random.randint(0, n_samples, n_samples)
            sample_X = X_train_scaled[indices]
            sample_y = y_train.iloc[indices]
            
            sample_model = xgb.XGBRegressor(random_state=42)
            sample_model.fit(sample_X, sample_y)
            importance_scores.append(sample_model.feature_importances_)
        
        importance_df = pd.DataFrame(importance_scores, columns=feature_cols)
        
        importance_stats = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance_df.mean(),
            'Lower': importance_df.quantile(0.025),
            'Upper': importance_df.quantile(0.975)
        }).sort_values('Importance', ascending=True)
        
        fig_importance = go.Figure()
        
        fig_importance.add_trace(go.Bar(
            x=importance_stats['Importance'],
            y=importance_stats['Feature'],
            orientation='h',
            error_x=dict(
                type='data',
                symmetric=False,
                array=importance_stats['Upper'] - importance_stats['Importance'],
                arrayminus=importance_stats['Importance'] - importance_stats['Lower']
            )
        ))
        
        fig_importance.update_layout(
            title="Feature Importance with 95% Confidence Intervals",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=400 + len(feature_cols) * 20
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Revenue Prediction Section
        st.header("Revenue Prediction")
        
        # Add prediction range information
        st.subheader("Reliable Prediction Ranges")
        st.write("The model is most reliable when predicting within these ranges:")
        
        range_data = []
        for feature in feature_cols:
            min_val = data[feature].min()
            max_val = data[feature].max()
            mean_val = data[feature].mean()
            std_val = data[feature].std()
            
            # Calculate reasonable ranges (within training data distribution)
            conservative_min = max(min_val, mean_val - 2*std_val)
            conservative_max = min(max_val, mean_val + 2*std_val)
            
            range_data.append({
                'Channel': feature,
                'Min Spend': f"${min_val:,.2f}",
                'Max Spend': f"${max_val:,.2f}",
                'Conservative Min': f"${conservative_min:,.2f}",
                'Conservative Max': f"${conservative_max:,.2f}",
                'Mean Spend': f"${mean_val:,.2f}"
            })
        
        range_df = pd.DataFrame(range_data)
        st.dataframe(range_df)
        
        st.write("""
        - **Min/Max Spend**: Absolute bounds from training data
        - **Conservative Min/Max**: Recommended range (within 2 standard deviations)
        - **Mean Spend**: Average spend level in training data
        
        ⚠️ Note: Predictions become less reliable:
        - Near the minimum spend (due to log transformation sensitivity)
        - Beyond the maximum spend (due to extrapolation)
        - Far from the mean (due to fewer training examples)
        """)
        
        st.write("Enter media spend values to predict revenue:")
        
        num_cols = 3
        cols = st.columns(num_cols)
        input_values = {}
        
        for i, feature in enumerate(feature_cols):
            col_idx = i % num_cols
            with cols[col_idx]:
                mean_val = data[feature].mean()
                input_values[feature] = st.number_input(
                    f"{feature}",
                    min_value=0.0,
                    value=float(mean_val),
                    format="%.2f",
                    help=f"Average value: {mean_val:.2f}"
                )
        
        if st.button("Predict Revenue"):
            input_df = pd.DataFrame([input_values])
            
            input_transformed = pd.DataFrame()
            for col in feature_cols:
                transformed = apply_diminishing_returns(np.array([input_df[col].iloc[0]]))
                input_transformed[col] = transformed
            
            input_scaled = scaler.transform(input_transformed)
            prediction = model.predict(input_scaled)[0]
            
            st.subheader("Predicted Revenue")
            
            predictions = []
            for _ in range(100):
                noise = np.random.normal(0, 0.1, input_scaled.shape)
                noisy_input = input_scaled + noise
                pred = model.predict(noisy_input)[0]
                predictions.append(pred)
            
            lower_bound = np.percentile(predictions, 2.5)
            upper_bound = np.percentile(predictions, 97.5)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Lower Bound (95% CI)", f"${lower_bound:,.2f}")
            with col2:
                st.metric("Predicted Revenue", f"${prediction:,.2f}")
            with col3:
                st.metric("Upper Bound (95% CI)", f"${upper_bound:,.2f}")
            
            st.subheader("Channel Contributions")
            
            contributions = []
            base_pred = model.predict(np.zeros_like(input_scaled))[0]
            
            for i, feature in enumerate(feature_cols):
                temp_input = np.zeros_like(input_scaled)
                temp_input[0, i] = input_scaled[0, i]
                feature_contribution = model.predict(temp_input)[0] - base_pred
                contributions.append({
                    'Channel': feature,
                    'Contribution': feature_contribution,
                    'Absolute Contribution': abs(feature_contribution)
                })
            
            contrib_df = pd.DataFrame(contributions)
            contrib_df = contrib_df.sort_values('Absolute Contribution', ascending=True)
            
            fig_contrib = go.Figure(go.Bar(
                x=contrib_df['Contribution'],
                y=contrib_df['Channel'],
                orientation='h'
            ))
            
            fig_contrib.update_layout(
                title="Channel Contributions to Predicted Revenue",
                xaxis_title="Contribution ($)",
                yaxis_title="Channel",
                height=400 + len(feature_cols) * 20
            )
            
            st.plotly_chart(fig_contrib, use_container_width=True)
        
        # After model training and predictions, add quality markers section
        st.header("Model Quality Assessment")
        
        quality_markers = ModelQualityMarkers(
            model=model,
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_cols
        )
        
        metrics = quality_markers.get_all_metrics()
        assessment = quality_markers.get_model_assessment()
        
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
        
        styled_metrics = metrics['basic_metrics'].style.apply(style_metric_row, axis=1)
        st.dataframe(styled_metrics)
        
        # Display feature significance
        st.subheader("Feature Significance Analysis")
        
        def style_significance_row(row):
            color_map = {
                'Significant': 'background-color: #90EE90',
                'Not Significant': 'background-color: #FFB6C1'
            }
            return [''] * len(row) if row.name != 'Assessment' else [color_map.get(x, '') for x in row]
        
        styled_significance = metrics['feature_significance'].style.apply(style_significance_row, axis=1)
        st.dataframe(styled_significance)
        
        # Display overall assessment
        st.subheader("Overall Model Assessment")
        for line in assessment['assessment']:
            st.write(line)
        
        st.subheader("Recommendations")
        for rec in assessment['recommendations']:
            st.write(rec)
        
        # Add explanatory text
        with st.expander("📚 Understanding Model Quality Metrics"):
            st.write("""
            **R-squared (R²)**: Measures how well the model explains the variance in the target variable.
            - Greater than 0.8: Excellent
            - Greater than 0.7: Good
            - Less than 0.7: Poor
            
            **Adjusted R-squared**: Similar to R², but penalizes adding predictors that don't help the model.
            - Greater than 0.75: Excellent
            - Greater than 0.65: Good
            - Less than 0.65: Poor
            
            **RMSE (Root Mean Square Error)**: Average prediction error in the same units as revenue.
            - Less than 10% of avg revenue: Excellent
            - Less than 15% of avg revenue: Good
            - Greater than 15% of avg revenue: Poor
            
            **MAPE (Mean Absolute Percentage Error)**: Average percentage error of predictions.
            - Less than 10%: Excellent
            - Less than 20%: Good
            - Greater than 20%: Poor
            
            **Feature Significance (p-value)**:
            - Less than 0.05: Feature has significant impact
            - Greater than 0.05: Feature might not be useful
            """) 
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please make sure your file is properly formatted and try again.") 