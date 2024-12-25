import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from scipy import stats
import warnings

class ModelQualityMarkers:
    def __init__(self, model, X_train, X_test, y_train, y_test, feature_names):
        """
        Initialize the ModelQualityMarkers class with model and data
        
        Parameters:
        -----------
        model : trained model object
            The trained model to evaluate
        X_train : array-like
            Training features
        X_test : array-like
            Test features
        y_train : array-like
            Training target values
        y_test : array-like
            Test target values
        feature_names : list
            List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        
        # Get predictions
        self.y_train_pred = model.predict(X_train)
        self.y_test_pred = model.predict(X_test)
        
        # Store number of features and samples
        self.n_features = X_train.shape[1]
        self.n_samples = X_train.shape[0]
    
    def calculate_r2(self):
        """Calculate R-squared for both training and test sets"""
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        test_r2 = r2_score(self.y_test, self.y_test_pred)
        
        assessment = "Excellent" if test_r2 > 0.8 else "Good" if test_r2 > 0.7 else "Poor"
        
        return {
            'Metric': 'R-squared',
            'Train Value': f"{train_r2:.3f}",
            'Test Value': f"{test_r2:.3f}",
            'Benchmark': "> 0.7",
            'Assessment': assessment
        }
    
    def calculate_adjusted_r2(self):
        """Calculate adjusted R-squared for both training and test sets"""
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        test_r2 = r2_score(self.y_test, self.y_test_pred)
        
        # Calculate adjusted R-squared
        train_adj_r2 = 1 - (1 - train_r2) * (self.n_samples - 1) / (self.n_samples - self.n_features - 1)
        test_adj_r2 = 1 - (1 - test_r2) * (len(self.y_test) - 1) / (len(self.y_test) - self.n_features - 1)
        
        assessment = "Excellent" if test_adj_r2 > 0.75 else "Good" if test_adj_r2 > 0.65 else "Poor"
        
        return {
            'Metric': 'Adjusted R-squared',
            'Train Value': f"{train_adj_r2:.3f}",
            'Test Value': f"{test_adj_r2:.3f}",
            'Benchmark': "> 0.65",
            'Assessment': assessment
        }
    
    def calculate_rmse(self):
        """Calculate RMSE and compare to average revenue"""
        train_rmse = np.sqrt(mean_squared_error(self.y_train, self.y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, self.y_test_pred))
        
        avg_revenue = np.mean(self.y_train)
        rmse_percentage = (test_rmse / avg_revenue) * 100
        
        assessment = "Excellent" if rmse_percentage < 10 else "Good" if rmse_percentage < 15 else "Poor"
        
        return {
            'Metric': 'RMSE',
            'Train Value': f"${train_rmse:,.2f}",
            'Test Value': f"${test_rmse:,.2f}",
            'Benchmark': "< 15% of avg revenue",
            'Assessment': assessment,
            'Additional Info': f"RMSE is {rmse_percentage:.1f}% of average revenue"
        }
    
    def calculate_mape(self):
        """Calculate MAPE for both training and test sets"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_mape = mean_absolute_percentage_error(self.y_train, self.y_train_pred) * 100
            test_mape = mean_absolute_percentage_error(self.y_test, self.y_test_pred) * 100
        
        assessment = "Excellent" if test_mape < 10 else "Good" if test_mape < 20 else "Poor"
        
        return {
            'Metric': 'MAPE',
            'Train Value': f"{train_mape:.1f}%",
            'Test Value': f"{test_mape:.1f}%",
            'Benchmark': "< 20%",
            'Assessment': assessment
        }
    
    def calculate_feature_significance(self):
        """Calculate p-values for feature significance using bootstrapping"""
        n_iterations = 1000
        n_samples = len(self.X_train)
        coefficients = []
        
        for _ in range(n_iterations):
            # Bootstrap sample
            indices = np.random.randint(0, n_samples, n_samples)
            sample_X = self.X_train[indices]
            sample_y = self.y_train.iloc[indices]
            
            # Fit model
            sample_model = type(self.model)()
            sample_model.fit(sample_X, sample_y)
            
            # Get feature importances
            if hasattr(sample_model, 'feature_importances_'):
                coefficients.append(sample_model.feature_importances_)
            else:
                coefficients.append(sample_model.coef_)
        
        coefficients = np.array(coefficients)
        
        # Calculate p-values using t-test
        p_values = []
        for i in range(coefficients.shape[1]):
            t_stat, p_val = stats.ttest_1samp(coefficients[:, i], 0)
            p_values.append(p_val)
        
        feature_significance = []
        for feature, p_val in zip(self.feature_names, p_values):
            assessment = "Significant" if p_val < 0.05 else "Not Significant"
            feature_significance.append({
                'Feature': feature,
                'p-value': f"{p_val:.4f}",
                'Benchmark': "< 0.05",
                'Assessment': assessment
            })
        
        return feature_significance
    
    def get_all_metrics(self):
        """Calculate and return all quality metrics"""
        metrics = []
        
        # Basic metrics
        metrics.append(self.calculate_r2())
        metrics.append(self.calculate_adjusted_r2())
        metrics.append(self.calculate_rmse())
        metrics.append(self.calculate_mape())
        
        # Feature significance
        feature_significance = self.calculate_feature_significance()
        
        return {
            'basic_metrics': pd.DataFrame(metrics),
            'feature_significance': pd.DataFrame(feature_significance)
        }
    
    def get_model_assessment(self):
        """Provide overall model assessment and recommendations"""
        metrics = self.get_all_metrics()
        basic_metrics = metrics['basic_metrics']
        feature_significance = metrics['feature_significance']
        
        # Count poor performances
        poor_count = sum(1 for x in basic_metrics['Assessment'] if x == 'Poor')
        insignificant_features = sum(1 for x in feature_significance['Assessment'] if x == 'Not Significant')
        
        assessment = []
        
        if poor_count == 0 and insignificant_features == 0:
            assessment.append("✅ Model Performance: Excellent")
            assessment.append("The model shows strong predictive power across all metrics.")
        elif poor_count <= 1 and insignificant_features <= 1:
            assessment.append("✅ Model Performance: Good")
            assessment.append("The model performs well but has room for improvement.")
        else:
            assessment.append("⚠️ Model Performance: Needs Improvement")
            assessment.append("Several metrics indicate suboptimal model performance.")
        
        # Add specific recommendations
        recommendations = []
        if poor_count > 0:
            recommendations.append("- Consider collecting more training data")
            recommendations.append("- Try feature engineering or selection")
            recommendations.append("- Experiment with different model parameters")
        
        if insignificant_features > 0:
            recommendations.append("- Review and possibly remove insignificant features")
            recommendations.append("- Investigate collinearity between features")
        
        return {
            'assessment': assessment,
            'recommendations': recommendations if recommendations else ["No specific recommendations needed."]
        } 