# Generate sample data
import pandas as pd
import numpy as np
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_data(n_days=90):
    # Updated parameters with more realistic ROAS and seasonality
    channels = {
        'Google_Ads_Spend': {
            'base': 2000, 
            'std': 200, 
            'multiplier': 4.2,  # Higher ROAS for search
            'daily_pattern': [0.8, 1.0, 1.1, 1.2, 1.3, 0.9, 0.7]  # Mon-Sun
        },
        'Facebook_Spend': {
            'base': 1500, 
            'std': 150, 
            'multiplier': 3.1,
            'daily_pattern': [0.9, 1.0, 1.1, 1.2, 1.4, 1.3, 1.0]
        },
        'Instagram_Spend': {
            'base': 800, 
            'std': 100, 
            'multiplier': 3.5,  # Higher for social engagement
            'daily_pattern': [0.8, 0.9, 1.0, 1.2, 1.4, 1.5, 1.2]
        },
        'TikTok_Spend': {
            'base': 1200, 
            'std': 180, 
            'multiplier': 2.8,
            'daily_pattern': [0.7, 0.8, 1.0, 1.2, 1.4, 1.6, 1.3]
        },
        'YouTube_Spend': {
            'base': 1800, 
            'std': 250, 
            'multiplier': 2.5,
            'daily_pattern': [1.0, 1.0, 1.1, 1.1, 1.2, 1.3, 1.3]
        }
    }
    
    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    data = {'Date': dates}
    
    # Generate spend data with daily patterns
    for channel, params in channels.items():
        # Apply daily seasonality pattern
        daily_pattern = np.tile(params['daily_pattern'], (n_days // 7) + 1)[:n_days]
        
        # Generate base spend with controlled randomness
        base_spend = np.random.normal(params['base'], params['std'], n_days)
        
        # Apply daily pattern and ensure no negative values
        data[channel] = np.maximum(
            base_spend * daily_pattern,
            params['base'] * 0.4  # Minimum spend floor
        )
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate revenue with more complex relationships
    revenue = sum(
        df[channel] * params['multiplier'] * (1 + np.sin(np.pi * np.arange(n_days)/30)/10)  # Monthly cyclical effect
        for channel, params in channels.items()
    )
    
    # Add interaction effects with diminishing returns
    revenue += (
        np.sqrt(df['Facebook_Spend'] * df['Instagram_Spend']) * 0.08 +  # Social synergy
        np.sqrt(df['Google_Ads_Spend'] * df['YouTube_Spend']) * 0.06 +  # Google synergy
        np.sqrt(df['TikTok_Spend'] * df['Instagram_Spend']) * 0.04      # Video social synergy
    )
    
    # Add seasonal effects
    monthly_seasonality = 1 + 0.2 * np.sin(2 * np.pi * np.arange(n_days)/30)  # Monthly cycle
    weekly_pattern = np.tile([1.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.2], (n_days // 7) + 1)[:n_days]
    
    # Combine all effects
    revenue *= monthly_seasonality * weekly_pattern
    
    # Add controlled noise with higher variance on weekends
    weekend_mask = dates.dayofweek >= 5
    noise = np.where(
        weekend_mask,
        np.random.normal(1, 0.03, n_days),  # 3% noise on weekends
        np.random.normal(1, 0.02, n_days)   # 2% noise on weekdays
    )
    
    df['Revenue'] = revenue * noise
    
    # Print validation metrics
    print("\nData Validation:")
    print(f"Average Revenue: ${df['Revenue'].mean():,.2f}")
    print("\nChannel Performance:")
    for channel in channels.keys():
        avg_spend = df[channel].mean()
        avg_roas = df['Revenue'].mean() / avg_spend
        print(f"{channel}:")
        print(f"  Avg Daily Spend: ${avg_spend:,.2f}")
        print(f"  Effective ROAS: {avg_roas:.2f}")
        print(f"  Correlation with Revenue: {df[channel].corr(df['Revenue']):.3f}")
    
    return df

# Generate and save the data
df = generate_sample_data()
df.to_csv('media_mix_sample.csv', index=False)