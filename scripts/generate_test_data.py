import numpy as np
import pandas as pd
import os

def generate_test_data(n_samples=100, seed=42):
    """Generate test data with known relationships and realistic values"""
    np.random.seed(seed)
    
    # Generate features with more realistic distributions
    # Using lognormal distribution to ensure positive values and right skew
    base_spend = {
        'TV_Spend': [1000000, 500000],     # TV: mean $1M, std $500k
        'Radio_Spend': [200000, 100000],    # Radio: mean $200k, std $100k
        'Social_Spend': [150000, 75000],    # Social: mean $150k, std $75k
        'Print_Spend': [100000, 50000]      # Print: mean $100k, std $50k
    }
    
    X = pd.DataFrame()
    for channel, (mean, std) in base_spend.items():
        # Convert mean and std to lognormal parameters
        mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
        sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
        X[channel] = np.random.lognormal(mu, sigma, n_samples)
    
    # Scale features to be in reasonable ranges
    # Generate target with known relationships (using scaled versions)
    scaled_X = X.copy()
    for col in X.columns:
        scaled_X[col] = X[col] / X[col].mean()  # Scale to relative changes
    
    y = (
        2.0 * scaled_X['TV_Spend'] +      # TV has strongest effect
        1.5 * scaled_X['Radio_Spend'] +   # Radio has second strongest
        0.5 * scaled_X['Social_Spend'] +  # Social has weakest effect
        0.8 * scaled_X['Print_Spend']     # Print has moderate effect
    )
    
    # Convert relative revenue changes to absolute values
    # Base revenue around $5M with noise
    base_revenue = 5000000
    revenue_std = 1000000
    X['Revenue'] = base_revenue * y + np.random.normal(0, revenue_std, n_samples)
    
    return X

def save_test_data(data, output_dir='data'):
    """Save test data to CSV and Excel formats"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'test_data.csv')
    data.to_csv(csv_path, index=False)
    print(f"Saved test data to CSV: {csv_path}")
    
    # Save to Excel
    excel_path = os.path.join(output_dir, 'test_data.xlsx')
    data.to_excel(excel_path, index=False)
    print(f"Saved test data to Excel: {excel_path}")

if __name__ == "__main__":
    # Generate data
    test_data = generate_test_data()
    
    # Add some metadata
    print("\nTest Data Summary:")
    print("-----------------")
    print(f"Number of samples: {len(test_data)}")
    print("\nFeature correlations with Revenue:")
    print(test_data.corr()['Revenue'].sort_values(ascending=False))
    print("\nChannel Spend Statistics:")
    print("------------------------")
    for channel in ['TV_Spend', 'Radio_Spend', 'Social_Spend', 'Print_Spend']:
        print(f"\n{channel}:")
        print(f"  Mean: ${test_data[channel].mean():,.2f}")
        print(f"  Median: ${test_data[channel].median():,.2f}")
        print(f"  Min: ${test_data[channel].min():,.2f}")
        print(f"  Max: ${test_data[channel].max():,.2f}")
    print("\nRevenue Statistics:")
    print("-----------------")
    print(f"Mean: ${test_data['Revenue'].mean():,.2f}")
    print(f"Median: ${test_data['Revenue'].median():,.2f}")
    print(f"Min: ${test_data['Revenue'].min():,.2f}")
    print(f"Max: ${test_data['Revenue'].max():,.2f}")
    
    # Save data
    save_test_data(test_data) 