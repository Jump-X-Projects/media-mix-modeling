import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(
    start_date: str = "2023-01-01",
    n_days: int = 365,
    channels: list = ["TV", "Radio", "Social", "Search"],
    seed: int = 42
) -> pd.DataFrame:
    """Generate sample media mix modeling data"""
    np.random.seed(seed)
    
    # Generate dates
    dates = [
        datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=x)
        for x in range(n_days)
    ]
    
    # Create base DataFrame
    data = pd.DataFrame({"Date": dates})
    
    # Generate spend data for each channel
    for channel in channels:
        # Base spend with some seasonality
        base = 1000 + 500 * np.sin(np.linspace(0, 4*np.pi, n_days))
        
        # Add random noise
        noise = np.random.normal(0, 100, n_days)
        
        # Add channel-specific patterns
        if channel == "TV":
            # TV has higher weekend spend
            weekend_boost = [200 if d.weekday() >= 5 else 0 for d in dates]
            spend = base + noise + weekend_boost
        elif channel == "Social":
            # Social has gradual increase over time
            trend = np.linspace(0, 500, n_days)
            spend = base + noise + trend
        elif channel == "Search":
            # Search has some correlation with TV
            tv_correlation = 0.3 * (base + noise)
            spend = base + noise + tv_correlation
        else:
            spend = base + noise
        
        data[f"{channel}_Spend"] = np.maximum(spend, 0)  # Ensure non-negative
    
    # Generate Revenue with lagged effects and diminishing returns
    revenue = np.zeros(n_days)
    
    for channel in channels:
        spend = data[f"{channel}_Spend"].values
        
        # Channel-specific parameters
        if channel == "TV":
            effect = 2.0
            lag = 3
        elif channel == "Social":
            effect = 1.5
            lag = 1
        elif channel == "Search":
            effect = 1.8
            lag = 0
        else:
            effect = 1.2
            lag = 2
        
        # Add lagged effect
        lagged_spend = np.pad(spend, (lag, 0))[:n_days]
        
        # Add diminishing returns using log transform
        revenue += effect * np.log1p(lagged_spend / 1000)
    
    # Scale revenue and add noise
    revenue = 10000 + 5000 * revenue
    revenue += np.random.normal(0, 1000, n_days)
    data["Revenue"] = np.maximum(revenue, 0)  # Ensure non-negative
    
    # Add control variables
    data["Weekend"] = [1 if d.weekday() >= 5 else 0 for d in dates]
    data["Month"] = [d.month for d in dates]
    data["DayOfWeek"] = [d.weekday() for d in dates]
    
    return data 