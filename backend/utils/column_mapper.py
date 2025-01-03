"""Utility for mapping and validating data columns"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import re
from datetime import datetime
from backend.config.column_config import (
    COLUMN_PATTERNS,
    VALIDATION_RULES,
    COLUMN_TYPES,
    DATE_FORMATS
)

class ColumnMapper:
    """Utility class for identifying and mapping column names"""
    
    def __init__(self):
        self.spend_patterns = ['spend', 'cost', 'investment', 'budget']
        self.revenue_patterns = ['revenue', 'sales', 'income', 'earnings']
        
    def identify_spend_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Identify columns that represent media spend
        
        Args:
            data: Input DataFrame
            
        Returns:
            List of column names identified as spend columns
        """
        spend_cols = []
        for pattern in self.spend_patterns:
            matches = [col for col in data.columns if pattern in col.lower()]
            spend_cols.extend(matches)
        return list(set(spend_cols))  # Remove duplicates
    
    def identify_revenue_column(self, data: pd.DataFrame) -> Optional[str]:
        """
        Identify the revenue column
        
        Args:
            data: Input DataFrame
            
        Returns:
            Name of identified revenue column or None if not found
        """
        for pattern in self.revenue_patterns:
            matches = [col for col in data.columns if pattern in col.lower()]
            if matches:
                return matches[0]
        return None
    
    def validate_column_names(self, data: pd.DataFrame) -> List[str]:
        """
        Validate column names and return any issues
        
        Args:
            data: Input DataFrame
            
        Returns:
            List of validation messages (empty if no issues)
        """
        messages = []
        
        # Check for spend columns
        spend_cols = self.identify_spend_columns(data)
        if not spend_cols:
            messages.append(
                "No spend columns identified. Expected columns containing: " + 
                ", ".join(self.spend_patterns)
            )
        
        # Check for revenue column
        revenue_col = self.identify_revenue_column(data)
        if not revenue_col:
            messages.append(
                "No revenue column identified. Expected column containing: " + 
                ", ".join(self.revenue_patterns)
            )
        
        # Check for date column
        if 'date' not in data.columns:
            messages.append("Missing required date column")
        
        return messages
    
    def suggest_mappings(self, data: pd.DataFrame) -> dict:
        """
        Suggest column mappings based on patterns
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary of suggested mappings
        """
        return {
            'spend_columns': self.identify_spend_columns(data),
            'revenue_column': self.identify_revenue_column(data),
            'date_column': 'date' if 'date' in data.columns else None
        } 