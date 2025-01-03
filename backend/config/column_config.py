"""Configuration for data validation and column mapping"""

# Column naming patterns for different types of data
COLUMN_PATTERNS = {
    'spend': [
        'spend',
        'cost',
        'investment',
        'budget',
        'expense',
        'media',
        'advertising',
        'ad_spend'
    ],
    'revenue': [
        'revenue',
        'sales',
        'conversion',
        'income',
        'earnings',
        'return',
        'value'
    ],
    'date': [
        'date',
        'day',
        'timestamp',
        'time',
        'period'
    ]
}

# Data validation rules
VALIDATION_RULES = {
    'min_rows': 10,  # Minimum number of rows required
    'required_columns': ['date', 'spend', 'revenue'],  # At least one of each type required
    'numeric_validation': {
        'allow_negative': False,
        'allow_zero': True,
        'min_unique_values': 3
    },
    'date_validation': {
        'allow_gaps': True,  # Whether to allow gaps in date sequence
        'min_date_range': 7,  # Minimum number of days required
        'default_format': '%Y-%m-%d'
    }
}

# Column type requirements
COLUMN_TYPES = {
    'spend': 'numeric',
    'revenue': 'numeric',
    'date': 'datetime'
}

# Default date formats to try
DATE_FORMATS = [
    '%Y-%m-%d',
    '%m/%d/%Y',
    '%d/%m/%Y',
    '%Y/%m/%d',
    '%m-%d-%Y',
    '%d-%m-%Y',
    '%Y%m%d'
] 