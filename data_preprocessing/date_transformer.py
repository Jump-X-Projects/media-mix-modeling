# data_preprocessing/date_transformer.py
import pandas as pd

class DateTransformer:
    def __init__(self, date_column, date_format):
        self.date_column = date_column
        self.date_format = date_format

    def transform(self, df):
        """Transforms the date column to the specified format."""
        try:
            df[self.date_column] = pd.to_datetime(
                df[self.date_column], format=self.date_format
            )
        except ValueError:
            print(
                f"Warning: Could not parse date column '{self.date_column}' with format '{self.date_format}'. "
                "Trying to infer format automatically."
            )
            try:
                df[self.date_column] = pd.to_datetime(df[self.date_column])
            except ValueError:
                print("Error: Could not parse date format automatically either.")
                return df
        return df