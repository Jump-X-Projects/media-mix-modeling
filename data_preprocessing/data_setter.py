# data_preprocessing/data_setter.py
import pandas as pd

class DataSetter:
    def __init__(self, date_column, media_channels, target_variable, uploaded_file=None):
        self.date_column = date_column
        self.media_channels = media_channels
        self.target_variable = target_variable
        self.uploaded_file = uploaded_file
        self.data = None

    def set_data(self):
        """Sets the data based on whether a user has uploaded a file or not."""
        if self.uploaded_file is not None:
            try:
                self.data = pd.read_csv(self.uploaded_file)
            except Exception as e:
                print(f"Error reading uploaded file: {e}")
                self.data = None
        else:
            # Default data if no file is uploaded - make sure this path is correct
            try:
                self.data = pd.read_csv("data/simulated_mmm.csv")
            except FileNotFoundError:
                print(
                    "Error: Default data file 'data/simulated_mmm.csv' not found. "
                    "Please upload a file or provide a valid default dataset."
                )
                self.data = None

    def get_data(self):
        """Returns the data."""
        return self.data