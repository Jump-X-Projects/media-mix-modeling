# data_preprocessing/data_preprocessor.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import numpy as np


class DataPreprocessor:
    def __init__(
        self,
        df,
        date_column,
        media_channels,
        non_media_cols,
        target_variable,
        adstock_type,
        saturation_type,
        spend_variables,
        scaler_type="MinMaxScaler",
        hyperparameters=None,
    ):
        self.df = df.copy()
        self.date_column = date_column
        self.media_channels = media_channels
        self.non_media_cols = non_media_cols
        self.organic_channels = list(set(media_channels) - set(spend_variables))
        self.target_variable = target_variable
        self.adstock_type = adstock_type
        self.saturation_type = saturation_type
        self.spend