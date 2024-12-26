import streamlit as st
import pandas as pd
import numpy as np
from data_preprocessing.data_preprocessor import DataPreprocessor
from data_preprocessing.date_transformer import DateTransformer
from data_preprocessing.data_setter import DataSetter
from model.mmm_model import MMMModel
from config.config import MEDIA_SPEND_COLUMNS, NON_MEDIA_COLUMNS, MEDIA_CHANNELS
from config.config import DATE_FORMAT, COSTS
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Custom CSS ---
st.markdown(
    """
<style>
/* Main background color */
.stApp {
    background-color: #f0f2f6;
}

/* Sidebar background color */
.css-1d391kg {
    background-color: #e0e2e6;
}

/* Header text */
h1, h2, h3 {
    color: #F63366; /* Your primary color */
    font-family: 'Arial';
}

/* Body text */
p, span, .st-eb, .st-en, .st-el, .st-eq {
    font-family: 'Arial';
}

/* Button styles */
.stButton>button {
    color: #ffffff;
    background-color: #F63366;
    border-radius: 8px;
    border: 2px solid #F63366;
    font-family: 'Arial';
}

.stButton>button:hover {
    background-color: #ffffff;
    color: #F63366;
    border: 2px solid #F63366;
}

/* Slider styles */
.stSlider>div>div>div>div {
    background-color: #F63366 !important; /* !important might be needed to override defaults */
}

/* Table styles */
.ReactTable {
  border-radius: 8px !important;
  overflow: hidden !important;
  border: 2px solid #F63366 !important;
}

.ReactTable .rt-thead {
  background-color: #F63366 !important;
  color: white !important;
  font-family: 'Arial';
}

.ReactTable .rt-tbody .rt-td {
  background-color: #f0f2f6 !important;
  border-bottom: 1px solid #e0e2e6 !important;
  font-family: 'Arial';
}

/* Input fields */
.stTextInput>div>div>input, .stNumberInput>div>div>input {
  border: 2px solid #F63366 !important;
  border-radius: 8px !important;
  font-family: 'Arial';
}

/* Selectbox */
.stSelectbox>div>div>div {
  border: 2px solid #F63366 !important;
  border-radius: 8px !important;
  font-family: 'Arial';
}

/* Radio buttons */
.stRadio>div>label>div>div {
  background-color: #F63366 !important;
  border: 2px solid #F63366 !important;
}

.stRadio>div>label>div {
  font-family: 'Arial';
}

</style>
""",
    unsafe_allow_html=True,
)


st.title("Media Mix Modeling")

# --- Sidebar for Data and Model Configuration ---
st.sidebar.header("Data Upload & Configuration")

# Allow users to upload their own data
uploaded_file = st.sidebar.file_uploader("Upload your CSV data", type=["csv"])

# Data and model configuration options
date_column = st.sidebar.selectbox("Select Date Column", ["Date"])
date_format = st.sidebar.selectbox("Select Date Format", list(DATE_FORMAT.keys()))
target_variable = st.sidebar.selectbox("Select Target Variable (Sales)", ["Sales"])

# --- Main Page Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Model Inputs")
    with st.expander("Media Channel Selection"):
        media_channels = st.multiselect(
            "Select Media Channels", MEDIA_SPEND_COLUMNS, default=MEDIA_CHANNELS
        )
    with st.expander("Model Parameters"):
        adstock_type = st.selectbox("Select Adstock Type", ["geometric", "delayed"])
        saturation_type = st.selectbox("Select Saturation Function", ["hill", "reach"])
        spend_variables = st.multiselect(
            "Select Spend Variables",
            media_channels,
            default=media_channels,
        )
        date_format_selected = DATE_FORMAT[date_format]
        scaler_type = st.selectbox("Select Scaler Type", ["MinMaxScaler", "MaxAbsScaler"])

        st.write("Hyperparameter Optimization")
        hyperparameter_tuning_df = pd.DataFrame(
            {
                "Channel": media_channels,
                "Alpha (Lag)": [0.5] * len(media_channels),
                "Beta (Saturation)": [0.5] * len(media_channels),
                "Gamma (Decay)": [0.5] * len(media_channels),
            }
        )
        edited_hyperparameter_df = st.data_editor(
            hyperparameter_tuning_df,
            column_config={
                "Channel": st.column_config.TextColumn(
                    "Channel",
                    help="Media Channel",
                    default="tv_S",
                    max_chars=50,
                    disabled=True,
                    required=True,
                ),
                "Alpha (Lag)": st.column_config.NumberColumn(
                    "Alpha (Lag)",
                    help="Lag effect for adstock transformation",
                    default=0.5,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    format="%.2f",
                ),
                "Beta (Saturation)": st.column_config.NumberColumn(
                    "Beta (Saturation)",
                    help="Saturation effect for diminishing returns",
                    default=0.5,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    format="%.2f",
                ),
                "Gamma (Decay)": st.column_config.NumberColumn(
                    "Gamma (Decay)",
                    help="Decay rate for adstock",
                    default=0.5,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    format="%.2f",
                ),
            },
            num_rows="dynamic",
            width=800,
        )

    with col2:
        st.header("Settings")
        with st.expander("Model Settings"):
            n_chains = st.slider("Number of Chains", min_value=1, max_value=5, value=2)
            n_draws = st.slider("Number of Samples", min_value=500, max_value=5000, value=1000)
            n_tune = st.slider("Number of Tuning Steps", min_value=500, max_value=5000, value=1000)

# Initialize data setter
data_setter = DataSetter(date_column, media_channels, target_variable, uploaded_file)
data_setter.set_data()
df = data_setter.get_data()

if df is not None:
    date_transformer = DateTransformer(date_column, date_format_selected)
    df = date_transformer.transform(df)

    data_preprocessor = DataPreprocessor(
        df,
        date_column,
        media_channels,
        NON_MEDIA_COLUMNS,
        target_variable,
        adstock_type,
        saturation_type,
        spend_variables,
        scaler_type,
        hyperparameters=edited_hyperparameter_df.set_index("Channel").to_dict(
            "index"
        ),
    )
    df = data_preprocessor.preprocess()

    # Model fitting
    mmm = MMMModel(
        df,
        target_variable,
        date_column,
        media_channels,
        NON_MEDIA_COLUMNS,
        n_draws,
        n_tune,
        n_chains,
        adstock_type,
        saturation_type,
        hyperparameters=edited_hyperparameter_df.set_index("Channel").to_dict(
            "index"
        ),
    )
    mmm.fit()

    # --- Plotting Functions ---
    def plot_actual_vs_predicted(df, y_true, y_pred):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add actual values trace
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[y_true],
                mode="lines",
                name="Actual",
                line=dict(color="#F63366"),
            ),
            secondary_y=False,
        )
        # Add predicted values trace
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[y_pred],
                mode="lines",
                name="Predicted",
                line=dict(color="#008080"),
            ),
            secondary_y=False,
        )
        # Customize layout
        fig.update_layout(
            title_text="Actual vs Predicted Values",
            title_x=0.5,
            xaxis_title="Date",
            yaxis_title="Value",
            font=dict(family="Arial", size=12, color="#262730"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="#ffffff",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        # Customize axes
        fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
        return fig

    def plot_channel_contribution(media_contribution_df, channel):
        fig = px.bar(
            media_contribution_df,
            x="date",
            y=channel,
            title=f"{channel} Contribution Over Time",
        )
        fig.update_layout(
            title_x=0.5,
            xaxis_title="Date",
            yaxis_title="Contribution",
            font=dict(family="Arial", size=12, color="#262730"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="#ffffff",
        )
        fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
        return fig

    def plot_response_curves(df, media_channels, target_variable, spend_variables):
        fig = go.Figure()
        for channel in media_channels:
            df_sorted = df.sort_values(by=channel)
            fig.add_trace(
                go.Scatter(
                    x=df_sorted[channel],
                    y=df_sorted[f"{channel}_{target_variable}_response_curve"],
                    mode="lines",
                    name=channel,
                )
            )
        fig.update_layout(
            title="Response Curves for Media Channels",
            xaxis_title="Media Spend",
            yaxis_title="Marginal Sales",
            font=dict(family="Arial", size=12, color="#262730"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="#ffffff",
        )
        fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
        return fig

    def plot_model_coefficients(mmm):
        # Extract coefficients (betas) and plot them
        coefficients = mmm.get_coefficients()
        fig = px.bar(
            coefficients,
            x=coefficients.index,
            y="Coefficient Value",
            title="Model Coefficients (Betas)",
        )
        fig.update_layout(
            title_x=0.5,
            xaxis_title="Variable",
            yaxis_title="Coefficient Value",
            font=dict(family="Arial", size=12, color="#262730"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="#ffffff",
        )
        fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
        return fig

    # --- Visualizations ---
    st.header("Model Results & Diagnostics")
    if mmm.trace is not None:
        # Model Coefficients
        st.plotly_chart(plot_model_coefficients(mmm))

        # Actual vs. Predicted
        st.plotly_chart(
            plot_actual_vs_predicted(
                mmm.df, mmm.target_variable, "target_prediction"
            )
        )

        # Media Channel Contributions
        media_contribution_df = mmm.get_media_contribution()
        for channel in media_channels:
            st.plotly_chart(plot_channel_contribution(media_contribution_df, channel))

        # Response Curves
        st.plotly_chart(
            plot_response_curves(
                mmm.df, media_channels, target_variable, spend_variables
            )
        )

        # Optimization (if implemented in MMMModel)
        if hasattr(mmm, "optimize_budget"):
            st.header("Budget Optimization")
            total_budget = st.number_input(
                "Enter Total Budget", min_value=0.0, value=100000.0
            )
            if st.button("Optimize"):
                optimal_allocations = mmm.optimize_budget(total_budget, spend_variables)
                st.write(optimal_allocations)