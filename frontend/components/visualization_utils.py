import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

def create_scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    add_trendline: bool = False,
    add_diagonal: bool = False
) -> go.Figure:
    """Create a scatter plot with optional trendline and diagonal line."""
    fig = px.scatter(
        x=x,
        y=y,
        labels={'x': x_label, 'y': y_label},
        title=title
    )
    
    if add_trendline:
        fig.add_traces(
            px.scatter(x=x, y=y, trendline="ols").data[1]
        )
    
    if add_diagonal:
        fig.add_shape(
            type='line',
            x0=min(x),
            y0=min(x),
            x1=max(x),
            y1=max(x),
            line=dict(color='red', dash='dash')
        )
    
    return fig

def create_bar_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    orientation: str = 'v'
) -> go.Figure:
    """Create a bar plot with specified orientation."""
    fig = px.bar(
        data,
        x=x_col if orientation == 'v' else y_col,
        y=y_col if orientation == 'v' else x_col,
        title=title,
        orientation=orientation
    )
    return fig

def create_line_plot(
    data: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    title: str,
    uncertainty_bands: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
) -> go.Figure:
    """Create a line plot with optional uncertainty bands."""
    fig = go.Figure()
    
    for y_col in y_cols:
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[y_col],
            name=y_col,
            mode='lines'
        ))
        
        if uncertainty_bands and y_col in uncertainty_bands:
            lower, upper = uncertainty_bands[y_col]
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=upper,
                name=f'{y_col} Upper CI',
                mode='lines',
                line=dict(dash='dash'),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=lower,
                name=f'{y_col} Lower CI',
                mode='lines',
                line=dict(dash='dash'),
                fill='tonexty',
                showlegend=False
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title="Value"
    )
    return fig

def create_heatmap(
    data: pd.DataFrame,
    title: str,
    colorscale: str = 'RdBu',
    zmin: float = -1,
    zmax: float = 1
) -> go.Figure:
    """Create a heatmap visualization."""
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.columns,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax
    ))
    
    fig.update_layout(
        title=title,
        height=600
    )
    return fig

def create_histogram(
    data: np.ndarray,
    title: str,
    x_label: str,
    add_vline: bool = False,
    vline_value: float = 0
) -> go.Figure:
    """Create a histogram with optional vertical line."""
    fig = px.histogram(
        data,
        title=title,
        labels={'value': x_label}
    )
    
    if add_vline:
        fig.add_vline(
            x=vline_value,
            line_dash="dash",
            line_color="red"
        )
    
    return fig

def create_roi_plot(
    roi_data: Dict[str, Dict[str, float]],
    title: str
) -> go.Figure:
    """Create a ROI visualization with uncertainty bands."""
    channels = list(roi_data.keys())
    roi_values = [d['roi'] for d in roi_data.values()]
    
    fig = go.Figure([
        go.Bar(
            name='ROI',
            x=channels,
            y=roi_values
        )
    ])
    
    # Add uncertainty bands if available
    if all('lower' in d and 'upper' in d for d in roi_data.values()):
        lower_values = [d['lower'] for d in roi_data.values()]
        upper_values = [d['upper'] for d in roi_data.values()]
        
        fig.add_trace(go.Scatter(
            name='Upper CI',
            x=channels,
            y=upper_values,
            mode='lines',
            line=dict(dash='dash')
        ))
        fig.add_trace(go.Scatter(
            name='Lower CI',
            x=channels,
            y=lower_values,
            mode='lines',
            line=dict(dash='dash'),
            fill='tonexty'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Channel",
        yaxis_title="ROI (Return per $ spent)"
    )
    return fig 