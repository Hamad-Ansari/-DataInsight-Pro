# src/time_series.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional, Tuple
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import io
import base64
import streamlit as st

class TimeSeriesAnalyzer:
    """Time series analysis utilities."""
    
    @staticmethod
    def analyze_stationarity(series: pd.Series, series_name: str = "Series") -> Dict:
        """Perform stationarity tests."""
        results = {}
        
        # ADF Test
        adf_result = adfuller(series.dropna())
        results['adf'] = {
            'statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05
        }
        
        # KPSS Test
        try:
            kpss_result = kpss(series.dropna(), regression='c')
            results['kpss'] = {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > 0.05
            }
        except:
            results['kpss'] = {'error': 'Test failed'}
        
        return results
    
    @staticmethod
    def create_decomposition_plot(series: pd.Series, period: int = 30, 
                                 model: str = 'additive') -> go.Figure:
        """Create seasonal decomposition plot."""
        try:
            decomposition = seasonal_decompose(series.dropna(), model=model, period=period)
            
            fig = go.Figure()
            
            # Original
            fig.add_trace(go.Scatter(
                x=decomposition.observed.index,
                y=decomposition.observed.values,
                mode='lines',
                name='Observed',
                line=dict(color='blue')
            ))
            
            # Trend
            fig.add_trace(go.Scatter(
                x=decomposition.trend.index,
                y=decomposition.trend.values,
                mode='lines',
                name='Trend',
                line=dict(color='red')
            ))
            
            # Seasonal
            fig.add_trace(go.Scatter(
                x=decomposition.seasonal.index,
                y=decomposition.seasonal.values,
                mode='lines',
                name='Seasonal',
                line=dict(color='green')
            ))
            
            # Residual
            fig.add_trace(go.Scatter(
                x=decomposition.resid.index,
                y=decomposition.resid.values,
                mode='lines',
                name='Residual',
                line=dict(color='orange')
            ))
            
            fig.update_layout(
                title='Seasonal Decomposition',
                xaxis_title='Date',
                yaxis_title='Value',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=600
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Decomposition failed: {str(e)}")
            return fig
    
    @staticmethod
    def create_acf_pacf_plots(series: pd.Series, lags: int = 40) -> Tuple[go.Figure, go.Figure]:
        """Create ACF and PACF plots."""
        # ACF Plot
        acf_values = acf(series.dropna(), nlags=lags)
        acf_fig = go.Figure()
        
        # Add bars
        acf_fig.add_trace(go.Bar(
            x=list(range(lags+1)),
            y=acf_values,
            name='ACF',
            marker_color='lightblue'
        ))
        
        # Add confidence interval
        conf_int = 1.96 / np.sqrt(len(series.dropna()))
        acf_fig.add_trace(go.Scatter(
            x=[-1, lags+1],
            y=[conf_int, conf_int],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='95% CI',
            showlegend=True
        ))
        
        acf_fig.add_trace(go.Scatter(
            x=[-1, lags+1],
            y=[-conf_int, -conf_int],
            mode='lines',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ))
        
        acf_fig.update_layout(
            title=f'Autocorrelation Function (ACF) - {lags} lags',
            xaxis_title='Lag',
            yaxis_title='ACF',
            showlegend=True
        )
        
        # PACF Plot
        pacf_values = pacf(series.dropna(), nlags=lags)
        pacf_fig = go.Figure()
        
        # Add bars
        pacf_fig.add_trace(go.Bar(
            x=list(range(lags+1)),
            y=pacf_values,
            name='PACF',
            marker_color='lightcoral'
        ))
        
        # Add confidence interval
        pacf_fig.add_trace(go.Scatter(
            x=[-1, lags+1],
            y=[conf_int, conf_int],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='95% CI',
            showlegend=True
        ))
        
        pacf_fig.add_trace(go.Scatter(
            x=[-1, lags+1],
            y=[-conf_int, -conf_int],
            mode='lines',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ))
        
        pacf_fig.update_layout(
            title=f'Partial Autocorrelation Function (PACF) - {lags} lags',
            xaxis_title='Lag',
            yaxis_title='PACF',
            showlegend=True
        )
        
        return acf_fig, pacf_fig
    
    @staticmethod
    def detect_seasonality(series: pd.Series, max_lag: int = 100) -> Dict:
        """Detect seasonality patterns."""
        from scipy import signal
        
        results = {}
        
        # Remove NaN values
        clean_series = series.dropna()
        
        if len(clean_series) < max_lag * 2:
            results['error'] = 'Series too short for seasonality detection'
            return results
        
        # Autocorrelation based seasonality detection
        acf_vals = acf(clean_series, nlags=max_lag)
        
        # Find peaks in ACF (potential seasonal periods)
        peaks, properties = signal.find_peaks(acf_vals, height=0.2, distance=5)
        
        if len(peaks) > 0:
            seasonal_periods = peaks.tolist()
            results['seasonal_periods'] = seasonal_periods
            results['peak_heights'] = acf_vals[peaks].tolist()
            
            # Suggest most likely seasonal period
            if len(seasonal_periods) > 0:
                results['suggested_period'] = seasonal_periods[np.argmax(results['peak_heights'])]
        
        # Fourier transform for frequency analysis
        try:
            n = len(clean_series)
            yf = np.fft.fft(clean_series.values - clean_series.values.mean())
            xf = np.fft.fftfreq(n, d=1)
            
            # Get dominant frequencies
            idx = np.argsort(np.abs(yf))[::-1][:5]
            dominant_freqs = xf[idx]
            results['dominant_frequencies'] = dominant_freqs.tolist()
            
        except:
            results['fourier_error'] = 'Fourier analysis failed'
        
        return results
    
    @staticmethod
    def create_rolling_statistics_plot(df: pd.DataFrame, date_col: str, 
                                      value_col: str, window: int = 30) -> go.Figure:
        """Create rolling statistics plot."""
        df_sorted = df.sort_values(date_col).copy()
        
        # Calculate rolling statistics
        df_sorted['rolling_mean'] = df_sorted[value_col].rolling(window=window).mean()
        df_sorted['rolling_std'] = df_sorted[value_col].rolling(window=window).std()
        df_sorted['rolling_min'] = df_sorted[value_col].rolling(window=window).min()
        df_sorted['rolling_max'] = df_sorted[value_col].rolling(window=window).max()
        
        fig = go.Figure()
        
        # Original series
        fig.add_trace(go.Scatter(
            x=df_sorted[date_col],
            y=df_sorted[value_col],
            mode='lines',
            name='Original',
            line=dict(color='blue', width=1),
            opacity=0.5
        ))
        
        # Rolling mean
        fig.add_trace(go.Scatter(
            x=df_sorted[date_col],
            y=df_sorted['rolling_mean'],
            mode='lines',
            name=f'Rolling Mean ({window} periods)',
            line=dict(color='red', width=2)
        ))
        
        # Rolling bounds
        fig.add_trace(go.Scatter(
            x=df_sorted[date_col],
            y=df_sorted['rolling_mean'] + df_sorted['rolling_std'],
            mode='lines',
            name='Upper Bound',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=df_sorted[date_col],
            y=df_sorted['rolling_mean'] - df_sorted['rolling_std'],
            mode='lines',
            name='Lower Bound',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(128, 128, 128, 0.2)',
            showlegend=False
        ))
        
        fig.update_layout(
            title=f'Rolling Statistics of {value_col} (Window: {window})',
            xaxis_title='Date',
            yaxis_title=value_col,
            hovermode='x unified'
        )
        
        return fig