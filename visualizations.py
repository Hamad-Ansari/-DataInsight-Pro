# src/visualizations.py
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import streamlit as st
from plotly.colors import qualitative, sequential
import warnings
warnings.filterwarnings('ignore')

class Visualizations:
    """Create various types of visualizations."""
    
    @staticmethod
    def set_theme():
        """Set visualization theme."""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    @staticmethod
    def create_distribution_plots(df: pd.DataFrame, column: str, plot_type: str = 'histogram') -> go.Figure:
        """Create distribution plots."""
        if column not in df.columns:
            return go.Figure()
        
        if plot_type == 'histogram':
            fig = px.histogram(
                df, x=column,
                title=f'Distribution of {column}',
                marginal='box',
                opacity=0.7,
                nbins=50,
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig.update_layout(
                xaxis_title=column,
                yaxis_title='Count',
                showlegend=False
            )
            
        elif plot_type == 'kde':
            # Create KDE plot
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=df[column].dropna(),
                histnorm='probability density',
                name='Histogram',
                opacity=0.5,
                marker_color='lightblue'
            ))
            
            # Add KDE curve
            from scipy import stats
            data = df[column].dropna()
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 1000)
            y_kde = kde(x_range)
            
            fig.add_trace(go.Scatter(
                x=x_range, y=y_kde,
                mode='lines',
                name='KDE',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title=f'KDE Plot of {column}',
                xaxis_title=column,
                yaxis_title='Density',
                showlegend=True
            )
            
        elif plot_type == 'box':
            fig = px.box(
                df, y=column,
                title=f'Box Plot of {column}',
                points='all',
                color_discrete_sequence=['lightseagreen']
            )
            fig.update_layout(showlegend=False)
            
        elif plot_type == 'violin':
            fig = px.violin(
                df, y=column,
                title=f'Violin Plot of {column}',
                box=True,
                points='all',
                color_discrete_sequence=['mediumpurple']
            )
            fig.update_layout(showlegend=False)
            
        elif plot_type == 'qq':
            # Q-Q Plot
            from scipy import stats
            data = df[column].dropna()
            (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=osm, y=osr,
                mode='markers',
                name='Data',
                marker=dict(color='blue', size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=osm, y=slope*osm + intercept,
                mode='lines',
                name='Theoretical',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title=f'Q-Q Plot of {column}',
                xaxis_title='Theoretical Quantiles',
                yaxis_title='Sample Quantiles',
                showlegend=True
            )
            
        else:
            fig = go.Figure()
            
        return fig
    
    @staticmethod
    def create_relationship_plots(df: pd.DataFrame, x_col: str, y_col: str, 
                                 plot_type: str = 'scatter', color_col: Optional[str] = None,
                                 size_col: Optional[str] = None) -> go.Figure:
        """Create relationship plots."""
        if x_col not in df.columns or y_col not in df.columns:
            return go.Figure()
        
        if plot_type == 'scatter':
            fig = px.scatter(
                df, x=x_col, y=y_col,
                color=color_col if color_col else None,
                size=size_col if size_col else None,
                title=f'{x_col} vs {y_col}',
                opacity=0.6,
                trendline='ols' if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]) else None
            )
            
        elif plot_type == 'line':
            fig = px.line(
                df, x=x_col, y=y_col,
                title=f'{x_col} vs {y_col}',
                markers=True
            )
            
        elif plot_type == 'bubble':
            fig = px.scatter(
                df, x=x_col, y=y_col,
                size=size_col if size_col else df[y_col],
                color=color_col if color_col else None,
                title=f'Bubble Chart: {x_col} vs {y_col}',
                hover_name=color_col,
                size_max=60
            )
            
        elif plot_type == 'heatmap':
            # For numeric columns correlation
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 0:
                corr_matrix = numeric_df.corr()
                fig = px.imshow(
                    corr_matrix,
                    title='Correlation Heatmap',
                    color_continuous_scale='RdBu_r',
                    aspect='auto'
                )
                fig.update_layout(
                    xaxis_title='Features',
                    yaxis_title='Features'
                )
            else:
                fig = go.Figure()
                fig.add_annotation(text="No numeric columns for heatmap")
                
        elif plot_type == 'pairplot':
            # Create pair plot for first 5 numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                numeric_cols = numeric_cols[:5]  # Limit to 5 columns
                fig = px.scatter_matrix(
                    df[numeric_cols],
                    title='Pair Plot',
                    dimensions=numeric_cols,
                    color=color_col if color_col else None
                )
                fig.update_traces(diagonal_visible=False)
            else:
                fig = go.Figure()
                fig.add_annotation(text="Need at least 2 numeric columns for pair plot")
                
        else:
            fig = go.Figure()
            
        return fig
    
    @staticmethod
    def create_composition_plots(df: pd.DataFrame, column: str, 
                                plot_type: str = 'pie', groupby_col: Optional[str] = None) -> go.Figure:
        """Create composition plots."""
        if column not in df.columns:
            return go.Figure()
        
        if plot_type == 'pie':
            if groupby_col:
                grouped = df.groupby(groupby_col)[column].sum().reset_index()
                fig = px.pie(
                    grouped, values=column, names=groupby_col,
                    title=f'{column} by {groupby_col}',
                    hole=0.3
                )
            else:
                value_counts = df[column].value_counts().reset_index()
                fig = px.pie(
                    value_counts, values='count', names=column,
                    title=f'Distribution of {column}',
                    hole=0.3
                )
                
        elif plot_type == 'bar':
            if groupby_col:
                grouped = df.groupby(groupby_col)[column].sum().reset_index()
                fig = px.bar(
                    grouped, x=groupby_col, y=column,
                    title=f'{column} by {groupby_col}',
                    color=groupby_col
                )
            else:
                value_counts = df[column].value_counts().reset_index()
                fig = px.bar(
                    value_counts, x=column, y='count',
                    title=f'Count of {column}',
                    color=column
                )
                
        elif plot_type == 'stacked_bar':
            if groupby_col:
                pivot_df = pd.crosstab(df[groupby_col], df[column])
                fig = px.bar(
                    pivot_df.reset_index().melt(id_vars=groupby_col),
                    x=groupby_col, y='value', color='variable',
                    title=f'Stacked Bar: {column} by {groupby_col}',
                    barmode='stack'
                )
            else:
                fig = go.Figure()
                fig.add_annotation(text="Need groupby column for stacked bar")
                
        elif plot_type == 'treemap':
            if groupby_col:
                grouped = df.groupby(groupby_col)[column].sum().reset_index()
                fig = px.treemap(
                    grouped, path=[groupby_col], values=column,
                    title=f'Treemap of {column} by {groupby_col}',
                    color=column,
                    color_continuous_scale='Viridis'
                )
            else:
                value_counts = df[column].value_counts().reset_index()
                fig = px.treemap(
                    value_counts, path=[column], values='count',
                    title=f'Treemap of {column}',
                    color='count',
                    color_continuous_scale='Viridis'
                )
                
        elif plot_type == 'sunburst':
            if groupby_col and len(df.columns) > 2:
                third_col = [c for c in df.columns if c not in [column, groupby_col]][0]
                fig = px.sunburst(
                    df, path=[groupby_col, column, third_col],
                    title='Sunburst Chart',
                    color=column if pd.api.types.is_numeric_dtype(df[column]) else None
                )
            else:
                fig = go.Figure()
                fig.add_annotation(text="Need at least 3 columns for sunburst")
                
        else:
            fig = go.Figure()
            
        return fig
    
    @staticmethod
    def create_time_series_plots(df: pd.DataFrame, date_col: str, value_col: str,
                                plot_type: str = 'line', groupby_col: Optional[str] = None) -> go.Figure:
        """Create time series plots."""
        if date_col not in df.columns or value_col not in df.columns:
            return go.Figure()
        
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col, value_col])
        
        if plot_type == 'line':
            fig = px.line(
                df, x=date_col, y=value_col,
                title=f'{value_col} over Time',
                markers=True
            )
            
        elif plot_type == 'area':
            fig = px.area(
                df, x=date_col, y=value_col,
                title=f'{value_col} over Time',
                line_shape='spline'
            )
            
        elif plot_type == 'seasonal':
            # Try to decompose time series
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                
                # Ensure regular frequency
                df_ts = df.set_index(date_col)[value_col]
                df_ts = df_ts.asfreq('D').fillna(method='ffill')
                
                decomposition = seasonal_decompose(df_ts, model='additive', period=30)
                
                fig = sp.make_subplots(
                    rows=4, cols=1,
                    subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'),
                    vertical_spacing=0.1
                )
                
                fig.add_trace(
                    go.Scatter(x=decomposition.observed.index, 
                             y=decomposition.observed.values,
                             mode='lines',
                             name='Observed'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=decomposition.trend.index, 
                             y=decomposition.trend.values,
                             mode='lines',
                             name='Trend'),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=decomposition.seasonal.index, 
                             y=decomposition.seasonal.values,
                             mode='lines',
                             name='Seasonal'),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=decomposition.resid.index, 
                             y=decomposition.resid.values,
                             mode='lines',
                             name='Residual'),
                    row=4, col=1
                )
                
                fig.update_layout(
                    height=800,
                    title_text=f'Seasonal Decomposition of {value_col}',
                    showlegend=False
                )
                
            except Exception as e:
                fig = go.Figure()
                fig.add_annotation(text=f"Cannot perform seasonal decomposition: {str(e)}")
                
        elif plot_type == 'rolling':
            df_sorted = df.sort_values(date_col)
            window = min(30, len(df_sorted) // 10)
            
            df_sorted['rolling_mean'] = df_sorted[value_col].rolling(window=window).mean()
            df_sorted['rolling_std'] = df_sorted[value_col].rolling(window=window).std()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_sorted[date_col], y=df_sorted[value_col],
                mode='lines',
                name='Original',
                line=dict(color='blue', width=1, dash='dot')
            ))
            
            fig.add_trace(go.Scatter(
                x=df_sorted[date_col], y=df_sorted['rolling_mean'],
                mode='lines',
                name=f'Rolling Mean ({window} periods)',
                line=dict(color='red', width=2)
            ))
            
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
                fillcolor='rgba(0,100,80,0.2)'
            ))
            
            fig.update_layout(
                title=f'Rolling Statistics of {value_col}',
                xaxis_title='Date',
                yaxis_title=value_col
            )
            
        else:
            fig = go.Figure()
            
        return fig
    
    @staticmethod
    def create_correlation_matrix(df: pd.DataFrame) -> go.Figure:
        """Create correlation matrix heatmap."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Need at least 2 numeric columns for correlation matrix")
            return fig
        
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect='auto',
            color_continuous_scale='RdBu_r',
            title='Correlation Matrix',
            labels=dict(color='Correlation')
        )
        
        fig.update_layout(
            xaxis_title='Features',
            yaxis_title='Features',
            width=800,
            height=800
        )
        
        return fig
    
    @staticmethod
    def create_missing_values_plot(df: pd.DataFrame) -> go.Figure:
        """Create missing values heatmap."""
        missing_df = df.isnull()
        
        fig = px.imshow(
            missing_df.T,
            title='Missing Values Heatmap',
            color_continuous_scale=['green', 'red'],
            labels=dict(x='Row Index', y='Columns', color='Missing')
        )
        
        fig.update_layout(
            width=1000,
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_feature_importance_plot(df: pd.DataFrame, target_col: str) -> go.Figure:
        """Create feature importance plot using Random Forest."""
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare data
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Train model
        if pd.api.types.is_numeric_dtype(y):
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            le = LabelEncoder()
            y = le.fit_transform(y)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        model.fit(X, y)
        
        # Get feature importances
        importances = model.feature_importances_
        feature_names = X.columns
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(20)
        
        # Create plot
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title=f'Feature Importance for {target_col}',
            color='importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_title='Importance',
            yaxis_title='Features',
            showlegend=False
        )
        
        return fig