# src/utils.py
import base64
import io
import pandas as pd
import numpy as np
from typing import Union, Dict, List, Optional
import streamlit as st
from datetime import datetime
import json

def get_download_link(df: pd.DataFrame, filename: str, file_format: str = 'csv'):
    """Generate download link for dataframes."""
    if file_format == 'csv':
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV</a>'
    elif file_format == 'excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">Download Excel</a>'
    elif file_format == 'json':
        json_str = df.to_json(orient='records', indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="{filename}.json">Download JSON</a>'
    
    return href

def download_plotly_fig(fig, filename: str, format: str = 'png'):
    """Download Plotly figure as image."""
    if format == 'png':
        img_bytes = fig.to_image(format="png", width=1200, height=800)
        b64 = base64.b64encode(img_bytes).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">Download PNG</a>'
    elif format == 'svg':
        img_bytes = fig.to_image(format="svg")
        b64 = base64.b64encode(img_bytes).decode()
        href = f'<a href="data:image/svg+xml;base64,{b64}" download="{filename}.svg">Download SVG</a>'
    elif format == 'pdf':
        img_bytes = fig.to_image(format="pdf")
        b64 = base64.b64encode(img_bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}.pdf">Download PDF</a>'
    else:
        href = "<p style='color:red'>Unsupported format</p>"
    
    return href

def convert_df_to_csv(df):
    """Convert dataframe to CSV for download."""
    return df.to_csv(index=False).encode('utf-8')

def get_file_size_str(size_in_bytes):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} TB"

def detect_date_columns(df: pd.DataFrame) -> List[str]:
    """Detect date columns in dataframe."""
    date_columns = []
    for col in df.columns:
        # Check if column has datetime dtype
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.append(col)
        else:
            # Try to convert to datetime
            try:
                pd.to_datetime(df[col], errors='raise')
                date_columns.append(col)
            except:
                continue
    return date_columns

def highlight_missing_values(val):
    """Highlight missing values in dataframe."""
    color = 'background-color: yellow' if pd.isna(val) else ''
    return color

def get_dataframe_info(df: pd.DataFrame) -> Dict:
    """Get comprehensive information about dataframe."""
    info = {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum(),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': detect_date_columns(df),
        'unique_counts': {col: df[col].nunique() for col in df.columns},
        'descriptive_stats': df.describe(include='all').to_dict() if not df.empty else {}
    }
    return info