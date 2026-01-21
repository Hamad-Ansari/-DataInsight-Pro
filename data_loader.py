# src/data_loader.py
import pandas as pd
import numpy as np
import streamlit as st
from typing import Union, Tuple, Dict
import io
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Handle all data loading operations."""
    
    @staticmethod
    def load_data(uploaded_file) -> Tuple[pd.DataFrame, Dict]:
        """
        Load data from uploaded file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (dataframe, file_info)
        """
        file_info = {
            'name': uploaded_file.name,
            'size': uploaded_file.size,
            'type': uploaded_file.type
        }
        
        try:
            # Determine file type and load accordingly
            if uploaded_file.name.endswith('.csv'):
                # Try different encodings
                for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        file_info['encoding'] = encoding
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # If all encodings fail, use default with error handling
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
                    file_info['encoding'] = 'utf-8 (with error replacement)'
                    
            elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file, engine='openpyxl')
                
            elif uploaded_file.name.endswith('.json'):
                uploaded_file.seek(0)
                # Try different JSON orientations
                try:
                    df = pd.read_json(uploaded_file)
                except:
                    uploaded_file.seek(0)
                    content = uploaded_file.read().decode('utf-8')
                    data = json.loads(content)
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    else:
                        df = pd.json_normalize(data)
            
            else:
                st.error(f"Unsupported file format: {uploaded_file.name}")
                return pd.DataFrame(), file_info
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Convert object columns to datetime if possible
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='ignore')
                except:
                    pass
            
            file_info['success'] = True
            file_info['rows'] = len(df)
            file_info['columns'] = len(df.columns)
            
            return df, file_info
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            file_info['success'] = False
            file_info['error'] = str(e)
            return pd.DataFrame(), file_info
    
    @staticmethod
    def get_sample_datasets() -> Dict[str, pd.DataFrame]:
        """Provide sample datasets for demonstration."""
        samples = {}
        
        # Sample 1: Iris dataset
        from sklearn.datasets import load_iris
        iris = load_iris()
        samples['Iris Dataset'] = pd.DataFrame(
            data=np.c_[iris['data'], iris['target']],
            columns=iris['feature_names'] + ['target']
        )
        
        # Sample 2: Titanic dataset
        titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        try:
            samples['Titanic Dataset'] = pd.read_csv(titanic_url)
        except:
            # Fallback to local generation
            titanic_data = {
                'PassengerId': range(1, 892),
                'Survived': np.random.randint(0, 2, 891),
                'Pclass': np.random.randint(1, 4, 891),
                'Name': [f'Passenger {i}' for i in range(1, 892)],
                'Sex': np.random.choice(['male', 'female'], 891),
                'Age': np.random.randint(1, 80, 891),
                'SibSp': np.random.randint(0, 5, 891),
                'Parch': np.random.randint(0, 4, 891),
                'Ticket': [f'Ticket_{i}' for i in range(1, 892)],
                'Fare': np.random.uniform(0, 300, 891),
                'Cabin': np.random.choice(['A', 'B', 'C', 'D', 'E', None], 891),
                'Embarked': np.random.choice(['S', 'C', 'Q', None], 891)
            }
            samples['Titanic Dataset'] = pd.DataFrame(titanic_data)
        
        # Sample 3: Time series data
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        time_series_data = {
            'Date': dates,
            'Sales': np.random.normal(100, 20, 365).cumsum(),
            'Temperature': np.sin(np.linspace(0, 4*np.pi, 365)) * 10 + 20,
            'Humidity': np.random.uniform(30, 90, 365),
            'Rainfall': np.random.exponential(5, 365)
        }
        samples['Time Series Dataset'] = pd.DataFrame(time_series_data)
        
        # Sample 4: Customer data
        customer_data = {
            'CustomerID': range(1, 1001),
            'Age': np.random.randint(18, 70, 1000),
            'Gender': np.random.choice(['Male', 'Female'], 1000),
            'Income': np.random.normal(50000, 15000, 1000),
            'SpendingScore': np.random.randint(1, 100, 1000),
            'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 1000),
            'MembershipYears': np.random.randint(0, 10, 1000)
        }
        samples['Customer Dataset'] = pd.DataFrame(customer_data)
        
        return samples
    
    @staticmethod
    def preprocess_data(df: pd.DataFrame, options: Dict = None) -> pd.DataFrame:
        """Preprocess data based on user options."""
        if options is None:
            options = {}
        
        df_processed = df.copy()
        
        # Handle missing values
        if options.get('handle_missing'):
            method = options.get('missing_method', 'mean')
            for col in df_processed.columns:
                if df_processed[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df_processed[col]):
                        if method == 'mean':
                            df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                        elif method == 'median':
                            df_processed[col].fillna(df_processed[col].median(), inplace=True)
                        elif method == 'mode':
                            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                        elif method == 'drop':
                            df_processed.dropna(subset=[col], inplace=True)
                    else:
                        # For categorical columns
                        df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
        
        # Remove duplicates
        if options.get('remove_duplicates'):
            df_processed.drop_duplicates(inplace=True)
        
        # Reset index
        df_processed.reset_index(drop=True, inplace=True)
        
        return df_processed