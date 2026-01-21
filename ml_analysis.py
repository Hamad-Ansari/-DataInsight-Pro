# src/ml_analysis.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class MLAnalyzer:
    """Machine Learning analysis utilities."""
    
    @staticmethod
    def analyze_feature_importance(df: pd.DataFrame, target_col: str, 
                                  method: str = 'random_forest') -> Tuple[pd.DataFrame, go.Figure]:
        """Analyze feature importance."""
        # Prepare data
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Check target type
        is_classification = not pd.api.types.is_numeric_dtype(y)
        
        if method == 'random_forest':
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                y_encoded = LabelEncoder().fit_transform(y)
                model.fit(X, y_encoded)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
            
            importances = model.feature_importances_
            
        elif method == 'mutual_info':
            if is_classification:
                y_encoded = LabelEncoder().fit_transform(y)
                importances = mutual_info_classif(X, y_encoded, random_state=42)
            else:
                importances = mutual_info_regression(X, y, random_state=42)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Create plot
        fig = px.bar(
            importance_df.head(20),
            x='importance',
            y='feature',
            orientation='h',
            title=f'Feature Importance for {target_col} ({method})',
            color='importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_title='Importance Score',
            yaxis_title='Features',
            showlegend=False
        )
        
        return importance_df, fig
    
    @staticmethod
    def analyze_multicollinearity(df: pd.DataFrame) -> Tuple[pd.DataFrame, go.Figure]:
        """Analyze multicollinearity using VIF."""
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return pd.DataFrame(), go.Figure()
        
        # Add constant for VIF calculation
        X = add_constant(numeric_df)
        
        # Calculate VIF
        vif_data = pd.DataFrame()
        vif_data['feature'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) 
                          for i in range(X.shape[1])]
        vif_data = vif_data[vif_data['feature'] != 'const']
        
        # Create plot
        fig = px.bar(
            vif_data.sort_values('VIF', ascending=False),
            x='VIF',
            y='feature',
            orientation='h',
            title='Variance Inflation Factor (VIF) - Multicollinearity Analysis',
            color='VIF',
            color_continuous_scale='RdYlGn_r'
        )
        
        # Add threshold lines
        fig.add_vline(x=5, line_dash='dash', line_color='orange', 
                     annotation_text='VIF > 5: Moderate', annotation_position='top')
        fig.add_vline(x=10, line_dash='dash', line_color='red',
                     annotation_text='VIF > 10: High', annotation_position='top')
        
        fig.update_layout(
            xaxis_title='VIF Score',
            yaxis_title='Features',
            showlegend=False
        )
        
        return vif_data, fig
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, method: str = 'iqr') -> Dict:
        """Detect outliers in numeric columns."""
        results = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = df[col].dropna()
            
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(data))
                outliers = data[z_scores > 3]
            
            else:
                continue
            
            results[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(data)) * 100,
                'outlier_values': outliers.tolist()
            }
        
        return results
    
    @staticmethod
    def analyze_clustering(df: pd.DataFrame, n_clusters: int = 3) -> Tuple[pd.DataFrame, go.Figure]:
        """Perform K-means clustering analysis."""
        # Select only numeric columns and handle missing values
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.fillna(numeric_df.mean())
        
        if len(numeric_df.columns) < 2:
            return pd.DataFrame(), go.Figure()
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Perform PCA for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(scaled_data, clusters)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'Cluster': clusters
        })
        
        # Add cluster centers
        pca_centers = pca.transform(kmeans.cluster_centers_)
        
        # Create plot
        fig = px.scatter(
            results_df,
            x='PC1',
            y='PC2',
            color='Cluster',
            title=f'K-means Clustering (n_clusters={n_clusters}, Silhouette Score: {silhouette_avg:.3f})',
            color_continuous_scale='Viridis',
            opacity=0.7
        )
        
        # Add cluster centers
        fig.add_trace(go.Scatter(
            x=pca_centers[:, 0],
            y=pca_centers[:, 1],
            mode='markers',
            marker=dict(size=15, color='red', symbol='x'),
            name='Cluster Centers'
        ))
        
        fig.update_layout(
            xaxis_title=f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
            yaxis_title=f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
            showlegend=True
        )
        
        # Create cluster statistics
        cluster_stats = {}
        for cluster in range(n_clusters):
            cluster_data = numeric_df[clusters == cluster]
            cluster_stats[f'Cluster_{cluster}'] = {
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(numeric_df)) * 100,
                'means': cluster_data.mean().to_dict(),
                'stds': cluster_data.std().to_dict()
            }
        
        return results_df, fig, cluster_stats, silhouette_avg
    
    @staticmethod
    def suggest_preprocessing(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict:
        """Suggest preprocessing steps based on data analysis."""
        suggestions = {
            'missing_values': {},
            'encoding': {},
            'scaling': {},
            'transformations': {},
            'feature_engineering': []
        }
        
        # Analyze missing values
        missing_stats = df.isnull().sum()
        total_rows = len(df)
        
        for col in df.columns:
            missing_count = missing_stats[col]
            missing_percentage = (missing_count / total_rows) * 100
            
            if missing_count > 0:
                suggestions['missing_values'][col] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': round(missing_percentage, 2),
                    'suggestion': 'impute_with_mean' if pd.api.types.is_numeric_dtype(df[col]) else 'impute_with_mode'
                }
            
            # Check data type for encoding suggestions
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                unique_count = df[col].nunique()
                if unique_count <= 10:
                    suggestions['encoding'][col] = {
                        'unique_count': int(unique_count),
                        'suggestion': 'one_hot_encoding'
                    }
                else:
                    suggestions['encoding'][col] = {
                        'unique_count': int(unique_count),
                        'suggestion': 'label_encoding_or_target_encoding'
                    }
            
            # Check for scaling
            if pd.api.types.is_numeric_dtype(df[col]):
                skewness = df[col].skew()
                if abs(skewness) > 1:
                    suggestions['transformations'][col] = {
                        'skewness': round(skewness, 2),
                        'suggestion': 'log_transform' if skewness > 0 else 'box_cox_transform'
                    }
                
                # Check if scaling is needed
                if df[col].std() > 10 * df[col].mean():
                    suggestions['scaling'][col] = {
                        'cv': round(df[col].std() / df[col].mean(), 2),
                        'suggestion': 'standard_scaling'
                    }
        
        # Feature engineering suggestions
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            suggestions['feature_engineering'].append({
                'type': 'interaction_terms',
                'suggestion': 'Create interaction terms between numeric features'
            })
        
        if target_col and target_col in df.columns:
            suggestions['feature_engineering'].append({
                'type': 'target_encoding',
                'suggestion': f'Create target-encoded features for categorical columns using {target_col}'
            })
        
        return suggestions