# app.py - Main Streamlit Application
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import base64
import io
import json
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import custom modules
from src.data_loader import DataLoader
from src.visualizations import Visualizations
from src.time_series import TimeSeriesAnalyzer
from src.ml_analysis import MLAnalyzer
from src.utils import (
    get_download_link, download_plotly_fig, convert_df_to_csv,
    get_file_size_str, detect_date_columns, highlight_missing_values,
    get_dataframe_info
)

# Page configuration
st.set_page_config(
    page_title="DataInsight Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
def load_css():
    try:
        with open('assets/custom.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        pass

load_css()

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_info' not in st.session_state:
    st.session_state.file_info = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'plotly_figs' not in st.session_state:
    st.session_state.plotly_figs = {}

# Suppress warnings
warnings.filterwarnings('ignore')

# App title and description
def show_header():
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("📊 DataInsight Pro")
        st.markdown("### Advanced EDA Platform for Machine Learning")
        st.markdown("*Uncover Insights, Build Better Models*")
    
    with col2:
        st.markdown("---")
        st.markdown("**By Hammad Zahid**")
        st.markdown("Data Scientist & Analyst")
        st.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/hammad-zahid-xyz)")
        st.markdown("🐙 [GitHub](https://github.com/Hamad-Ansari)")
        st.markdown("✉️ [Email](mailto:Hammadzahid24@gmail.com)")
    
    st.markdown("---")

# Sidebar navigation
def sidebar_navigation():
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/data-configuration.png", width=80)
        st.title("Navigation")
        
        # File upload section
        st.markdown("### 📁 Upload Data")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'json'],
            help="Upload CSV, Excel, or JSON files up to 500MB"
        )
        
        # Sample data
        st.markdown("### 📊 Sample Datasets")
        sample_options = ["None", "Iris Dataset", "Titanic Dataset", 
                         "Time Series Dataset", "Customer Dataset"]
        selected_sample = st.selectbox("Try sample data", sample_options)
        
        # Load data
        if uploaded_file is not None:
            load_data(uploaded_file)
        elif selected_sample != "None":
            load_sample_data(selected_sample)
        
        # Navigation menu
        st.markdown("---")
        st.markdown("### 🧭 Analysis Menu")
        
        menu_options = [
            "🏠 Home",
            "📈 Data Overview",
            "🔍 Automated EDA",
            "📊 Visualization Studio",
            "⏰ Time Series Analysis",
            "🤖 ML Insights",
            "📥 Export Center"
        ]
        
        selected_menu = st.radio("Select Section", menu_options)
        
        # Map menu to page
        page_map = {
            "🏠 Home": "home",
            "📈 Data Overview": "overview",
            "🔍 Automated EDA": "eda",
            "📊 Visualization Studio": "visualization",
            "⏰ Time Series Analysis": "timeseries",
            "🤖 ML Insights": "ml",
            "📥 Export Center": "export"
        }
        
        if selected_menu in page_map:
            st.session_state.current_page = page_map[selected_menu]
        
        # Show data info if loaded
        if st.session_state.df is not None:
            st.markdown("---")
            st.markdown("### 📋 Data Info")
            st.info(f"**Rows:** {len(st.session_state.df):,}")
            st.info(f"**Columns:** {len(st.session_state.df.columns):,}")
            
            if st.session_state.file_info:
                st.info(f"**File:** {st.session_state.file_info.get('name', 'N/A')}")
                st.info(f"**Size:** {get_file_size_str(st.session_state.file_info.get('size', 0))}")
        
        st.markdown("---")
        st.markdown("*Built with Streamlit*")
        st.markdown("Version 1.0.0")

def load_data(uploaded_file):
    """Load data from uploaded file."""
    with st.spinner("Loading data..."):
        df, file_info = DataLoader.load_data(uploaded_file)
        
        if not df.empty:
            st.session_state.df = df
            st.session_state.file_info = file_info
            st.session_state.processed_df = df.copy()
            st.success(f"✅ Data loaded successfully! {len(df):,} rows × {len(df.columns):,} columns")
        else:
            st.error("❌ Failed to load data. Please check the file format.")

def load_sample_data(sample_name):
    """Load sample dataset."""
    with st.spinner(f"Loading {sample_name}..."):
        samples = DataLoader.get_sample_datasets()
        if sample_name in samples:
            st.session_state.df = samples[sample_name]
            st.session_state.file_info = {
                'name': f"{sample_name}.csv",
                'size': 0,
                'type': 'sample',
                'success': True,
                'rows': len(samples[sample_name]),
                'columns': len(samples[sample_name].columns)
            }
            st.session_state.processed_df = samples[sample_name].copy()
            st.success(f"✅ {sample_name} loaded! {len(samples[sample_name]):,} rows × {len(samples[sample_name].columns):,} columns")
        else:
            st.error("Sample dataset not found.")

# Home Page
def home_page():
    st.markdown("""
    ## 🎯 Welcome to DataInsight Pro
    
    **DataInsight Pro** is a comprehensive Exploratory Data Analysis (EDA) platform designed for 
    data scientists and analysts. It provides powerful tools to understand, visualize, and 
    prepare your data for machine learning.
    
    ### 🚀 Key Features:
    
    #### 📁 **Data Loading & Processing**
    - Support for CSV, Excel, and JSON files
    - Automatic encoding detection
    - Large dataset handling (up to 500MB)
    - Real-time data preview
    
    #### 📊 **Automated EDA**
    - **ydata-profiling**: Comprehensive automated reports
    - **Dtale**: Interactive data exploration
    - Missing value analysis
    - Correlation matrices
    
    #### 🎨 **Advanced Visualization**
    - **30+ Plot Types**: Histograms, scatter plots, box plots, heatmaps, and more
    - **Interactive Charts**: Plotly-powered interactive visualizations
    - **Customization**: Color schemes, annotations, layouts
    - **Multi-plot Dashboards**: Compare multiple visualizations side-by-side
    
    #### ⏰ **Time Series Analysis**
    - Seasonality detection
    - Trend decomposition
    - Stationarity testing
    - Rolling statistics
    - ACF/PACF plots
    
    #### 🤖 **Machine Learning Insights**
    - Feature importance analysis
    - Outlier detection
    - Multicollinearity checking
    - Clustering analysis
    - Preprocessing suggestions
    
    #### 📥 **Export & Sharing**
    - Download plots as PNG, SVG, PDF
    - Export reports as HTML
    - Save processed data
    - Generate PDF reports
    
    ### 📋 **Getting Started:**
    1. **Upload your data** using the sidebar
    2. **Explore sample datasets** to see the platform in action
    3. **Navigate through sections** using the menu
    4. **Generate insights** with automated tools
    5. **Export your results** for reports and presentations
    
    ### 🛠️ **Technical Stack:**
    - **Framework**: Streamlit
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **Data Processing**: Pandas, NumPy
    - **Machine Learning**: Scikit-learn, TensorFlow
    - **EDA Tools**: ydata-profiling, dtale, missingno
    
    ### 📈 **Sample Analysis Workflow:**
    """)
    
    # Show workflow diagram
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("**1. Upload**")
        st.markdown("📁")
    with col2:
        st.markdown("**2. Explore**")
        st.markdown("🔍")
    with col3:
        st.markdown("**3. Visualize**")
        st.markdown("📊")
    with col4:
        st.markdown("**4. Analyze**")
        st.markdown("📈")
    with col5:
        st.markdown("**5. Export**")
        st.markdown("💾")
    
    st.markdown("---")
    
    # Quick stats if data is loaded
    if st.session_state.df is not None:
        st.markdown("### 📋 Quick Stats")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", f"{len(st.session_state.df):,}")
        with col2:
            st.metric("Columns", len(st.session_state.df.columns))
        with col3:
            numeric_cols = len(st.session_state.df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Columns", numeric_cols)
        with col4:
            missing = st.session_state.df.isnull().sum().sum()
            st.metric("Missing Values", f"{missing:,}")
    
    st.markdown("---")
    st.markdown("""
    ### ❓ Need Help?
    - Check out the sample datasets to explore features
    - Hover over buttons and options for tooltips
    - Each visualization includes customization options
    - All exports are available in the Export Center
    
    **Happy Analyzing!** 🎉
    """)

# Data Overview Page
def overview_page():
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    st.header("📈 Data Overview")
    
    # Create tabs for different overview sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Basic Info", 
        "🔢 Statistics", 
        "❓ Missing Values", 
        "📊 Data Preview", 
        "⚙️ Preprocessing"
    ])
    
    with tab1:
        st.subheader("Dataset Information")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{len(st.session_state.df):,}")
        with col2:
            st.metric("Total Columns", len(st.session_state.df.columns))
        with col3:
            numeric_cols = len(st.session_state.df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Columns", numeric_cols)
        with col4:
            categorical_cols = len(st.session_state.df.select_dtypes(include=['object', 'category']).columns)
            st.metric("Categorical Columns", categorical_cols)
        
        # Memory usage
        memory_usage = st.session_state.df.memory_usage(deep=True).sum()
        st.metric("Memory Usage", get_file_size_str(memory_usage))
        
        # Column information
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Data Type': st.session_state.df.dtypes.astype(str),
            'Non-Null Count': st.session_state.df.notnull().sum().values,
            'Null Count': st.session_state.df.isnull().sum().values,
            'Null Percentage': (st.session_state.df.isnull().sum() / len(st.session_state.df) * 100).round(2).values,
            'Unique Values': [st.session_state.df[col].nunique() for col in st.session_state.df.columns]
        })
        
        st.dataframe(col_info, use_container_width=True)
        
        # Download column info
        csv = col_info.to_csv(index=False)
        st.download_button(
            label="📥 Download Column Info",
            data=csv,
            file_name="column_information.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("Descriptive Statistics")
        
        # Numeric statistics
        numeric_df = st.session_state.df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            st.write("#### Numeric Columns Statistics")
            stats_df = numeric_df.describe().T
            stats_df['skewness'] = numeric_df.skew()
            stats_df['kurtosis'] = numeric_df.kurtosis()
            stats_df['missing'] = numeric_df.isnull().sum()
            stats_df['missing_pct'] = (numeric_df.isnull().sum() / len(numeric_df) * 100).round(2)
            
            st.dataframe(stats_df, use_container_width=True)
            
            # Download numeric stats
            csv = stats_df.to_csv()
            st.download_button(
                label="📥 Download Numeric Stats",
                data=csv,
                file_name="numeric_statistics.csv",
                mime="text/csv"
            )
        
        # Categorical statistics
        categorical_df = st.session_state.df.select_dtypes(include=['object', 'category'])
        if not categorical_df.empty:
            st.write("#### Categorical Columns Statistics")
            cat_stats = []
            for col in categorical_df.columns:
                value_counts = categorical_df[col].value_counts()
                cat_stats.append({
                    'Column': col,
                    'Unique Values': categorical_df[col].nunique(),
                    'Most Common': value_counts.index[0] if len(value_counts) > 0 else 'N/A',
                    'Most Common Count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'Most Common %': (value_counts.iloc[0] / len(categorical_df) * 100).round(2) if len(value_counts) > 0 else 0,
                    'Missing Values': categorical_df[col].isnull().sum()
                })
            
            cat_stats_df = pd.DataFrame(cat_stats)
            st.dataframe(cat_stats_df, use_container_width=True)
            
            # Download categorical stats
            csv = cat_stats_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Categorical Stats",
                data=csv,
                file_name="categorical_statistics.csv",
                mime="text/csv"
            )
    
    with tab3:
        st.subheader("Missing Values Analysis")
        
        # Missing values summary
        missing_total = st.session_state.df.isnull().sum().sum()
        missing_percentage = (missing_total / (len(st.session_state.df) * len(st.session_state.df.columns)) * 100)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Missing Values", f"{missing_total:,}")
        with col2:
            st.metric("Missing Percentage", f"{missing_percentage:.2f}%")
        
        # Missing values by column
        missing_by_col = st.session_state.df.isnull().sum()
        missing_by_col = missing_by_col[missing_by_col > 0]
        
        if len(missing_by_col) > 0:
            missing_df = pd.DataFrame({
                'Column': missing_by_col.index,
                'Missing Count': missing_by_col.values,
                'Missing Percentage': (missing_by_col.values / len(st.session_state.df) * 100).round(2)
            }).sort_values('Missing Count', ascending=False)
            
            st.dataframe(missing_df, use_container_width=True)
            
            # Create missing values heatmap
            st.subheader("Missing Values Heatmap")
            fig = Visualizations.create_missing_values_plot(st.session_state.df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Download missing values info
            csv = missing_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Missing Values Report",
                data=csv,
                file_name="missing_values_report.csv",
                mime="text/csv"
            )
        else:
            st.success("🎉 No missing values found in the dataset!")
    
    with tab4:
        st.subheader("Data Preview")
        
        # Show head and tail
        preview_option = st.radio(
            "Preview Options",
            ["First 100 Rows", "Last 100 Rows", "Random 100 Rows"],
            horizontal=True
        )
        
        if preview_option == "First 100 Rows":
            preview_df = st.session_state.df.head(100)
        elif preview_option == "Last 100 Rows":
            preview_df = st.session_state.df.tail(100)
        else:
            preview_df = st.session_state.df.sample(min(100, len(st.session_state.df)))
        
        # Highlight missing values
        styled_df = preview_df.style.applymap(
            lambda x: 'background-color: yellow' if pd.isna(x) else ''
        )
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Column selection for detailed view
        st.subheader("Column Details")
        selected_col = st.selectbox("Select column to view details", st.session_state.df.columns)
        
        if selected_col:
            col_data = st.session_state.df[selected_col]
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Summary Statistics:**")
                if pd.api.types.is_numeric_dtype(col_data):
                    st.write(col_data.describe())
                else:
                    st.write(f"**Data Type:** {col_data.dtype}")
                    st.write(f"**Unique Values:** {col_data.nunique()}")
                    st.write(f"**Missing Values:** {col_data.isnull().sum()} ({col_data.isnull().sum()/len(col_data)*100:.2f}%)")
            
            with col2:
                st.write("**Value Distribution:**")
                if col_data.nunique() <= 20:
                    value_counts = col_data.value_counts().head(10)
                    fig = px.bar(
                        x=value_counts.index.astype(str),
                        y=value_counts.values,
                        title=f"Top Values in {selected_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    if pd.api.types.is_numeric_dtype(col_data):
                        fig = px.histogram(col_data.dropna(), title=f"Distribution of {selected_col}")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("Too many unique values to display")
    
    with tab5:
        st.subheader("Data Preprocessing")
        
        st.info("""
        Apply preprocessing steps to clean your data. The processed data will be available 
        for all analysis and can be downloaded from the Export Center.
        """)
        
        preprocessing_options = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            preprocessing_options['handle_missing'] = st.checkbox("Handle Missing Values", value=True)
            if preprocessing_options['handle_missing']:
                preprocessing_options['missing_method'] = st.selectbox(
                    "Missing Values Handling Method",
                    ["mean", "median", "mode", "drop"],
                    help="Mean/Median for numeric, Mode for categorical"
                )
            
            preprocessing_options['remove_duplicates'] = st.checkbox("Remove Duplicate Rows", value=True)
        
        with col2:
            preprocessing_options['standardize_names'] = st.checkbox("Standardize Column Names", value=True)
            if preprocessing_options['standardize_names']:
                preprocessing_options['name_format'] = st.selectbox(
                    "Column Name Format",
                    ["lowercase", "UPPERCASE", "Title Case", "snake_case"]
                )
            
            preprocessing_options['reset_index'] = st.checkbox("Reset Index", value=True)
        
        if st.button("🚀 Apply Preprocessing", type="primary"):
            with st.spinner("Applying preprocessing..."):
                # Apply preprocessing
                processed_df = DataLoader.preprocess_data(
                    st.session_state.df, 
                    preprocessing_options
                )
                
                # Standardize column names if requested
                if preprocessing_options.get('standardize_names'):
                    format_method = preprocessing_options.get('name_format', 'lowercase')
                    if format_method == 'lowercase':
                        processed_df.columns = [col.lower().replace(' ', '_') for col in processed_df.columns]
                    elif format_method == 'UPPERCASE':
                        processed_df.columns = [col.upper().replace(' ', '_') for col in processed_df.columns]
                    elif format_method == 'Title Case':
                        processed_df.columns = [col.title().replace(' ', '_') for col in processed_df.columns]
                    elif format_method == 'snake_case':
                        processed_df.columns = [col.lower().replace(' ', '_') for col in processed_df.columns]
                
                # Reset index if requested
                if preprocessing_options.get('reset_index'):
                    processed_df.reset_index(drop=True, inplace=True)
                
                # Update session state
                st.session_state.processed_df = processed_df
                
                # Show results
                st.success(f"✅ Preprocessing applied successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Rows", len(st.session_state.df))
                    st.metric("Original Columns", len(st.session_state.df.columns))
                with col2:
                    st.metric("Processed Rows", len(processed_df))
                    st.metric("Processed Columns", len(processed_df.columns))
                
                # Show preview of processed data
                st.subheader("Processed Data Preview")
                st.dataframe(processed_df.head(10), use_container_width=True)

# Automated EDA Page
def eda_page():
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    st.header("🔍 Automated EDA")
    
    # Create tabs for different EDA tools
    tab1, tab2, tab3 = st.tabs(["📊 ydata-profiling", "🔍 Dtale", "📈 SweetViz"])
    
    with tab1:
        st.subheader("ydata-profiling Report")
        st.info("""
        ydata-profiling generates comprehensive reports with statistics, visualizations, 
        and insights about your dataset. This may take a few moments for large datasets.
        """)
        
        if st.button("🚀 Generate ydata-profiling Report", type="primary"):
            with st.spinner("Generating report... This may take a few moments."):
                try:
                    from ydata_profiling import ProfileReport
                    
                    # Create profile report
                    profile = ProfileReport(
                        st.session_state.df,
                        title="DataInsight Pro - EDA Report",
                        explorative=True,
                        minimal=False
                    )
                    
                    # Save to HTML
                    profile.to_file("temp_profile_report.html")
                    
                    # Display in Streamlit
                    with open("temp_profile_report.html", "r", encoding="utf-8") as f:
                        html_content = f.read()
                    
                    st.components.v1.html(html_content, height=800, scrolling=True)
                    
                    # Provide download link
                    st.download_button(
                        label="📥 Download Full Report (HTML)",
                        data=html_content,
                        file_name="ydata_profiling_report.html",
                        mime="text/html"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
    
    with tab2:
        st.subheader("Dtale Interactive Analysis")
        st.info("""
        Dtale provides an interactive interface for exploring your data. 
        It includes features like filtering, sorting, and statistical analysis.
        """)
        
        if st.button("🚀 Launch Dtale", type="primary"):
            with st.spinner("Launching Dtale..."):
                try:
                    import dtale
                    import dtale.app as dtale_app
                    
                    # Launch Dtale
                    d = dtale.show(st.session_state.df)
                    
                    # Get the URL
                    dtale_url = d._url
                    
                    # Display in iframe
                    st.components.v1.iframe(dtale_url, height=800)
                    
                    st.success(f"Dtale launched successfully!")
                    st.markdown(f"Access Dtale at: {dtale_url}")
                    
                except Exception as e:
                    st.error(f"Error launching Dtale: {str(e)}")
    
    with tab3:
        st.subheader("SweetViz Comparison Reports")
        st.info("""
        SweetViz generates comparison reports between datasets. You can compare:
        - Train vs Test datasets
        - Original vs Processed data
        - Different subsets of your data
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            compare_option = st.selectbox(
                "Comparison Type",
                ["Train vs Test", "Original vs Processed", "Custom Split"]
            )
            
            if compare_option == "Custom Split":
                split_ratio = st.slider("Split Ratio", 0.1, 0.9, 0.7)
        
        with col2:
            target_col = st.selectbox(
                "Target Column (for analysis)",
                ["None"] + list(st.session_state.df.columns)
            )
        
        if st.button("🚀 Generate SweetViz Report", type="primary"):
            with st.spinner("Generating SweetViz report..."):
                try:
                    import sweetviz as sv
                    
                    if compare_option == "Train vs Test":
                        # Split data
                        from sklearn.model_selection import train_test_split
                        train_df, test_df = train_test_split(
                            st.session_state.df, 
                            test_size=0.3, 
                            random_state=42
                        )
                        
                        # Generate report
                        report = sv.compare(
                            [train_df, "Train"],
                            [test_df, "Test"],
                            target_feat=target_col if target_col != "None" else None
                        )
                        
                    elif compare_option == "Original vs Processed":
                        if st.session_state.processed_df is not None:
                            report = sv.compare(
                                [st.session_state.df, "Original"],
                                [st.session_state.processed_df, "Processed"],
                                target_feat=target_col if target_col != "None" else None
                            )
                        else:
                            st.warning("No processed data available. Please preprocess data first.")
                            return
                    
                    else:  # Custom Split
                        split_idx = int(len(st.session_state.df) * split_ratio)
                        part1 = st.session_state.df.iloc[:split_idx]
                        part2 = st.session_state.df.iloc[split_idx:]
                        
                        report = sv.compare(
                            [part1, f"First {split_ratio*100:.0f}%"],
                            [part2, f"Last {(1-split_ratio)*100:.0f}%"],
                            target_feat=target_col if target_col != "None" else None
                        )
                    
                    # Save and display report
                    report.show_html("temp_sweetviz_report.html")
                    
                    with open("temp_sweetviz_report.html", "r", encoding="utf-8") as f:
                        html_content = f.read()
                    
                    st.components.v1.html(html_content, height=800, scrolling=True)
                    
                    # Download link
                    st.download_button(
                        label="📥 Download SweetViz Report (HTML)",
                        data=html_content,
                        file_name="sweetviz_report.html",
                        mime="text/html"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating SweetViz report: {str(e)}")

# Visualization Studio Page
def visualization_page():
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    st.header("📊 Visualization Studio")
    
    # Use processed data if available
    df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.df
    
    # Create tabs for different visualization types
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Distribution Plots", 
        "🔗 Relationship Plots", 
        "🥧 Composition Plots", 
        "📊 Statistical Plots", 
        "🎨 Custom Dashboard", 
        "📋 Plot Gallery"
    ])
    
    with tab1:
        st.subheader("Distribution Plots")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Column selection
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select Column", numeric_cols)
                
                # Plot type selection
                plot_type = st.selectbox(
                    "Plot Type",
                    ["histogram", "kde", "box", "violin", "qq"]
                )
                
                # Additional options
                if plot_type in ['histogram', 'kde']:
                    bins = st.slider("Number of Bins", 10, 100, 30)
                
                if st.button("Generate Plot", type="primary"):
                    with col2:
                        with st.spinner("Generating plot..."):
                            fig = Visualizations.create_distribution_plots(
                                df, selected_col, plot_type
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Save figure to session state
                            fig_name = f"distribution_{selected_col}_{plot_type}"
                            st.session_state.plotly_figs[fig_name] = fig
                            
                            # Download options
                            col_d1, col_d2, col_d3 = st.columns(3)
                            with col_d1:
                                st.markdown(download_plotly_fig(fig, fig_name, 'png'), unsafe_allow_html=True)
                            with col_d2:
                                st.markdown(download_plotly_fig(fig, fig_name, 'svg'), unsafe_allow_html=True)
                            with col_d3:
                                st.markdown(download_plotly_fig(fig, fig_name, 'pdf'), unsafe_allow_html=True)
            else:
                st.warning("No numeric columns found for distribution plots")
    
    with tab2:
        st.subheader("Relationship Plots")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Column selection
            all_cols = df.columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            x_col = st.selectbox("X-axis Column", all_cols, key="rel_x")
            y_col = st.selectbox("Y-axis Column", all_cols, key="rel_y")
            
            # Plot type selection
            plot_type = st.selectbox(
                "Plot Type",
                ["scatter", "line", "bubble", "heatmap", "pairplot"],
                key="rel_plot_type"
            )
            
            # Additional options based on plot type
            color_col = None
            size_col = None
            
            if plot_type in ['scatter', 'bubble']:
                color_col = st.selectbox(
                    "Color By (optional)",
                    ["None"] + all_cols,
                    key="rel_color"
                )
                if color_col == "None":
                    color_col = None
            
            if plot_type == 'bubble':
                size_col = st.selectbox(
                    "Size By (optional)",
                    ["None"] + numeric_cols,
                    key="rel_size"
                )
                if size_col == "None":
                    size_col = None
            
            if st.button("Generate Plot", type="primary", key="rel_btn"):
                with col2:
                    with st.spinner("Generating plot..."):
                        fig = Visualizations.create_relationship_plots(
                            df, x_col, y_col, plot_type, color_col, size_col
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Save figure to session state
                        fig_name = f"relationship_{x_col}_{y_col}_{plot_type}"
                        st.session_state.plotly_figs[fig_name] = fig
                        
                        # Download options
                        col_d1, col_d2, col_d3 = st.columns(3)
                        with col_d1:
                            st.markdown(download_plotly_fig(fig, fig_name, 'png'), unsafe_allow_html=True)
                        with col_d2:
                            st.markdown(download_plotly_fig(fig, fig_name, 'svg'), unsafe_allow_html=True)
                        with col_d3:
                            st.markdown(download_plotly_fig(fig, fig_name, 'pdf'), unsafe_allow_html=True)
    
    with tab3:
        st.subheader("Composition Plots")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Column selection
            all_cols = df.columns.tolist()
            
            selected_col = st.selectbox("Primary Column", all_cols, key="comp_primary")
            
            # Plot type selection
            plot_type = st.selectbox(
                "Plot Type",
                ["pie", "bar", "stacked_bar", "treemap", "sunburst"],
                key="comp_plot_type"
            )
            
            # Group by option
            groupby_col = st.selectbox(
                "Group By (optional)",
                ["None"] + [col for col in all_cols if col != selected_col],
                key="comp_group"
            )
            if groupby_col == "None":
                groupby_col = None
            
            if st.button("Generate Plot", type="primary", key="comp_btn"):
                with col2:
                    with st.spinner("Generating plot..."):
                        fig = Visualizations.create_composition_plots(
                            df, selected_col, plot_type, groupby_col
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Save figure to session state
                        fig_name = f"composition_{selected_col}_{plot_type}"
                        st.session_state.plotly_figs[fig_name] = fig
                        
                        # Download options
                        col_d1, col_d2, col_d3 = st.columns(3)
                        with col_d1:
                            st.markdown(download_plotly_fig(fig, fig_name, 'png'), unsafe_allow_html=True)
                        with col_d2:
                            st.markdown(download_plotly_fig(fig, fig_name, 'svg'), unsafe_allow_html=True)
                        with col_d3:
                            st.markdown(download_plotly_fig(fig, fig_name, 'pdf'), unsafe_allow_html=True)
    
    with tab4:
        st.subheader("Statistical Plots")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Correlation matrix
            st.write("### Correlation Matrix")
            if st.button("Generate Correlation Matrix", key="corr_btn"):
                fig = Visualizations.create_correlation_matrix(df)
                st.plotly_chart(fig, use_container_width=True)
                
                fig_name = "correlation_matrix"
                st.session_state.plotly_figs[fig_name] = fig
                
                # Download options
                st.markdown(download_plotly_fig(fig, fig_name, 'png'), unsafe_allow_html=True)
            
            # Missing values heatmap
            st.write("### Missing Values Heatmap")
            if st.button("Generate Missing Values Plot", key="missing_btn"):
                fig = Visualizations.create_missing_values_plot(df)
                st.plotly_chart(fig, use_container_width=True)
                
                fig_name = "missing_values_heatmap"
                st.session_state.plotly_figs[fig_name] = fig
                
                # Download options
                st.markdown(download_plotly_fig(fig, fig_name, 'png'), unsafe_allow_html=True)
        
        with col2:
            # Feature importance
            st.write("### Feature Importance")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                target_col = st.selectbox("Target Column", numeric_cols, key="feature_target")
                
                if st.button("Generate Feature Importance", key="feature_btn"):
                    fig = Visualizations.create_feature_importance_plot(df, target_col)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    fig_name = f"feature_importance_{target_col}"
                    st.session_state.plotly_figs[fig_name] = fig
                    
                    # Download options
                    st.markdown(download_plotly_fig(fig, fig_name, 'png'), unsafe_allow_html=True)
            else:
                st.warning("Need numeric columns for feature importance")
    
    with tab5:
        st.subheader("Custom Dashboard")
        
        st.info("""
        Create a custom dashboard with multiple plots. Select up to 4 plots to display together.
        """)
        
        # Get available figures from session state
        available_figs = list(st.session_state.plotly_figs.keys())
        
        if available_figs:
            # Select plots for dashboard
            selected_plots = st.multiselect(
                "Select plots for dashboard (max 4)",
                available_figs,
                max_selections=4
            )
            
            if selected_plots:
                # Create subplot layout based on number of selected plots
                n_plots = len(selected_plots)
                
                if n_plots == 1:
                    fig = st.session_state.plotly_figs[selected_plots[0]]
                    st.plotly_chart(fig, use_container_width=True)
                
                elif n_plots == 2:
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=selected_plots
                    )
                    
                    for i, plot_name in enumerate(selected_plots):
                        plot_fig = st.session_state.plotly_figs[plot_name]
                        for trace in plot_fig.data:
                            fig.add_trace(trace, row=1, col=i+1)
                    
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif n_plots == 3:
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=selected_plots + [''] if n_plots == 3 else selected_plots
                    )
                    
                    positions = [(1,1), (1,2), (2,1)]
                    for i, plot_name in enumerate(selected_plots[:3]):
                        plot_fig = st.session_state.plotly_figs[plot_name]
                        for trace in plot_fig.data:
                            fig.add_trace(trace, row=positions[i][0], col=positions[i][1])
                    
                    fig.update_layout(height=800)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif n_plots == 4:
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=selected_plots
                    )
                    
                    positions = [(1,1), (1,2), (2,1), (2,2)]
                    for i, plot_name in enumerate(selected_plots):
                        plot_fig = st.session_state.plotly_figs[plot_name]
                        for trace in plot_fig.data:
                            fig.add_trace(trace, row=positions[i][0], col=positions[i][1])
                    
                    fig.update_layout(height=800)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download dashboard
                dashboard_name = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    st.markdown(download_plotly_fig(fig, dashboard_name, 'png'), unsafe_allow_html=True)
                with col_d2:
                    st.markdown(download_plotly_fig(fig, dashboard_name, 'svg'), unsafe_allow_html=True)
                with col_d3:
                    st.markdown(download_plotly_fig(fig, dashboard_name, 'pdf'), unsafe_allow_html=True)
        else:
            st.info("Generate some plots first to create a dashboard")
    
    with tab6:
        st.subheader("Plot Gallery")
        
        # Display all generated plots
        if st.session_state.plotly_figs:
            st.write(f"**Total Plots Generated:** {len(st.session_state.plotly_figs)}")
            
            for plot_name, fig in st.session_state.plotly_figs.items():
                with st.expander(f"📊 {plot_name}"):
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download options for each plot
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(download_plotly_fig(fig, plot_name, 'png'), unsafe_allow_html=True)
                    with col2:
                        st.markdown(download_plotly_fig(fig, plot_name, 'svg'), unsafe_allow_html=True)
                    with col3:
                        st.markdown(download_plotly_fig(fig, plot_name, 'pdf'), unsafe_allow_html=True)
        else:
            st.info("No plots generated yet. Create some visualizations first!")

# Time Series Analysis Page
def timeseries_page():
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.df
    
    st.header("⏰ Time Series Analysis")
    
    # Detect date columns
    date_cols = detect_date_columns(df)
    
    if not date_cols:
        st.warning("No date columns detected. Time series analysis requires datetime columns.")
        
        # Try to convert potential date columns
        st.info("Try converting columns to datetime:")
        potential_date_cols = st.multiselect(
            "Select columns that might contain dates",
            df.columns.tolist()
        )
        
        if potential_date_cols and st.button("Convert to Datetime"):
            for col in potential_date_cols:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    st.success(f"Converted {col} to datetime")
                except:
                    st.error(f"Failed to convert {col}")
            
            # Update session state
            st.session_state.df = df
            if st.session_state.processed_df is not None:
                st.session_state.processed_df = df
            
            st.rerun()
        
        return
    
    # Create tabs for time series analysis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📅 Basic Analysis", 
        "📈 Decomposition", 
        "📊 Stationarity", 
        "🔍 ACF/PACF", 
        "🎯 Forecasting Prep"
    ])
    
    with tab1:
        st.subheader("Basic Time Series Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            date_col = st.selectbox("Date Column", date_cols, key="ts_date")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            value_col = st.selectbox("Value Column", numeric_cols, key="ts_value")
        
        with col2:
            plot_type = st.selectbox(
                "Plot Type",
                ["line", "area", "rolling", "seasonal"],
                key="ts_plot_type"
            )
            
            if plot_type == 'rolling':
                window = st.slider("Rolling Window", 7, 365, 30)
        
        if st.button("Generate Time Series Plot", type="primary", key="ts_btn"):
            with st.spinner("Generating plot..."):
                if plot_type == 'rolling':
                    fig = TimeSeriesAnalyzer.create_rolling_statistics_plot(
                        df, date_col, value_col, window
                    )
                else:
                    fig = Visualizations.create_time_series_plots(
                        df, date_col, value_col, plot_type
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Save figure
                fig_name = f"timeseries_{value_col}_{plot_type}"
                st.session_state.plotly_figs[fig_name] = fig
                
                # Download options
                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    st.markdown(download_plotly_fig(fig, fig_name, 'png'), unsafe_allow_html=True)
                with col_d2:
                    st.markdown(download_plotly_fig(fig, fig_name, 'svg'), unsafe_allow_html=True)
                with col_d3:
                    st.markdown(download_plotly_fig(fig, fig_name, 'pdf'), unsafe_allow_html=True)
        
        # Basic statistics
        if date_col and value_col:
            st.subheader("Time Series Statistics")
            
            # Create time series
            ts_df = df[[date_col, value_col]].copy()
            ts_df = ts_df.sort_values(date_col)
            ts_df = ts_df.set_index(date_col)[value_col]
            
            # Calculate statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Start Date", ts_df.index.min().strftime('%Y-%m-%d'))
                st.metric("End Date", ts_df.index.max().strftime('%Y-%m-%d'))
            
            with col2:
                st.metric("Duration (days)", (ts_df.index.max() - ts_df.index.min()).days)
                st.metric("Frequency", pd.infer_freq(ts_df.index) or "Irregular")
            
            with col3:
                st.metric("Mean", f"{ts_df.mean():.2f}")
                st.metric("Std Dev", f"{ts_df.std():.2f}")
            
            with col4:
                st.metric("Min", f"{ts_df.min():.2f}")
                st.metric("Max", f"{ts_df.max():.2f}")
    
    with tab2:
        st.subheader("Seasonal Decomposition")
        
        col1, col2 = st.columns(2)
        
        with col1:
            date_col = st.selectbox("Date Column", date_cols, key="decomp_date")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            value_col = st.selectbox("Value Column", numeric_cols, key="decomp_value")
        
        with col2:
            model_type = st.selectbox(
                "Decomposition Model",
                ["additive", "multiplicative"],
                help="Additive: trend + seasonal + residual. Multiplicative: trend * seasonal * residual."
            )
            
            period = st.number_input(
                "Seasonal Period",
                min_value=2,
                max_value=365,
                value=30,
                help="Number of periods in a season (e.g., 7 for weekly, 30 for monthly)"
            )
        
        if st.button("Decompose Time Series", type="primary", key="decomp_btn"):
            with st.spinner("Performing decomposition..."):
                # Prepare time series
                ts_df = df[[date_col, value_col]].copy()
                ts_df = ts_df.sort_values(date_col)
                ts_df = ts_df.set_index(date_col)[value_col]
                
                # Fill missing dates and values
                ts_df = ts_df.asfreq('D')
                ts_df = ts_df.interpolate()
                
                # Perform decomposition
                try:
                    fig = TimeSeriesAnalyzer.create_decomposition_plot(ts_df, period, model_type)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save figure
                    fig_name = f"decomposition_{value_col}_{model_type}"
                    st.session_state.plotly_figs[fig_name] = fig
                    
                    # Download options
                    col_d1, col_d2, col_d3 = st.columns(3)
                    with col_d1:
                        st.markdown(download_plotly_fig(fig, fig_name, 'png'), unsafe_allow_html=True)
                    with col_d2:
                        st.markdown(download_plotly_fig(fig, fig_name, 'svg'), unsafe_allow_html=True)
                    with col_d3:
                        st.markdown(download_plotly_fig(fig, fig_name, 'pdf'), unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Decomposition failed: {str(e)}")
    
    with tab3:
        st.subheader("Stationarity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            date_col = st.selectbox("Date Column", date_cols, key="stat_date")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            value_col = st.selectbox("Value Column", numeric_cols, key="stat_value")
        
        with col2:
            diff_order = st.slider(
                "Differencing Order",
                0, 3, 0,
                help="Apply differencing to make series stationary"
            )
        
        if st.button("Test Stationarity", type="primary", key="stat_btn"):
            with st.spinner("Testing stationarity..."):
                # Prepare time series
                ts_df = df[[date_col, value_col]].copy()
                ts_df = ts_df.sort_values(date_col)
                ts_df = ts_df.set_index(date_col)[value_col]
                
                # Apply differencing if requested
                if diff_order > 0:
                    for i in range(diff_order):
                        ts_df = ts_df.diff().dropna()
                
                # Test stationarity
                results = TimeSeriesAnalyzer.analyze_stationarity(ts_df, value_col)
                
                # Display results
                st.subheader("ADF Test Results")
                adf_results = results.get('adf', {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ADF Statistic", f"{adf_results.get('statistic', 0):.4f}")
                with col2:
                    p_value = adf_results.get('p_value', 1)
                    st.metric("p-value", f"{p_value:.6f}")
                with col3:
                    is_stationary = adf_results.get('is_stationary', False)
                    st.metric("Stationary", "✅ Yes" if is_stationary else "❌ No")
                
                # Critical values
                st.write("**Critical Values:**")
                crit_vals = adf_results.get('critical_values', {})
                crit_df = pd.DataFrame(list(crit_vals.items()), columns=['Significance', 'Value'])
                st.dataframe(crit_df, use_container_width=True)
                
                # KPSS Test results if available
                kpss_results = results.get('kpss', {})
                if 'error' not in kpss_results:
                    st.subheader("KPSS Test Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("KPSS Statistic", f"{kpss_results.get('statistic', 0):.4f}")
                    with col2:
                        kpss_p = kpss_results.get('p_value', 1)
                        st.metric("p-value", f"{kpss_p:.6f}")
                    with col3:
                        kpss_stationary = kpss_results.get('is_stationary', False)
                        st.metric("Stationary", "✅ Yes" if kpss_stationary else "❌ No")
                
                # Interpretation
                st.subheader("Interpretation")
                if adf_results.get('is_stationary', False):
                    st.success("""
                    **✅ Series is Stationary**
                    
                    The time series appears to have constant statistical properties over time.
                    This is good for many time series models.
                    """)
                else:
                    st.warning("""
                    **⚠️ Series is Non-Stationary**
                    
                    Consider:
                    1. Applying differencing (try increasing differencing order)
                    2. Applying transformations (log, sqrt)
                    3. Using models that handle non-stationarity
                    """)
    
    with tab4:
        st.subheader("ACF and PACF Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            date_col = st.selectbox("Date Column", date_cols, key="acf_date")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            value_col = st.selectbox("Value Column", numeric_cols, key="acf_value")
        
        with col2:
            lags = st.slider("Number of Lags", 10, 100, 40)
            diff_order = st.slider(
                "Differencing Order",
                0, 3, 0,
                key="acf_diff",
                help="Apply differencing before ACF/PACF"
            )
        
        if st.button("Generate ACF/PACF Plots", type="primary", key="acf_btn"):
            with st.spinner("Calculating autocorrelations..."):
                # Prepare time series
                ts_df = df[[date_col, value_col]].copy()
                ts_df = ts_df.sort_values(date_col)
                ts_df = ts_df.set_index(date_col)[value_col]
                
                # Apply differencing if requested
                if diff_order > 0:
                    for i in range(diff_order):
                        ts_df = ts_df.diff().dropna()
                
                # Generate plots
                acf_fig, pacf_fig = TimeSeriesAnalyzer.create_acf_pacf_plots(ts_df, lags)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(acf_fig, use_container_width=True)
                    
                    # Save figure
                    fig_name = f"acf_{value_col}_lags{lags}"
                    st.session_state.plotly_figs[fig_name] = acf_fig
                    
                    # Download options
                    st.markdown(download_plotly_fig(acf_fig, fig_name, 'png'), unsafe_allow_html=True)
                
                with col2:
                    st.plotly_chart(pacf_fig, use_container_width=True)
                    
                    # Save figure
                    fig_name = f"pacf_{value_col}_lags{lags}"
                    st.session_state.plotly_figs[fig_name] = pacf_fig
                    
                    # Download options
                    st.markdown(download_plotly_fig(pacf_fig, fig_name, 'png'), unsafe_allow_html=True)
                
                # Interpretation guide
                st.subheader("Interpretation Guide")
                st.info("""
                **ACF (Autocorrelation Function):**
                - Shows correlation between series and its lags
                - Slow decay suggests non-stationarity
                - Significant spikes at seasonal lags indicate seasonality
                
                **PACF (Partial Autocorrelation Function):**
                - Shows direct correlation between series and its lags
                - Helps identify AR (AutoRegressive) order in ARIMA models
                - Cutoff after p lags suggests AR(p) process
                
                **For ARIMA modeling:**
                - AR order (p): Look at PACF cutoff
                - MA order (q): Look at ACF cutoff
                - Differencing (d): Use ADF test results
                """)
    
    with tab5:
        st.subheader("Forecasting Preparation")
        
        st.info("""
        This section helps prepare your data for forecasting models.
        Select features that might be useful for predicting your target variable.
        """)
        
        # Feature selection for forecasting
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            date_col = st.selectbox("Date Column", date_cols, key="forecast_date")
            target_col = st.selectbox("Target Column", numeric_cols, key="forecast_target")
        
        with col2:
            # Feature engineering options
            st.write("**Feature Engineering Options:**")
            
            create_lags = st.checkbox("Create Lag Features", value=True)
            if create_lags:
                lag_periods = st.multiselect(
                    "Lag Periods",
                    [1, 2, 3, 7, 14, 30, 90],
                    default=[1, 7, 30]
                )
            
            create_rolling = st.checkbox("Create Rolling Statistics", value=True)
            if create_rolling:
                rolling_windows = st.multiselect(
                    "Rolling Windows",
                    [3, 7, 14, 30, 60, 90],
                    default=[7, 30]
                )
            
            create_seasonal = st.checkbox("Create Seasonal Features", value=True)
        
        if st.button("Prepare for Forecasting", type="primary", key="forecast_btn"):
            with st.spinner("Preparing data for forecasting..."):
                # Create a copy for forecasting preparation
                forecast_df = df.copy()
                
                # Ensure date column is datetime
                forecast_df[date_col] = pd.to_datetime(forecast_df[date_col])
                forecast_df = forecast_df.sort_values(date_col)
                
                # Create lag features
                if create_lags and lag_periods:
                    for lag in lag_periods:
                        forecast_df[f'lag_{lag}'] = forecast_df[target_col].shift(lag)
                        st.success(f"Created lag_{lag} feature")
                
                # Create rolling statistics
                if create_rolling and rolling_windows:
                    for window in rolling_windows:
                        forecast_df[f'rolling_mean_{window}'] = forecast_df[target_col].rolling(window=window).mean()
                        forecast_df[f'rolling_std_{window}'] = forecast_df[target_col].rolling(window=window).std()
                        st.success(f"Created rolling statistics for window {window}")
                
                # Create seasonal features
                if create_seasonal:
                    forecast_df['year'] = forecast_df[date_col].dt.year
                    forecast_df['month'] = forecast_df[date_col].dt.month
                    forecast_df['day'] = forecast_df[date_col].dt.day
                    forecast_df['dayofweek'] = forecast_df[date_col].dt.dayofweek
                    forecast_df['quarter'] = forecast_df[date_col].dt.quarter
                    
                    st.success("Created seasonal features (year, month, day, dayofweek, quarter)")
                
                # Display prepared data
                st.subheader("Prepared Data Preview")
                st.dataframe(forecast_df.head(), use_container_width=True)
                
                # Statistics
                st.subheader("Prepared Data Statistics")
                st.write(f"**Original columns:** {len(df.columns)}")
                st.write(f"**New columns after feature engineering:** {len(forecast_df.columns)}")
                st.write(f"**New features created:** {len(forecast_df.columns) - len(df.columns)}")
                
                # Save prepared data to session state
                st.session_state.forecast_df = forecast_df
                
                # Download prepared data
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Prepared Data (CSV)",
                    data=csv,
                    file_name="forecasting_prepared_data.csv",
                    mime="text/csv"
                )

# ML Insights Page
def ml_page():
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.df
    
    st.header("🤖 Machine Learning Insights")
    
    # Create tabs for different ML analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Feature Importance", 
        "📊 Multicollinearity", 
        "🔍 Outlier Detection", 
        "📈 Clustering", 
        "⚙️ Preprocessing Suggestions"
    ])
    
    with tab1:
        st.subheader("Feature Importance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Target selection
            all_cols = df.columns.tolist()
            target_col = st.selectbox(
                "Select Target Column",
                all_cols,
                key="feature_target"
            )
            
            # Method selection
            method = st.selectbox(
                "Importance Method",
                ["random_forest", "mutual_info"],
                help="Random Forest: Tree-based importance. Mutual Info: Information theory based."
            )
        
        with col2:
            # Display target information
            if target_col:
                target_data = df[target_col]
                st.write("**Target Information:**")
                st.write(f"**Data Type:** {target_data.dtype}")
                st.write(f"**Missing Values:** {target_data.isnull().sum()} ({target_data.isnull().sum()/len(target_data)*100:.2f}%)")
                
                if pd.api.types.is_numeric_dtype(target_data):
                    st.write(f"**Range:** {target_data.min():.2f} to {target_data.max():.2f}")
                    st.write(f"**Mean:** {target_data.mean():.2f}")
                    st.write("**Problem Type:** Regression")
                else:
                    st.write(f"**Unique Values:** {target_data.nunique()}")
                    st.write(f"**Most Common:** {target_data.mode().iloc[0] if not target_data.mode().empty else 'N/A'}")
                    st.write("**Problem Type:** Classification")
        
        if st.button("Analyze Feature Importance", type="primary", key="feature_btn"):
            with st.spinner("Calculating feature importance..."):
                try:
                    # Get feature importance
                    importance_df, fig = MLAnalyzer.analyze_feature_importance(
                        df, target_col, method
                    )
                    
                    # Display results
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save figure
                    fig_name = f"feature_importance_{target_col}_{method}"
                    st.session_state.plotly_figs[fig_name] = fig
                    
                    # Download options
                    col_d1, col_d2, col_d3 = st.columns(3)
                    with col_d1:
                        st.markdown(download_plotly_fig(fig, fig_name, 'png'), unsafe_allow_html=True)
                    with col_d2:
                        st.markdown(download_plotly_fig(fig, fig_name, 'svg'), unsafe_allow_html=True)
                    with col_d3:
                        st.markdown(download_plotly_fig(fig, fig_name, 'pdf'), unsafe_allow_html=True)
                    
                    # Show top features table
                    st.subheader("Top 10 Important Features")
                    st.dataframe(importance_df.head(10), use_container_width=True)
                    
                    # Download importance data
                    csv = importance_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Feature Importance (CSV)",
                        data=csv,
                        file_name=f"feature_importance_{target_col}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error calculating feature importance: {str(e)}")
    
    with tab2:
        st.subheader("Multicollinearity Analysis (VIF)")
        
        st.info("""
        **Variance Inflation Factor (VIF)** measures multicollinearity:
        - **VIF < 5**: Low correlation (OK)
        - **5 ≤ VIF ≤ 10**: Moderate correlation (Consider removing)
        - **VIF > 10**: High correlation (Remove feature)
        """)
        
        if st.button("Calculate VIF", type="primary", key="vif_btn"):
            with st.spinner("Calculating VIF..."):
                try:
                    vif_data, fig = MLAnalyzer.analyze_multicollinearity(df)
                    
                    if not vif_data.empty:
                        # Display results
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Save figure
                        fig_name = "multicollinearity_vif"
                        st.session_state.plotly_figs[fig_name] = fig
                        
                        # Download options
                        col_d1, col_d2, col_d3 = st.columns(3)
                        with col_d1:
                            st.markdown(download_plotly_fig(fig, fig_name, 'png'), unsafe_allow_html=True)
                        with col_d2:
                            st.markdown(download_plotly_fig(fig, fig_name, 'svg'), unsafe_allow_html=True)
                        with col_d3:
                            st.markdown(download_plotly_fig(fig, fig_name, 'pdf'), unsafe_allow_html=True)
                        
                        # Show features with high VIF
                        high_vif = vif_data[vif_data['VIF'] > 5]
                        if not high_vif.empty:
                            st.warning(f"**{len(high_vif)} features with VIF > 5 detected:**")
                            st.dataframe(high_vif.sort_values('VIF', ascending=False), use_container_width=True)
                            
                            # Suggestions for handling high VIF
                            st.subheader("Suggestions for High VIF Features:")
                            st.info("""
                            1. **Remove one of the correlated features**
                            2. **Use PCA** to create uncorrelated components
                            3. **Apply regularization** (Lasso/Ridge regression)
                            4. **Combine correlated features** into a single feature
                            """)
                        else:
                            st.success("✅ No features with problematic multicollinearity detected!")
                        
                        # Download VIF data
                        csv = vif_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download VIF Results (CSV)",
                            data=csv,
                            file_name="vif_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Not enough numeric columns for VIF calculation")
                        
                except Exception as e:
                    st.error(f"Error calculating VIF: {str(e)}")
    
    with tab3:
        st.subheader("Outlier Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            method = st.selectbox(
                "Outlier Detection Method",
                ["iqr", "zscore"],
                help="IQR: Uses interquartile range. Z-score: Uses standard deviations."
            )
        
        with col2:
            # Select columns for outlier detection
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_cols = st.multiselect(
                "Select Columns to Analyze",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
        
        if st.button("Detect Outliers", type="primary", key="outlier_btn"):
            with st.spinner("Detecting outliers..."):
                if selected_cols:
                    # Analyze selected columns
                    results = MLAnalyzer.detect_outliers(df[selected_cols], method)
                    
                    if results:
                        # Create summary table
                        summary_data = []
                        for col, stats in results.items():
                            summary_data.append({
                                'Column': col,
                                'Outliers': stats['outlier_count'],
                                'Percentage': f"{stats['outlier_percentage']:.2f}%",
                                'Status': '⚠️ High' if stats['outlier_percentage'] > 5 else '✅ Normal'
                            })
                        
                        summary_df = pd.DataFrame(summary_data)
                        
                        # Display summary
                        st.subheader("Outlier Summary")
                        st.dataframe(summary_df, use_container_width=True)
                        
                        # Visualize outliers for each column
                        for col, stats in results.items():
                            if stats['outlier_count'] > 0:
                                with st.expander(f"Outliers in {col}"):
                                    # Create box plot
                                    fig = px.box(df, y=col, title=f"Outliers in {col}")
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Show outlier values
                                    st.write(f"**{stats['outlier_count']} outliers detected ({stats['outlier_percentage']:.2f}%)**")
                                    
                                    if len(stats['outlier_values']) <= 20:
                                        st.write("**Outlier values:**", stats['outlier_values'])
                                    
                                    # Download outlier data for this column
                                    outlier_df = df[df[col].isin(stats['outlier_values'])][[col]]
                                    csv = outlier_df.to_csv()
                                    st.download_button(
                                        label=f"📥 Download Outliers for {col}",
                                        data=csv,
                                        file_name=f"outliers_{col}.csv",
                                        mime="text/csv",
                                        key=f"outlier_dl_{col}"
                                    )
                        
                        # Suggestions for handling outliers
                        st.subheader("Handling Outliers Suggestions:")
                        st.info("""
                        **For moderate outliers (< 5%):**
                        1. **Keep them** if they represent valid data points
                        2. **Transform data** (log, square root)
                        3. **Use robust models** (Random Forest, SVM)
                        
                        **For high outliers (> 5%):**
                        1. **Investigate** if they are data errors
                        2. **Cap/winsorize** extreme values
                        3. **Remove** if they are errors and small in number
                        4. **Use models** that are outlier-resistant
                        """)
                        
                        # Download complete outlier report
                        report_data = []
                        for col, stats in results.items():
                            report_data.append({
                                'column': col,
                                'outlier_count': stats['outlier_count'],
                                'outlier_percentage': stats['outlier_percentage'],
                                'method': method
                            })
                        
                        report_df = pd.DataFrame(report_data)
                        csv = report_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Outlier Report",
                            data=csv,
                            file_name="outlier_report.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No outliers detected with selected method")
                else:
                    st.warning("Please select at least one column")
    
    with tab4:
        st.subheader("Clustering Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Number of clusters
            n_clusters = st.slider(
                "Number of Clusters",
                min_value=2,
                max_value=10,
                value=3
            )
            
            # Clustering method
            method = st.selectbox(
                "Clustering Method",
                ["kmeans"],
                disabled=True,
                help="K-means clustering with PCA visualization"
            )
        
        with col2:
            # Feature selection for clustering
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            clustering_features = st.multiselect(
                "Select Features for Clustering",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
        
        if st.button("Perform Clustering", type="primary", key="cluster_btn"):
            with st.spinner("Performing clustering analysis..."):
                if len(clustering_features) >= 2:
                    try:
                        # Perform clustering
                        results_df, fig, cluster_stats, silhouette_score = MLAnalyzer.analyze_clustering(
                            df[clustering_features], n_clusters
                        )
                        
                        # Display results
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Save figure
                        fig_name = f"clustering_{n_clusters}_clusters"
                        st.session_state.plotly_figs[fig_name] = fig
                        
                        # Download options
                        col_d1, col_d2, col_d3 = st.columns(3)
                        with col_d1:
                            st.markdown(download_plotly_fig(fig, fig_name, 'png'), unsafe_allow_html=True)
                        with col_d2:
                            st.markdown(download_plotly_fig(fig, fig_name, 'svg'), unsafe_allow_html=True)
                        with col_d3:
                            st.markdown(download_plotly_fig(fig, fig_name, 'pdf'), unsafe_allow_html=True)
                        
                        # Display clustering statistics
                        st.subheader("Clustering Performance")
                        st.metric("Silhouette Score", f"{silhouette_score:.3f}")
                        
                        # Interpretation of silhouette score
                        if silhouette_score > 0.7:
                            st.success("**Strong structure** - Clusters are well separated")
                        elif silhouette_score > 0.5:
                            st.info("**Reasonable structure** - Clusters are somewhat separated")
                        elif silhouette_score > 0.25:
                            st.warning("**Weak structure** - Clusters are not well separated")
                        else:
                            st.error("**No substantial structure** - Data may not have natural clusters")
                        
                        # Cluster statistics
                        st.subheader("Cluster Statistics")
                        for cluster, stats in cluster_stats.items():
                            with st.expander(f"{cluster} - {stats['size']} samples ({stats['percentage']:.1f}%)"):
                                # Create metrics for this cluster
                                cols = st.columns(4)
                                for idx, (feat, mean_val) in enumerate(list(stats['means'].items())[:4]):
                                    with cols[idx]:
                                        st.metric(feat, f"{mean_val:.2f}")
                                
                                # Show all means in a table
                                means_df = pd.DataFrame({
                                    'Feature': list(stats['means'].keys()),
                                    'Mean': list(stats['means'].values()),
                                    'Std': list(stats['stds'].values())
                                })
                                st.dataframe(means_df, use_container_width=True)
                        
                        # Download clustering results
                        clustering_results = df.copy()
                        clustering_results['Cluster'] = results_df['Cluster']
                        csv = clustering_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Clustering Results",
                            data=csv,
                            file_name="clustering_results.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error performing clustering: {str(e)}")
                else:
                    st.warning("Please select at least 2 features for clustering")
    
    with tab5:
        st.subheader("Preprocessing Suggestions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_col = st.selectbox(
                "Target Column (for suggestions)",
                ["None"] + df.columns.tolist(),
                key="preproc_target"
            )
        
        with col2:
            detail_level = st.select_slider(
                "Detail Level",
                options=["Basic", "Detailed", "Comprehensive"]
            )
        
        if st.button("Generate Suggestions", type="primary", key="preproc_btn"):
            with st.spinner("Analyzing data for preprocessing suggestions..."):
                try:
                    # Get preprocessing suggestions
                    suggestions = MLAnalyzer.suggest_preprocessing(
                        df, 
                        target_col if target_col != "None" else None
                    )
                    
                    # Display suggestions
                    st.subheader("📋 Preprocessing Recommendations")
                    
                    # Missing Values
                    if suggestions['missing_values']:
                        st.write("### 🚨 Missing Values")
                        missing_df = pd.DataFrame.from_dict(
                            suggestions['missing_values'], 
                            orient='index'
                        ).reset_index()
                        missing_df.columns = ['Column', 'Missing Count', 'Missing %', 'Suggestion']
                        st.dataframe(missing_df, use_container_width=True)
                    else:
                        st.success("✅ No missing values detected!")
                    
                    # Encoding Suggestions
                    if suggestions['encoding']:
                        st.write("### 🔤 Encoding Suggestions")
                        encoding_df = pd.DataFrame.from_dict(
                            suggestions['encoding'], 
                            orient='index'
                        ).reset_index()
                        encoding_df.columns = ['Column', 'Unique Values', 'Suggestion']
                        st.dataframe(encoding_df, use_container_width=True)
                    
                    # Scaling Suggestions
                    if suggestions['scaling']:
                        st.write("### ⚖️ Scaling Suggestions")
                        scaling_df = pd.DataFrame.from_dict(
                            suggestions['scaling'], 
                            orient='index'
                        ).reset_index()
                        scaling_df.columns = ['Column', 'CV', 'Suggestion']
                        st.dataframe(scaling_df, use_container_width=True)
                    
                    # Transformation Suggestions
                    if suggestions['transformations']:
                        st.write("### 📊 Transformation Suggestions")
                        transform_df = pd.DataFrame.from_dict(
                            suggestions['transformations'], 
                            orient='index'
                        ).reset_index()
                        transform_df.columns = ['Column', 'Skewness', 'Suggestion']
                        st.dataframe(transform_df, use_container_width=True)
                    
                    # Feature Engineering
                    if suggestions['feature_engineering']:
                        st.write("### 🛠️ Feature Engineering Ideas")
                        for idea in suggestions['feature_engineering']:
                            st.info(f"**{idea['type'].replace('_', ' ').title()}**: {idea['suggestion']}")
                    
                    # Download suggestions
                    import json
                    suggestions_json = json.dumps(suggestions, indent=2)
                    st.download_button(
                        label="📥 Download Preprocessing Suggestions (JSON)",
                        data=suggestions_json,
                        file_name="preprocessing_suggestions.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating suggestions: {str(e)}")

# Export Center Page
def export_page():
    st.header("📥 Export Center")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    # Create tabs for different export types
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Visualizations", 
        "📈 Reports", 
        "💾 Data", 
        "📋 Summary"
    ])
    
    with tab1:
        st.subheader("Export Visualizations")
        
        if st.session_state.plotly_figs:
            st.write(f"**Total plots available for export:** {len(st.session_state.plotly_figs)}")
            
            # Batch export options
            st.write("### Batch Export")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                export_format = st.selectbox(
                    "Export Format",
                    ["PNG", "SVG", "PDF"]
                )
            
            with col2:
                export_quality = st.slider(
                    "Quality/DPI",
                    min_value=72,
                    max_value=300,
                    value=150,
                    help="Higher DPI for better print quality"
                )
            
            with col3:
                if st.button("🚀 Export All Plots", type="primary"):
                    with st.spinner(f"Exporting {len(st.session_state.plotly_figs)} plots..."):
                        # Create zip file with all plots
                        import zipfile
                        import io
                        
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for plot_name, fig in st.session_state.plotly_figs.items():
                                # Convert plot to image
                                img_bytes = fig.to_image(
                                    format=export_format.lower(),
                                    width=1200,
                                    height=800,
                                    scale=export_quality/72
                                )
                                
                                # Add to zip
                                zip_file.writestr(f"{plot_name}.{export_format.lower()}", img_bytes)
                        
                        # Offer download
                        zip_buffer.seek(0)
                        st.download_button(
                            label=f"📥 Download All Plots ({export_format})",
                            data=zip_buffer,
                            file_name=f"all_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip"
                        )
            
            # Individual plot export
            st.write("### Individual Plot Export")
            plot_list = list(st.session_state.plotly_figs.keys())
            
            if plot_list:
                selected_plot = st.selectbox("Select plot to export", plot_list)
                
                if selected_plot:
                    fig = st.session_state.plotly_figs[selected_plot]
                    
                    # Display plot
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Export options
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(download_plotly_fig(fig, selected_plot, 'png'), unsafe_allow_html=True)
                    with col2:
                        st.markdown(download_plotly_fig(fig, selected_plot, 'svg'), unsafe_allow_html=True)
                    with col3:
                        st.markdown(download_plotly_fig(fig, selected_plot, 'pdf'), unsafe_allow_html=True)
                    with col4:
                        # Export as HTML (interactive)
                        html_bytes = fig.to_html().encode()
                        b64 = base64.b64encode(html_bytes).decode()
                        href = f'<a href="data:text/html;base64,{b64}" download="{selected_plot}.html">Download HTML</a>'
                        st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("No plots generated yet. Create visualizations first!")
    
    with tab2:
        st.subheader("Export Reports")
        
        st.info("Generate comprehensive reports of your analysis.")
        
        # Report options
        report_type = st.selectbox(
            "Report Type",
            ["Data Summary", "EDA Report", "ML Insights", "Complete Analysis"]
        )
        
        report_format = st.selectbox(
            "Format",
            ["HTML", "PDF", "Markdown"]
        )
        
        if st.button("📄 Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                try:
                    # Create report content
                    report_content = generate_report(report_type)
                    
                    if report_format == "HTML":
                        # Create HTML report
                        html_report = create_html_report(report_content)
                        
                        st.success("✅ HTML report generated!")
                        st.download_button(
                            label="📥 Download HTML Report",
                            data=html_report,
                            file_name=f"{report_type.replace(' ', '_')}_report.html",
                            mime="text/html"
                        )
                        
                        # Preview
                        with st.expander("Preview Report"):
                            st.components.v1.html(html_report, height=600, scrolling=True)
                    
                    elif report_format == "PDF":
                        st.warning("PDF export requires additional libraries. Using HTML instead.")
                        html_report = create_html_report(report_content)
                        st.download_button(
                            label="📥 Download HTML Report (PDF not available)",
                            data=html_report,
                            file_name=f"{report_type.replace(' ', '_')}_report.html",
                            mime="text/html"
                        )
                    
                    else:  # Markdown
                        st.text_area("Markdown Report", report_content, height=400)
                        st.download_button(
                            label="📥 Download Markdown Report",
                            data=report_content,
                            file_name=f"{report_type.replace(' ', '_')}_report.md",
                            mime="text/markdown"
                        )
                        
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
    
    with tab3:
        st.subheader("Export Data")
        
        st.info("Export your original or processed data in various formats.")
        
        # Data selection
        data_options = ["Original Data", "Processed Data"]
        if hasattr(st.session_state, 'forecast_df'):
            data_options.append("Forecasting Prepared Data")
        
        selected_data = st.selectbox("Select dataset to export", data_options)
        
        # Get the selected dataframe
        if selected_data == "Original Data":
            export_df = st.session_state.df
        elif selected_data == "Processed Data":
            export_df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.df
        else:
            export_df = st.session_state.forecast_df
        
        if export_df is not None:
            # Show preview
            st.write(f"**Dataset:** {selected_data}")
            st.write(f"**Shape:** {export_df.shape[0]} rows × {export_df.shape[1]} columns")
            st.dataframe(export_df.head(10), use_container_width=True)
            
            # Export formats
            st.write("### Export Formats")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # CSV
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 CSV",
                    data=csv,
                    file_name=f"{selected_data.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    export_df.to_excel(writer, index=False, sheet_name='Data')
                excel_data = output.getvalue()
                st.download_button(
                    label="📥 Excel",
                    data=excel_data,
                    file_name=f"{selected_data.replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col3:
                # JSON
                json_str = export_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 JSON",
                    data=json_str,
                    file_name=f"{selected_data.replace(' ', '_')}.json",
                    mime="application/json"
                )
            
            with col4:
                # Parquet
                try:
                    parquet_buffer = io.BytesIO()
                    export_df.to_parquet(parquet_buffer, index=False)
                    parquet_data = parquet_buffer.getvalue()
                    st.download_button(
                        label="📥 Parquet",
                        data=parquet_data,
                        file_name=f"{selected_data.replace(' ', '_')}.parquet",
                        mime="application/octet-stream"
                    )
                except:
                    st.info("Parquet requires pyarrow")
        
        else:
            st.warning(f"{selected_data} is not available")
    
    with tab4:
        st.subheader("Project Summary")
        
        # Generate summary statistics
        summary_data = generate_summary()
        
        # Display summary
        st.json(summary_data)
        
        # Download summary
        summary_json = json.dumps(summary_data, indent=2, default=str)
        st.download_button(
            label="📥 Download Project Summary (JSON)",
            data=summary_json,
            file_name="project_summary.json",
            mime="application/json"
        )
        
        # Quick statistics
        st.write("### 📊 Quick Statistics")
        
        if st.session_state.df is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Dataset Size", f"{len(st.session_state.df):,} rows")
            with col2:
                st.metric("Features", len(st.session_state.df.columns))
            with col3:
                plots_count = len(st.session_state.plotly_figs)
                st.metric("Plots Generated", plots_count)
            with col4:
                if st.session_state.file_info:
                    size_str = get_file_size_str(st.session_state.file_info.get('size', 0))
                    st.metric("File Size", size_str)

def generate_report(report_type):
    """Generate report content based on type."""
    report = f"# DataInsight Pro - {report_type} Report\n\n"
    report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    if st.session_state.df is not None:
        report += f"## Dataset Information\n\n"
        report += f"- **Rows:** {len(st.session_state.df):,}\n"
        report += f"- **Columns:** {len(st.session_state.df.columns)}\n"
        report += f"- **File:** {st.session_state.file_info.get('name', 'N/A') if st.session_state.file_info else 'N/A'}\n\n"
        
        if report_type in ["EDA Report", "Complete Analysis"]:
            report += "## Data Types\n\n"
            dtype_counts = st.session_state.df.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                report += f"- {dtype}: {count} columns\n"
            report += "\n"
        
        if report_type in ["ML Insights", "Complete Analysis"]:
            report += "## Missing Values Summary\n\n"
            missing_total = st.session_state.df.isnull().sum().sum()
            missing_pct = (missing_total / (len(st.session_state.df) * len(st.session_state.df.columns)) * 100)
            report += f"- **Total Missing Values:** {missing_total:,}\n"
            report += f"- **Missing Percentage:** {missing_pct:.2f}%\n\n"
        
        if report_type == "Complete Analysis":
            report += "## Analysis Summary\n\n"
            report += f"- **Plots Generated:** {len(st.session_state.plotly_figs)}\n"
            if hasattr(st.session_state, 'processed_df'):
                report += "- **Data Preprocessing:** Applied\n"
            report += "\n"
    
    return report

def create_html_report(report_content):
    """Create HTML report from content."""
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>DataInsight Pro Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #1f77b4; }}
            h2 {{ color: #2ca02c; }}
            .summary {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .footer {{ margin-top: 50px; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 DataInsight Pro Analysis Report</h1>
            <p>Generated with DataInsight Pro - Advanced EDA Platform</p>
            <hr>
        </div>
        <div class="content">
            {report_content.replace('\n', '<br>').replace('## ', '<h2>').replace('\n# ', '</h2>')}
        </div>
        <div class="footer">
            <hr>
            <p>Report generated by DataInsight Pro | By Hammad Zahid</p>
            <p>🔗 LinkedIn: https://www.linkedin.com/in/hammad-zahid-xyz</p>
            <p>🐙 GitHub: https://github.com/Hamad-Ansari</p>
            <p>✉️ Email: Hammadzahid24@gmail.com</p>
        </div>
    </body>
    </html>
    """
    return html_template

def generate_summary():
    """Generate project summary."""
    summary = {
        "project": "DataInsight Pro",
        "version": "1.0.0",
        "generated": datetime.now().isoformat(),
        "developer": {
            "name": "Hammad Zahid",
            "linkedin": "https://www.linkedin.com/in/hammad-zahid-xyz",
            "github": "https://github.com/Hamad-Ansari",
            "email": "Hammadzahid24@gmail.com"
        }
    }
    
    if st.session_state.df is not None:
        summary["dataset"] = {
            "name": st.session_state.file_info.get('name', 'Unknown') if st.session_state.file_info else 'Unknown',
            "rows": len(st.session_state.df),
            "columns": len(st.session_state.df.columns),
            "size": st.session_state.file_info.get('size', 0) if st.session_state.file_info else 0
        }
        
        summary["analysis"] = {
            "plots_generated": len(st.session_state.plotly_figs),
            "data_preprocessed": st.session_state.processed_df is not None,
            "forecasting_prepared": hasattr(st.session_state, 'forecast_df')
        }
    
    return summary

# Main app function
def main():
    # Show header
    show_header()
    
    # Sidebar navigation
    sidebar_navigation()
    
    # Main content based on current page
    if st.session_state.current_page == 'home':
        home_page()
    elif st.session_state.current_page == 'overview':
        overview_page()
    elif st.session_state.current_page == 'eda':
        eda_page()
    elif st.session_state.current_page == 'visualization':
        visualization_page()
    elif st.session_state.current_page == 'timeseries':
        timeseries_page()
    elif st.session_state.current_page == 'ml':
        ml_page()
    elif st.session_state.current_page == 'export':
        export_page()

if __name__ == "__main__":
    main()