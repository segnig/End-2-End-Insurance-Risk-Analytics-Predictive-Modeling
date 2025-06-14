import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import pandas as pd

def data_loading(path: str) -> pd.DataFrame:
    """
    Loads pipe-delimited text file using pandas.read_csv().
    
    Args:
        path (str): File path to the pipe-delimited text file.
    
    Returns:
        pd.DataFrame: Loaded data. Empty DataFrame if error occurs.
    """
    try:
        # Convert 'None' and empty strings to NaN
        df = pd.read_csv(
            path,
            sep='|',
            na_values=['None', 'none', ''],  # Treat these as NaN
            keep_default_na=True,  # Also handle standard NA values
            dtype=str,  # Read all as strings first (optional)
            skipinitialspace=True,  # Skip extra whitespace after delimiter
            on_bad_lines='warn'  # Skip bad lines (or use 'error' to raise)
        )
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()
    
def detect_dtype(dataframe):
    """
    Detects the data type of each column in the DataFrame by inspecting its values.
    Returns a dictionary mapping column names to detected types: 'int', 'float', 'bool', 'datetime', or 'string'.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame to analyze
        
    Returns:
        dict: Column name to detected type mapping
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    dtype_map = {}
    common_date_formats = [
        '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d',
        '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S'
    ]
    
    for col in dataframe.columns:
        # Skip empty columns
        if dataframe[col].empty:
            dtype_map[col] = 'string'
            continue
            
        # Get sample of non-null values (for efficiency)
        sample = dataframe[col]
        
        # Initialize detected type
        detected_type = 'string'
        
        # Test for bool first (since bool is subclass of int in Python)
        bool_test = sample.astype(str).str.lower().isin(['true', 'false', 't', 'f', '1', '0', 'yes', 'no'])
        if all(bool_test):
            detected_type = 'bool'
        else:
            # Test for datetime with common formats
            is_date = False
            for fmt in common_date_formats:
                try:
                    if pd.to_datetime(sample, format=fmt, errors='raise').notna().all():
                        is_date = True
                        break
                except (ValueError, TypeError):
                    continue
            
            if is_date:
                detected_type = 'datetime'
            else:
                # Test for numeric types
                numeric_count = 0
                int_count = 0
                
                for val in sample:
                    if str(val).strip() == '':  # Skip empty strings
                        continue
                    try:
                        float_val = float(val)
                        numeric_count += 1
                        if float_val.is_integer():
                            int_count += 1
                    except (ValueError, TypeError):
                        pass
                
                # Determine numeric type
                if numeric_count > 0 and numeric_count == len(sample):
                    if int_count == len(sample):
                        detected_type = 'int'
                    else:
                        detected_type = 'float'
        
        dtype_map[col] = detected_type
    
    return dtype_map



# ====================== DATA TYPE HANDLING ======================
def convert_dtypes(df, columns_dtype):
    """
    Convert DataFrame columns to specified dtypes with error handling
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns_dtype (dict): Dictionary of {column: dtype} pairs
        
    Returns:
        pd.DataFrame: DataFrame with converted dtypes
    """
    for col, dtype in columns_dtype.items():
        try:
            if dtype == "int":
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            elif dtype == "float":
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif dtype == "bool":
                df[col] = df[col].astype(bool)
            elif dtype == "datetime":
                df[col] = pd.to_datetime(df[col], errors='coerce')
        except Exception as e:
            print(f"Cannot cast {col} to {dtype}: {e}")
    return df

# ====================== BASIC EDA FUNCTIONS ======================
def calculate_loss_ratio(df, groupby_cols=None):
    """
    Calculate loss ratio (TotalClaims/TotalPremium) with optional grouping
    
    Args:
        df (pd.DataFrame): Input DataFrame
        groupby_cols (list): Columns to group by (e.g., ['Province', 'VehicleType'])
        
    Returns:
        pd.DataFrame: Loss ratio results
    """
    if groupby_cols:
        result = df.groupby(groupby_cols).apply(
            lambda x: x['TotalClaims'].sum() / x['TotalPremium'].sum()
        ).reset_index(name='LossRatio')
    else:
        result = pd.DataFrame({
            'OverallLossRatio': [df['TotalClaims'].sum() / df['TotalPremium'].sum()]
        })
    return result

def plot_distributions(df, numerical_cols, categorical_cols, figsize=(15, 10)):
    """
    Plot distributions for numerical and categorical columns
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numerical_cols (list): Numerical columns to plot
        categorical_cols (list): Categorical columns to plot
        figsize (tuple): Figure size
    """
    # Numerical distributions
    if numerical_cols:
        df[numerical_cols].hist(bins=30, figsize=figsize)
        plt.tight_layout()
        plt.show()
    
    # Categorical distributions (top 20 categories only)
    if categorical_cols:
        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            df[col].value_counts().head(20).plot(kind='bar')
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=45)
            plt.show()

# ====================== DATA QUALITY FUNCTIONS ======================
def check_missing_values(df):
    """Return DataFrame with missing value statistics"""
    missing = df.isnull().sum().to_frame(name='MissingCount')
    missing['Percentage'] = (missing['MissingCount'] / len(df)) * 100
    return missing[missing['MissingCount'] > 0]


def impute_missing_values(df, categorical_features=None, numerical_features=None, strategy='auto'):
    """
    Impute missing values in a DataFrame with appropriate strategies.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        categorical_features (list): List of categorical column names
        numerical_features (list): List of numerical column names
        strategy (str): 'auto' (default), 'mode' for categorical/mean for numerical,
                       or specify 'mode', 'mean', 'median', 'constant'
    
    Returns:
        pd.DataFrame: DataFrame with missing values imputed
    """
    # Auto-detect feature types if not provided
    if categorical_features is None or numerical_features is None:
        categorical_features = df.select_dtypes(include=['object', 'category', 'bool']).columns
        numerical_features = df.select_dtypes(include=['number']).columns
    
    # Create copies of the feature lists to avoid modifying the input
    cat_features = list(categorical_features)
    num_features = list(numerical_features)
    
    # Track which columns had missing values
    cols_with_missing = df.columns[df.isnull().any()].tolist()
    
    # Impute categorical features
    for col in cat_features:
        if col in cols_with_missing:
            mode_value = df[col].mode(dropna=True)
            if not mode_value.empty:
                # Use the first mode if multiple modes exist
                df[col].fillna(mode_value.iloc[0], inplace=True)
            else:
                # If no mode exists (all values were NA), fill with 'Unknown'
                df[col].fillna('Unknown', inplace=True)
    
    # Impute numerical features
    for col in num_features:
        if col in cols_with_missing:
            if strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'constant':
                fill_value = 0
            else:  # default to mean
                fill_value = df[col].mean()
            
            # Only fill if we got a valid fill value
            if pd.notna(fill_value):
                df[col].fillna(fill_value, inplace=True)
            else:
                # If all values were NA, fill with 0
                df[col].fillna(0, inplace=True)
    
    # Report on imputation
    print(f"Imputed missing values in {len(cols_with_missing)} columns")
    if cols_with_missing:
        print("Columns imputed:", ", ".join(cols_with_missing))
    
    return df

def detect_outliers(df, numerical_cols, threshold=3):
    """
    Detect outliers using z-score method
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numerical_cols (list): Numerical columns to check
        threshold (float): Z-score threshold
        
    Returns:
        dict: Outlier counts per column
    """
    outliers = {}
    for col in numerical_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers[col] = len(z_scores[z_scores > threshold])
    return outliers

# ====================== TEMPORAL ANALYSIS ======================
def analyze_temporal_trends(df, date_col='TransactionMonth', value_cols=['TotalPremium', 'TotalClaims']):
    """
    Analyze monthly trends for specified value columns
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_col (str): Date column name
        value_cols (list): Value columns to analyze
        
    Returns:
        pd.DataFrame: Monthly aggregated results
    """
    # Ensure proper datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Group by month
    monthly = df.groupby(pd.Grouper(key=date_col, freq='M'))[value_cols].sum()
    monthly['LossRatio'] = monthly['TotalClaims'] / monthly['TotalPremium']
    
    # Plot trends
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    for i, col in enumerate(value_cols + ['LossRatio']):
        monthly[col].plot(ax=axes[i], title=f'Monthly {col}', marker='o')
        axes[i].grid(True)
    plt.tight_layout()
    plt.show()
    
    return monthly

# ====================== GEOGRAPHICAL ANALYSIS ======================
def analyze_by_geography(df, geo_col='Province', value_cols=['TotalPremium', 'TotalClaims']):
    """
    Analyze data by geographical region
    
    Args:
        df (pd.DataFrame): Input DataFrame
        geo_col (str): Geographical column name
        value_cols (list): Value columns to analyze
        
    Returns:
        pd.DataFrame: Geographical analysis results
    """
    geo_analysis = df.groupby(geo_col)[value_cols].sum()
    geo_analysis['LossRatio'] = geo_analysis['TotalClaims'] / geo_analysis['TotalPremium']
    geo_analysis = geo_analysis.sort_values('LossRatio', ascending=False)
    
    # Plot top 10 provinces by loss ratio
    plt.figure(figsize=(12, 6))
    geo_analysis['LossRatio'].head(10).plot(kind='bar')
    plt.title(f'Top 10 {geo_col} by Loss Ratio')
    plt.ylabel('Loss Ratio (Claims/Premium)')
    plt.xticks(rotation=45)
    plt.show()
    
    return geo_analysis

# ====================== VEHICLE ANALYSIS ======================
def analyze_vehicle_metrics(df, make_col='make', model_col='Model'):
    """
    Analyze vehicle makes/models by claims and premiums
    
    Args:
        df (pd.DataFrame): Input DataFrame
        make_col (str): Make column name
        model_col (str): Model column name
        
    Returns:
        tuple: (make_analysis, model_analysis) DataFrames
    """
    # By make
    make_analysis = df.groupby(make_col).agg({
        'TotalPremium': 'sum',
        'TotalClaims': 'sum',
        'CustomValueEstimate': 'mean'
    })
    make_analysis['LossRatio'] = make_analysis['TotalClaims'] / make_analysis['TotalPremium']
    make_analysis = make_analysis.sort_values('LossRatio', ascending=False)
    
    # By model (top 20 only for visualization)
    model_analysis = df.groupby([make_col, model_col]).agg({
        'TotalPremium': 'sum',
        'TotalClaims': 'sum'
    }).nlargest(20, 'TotalClaims')
    model_analysis['LossRatio'] = model_analysis['TotalClaims'] / model_analysis['TotalPremium']
    
    # Plot top 10 makes by loss ratio
    plt.figure(figsize=(12, 6))
    make_analysis['LossRatio'].head(10).plot(kind='bar')
    plt.title('Top 10 Vehicle Makes by Loss Ratio')
    plt.ylabel('Loss Ratio (Claims/Premium)')
    plt.xticks(rotation=45)
    plt.show()
    
    return make_analysis, model_analysis

# ====================== CREATIVE VISUALIZATIONS ======================
def plot_creative_visualizations(df):
    """Generate 3 creative visualizations for key insights"""
    # 1. Heatmap of loss ratio by province and vehicle type
    plt.figure(figsize=(15, 8))
    loss_by_province_type = df.groupby(['Province', 'VehicleType']).apply(
        lambda x: x['TotalClaims'].sum() / x['TotalPremium'].sum()
    ).unstack()
    sns.heatmap(loss_by_province_type, cmap='coolwarm', annot=True, fmt='.2f')
    plt.title('Loss Ratio by Province and Vehicle Type')
    plt.tight_layout()
    plt.show()
    
    # 2. Bubble plot of makes by premium and claims
    make_stats = df.groupby('make').agg({
        'TotalPremium': 'sum',
        'TotalClaims': 'sum',
        'PolicyID': 'count'
    }).sort_values('TotalPremium', ascending=False).head(20)
    make_stats['LossRatio'] = make_stats['TotalClaims'] / make_stats['TotalPremium']
    
    plt.figure(figsize=(15, 10))
    plt.scatter(
        x=make_stats['TotalPremium'],
        y=make_stats['TotalClaims'],
        s=make_stats['PolicyID']/100,
        c=make_stats['LossRatio'],
        cmap='viridis',
        alpha=0.6
    )
    plt.colorbar(label='Loss Ratio')
    plt.xlabel('Total Premium')
    plt.ylabel('Total Claims')
    plt.title('Vehicle Makes: Premium vs Claims (Size = Policy Count)')
    for i, txt in enumerate(make_stats.index):
        plt.annotate(txt, (make_stats['TotalPremium'].iloc[i], make_stats['TotalClaims'].iloc[i]))
    plt.grid(True)
    plt.show()
    
    # 3. Temporal trends with gender breakdown
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
    gender_trends = df.groupby([pd.Grouper(key='TransactionMonth', freq='M'), 'Gender']).agg({
        'TotalPremium': 'sum',
        'TotalClaims': 'sum'
    }).reset_index()
    gender_trends['LossRatio'] = gender_trends['TotalClaims'] / gender_trends['TotalPremium']
    
    plt.figure(figsize=(15, 6))
    sns.lineplot(data=gender_trends, x='TransactionMonth', y='LossRatio', hue='Gender')
    plt.title('Monthly Loss Ratio Trend by Gender')
    plt.ylabel('Loss Ratio')
    plt.grid(True)
    plt.show()