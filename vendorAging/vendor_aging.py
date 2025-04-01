# Standard library imports
import os
import glob
import json
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical and machine learning imports
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import LocalOutlierFactor

# =====================================================
# SNAPSHOT-AWARE DATA PREPARATION FUNCTIONS
# =====================================================

def prepare_time_series_data(df, snapshot_date_col='Snapshot_Date', vendor_id_col='Vendor ID'):
    """
    Prepare vendor aging data for time series analysis by handling snapshot dates.

    Args:
        df (pandas.DataFrame): Raw vendor aging data with multiple snapshots
        snapshot_date_col (str): Column containing snapshot date
        vendor_id_col (str): Column containing vendor identifier

    Returns:
        pandas.DataFrame: Prepared DataFrame with proper date formats
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Convert snapshot date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(result_df[snapshot_date_col]):
        result_df[snapshot_date_col] = pd.to_datetime(result_df[snapshot_date_col], errors='coerce')

    # Convert last payment date to datetime if it exists
    if 'LP Date' in result_df.columns and not pd.api.types.is_datetime64_any_dtype(result_df['LP Date']):
        result_df['LP Date'] = pd.to_datetime(result_df['LP Date'], errors='coerce')

    # Sort data by vendor and date
    result_df = result_df.sort_values([vendor_id_col, snapshot_date_col])

    # Add a year-month column for easier grouping
    result_df['Year_Month'] = result_df[snapshot_date_col].dt.to_period('M')

    return result_df


def create_vendor_snapshot_pivot(df, value_col='Balance Outstanding', snapshot_date_col='Snapshot_Date',
                               vendor_id_col='Vendor ID', vendor_name_col='Vendor',
                               fill_method='ffill'):
    """
    Create a pivot table with vendors as rows and snapshot dates as columns.

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        value_col (str): Column to use as the values in the pivot
        snapshot_date_col (str): Column containing snapshot date
        vendor_id_col (str): Column containing vendor identifier
        vendor_name_col (str): Column containing vendor name
        fill_method (str): Method to fill missing values ('ffill', 'bfill', None)

    Returns:
        pandas.DataFrame: Pivot table with vendors as rows and dates as columns
    """
    # Create a copy of the data with just the needed columns
    pivot_data = df[[vendor_id_col, vendor_name_col, snapshot_date_col, value_col]].copy()

    # Create the pivot table
    pivot = pivot_data.pivot_table(
        index=[vendor_id_col, vendor_name_col],
        columns=snapshot_date_col,
        values=value_col,
        aggfunc='first'  # Use first value in case of duplicates
    )

    # Fill missing values if specified
    if fill_method:
        pivot = pivot.fillna(method=fill_method)

    return pivot


def calculate_aging_metrics_over_time(df, snapshot_date_col='Snapshot_Date',
                                    aging_cols=None, vendor_id_col='Vendor ID'):
    """
    Calculate aging metrics for each vendor across snapshots.

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        snapshot_date_col (str): Column containing snapshot date
        aging_cols (list): List of aging bucket columns
        vendor_id_col (str): Column containing vendor identifier

    Returns:
        pandas.DataFrame: DataFrame with calculated metrics by vendor and snapshot
    """
    # Define default aging columns if not specified
    if aging_cols is None:
        aging_cols = [
            'Future_Aging', 'Aging_0_30', 'Aging_31_60', 'Aging_61_90',
            'Aging_91_120', 'Aging_121_180', 'Aging_181_360', 'Above_361_Aging'
        ]

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Calculate total aging for each snapshot
    result_df['Total_Aging'] = result_df[aging_cols].sum(axis=1)

    # Calculate aging beyond 90 days
    aging_beyond_90_cols = [col for col in aging_cols if '91' in col or '120' in col or
                           '180' in col or '360' in col or 'Above_361' in col]

    result_df['Aging_Beyond_90'] = result_df[aging_beyond_90_cols].sum(axis=1)

    # Calculate aging percentages
    for col in aging_cols:
        result_df[f'{col}_Pct'] = (result_df[col] / result_df['Total_Aging'].replace(0, np.nan)) * 100
        result_df[f'{col}_Pct'] = result_df[f'{col}_Pct'].fillna(0)

    # Calculate percentage of aging beyond 90 days
    result_df['Pct_Aging_Beyond_90'] = (result_df['Aging_Beyond_90'] /
                                      result_df['Total_Aging'].replace(0, np.nan)) * 100
    result_df['Pct_Aging_Beyond_90'] = result_df['Pct_Aging_Beyond_90'].fillna(0)

    return result_df


# =====================================================
# TIME SERIES ANALYSIS FUNCTIONS
# =====================================================

def calculate_aging_trends(df, snapshot_date_col='Snapshot_Date', vendor_id_col='Vendor ID',
                          aging_cols=None, window=3):
    """
    Calculate aging trends for each vendor over time.

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        snapshot_date_col (str): Column containing snapshot date
        vendor_id_col (str): Column containing vendor identifier
        aging_cols (list): List of aging bucket columns
        window (int): Window size for rolling calculations

    Returns:
        pandas.DataFrame: DataFrame with trend metrics for each vendor and snapshot
    """
    # Define default aging columns if not specified
    if aging_cols is None:
        aging_cols = [
            'Future_Aging', 'Aging_0_30', 'Aging_31_60', 'Aging_61_90',
            'Aging_91_120', 'Aging_121_180', 'Aging_181_360', 'Above_361_Aging'
        ]

    # Calculate metrics over time
    result_df = calculate_aging_metrics_over_time(df, snapshot_date_col, aging_cols, vendor_id_col)

    # Group by vendor and sort by date for time series operations
    vendor_groups = result_df.groupby(vendor_id_col)

    # Initialize columns for trends
    trend_cols = ['Balance_MoM_Change', 'Aging_Beyond_90_MoM_Change',
                 'Balance_Trend', 'Aging_Beyond_90_Trend']

    for col in trend_cols:
        result_df[col] = np.nan

    # Calculate trends for each vendor
    for vendor_id, group in vendor_groups:
        # Sort by snapshot date
        vendor_data = group.sort_values(snapshot_date_col)

        if len(vendor_data) >= 2:  # Need at least 2 snapshots for trend
            # Calculate month-over-month changes
            vendor_data['Balance_MoM_Change'] = vendor_data['Balance Outstanding'].pct_change() * 100
            vendor_data['Aging_Beyond_90_MoM_Change'] = vendor_data['Aging_Beyond_90'].pct_change() * 100

            # Calculate rolling average trend if enough data points
            if len(vendor_data) >= window:
                # Simple linear regression slope would be better, but for simplicity:
                vendor_data['Balance_Trend'] = vendor_data['Balance Outstanding'].rolling(window=window).apply(
                    lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100 if x.iloc[0] != 0 else 0
                )

                vendor_data['Aging_Beyond_90_Trend'] = vendor_data['Aging_Beyond_90'].rolling(window=window).apply(
                    lambda x: (x.iloc[-1] - x.iloc[0]) / (x.iloc[0] or 1) * 100
                )

            # Update the main dataframe with trend calculations
            result_df.loc[vendor_data.index, 'Balance_MoM_Change'] = vendor_data['Balance_MoM_Change']
            result_df.loc[vendor_data.index, 'Aging_Beyond_90_MoM_Change'] = vendor_data['Aging_Beyond_90_MoM_Change']
            result_df.loc[vendor_data.index, 'Balance_Trend'] = vendor_data['Balance_Trend']
            result_df.loc[vendor_data.index, 'Aging_Beyond_90_Trend'] = vendor_data['Aging_Beyond_90_Trend']

    # Categorize the trends
    result_df['Balance_Trend_Category'] = pd.cut(
        result_df['Balance_Trend'],
        bins=[-float('inf'), -10, -3, 3, 10, float('inf')],
        labels=['Rapidly Decreasing', 'Decreasing', 'Stable', 'Increasing', 'Rapidly Increasing']
    )

    result_df['Aging_Trend_Category'] = pd.cut(
        result_df['Aging_Beyond_90_Trend'],
        bins=[-float('inf'), -10, -3, 3, 10, float('inf')],
        labels=['Rapidly Improving', 'Improving', 'Stable', 'Worsening', 'Rapidly Worsening']
    )

    return result_df


def vendor_payment_history_analysis(df, snapshot_date_col='Snapshot_Date', vendor_id_col='Vendor ID',
                                  payment_date_col='LP Date', payment_amount_col='Vendor LP Amount'):
    """
    Analyze vendor payment history across snapshots.

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        snapshot_date_col (str): Column containing snapshot date
        vendor_id_col (str): Column containing vendor identifier
        payment_date_col (str): Column containing last payment date
        payment_amount_col (str): Column containing last payment amount

    Returns:
        pandas.DataFrame: DataFrame with payment history metrics
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Ensure date columns are datetime
    if not pd.api.types.is_datetime64_any_dtype(result_df[snapshot_date_col]):
        result_df[snapshot_date_col] = pd.to_datetime(result_df[snapshot_date_col], errors='coerce')

    if not pd.api.types.is_datetime64_any_dtype(result_df[payment_date_col]):
        result_df[payment_date_col] = pd.to_datetime(result_df[payment_date_col], errors='coerce')

    # Calculate days between snapshot and last payment
    result_df['Days_Since_Payment'] = (result_df[snapshot_date_col] - result_df[payment_date_col]).dt.days

    # Group by vendor and extract payment history
    payment_history = []

    for vendor_id, group in result_df.groupby(vendor_id_col):
        vendor_history = group.sort_values(snapshot_date_col)

        # Track payment dates and amounts across snapshots
        payment_dates = vendor_history[payment_date_col].dropna().unique()
        payment_dates = sorted(payment_dates)

        # Calculate frequency metrics if we have multiple payments
        if len(payment_dates) >= 2:
            # Calculate average days between payments
            payment_intervals = [(payment_dates[i] - payment_dates[i-1]).days
                               for i in range(1, len(payment_dates))]
            avg_payment_interval = sum(payment_intervals) / len(payment_intervals)

            # Count payments in last 90 days (relative to each snapshot date)
            for idx, row in vendor_history.iterrows():
                snapshot_date = row[snapshot_date_col]
                payments_90d = sum(1 for date in payment_dates
                                  if snapshot_date - timedelta(days=90) <= date <= snapshot_date)

                result_df.loc[idx, 'Payments_Last_90d'] = payments_90d
                result_df.loc[idx, 'Avg_Payment_Interval_Days'] = avg_payment_interval
        else:
            # Only one payment date or none
            vendor_history_indexes = vendor_history.index
            result_df.loc[vendor_history_indexes, 'Payments_Last_90d'] = 0 if len(payment_dates) == 0 else 1
            result_df.loc[vendor_history_indexes, 'Avg_Payment_Interval_Days'] = np.nan

    # Calculate payment regularity score (higher is better)
    # Consider both frequency and consistency
    result_df['Payment_Regularity_Score'] = np.nan

    for vendor_id, group in result_df.groupby(vendor_id_col):
        if 'Avg_Payment_Interval_Days' in group.columns:
            avg_interval = group['Avg_Payment_Interval_Days'].iloc[0]
            if not pd.isna(avg_interval):
                # More regular = lower standard deviation in payment intervals
                # This is a placeholder calculation - in a real implementation,
                # you would use actual payment intervals
                regularity_score = 100 - min(100, avg_interval / 3)  # Higher score for shorter intervals
                result_df.loc[group.index, 'Payment_Regularity_Score'] = regularity_score

    return result_df



def detect_time_based_anomalies(df, snapshot_date_col='Snapshot_Date', vendor_id_col='Vendor ID',
                              value_cols=None, zscore_threshold=5, window_size=12): # This function needs more work
    """
    Detect anomalies in vendor data based on time series patterns.

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        snapshot_date_col (str): Column containing snapshot date
        vendor_id_col (str): Column containing vendor identifier
        value_cols (list): List of columns to check for anomalies
        zscore_threshold (float): Z-score threshold for flagging anomalies
        window_size (int): Size of the window for local anomaly detection

    Returns:
        pandas.DataFrame: DataFrame with time-based anomaly flags
    """
    # Define default value columns if not specified
    if value_cols is None:
        value_cols = [
            'Balance Outstanding', 
            'Aging_Beyond_90', 
            'Pct_Aging_Beyond_90',
            'Future_Aging',
            'Aging_0_30',
            'Aging_31_60',
            'Aging_61_90',
            'Aging_91_120',
            'Aging_121_180',
            'Aging_181_360',
            'Above_361_Aging'
        ]

    # Ensure we have time-based metrics
    if 'Aging_Beyond_90' not in df.columns:
        df = calculate_aging_metrics_over_time(df)

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Initialize anomaly columns
    for col in value_cols:
        if col in result_df.columns:
            result_df[f'{col}_Anomaly'] = False
            result_df[f'{col}_Zscore'] = np.nan

    result_df['Time_Anomaly'] = False
    result_df['Anomaly_Details'] = ''

    # Detect anomalies for each vendor over time
    for vendor_id, group in result_df.groupby(vendor_id_col):
        if len(group) >= window_size:  # Need at least window_size data points
            vendor_data = group.sort_values(snapshot_date_col)
            
            for col in value_cols:
                if col in vendor_data.columns:
                    # First check: Look for sudden changes between consecutive points
                    if len(vendor_data) >= 2:
                        pct_changes = vendor_data[col].pct_change().abs()
                        # Flag large percentage changes (more than 100%)
                        large_changes = pct_changes > 1.0
                        
                        # Update anomalies based on large changes
                        for idx, is_change in zip(vendor_data.index[1:], large_changes[1:]):
                            if is_change:
                                result_df.loc[idx, f'{col}_Anomaly'] = True
                                result_df.loc[idx, f'{col}_Zscore'] = pct_changes.loc[idx]
                    
                    # Second check: Look for global outliers using z-score
                    mean_val = vendor_data[col].mean()
                    std_val = vendor_data[col].std()
                    
                    if std_val > 0:  # Avoid division by zero
                        zscores = np.abs((vendor_data[col] - mean_val) / std_val)
                        
                        # Flag global anomalies
                        global_anomalies = zscores > zscore_threshold
                        result_df.loc[vendor_data.index, f'{col}_Anomaly'] = (
                            result_df.loc[vendor_data.index, f'{col}_Anomaly'] | global_anomalies
                        )
                        
                        # Update z-scores (only if we haven't set them from pct_change)
                        for idx in vendor_data.index:
                            if pd.isna(result_df.loc[idx, f'{col}_Zscore']):
                                result_df.loc[idx, f'{col}_Zscore'] = zscores.loc[idx]
                    
                    # Third check: Look for local anomalies using rolling window
                    if len(vendor_data) >= window_size:
                        for i in range(window_size, len(vendor_data)):
                            window = vendor_data.iloc[i-window_size:i]
                            current_val = vendor_data.iloc[i][col]
                            
                            window_mean = window[col].mean()
                            window_std = window[col].std()
                            
                            if window_std > 0:
                                local_zscore = abs((current_val - window_mean) / window_std)
                                current_idx = vendor_data.index[i]
                                
                                # Flag if it's a local anomaly and not already flagged
                                if local_zscore > zscore_threshold and not result_df.loc[current_idx, f'{col}_Anomaly']:
                                    result_df.loc[current_idx, f'{col}_Anomaly'] = True
                                    result_df.loc[current_idx, f'{col}_Zscore'] = local_zscore

            # Combine anomalies across columns
            for idx, row in vendor_data.iterrows():
                anomaly_cols = []
                details = []
                
                for col in value_cols:
                    if col in vendor_data.columns:
                        anomaly_col = f'{col}_Anomaly'
                        zscore_col = f'{col}_Zscore'
                        
                        if anomaly_col in result_df.columns and result_df.loc[idx, anomaly_col]:
                            anomaly_cols.append(col)
                            
                            # Get direction
                            if vendor_data.index.get_loc(idx) > 0:
                                prev_idx = vendor_data.index[vendor_data.index.get_loc(idx) - 1]
                                direction = "increase" if row[col] > vendor_data.loc[prev_idx, col] else "decrease"
                            else:
                                direction = "abnormal value"
                                
                            z_val = result_df.loc[idx, zscore_col]
                            details.append(f"Unusual {direction} in {col} (score={z_val:.2f})")
                
                if anomaly_cols:
                    result_df.loc[idx, 'Time_Anomaly'] = True
                    result_df.loc[idx, 'Anomaly_Details'] = "; ".join(details)

    return result_df

def detect_trend_shifts(df, snapshot_date_col='Snapshot_Date', vendor_id_col='Vendor ID',
                      value_col='Balance Outstanding', min_snapshots=4, window=3):
    """
    Detect significant shifts in trends over time for each vendor.

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        snapshot_date_col (str): Column containing snapshot date
        vendor_id_col (str): Column containing vendor identifier
        value_col (str): Column to analyze for trend shifts
        min_snapshots (int): Minimum number of snapshots required
        window (int): Window size for rolling trend calculations

    Returns:
        pandas.DataFrame: DataFrame with trend shift indicators
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Initialize columns for trend shifts
    result_df['Trend_Shift'] = False
    result_df['Trend_Shift_Direction'] = np.nan
    result_df['Trend_Shift_Magnitude'] = np.nan

    # Analyze each vendor separately
    for vendor_id, group in result_df.groupby(vendor_id_col):
        # Need enough snapshots for meaningful trend analysis
        if len(group) >= min_snapshots:
            vendor_data = group.sort_values(snapshot_date_col)

            # Calculate rolling slopes
            if window < len(vendor_data) - 1:
                slopes = []
                dates = []

                for i in range(len(vendor_data) - window):
                    window_data = vendor_data.iloc[i:i+window]

                    # Simple slope calculation (could use linear regression for more accuracy)
                    if window_data[value_col].iloc[0] != 0:
                        slope = ((window_data[value_col].iloc[-1] - window_data[value_col].iloc[0]) /
                               window_data[value_col].iloc[0])
                    else:
                        slope = 0

                    slopes.append(slope)
                    dates.append(window_data[snapshot_date_col].iloc[-1])

                # Detect shifts in the slope direction
                for i in range(1, len(slopes)):
                    prev_slope = slopes[i-1]
                    curr_slope = slopes[i]

                    # Check if the sign changed or magnitude changed significantly
                    sign_change = (prev_slope * curr_slope < 0)  # Different signs
                    magnitude_change = abs(curr_slope - prev_slope) > 0.1  # Arbitrary threshold

                    if sign_change or magnitude_change:
                        # Find the corresponding index in the original dataframe
                        shift_date = dates[i]
                        shift_idx = vendor_data[vendor_data[snapshot_date_col] == shift_date].index

                        if len(shift_idx) > 0:
                            result_df.loc[shift_idx, 'Trend_Shift'] = True
                            result_df.loc[shift_idx, 'Trend_Shift_Direction'] = 'Positive' if curr_slope > prev_slope else 'Negative'
                            result_df.loc[shift_idx, 'Trend_Shift_Magnitude'] = abs(curr_slope - prev_slope)

    return result_df


def analyze_seasonal_patterns(df, snapshot_date_col='Snapshot_Date', vendor_id_col='Vendor ID',
                            value_col='Balance Outstanding', period=12, min_periods=24):
    """
    Analyze seasonal patterns in vendor aging data.

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        snapshot_date_col (str): Column containing snapshot date
        vendor_id_col (str): Column containing vendor identifier
        value_col (str): Column to analyze for seasonal patterns
        period (int): Expected seasonality period (e.g., 12 for annual)
        min_periods (int): Minimum number of periods required for analysis

    Returns:
        dict: Dictionary with seasonal analysis results by vendor
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Ensure date columns are datetime
    if not pd.api.types.is_datetime64_any_dtype(result_df[snapshot_date_col]):
        result_df[snapshot_date_col] = pd.to_datetime(result_df[snapshot_date_col], errors='coerce')

    # Results container
    seasonal_results = {}
    
    # Check if statsmodels is available
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.tsa.stattools import adfuller
        statsmodels_available = True
    except ImportError:
        statsmodels_available = False
        print("Statsmodels not available, using simplified seasonal analysis")

    # If statsmodels is not available, use simplified approach
    if not statsmodels_available:
        for vendor_id, group in result_df.groupby(vendor_id_col):
            if len(group) >= min_periods // 2:  # Relaxed requirement
                vendor_data = group.sort_values(snapshot_date_col)
                
                # Create a month field for grouping
                vendor_data['month'] = vendor_data[snapshot_date_col].dt.month
                
                # Calculate average by month
                monthly_avg = vendor_data.groupby('month')[value_col].mean()
                
                # Calculate simple seasonal strength (max month - min month) / mean
                seasonal_strength = (monthly_avg.max() - monthly_avg.min()) / monthly_avg.mean() if monthly_avg.mean() != 0 else 0
                
                seasonal_results[vendor_id] = {
                    'has_seasonality': seasonal_strength > 0.2,  # Arbitrary threshold
                    'seasonal_strength': seasonal_strength,
                    'peak_month': monthly_avg.idxmax(),
                    'trough_month': monthly_avg.idxmin(),
                    'monthly_averages': monthly_avg.to_dict()
                }
        
        return seasonal_results

    # If statsmodels is available, use the full analysis
    for vendor_id, group in result_df.groupby(vendor_id_col):
        # Need enough data points for seasonal analysis
        if len(group) >= min_periods:
            vendor_data = group.sort_values(snapshot_date_col)

            # Create a time series
            ts = vendor_data.set_index(snapshot_date_col)[value_col]

            try:
                # Check if data is stationary
                adf_result = adfuller(ts.values, autolag='AIC')
                is_stationary = adf_result[1] < 0.05  # p-value < 0.05 indicates stationarity

                # Perform seasonal decomposition if we have enough data points
                decomposition = seasonal_decompose(ts, model='additive', period=period, extrapolate_trend='freq')

                # Extract components
                trend = decomposition.trend
                seasonal = decomposition.seasonal
                residual = decomposition.resid

                # Calculate strength of seasonality
                # (1 - variance of residual / variance of detrended series)
                detrended = ts - trend
                seasonal_strength = max(0, 1 - residual.var() / detrended.var())

                seasonal_results[vendor_id] = {
                    'has_seasonality': seasonal_strength > 0.3,  # Arbitrary threshold
                    'seasonal_strength': seasonal_strength,
                    'is_stationary': is_stationary,
                    'peak_month': seasonal.groupby(seasonal.index.month).mean().idxmax(),
                    'trough_month': seasonal.groupby(seasonal.index.month).mean().idxmin(),
                    'components': {
                        'trend': trend,
                        'seasonal': seasonal,
                        'residual': residual
                    }
                }

            except Exception as e:
                # Not enough data or other issues
                seasonal_results[vendor_id] = {
                    'has_seasonality': False,
                    'error': str(e)
                }

    return seasonal_results


def detect_pattern_anomalies(df, snapshot_date_col='Snapshot_Date', vendor_id_col='Vendor ID',
                           value_col='Balance Outstanding', window=3, threshold=2.0):
    """
    Detect anomalies in time series patterns for each vendor.

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        snapshot_date_col (str): Column containing snapshot date
        vendor_id_col (str): Column containing vendor identifier
        value_col (str): Column to analyze for pattern anomalies
        window (int): Window size for pattern detection
        threshold (float): Z-score threshold for flagging anomalies

    Returns:
        pandas.DataFrame: DataFrame with pattern anomaly flags
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Initialize anomaly columns
    result_df['Pattern_Anomaly'] = False
    result_df['Pattern_Zscore'] = np.nan
    result_df['Pattern_Change'] = np.nan

    # Process each vendor separately
    for vendor_id, vendor_df in result_df.groupby(vendor_id_col):
        # Need enough data points for pattern analysis
        if len(vendor_df) < window + 1:
            continue

        # Sort by snapshot date
        vendor_data = vendor_df.sort_values(snapshot_date_col)

        # Calculate rolling statistics
        values = vendor_data[value_col].values

        # Calculate sequential changes
        changes = np.diff(values)

        # Calculate rolling pattern changes if we have enough data
        if len(changes) >= window:
            pattern_changes = []

            for i in range(len(changes) - window + 1):
                window_pattern = changes[i:i+window]

                # Calculate pattern change metric
                # (mean absolute change within the window)
                pattern_change = np.mean(np.abs(window_pattern))
                pattern_changes.append(pattern_change)

            # Calculate z-scores of pattern changes
            pattern_mean = np.mean(pattern_changes)
            pattern_std = np.std(pattern_changes)

            if pattern_std > 0:  # Avoid division by zero
                z_scores = np.abs((pattern_changes - pattern_mean) / pattern_std)

                # Flag anomalies
                anomaly_flags = z_scores > threshold

                # Update the main dataframe
                # We need to align the indices correctly
                start_idx = window  # Skip first "window" snapshots since we need diff and rolling window
                for i, anomaly in enumerate(anomaly_flags):
                    idx = vendor_data.index[start_idx + i]
                    result_df.loc[idx, 'Pattern_Anomaly'] = anomaly
                    result_df.loc[idx, 'Pattern_Zscore'] = z_scores[i]
                    result_df.loc[idx, 'Pattern_Change'] = pattern_changes[i]

    return result_df


# =====================================================
# VENDOR COMPARISON AND COHORT ANALYSIS
# =====================================================

def create_vendor_cohorts(df, snapshot_date_col='Snapshot_Date', vendor_id_col='Vendor ID',
                        aging_cols=None, n_clusters=4):
    """
    Create vendor cohorts based on aging patterns over time.

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        snapshot_date_col (str): Column containing snapshot date
        vendor_id_col (str): Column containing vendor identifier
        aging_cols (list): List of aging bucket columns
        n_clusters (int): Number of clusters/cohorts to create

    Returns:
        tuple: (DataFrame with cohort assignments, cohort profiles)
    """
    # Define default aging columns if not specified
    if aging_cols is None:
        aging_cols = [
            'Future_Aging', 'Aging_0_30', 'Aging_31_60', 'Aging_61_90',
            'Aging_91_120', 'Aging_121_180', 'Aging_181_360', 'Above_361_Aging'
        ]

    # Calculate metrics over time
    result_df = calculate_aging_metrics_over_time(df, snapshot_date_col, aging_cols, vendor_id_col)

    # Add trend metrics
    result_df = calculate_aging_trends(result_df, snapshot_date_col, vendor_id_col, aging_cols)

    # Get the latest snapshot for each vendor for cohort assignment
    latest_snapshots = result_df.sort_values(snapshot_date_col).groupby(vendor_id_col).last().reset_index()

    # Create features for clustering
    features = [
        'Pct_Aging_Beyond_90',  # Current aging percentage
        'Aging_Beyond_90_Trend'  # Trend in aging
    ]

    # Add aging bucket percentages
    for col in aging_cols:
        pct_col = f'{col}_Pct'
        if pct_col in latest_snapshots.columns:
            features.append(pct_col)

    # Prepare feature matrix
    X = latest_snapshots[features].fillna(0)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cohort_labels = kmeans.fit_predict(X_scaled)

    # Add cohort labels to the latest snapshot data
    latest_snapshots['Cohort'] = cohort_labels

    # Create cohort profiles
    cohort_profiles = {}
    for i in range(n_clusters):
        cohort_vendors = latest_snapshots[latest_snapshots['Cohort'] == i]

        # Calculate key metrics for this cohort
        profile = {
            'size': len(cohort_vendors),
            'avg_balance': cohort_vendors['Balance Outstanding'].mean(),
            'avg_aging_beyond_90': cohort_vendors['Aging_Beyond_90'].mean(),
            'avg_pct_aging_beyond_90': cohort_vendors['Pct_Aging_Beyond_90'].mean(),
            'avg_trend': cohort_vendors['Aging_Beyond_90_Trend'].mean()
        }

        # Assign a descriptive label
        trend_direction = "Improving" if profile['avg_trend'] < 0 else "Worsening"
        aging_level = "High Aging" if profile['avg_pct_aging_beyond_90'] > 30 else "Low Aging"

        profile['label'] = f"Cohort {i+1}: {aging_level}, {trend_direction}"
        cohort_profiles[i] = profile

    # Map cohort back to all snapshots
    vendor_cohort_map = latest_snapshots[[vendor_id_col, 'Cohort']].set_index(vendor_id_col)['Cohort'].to_dict()
    result_df['Cohort'] = result_df[vendor_id_col].map(vendor_cohort_map)

    # Map labels
    cohort_labels_map = {i: profile['label'] for i, profile in cohort_profiles.items()}
    result_df['Cohort_Label'] = result_df['Cohort'].map(cohort_labels_map)

    return result_df, cohort_profiles


def compare_vendor_to_peers(df, target_vendor_id, snapshot_date_col='Snapshot_Date',
                          vendor_id_col='Vendor ID', metrics=None,
                          latest_only=True, percentile=True):
    """
    Compare a specific vendor against peers over time.

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        target_vendor_id: ID of the vendor to compare
        snapshot_date_col (str): Column containing snapshot date
        vendor_id_col (str): Column containing vendor identifier
        metrics (list): List of metrics to compare
        latest_only (bool): Whether to compare only the latest snapshot
        percentile (bool): Whether to return percentile rankings

    Returns:
        pandas.DataFrame or dict: Comparison results
    """
    # Define default metrics if not specified
    if metrics is None:
        metrics = [
            'Balance Outstanding', 'Aging_Beyond_90', 'Pct_Aging_Beyond_90',
            'Balance_Trend', 'Aging_Beyond_90_Trend'
        ]

    # Ensure we have required metrics
    if 'Aging_Beyond_90' not in df.columns:
        df = calculate_aging_metrics_over_time(df)

    if 'Balance_Trend' not in df.columns:
        df = calculate_aging_trends(df)

    # Create a copy to work with
    result_df = df.copy()

    # Filter to specific vendor
    target_vendor_data = result_df[result_df[vendor_id_col] == target_vendor_id]

    if latest_only:
        # Get only the latest snapshot for each vendor
        latest_snapshot = result_df.sort_values(snapshot_date_col).groupby(vendor_id_col).last().reset_index()

        # Get the latest date for the target vendor
        if not target_vendor_data.empty:
            latest_target_date = target_vendor_data[snapshot_date_col].max()
            target_data = target_vendor_data[target_vendor_data[snapshot_date_col] == latest_target_date]

            # Calculate comparison metrics
            comparison = {}

            for metric in metrics:
                if metric in latest_snapshot.columns:
                    # Get target vendor's value
                    target_value = target_data[metric].iloc[0] if not target_data.empty else None

                    if target_value is not None:
                        # Calculate peer statistics
                        peer_mean = latest_snapshot[metric].mean()
                        peer_median = latest_snapshot[metric].median()
                        peer_std = latest_snapshot[metric].std()

                        # Calculate percentile if requested
                        if percentile:
                            percentile_rank = (latest_snapshot[metric] < target_value).mean() * 100
                        else:
                            percentile_rank = None

                        comparison[metric] = {
                            'target_value': target_value,
                            'peer_mean': peer_mean,
                            'peer_median': peer_median,
                            'peer_std': peer_std,
                            'percentile_rank': percentile_rank,
                            'difference_from_mean': target_value - peer_mean,
                            'standard_deviations_from_mean': (target_value - peer_mean) / peer_std if peer_std > 0 else 0
                        }

            return comparison
    else:
        # Compare across all snapshots
        comparison_by_snapshot = {}

        for snapshot_date, snapshot_group in result_df.groupby(snapshot_date_col):
            # Check if target vendor exists in this snapshot
            target_in_snapshot = target_vendor_data[target_vendor_data[snapshot_date_col] == snapshot_date]

            if not target_in_snapshot.empty:
                # Calculate comparison for this snapshot
                snapshot_comparison = {}

                for metric in metrics:
                    if metric in snapshot_group.columns:
                        # Get target vendor's value
                        target_value = target_in_snapshot[metric].iloc[0]

                        # Calculate peer statistics
                        peer_mean = snapshot_group[metric].mean()
                        peer_median = snapshot_group[metric].median()
                        peer_std = snapshot_group[metric].std()

                        # Calculate percentile if requested
                        if percentile:
                            percentile_rank = (snapshot_group[metric] < target_value).mean() * 100
                        else:
                            percentile_rank = None

                        snapshot_comparison[metric] = {
                            'target_value': target_value,
                            'peer_mean': peer_mean,
                            'peer_median': peer_median,
                            'peer_std': peer_std,
                            'percentile_rank': percentile_rank,
                            'difference_from_mean': target_value - peer_mean,
                            'standard_deviations_from_mean': (target_value - peer_mean) / peer_std if peer_std > 0 else 0
                        }

                comparison_by_snapshot[snapshot_date] = snapshot_comparison

        return comparison_by_snapshot


def aging_waterfall_analysis(df, snapshot_date_col='Snapshot_Date', vendor_id_col='Vendor ID',
                           aging_cols=None, include_total=True):
    """
    Create a waterfall analysis showing how aging buckets shift between snapshots.

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        snapshot_date_col (str): Column containing snapshot date
        vendor_id_col (str): Column containing vendor identifier
        aging_cols (list): List of aging bucket columns in chronological order
        include_total (bool): Whether to include total balance in the analysis

    Returns:
        dict: Waterfall analysis results by vendor
    """
    # Define default aging columns if not specified (in chronological order)
    if aging_cols is None:
        aging_cols = [
            'Future_Aging', 'Aging_0_30', 'Aging_31_60', 'Aging_61_90',
            'Aging_91_120', 'Aging_121_180', 'Aging_181_360', 'Above_361_Aging'
        ]

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(result_df[snapshot_date_col]):
        result_df[snapshot_date_col] = pd.to_datetime(result_df[snapshot_date_col], errors='coerce')

    # Add balance column to aging columns if requested
    analysis_cols = aging_cols.copy()
    if include_total:
        analysis_cols.append('Balance Outstanding')

    # Results container
    waterfall_results = {}

    # Analyze each vendor
    for vendor_id, group in result_df.groupby(vendor_id_col):
        # Sort by snapshot date
        vendor_data = group.sort_values(snapshot_date_col)

        # Need at least two snapshots for waterfall analysis
        if len(vendor_data) >= 2:
            # Store transitions between snapshots
            transitions = []

            for i in range(1, len(vendor_data)):
                prev_snapshot = vendor_data.iloc[i-1]
                curr_snapshot = vendor_data.iloc[i]

                prev_date = prev_snapshot[snapshot_date_col]
                curr_date = curr_snapshot[snapshot_date_col]

                # Calculate changes in each bucket
                changes = {}

                for col in analysis_cols:
                    prev_value = prev_snapshot[col]
                    curr_value = curr_snapshot[col]
                    changes[col] = curr_value - prev_value

                # Calculate implied bucket transitions
                # (This is a simplified approximation since we don't have actual cash flows)
                bucket_transitions = {}

                # For each aging bucket (except the last one)
                for j in range(len(aging_cols) - 1):
                    from_bucket = aging_cols[j]
                    to_bucket = aging_cols[j+1]

                    # Estimate amount that moved from one bucket to the next
                    # This is a very rough approximation
                    implied_flow = min(prev_snapshot[from_bucket], curr_snapshot[to_bucket])
                    bucket_transitions[f'{from_bucket}_to_{to_bucket}'] = implied_flow

                transitions.append({
                    'from_date': prev_date,
                    'to_date': curr_date,
                    'days_between': (curr_date - prev_date).days,
                    'changes': changes,
                    'implied_transitions': bucket_transitions
                })

            waterfall_results[vendor_id] = {
                'transitions': transitions,
                'latest_snapshot': vendor_data.iloc[-1][analysis_cols].to_dict(),
                'first_snapshot': vendor_data.iloc[0][analysis_cols].to_dict(),
                'total_change': {
                    col: vendor_data.iloc[-1][col] - vendor_data.iloc[0][col]
                    for col in analysis_cols
                }
            }

    return waterfall_results


def snapshot_comparison_analysis(df, snapshot_date_col='Snapshot_Date',
                               base_date=None, compare_date=None,
                               aging_cols=None, vendor_id_col='Vendor ID'):
    """
    Compare aging metrics between two snapshots.

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        snapshot_date_col (str): Column containing snapshot date
        base_date: Base snapshot date (if None, uses earliest date)
        compare_date: Comparison snapshot date (if None, uses latest date)
        aging_cols (list): List of aging bucket columns
        vendor_id_col (str): Column containing vendor identifier

    Returns:
        pandas.DataFrame: DataFrame with snapshot comparison metrics
    """
    # Define default aging columns if not specified
    if aging_cols is None:
        aging_cols = [
            'Future_Aging', 'Aging_0_30', 'Aging_31_60', 'Aging_61_90',
            'Aging_91_120', 'Aging_121_180', 'Aging_181_360', 'Above_361_Aging'
        ]

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(result_df[snapshot_date_col]):
        result_df[snapshot_date_col] = pd.to_datetime(result_df[snapshot_date_col], errors='coerce')

    # Determine base and comparison dates if not provided
    if base_date is None:
        base_date = result_df[snapshot_date_col].min()

    if compare_date is None:
        compare_date = result_df[snapshot_date_col].max()

    # Convert dates to datetime if they're not already
    base_date = pd.to_datetime(base_date)
    compare_date = pd.to_datetime(compare_date)

    # Create Aging_Beyond_90 column if it doesn't exist
    if 'Aging_Beyond_90' not in result_df.columns:
        aging_beyond_90_cols = [col for col in aging_cols if '91' in col or '120' in col or
                               '180' in col or '360' in col or 'Above_361' in col]
        result_df['Aging_Beyond_90'] = result_df[aging_beyond_90_cols].sum(axis=1)

    # Get data for base and comparison dates
    base_data = result_df[result_df[snapshot_date_col] == base_date]
    compare_data = result_df[result_df[snapshot_date_col] == compare_date]

    # Create pivot for easy comparison
    comparison_columns = ['Balance Outstanding', 'Aging_Beyond_90'] + aging_cols

    # Create a merged dataframe for comparison
    base_subset = base_data[[vendor_id_col] + comparison_columns].set_index(vendor_id_col)
    base_subset.columns = [f'{col}_Base' for col in base_subset.columns]

    compare_subset = compare_data[[vendor_id_col] + comparison_columns].set_index(vendor_id_col)
    compare_subset.columns = [f'{col}_Compare' for col in compare_subset.columns]

    # Merge the data
    comparison_df = base_subset.join(compare_subset, how='outer')

    # Calculate changes
    for col in comparison_columns:
        base_col = f'{col}_Base'
        compare_col = f'{col}_Compare'
        change_col = f'{col}_Change'
        pct_change_col = f'{col}_Pct_Change'

        comparison_df[change_col] = comparison_df[compare_col] - comparison_df[base_col]

        # Calculate percentage change, handling division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_change = (comparison_df[compare_col] - comparison_df[base_col]) / comparison_df[base_col].abs() * 100
            comparison_df[pct_change_col] = np.where(
                comparison_df[base_col] == 0,
                np.nan,
                pct_change
            )

    # Reset index to make vendor ID a regular column
    comparison_df = comparison_df.reset_index()

    # Add a direction column for key metrics
    for col in ['Balance Outstanding', 'Aging_Beyond_90']:
        comparison_df[f'{col}_Direction'] = np.where(
            comparison_df[f'{col}_Change'] > 0,
            'Increased',
            np.where(
                comparison_df[f'{col}_Change'] < 0,
                'Decreased',
                'Unchanged'
            )
        )

    return comparison_df


# =====================================================
# ANOMALY DETECTION FUNCTIONS
# =====================================================

def detect_multivariate_anomalies(df, snapshot_date_col='Snapshot_Date', vendor_id_col='Vendor ID',
                                features=None, contamination=0.05):
    """
    Detect multivariate anomalies using Isolation Forest across snapshots.

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        snapshot_date_col (str): Column containing snapshot date
        vendor_id_col (str): Column containing vendor identifier
        features (list): List of columns to use for anomaly detection
        contamination (float): Expected proportion of outliers (0.0 to 0.5)

    Returns:
        pandas.DataFrame: DataFrame with multivariate anomaly flags
    """
    # Define default features if not specified
    if features is None:
        features = [
            'Balance Outstanding', 'Future_Aging', 'Aging_0_30', 'Aging_31_60',
            'Aging_61_90', 'Aging_91_120', 'Aging_121_180', 'Aging_181_360',
            'Above_361_Aging'
        ]

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Initialize anomaly columns
    result_df['Multivariate_Anomaly'] = False
    result_df['Anomaly_Score'] = np.nan
    result_df['Anomaly_Reason'] = ""

    # Process each snapshot separately
    for snapshot_date, snapshot_df in result_df.groupby(snapshot_date_col):
        # Need enough data points for meaningful analysis
        if len(snapshot_df) < 20:
            continue

        # Prepare features
        # Make sure to use only columns that exist in the dataframe
        available_features = [f for f in features if f in snapshot_df.columns]
        if len(available_features) < 2:
            print(f"Warning: Not enough features available for snapshot {snapshot_date}")
            continue

        X = snapshot_df[available_features].fillna(0)

        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply Isolation Forest
        model = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = model.fit_predict(X_scaled)

        # Convert predictions to binary outlier flag (-1 for outliers, 1 for inliers)
        is_outlier = outlier_labels == -1

        # Calculate anomaly scores
        anomaly_scores = model.decision_function(X_scaled)
        normalized_scores = 1 - (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))

        # Update the dataframe
        snapshot_indices = snapshot_df.index
        result_df.loc[snapshot_indices, 'Multivariate_Anomaly'] = is_outlier
        result_df.loc[snapshot_indices, 'Anomaly_Score'] = normalized_scores

        # For outliers, determine which features contributed most
        if sum(is_outlier) > 0:
            # Calculate z-scores for each feature using numpy arrays
            feature_values = X.values
            feature_means = np.mean(feature_values, axis=0)
            feature_stds = np.std(feature_values, axis=0)

            # Replace zero std with 1 to avoid division by zero
            feature_stds[feature_stds == 0] = 1

            # Calculate z-scores
            z_scores_array = np.abs((feature_values - feature_means) / feature_stds)

            # Create anomaly reasons
            anomaly_reasons = []
            for i, (idx, is_out) in enumerate(zip(snapshot_indices, is_outlier)):
                if is_out:
                    # Get top contributing features for this outlier
                    feature_scores = []
                    for j, feature in enumerate(available_features):
                        feature_scores.append((feature, z_scores_array[i, j]))

                    # Sort by z-score and take top 3
                    feature_scores.sort(key=lambda x: x[1], reverse=True)
                    top_features = feature_scores[:3]

                    # Create reason string
                    reason = "; ".join([f"{feature} (z={score:.2f})" for feature, score in top_features])
                    result_df.loc[idx, 'Anomaly_Reason'] = reason
                else:
                    result_df.loc[idx, 'Anomaly_Reason'] = ""

    return result_df


def detect_velocity_anomalies(df, snapshot_date_col='Snapshot_Date', vendor_id_col='Vendor ID',
                            aging_cols=None, threshold=0.3):
    """
    Detect anomalies in the velocity of aging movement between buckets.

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        snapshot_date_col (str): Column containing snapshot date
        vendor_id_col (str): Column containing vendor identifier
        aging_cols (list): List of aging bucket columns in chronological order
        threshold (float): Threshold for flagging velocity anomalies

    Returns:
        pandas.DataFrame: DataFrame with velocity anomaly flags
    """
    # Define default aging columns if not specified
    if aging_cols is None:
        aging_cols = [
            'Future_Aging', 'Aging_0_30', 'Aging_31_60', 'Aging_61_90',
            'Aging_91_120', 'Aging_121_180', 'Aging_181_360', 'Above_361_Aging'
        ]

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Initialize anomaly columns
    result_df['Velocity_Anomaly'] = False
    result_df['Velocity_Anomaly_Score'] = np.nan
    result_df['Velocity_Anomaly_Reason'] = ""

    # Process each vendor separately
    for vendor_id, vendor_df in result_df.groupby(vendor_id_col):
        # Need at least two snapshots for velocity analysis
        if len(vendor_df) < 2:
            continue

        # Sort by snapshot date
        vendor_data = vendor_df.sort_values(snapshot_date_col)

        # Calculate aging distribution for each snapshot
        for i in range(len(vendor_data)):
            total_aging = sum(vendor_data.iloc[i][aging_cols])

            if total_aging != 0:
                for col in aging_cols:
                    vendor_data.loc[vendor_data.index[i], f'{col}_Pct'] = vendor_data.iloc[i][col] / total_aging
            else:
                for col in aging_cols:
                    vendor_data.loc[vendor_data.index[i], f'{col}_Pct'] = 0

        # Calculate velocity of change between snapshots
        if len(vendor_data) >= 2:
            for i in range(1, len(vendor_data)):
                velocity_anomalies = []

                for col in aging_cols:
                    pct_col = f'{col}_Pct'
                    prev_pct = vendor_data.iloc[i-1][pct_col]
                    curr_pct = vendor_data.iloc[i][pct_col]

                    # Calculate velocity of change
                    velocity = curr_pct - prev_pct

                    # Check if velocity exceeds threshold
                    if abs(velocity) > threshold:
                        direction = "increase" if velocity > 0 else "decrease"
                        velocity_anomalies.append(f"{col} {direction} by {abs(velocity):.1%}")

                if velocity_anomalies:
                    result_df.loc[vendor_data.index[i], 'Velocity_Anomaly'] = True
                    result_df.loc[vendor_data.index[i], 'Velocity_Anomaly_Score'] = len(velocity_anomalies) / len(aging_cols)
                    result_df.loc[vendor_data.index[i], 'Velocity_Anomaly_Reason'] = "; ".join(velocity_anomalies)

    return result_df


def detect_cohort_anomalies(df, snapshot_date_col='Snapshot_Date', vendor_id_col='Vendor ID',
                          group_by_col='country_name', value_col='Aging_Beyond_90',
                          threshold=2.0):
    """
    Detect vendors that are anomalous compared to their cohort (e.g., country).

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        snapshot_date_col (str): Column containing snapshot date
        vendor_id_col (str): Column containing vendor identifier
        group_by_col (str): Column defining the cohort (e.g., country, sales person)
        value_col (str): Column to analyze for anomalies
        threshold (float): Z-score threshold for flagging anomalies

    Returns:
        pandas.DataFrame: DataFrame with cohort anomaly flags
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Initialize anomaly columns
    result_df['Cohort_Anomaly'] = False
    result_df['Cohort_Zscore'] = np.nan
    result_df['Cohort_Deviation'] = np.nan

    # Process each snapshot separately
    for snapshot_date, snapshot_df in result_df.groupby(snapshot_date_col):
        # Process each cohort within the snapshot
        for group_value, group_df in snapshot_df.groupby(group_by_col):
            # Need enough data points for meaningful analysis
            if len(group_df) < 5:
                continue

            # Calculate cohort statistics
            cohort_mean = group_df[value_col].mean()
            cohort_std = group_df[value_col].std()

            if cohort_std > 0:  # Avoid division by zero
                # Calculate z-scores within cohort
                z_scores = np.abs((group_df[value_col] - cohort_mean) / cohort_std)

                # Flag anomalies
                anomaly_flags = z_scores > threshold

                # Update the main dataframe
                result_df.loc[group_df.index, 'Cohort_Anomaly'] = anomaly_flags
                result_df.loc[group_df.index, 'Cohort_Zscore'] = z_scores
                result_df.loc[group_df.index, 'Cohort_Deviation'] = group_df[value_col] - cohort_mean

    return result_df


def combine_anomaly_detectors(df, snapshot_date_col='Snapshot_Date', vendor_id_col='Vendor ID'):
    """
    Combine multiple anomaly detection methods for a comprehensive view.

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        snapshot_date_col (str): Column containing snapshot date
        vendor_id_col (str): Column containing vendor identifier

    Returns:
        pandas.DataFrame: DataFrame with combined anomaly results
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Apply each anomaly detection method
    print("Detecting multivariate anomalies...")
    multivariate_df = detect_multivariate_anomalies(result_df)

    print("Detecting velocity anomalies...")
    velocity_df = detect_velocity_anomalies(result_df)

    print("Detecting cohort anomalies...")
    cohort_df = detect_cohort_anomalies(result_df)

    # Combine results into a single dataframe
    result_df['Multivariate_Anomaly'] = multivariate_df['Multivariate_Anomaly']
    result_df['Multivariate_Score'] = multivariate_df['Anomaly_Score']
    result_df['Multivariate_Reason'] = multivariate_df['Anomaly_Reason'] if 'Anomaly_Reason' in multivariate_df.columns else ""

    result_df['Velocity_Anomaly'] = velocity_df['Velocity_Anomaly']
    result_df['Velocity_Score'] = velocity_df['Velocity_Anomaly_Score']
    result_df['Velocity_Reason'] = velocity_df['Velocity_Anomaly_Reason']

    result_df['Cohort_Anomaly'] = cohort_df['Cohort_Anomaly']
    result_df['Cohort_Score'] = cohort_df['Cohort_Zscore']

    # Create a combined anomaly flag and score
    result_df['Any_Anomaly'] = (
        result_df['Multivariate_Anomaly'] |
        result_df['Velocity_Anomaly'] |
        result_df['Cohort_Anomaly']
    )

    # Create a combined anomaly score (average of available scores)
    score_columns = [col for col in ['Multivariate_Score', 'Velocity_Score', 'Cohort_Score']
                    if col in result_df.columns]

    if score_columns:
        result_df['Combined_Anomaly_Score'] = result_df[score_columns].mean(axis=1)
    else:
        result_df['Combined_Anomaly_Score'] = np.nan

    # Create a comprehensive anomaly reason
    anomaly_reasons = []
    for idx, row in result_df.iterrows():
        reasons = []

        if row.get('Multivariate_Anomaly') and 'Multivariate_Reason' in row and row['Multivariate_Reason']:
            reasons.append(f"Multivariate: {row['Multivariate_Reason']}")

        if row.get('Velocity_Anomaly') and 'Velocity_Reason' in row and row['Velocity_Reason']:
            reasons.append(f"Velocity: {row['Velocity_Reason']}")

        if row.get('Cohort_Anomaly'):
            if 'Cohort_Deviation' in row:
                direction = "higher" if row['Cohort_Deviation'] > 0 else "lower"
                reasons.append(f"Differs from cohort: {direction} than peers")
            else:
                reasons.append("Differs from cohort")

        anomaly_reasons.append("; ".join(reasons))

    result_df['Anomaly_Details'] = anomaly_reasons

    return result_df


def segment_vendors_by_anomaly_patterns(df, snapshot_date_col='Snapshot_Date', vendor_id_col='Vendor ID',
                                     n_clusters=4):
    """
    Segment vendors based on their anomaly patterns over time.

    Args:
        df (pandas.DataFrame): Dataframe with anomaly detection results
        snapshot_date_col (str): Column containing snapshot date
        vendor_id_col (str): Column containing vendor identifier
        n_clusters (int): Number of clusters to create

    Returns:
        tuple: (DataFrame with anomaly pattern segments, cluster profiles)
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Calculate anomaly frequency for each vendor across snapshots
    vendor_anomaly_stats = []

    for vendor_id, vendor_df in result_df.groupby(vendor_id_col):
        # Count snapshots with different types of anomalies
        total_snapshots = len(vendor_df)

        if total_snapshots == 0:
            continue

        multivariate_count = sum(vendor_df.get('Multivariate_Anomaly', 0))
        velocity_count = sum(vendor_df.get('Velocity_Anomaly', 0))
        cohort_count = sum(vendor_df.get('Cohort_Anomaly', 0))
        any_count = sum(vendor_df.get('Any_Anomaly', 0))

        # Calculate frequencies
        vendor_stats = {
            'Vendor_ID': vendor_id,
            'Total_Snapshots': total_snapshots,
            'Multivariate_Frequency': multivariate_count / total_snapshots,
            'Velocity_Frequency': velocity_count / total_snapshots,
            'Cohort_Frequency': cohort_count / total_snapshots,
            'Any_Anomaly_Frequency': any_count / total_snapshots,
            'Max_Anomaly_Score': vendor_df.get('Combined_Anomaly_Score', 0).max() if 'Combined_Anomaly_Score' in vendor_df.columns else 0
        }

        vendor_anomaly_stats.append(vendor_stats)

# Create dataframe with vendor anomaly statistics
    anomaly_stats_df = pd.DataFrame(vendor_anomaly_stats)

    # Apply clustering to segment vendors
    if len(anomaly_stats_df) >= n_clusters:
        # Features for clustering
        features = [
            'Multivariate_Frequency', 'Velocity_Frequency',
            'Cohort_Frequency', 'Any_Anomaly_Frequency', 'Max_Anomaly_Score'
        ]

        # Prepare feature matrix
        X = anomaly_stats_df[features].fillna(0)

        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Add cluster assignments
        anomaly_stats_df['Anomaly_Cluster'] = clusters

        # Create descriptive labels
        cluster_profiles = {}
        for i in range(n_clusters):
            cluster_data = anomaly_stats_df[anomaly_stats_df['Anomaly_Cluster'] == i]

            # Determine key characteristics
            avg_frequency = cluster_data['Any_Anomaly_Frequency'].mean()
            max_score = cluster_data['Max_Anomaly_Score'].mean()

            # Primary anomaly type
            anomaly_types = ['Multivariate', 'Velocity', 'Cohort']
            type_frequencies = [
                cluster_data[f'{t}_Frequency'].mean() for t in anomaly_types
            ]
            primary_type = anomaly_types[np.argmax(type_frequencies)]

            # Create label
            if avg_frequency > 0.7:
                frequency_label = "Frequent"
            elif avg_frequency > 0.3:
                frequency_label = "Occasional"
            else:
                frequency_label = "Rare"

            if max_score > 0.7:
                severity_label = "Severe"
            elif max_score > 0.4:
                severity_label = "Moderate"
            else:
                severity_label = "Minor"

            label = f"Cluster {i+1}: {frequency_label} {severity_label} Anomalies ({primary_type})"

            cluster_profiles[i] = {
                'label': label,
                'size': len(cluster_data),
                'avg_frequency': avg_frequency,
                'avg_max_score': max_score,
                'primary_type': primary_type
            }

        # Map clusters back to the main dataframe
        vendor_cluster_map = anomaly_stats_df[['Vendor_ID', 'Anomaly_Cluster']].set_index('Vendor_ID')

        # Add cluster and label to each snapshot
        result_df['Anomaly_Cluster'] = result_df[vendor_id_col].map(vendor_cluster_map['Anomaly_Cluster'])
        result_df['Anomaly_Cluster_Label'] = result_df['Anomaly_Cluster'].map({i: profile['label'] for i, profile in cluster_profiles.items()})

        return result_df, cluster_profiles
    else:
        # Not enough vendors for clustering
        return result_df, {}


# =====================================================
# DECISION SUPPORT FUNCTIONS
# =====================================================

def calculate_collection_priority_score(df, snapshot_date_col='Snapshot_Date',
                                      balance_col='Balance Outstanding',
                                      aging_beyond_90_col='Aging_Beyond_90',
                                      days_since_payment_col='Days_Since_Payment',
                                      latest_snapshot_only=True):
    """
    Calculate a collection priority score for each vendor based on multiple factors.

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        snapshot_date_col (str): Column containing snapshot date
        balance_col (str): Column containing outstanding balance
        aging_beyond_90_col (str): Column containing aging beyond 90 days
        days_since_payment_col (str): Column containing days since last payment
        latest_snapshot_only (bool): Whether to use only the latest snapshot

    Returns:
        pandas.DataFrame: DataFrame with collection priority scores
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Ensure we have required columns
    if aging_beyond_90_col not in result_df.columns:
        # Calculate it if needed
        aging_beyond_90_cols = [col for col in result_df.columns
                              if '91' in col or '120' in col or '180' in col or '360' in col or 'Above_361' in col]
        result_df[aging_beyond_90_col] = result_df[aging_beyond_90_cols].sum(axis=1)

    # Use only the latest snapshot if requested
    if latest_snapshot_only:
        # Get the latest snapshot for each vendor
        result_df = result_df.sort_values(snapshot_date_col).groupby('Vendor ID').last().reset_index()

    # Calculate components of the priority score

    # 1. Balance component (higher absolute balance = higher priority)
    result_df['Balance_Component'] = result_df[balance_col].abs() / result_df[balance_col].abs().max()

    # 2. Aging component (higher aging ratio = higher priority)
    with np.errstate(divide='ignore', invalid='ignore'):
        result_df['Aging_Ratio'] = result_df[aging_beyond_90_col] / result_df[balance_col].abs()

    result_df['Aging_Ratio'] = result_df['Aging_Ratio'].fillna(0).clip(0, 1)
    result_df['Aging_Component'] = result_df['Aging_Ratio']

    # 3. Time component (more days since payment = higher priority)
    if days_since_payment_col in result_df.columns:
        # Normalize to 0-1 range
        max_days = max(365, result_df[days_since_payment_col].max())  # Cap at 1 year
        result_df['Time_Component'] = result_df[days_since_payment_col] / max_days
        result_df['Time_Component'] = result_df['Time_Component'].fillna(0.5).clip(0, 1)  # Default if missing
    else:
        result_df['Time_Component'] = 0.5  # Default mid-value if not available

    # 4. Trend component (worsening trend = higher priority)
    if 'Aging_Beyond_90_MoM_Change' in result_df.columns:
        # Convert to 0-1 range where 1 = rapid worsening
        max_change = 100  # Cap at 100% increase
        result_df['Trend_Component'] = (
            result_df['Aging_Beyond_90_MoM_Change'].clip(-max_change, max_change) + max_change
        ) / (2 * max_change)
    else:
        result_df['Trend_Component'] = 0.5  # Default mid-value if not available

    # Calculate the combined score (weighted average)
    result_df['Collection_Priority_Score'] = (
        0.35 * result_df['Balance_Component'] +
        0.30 * result_df['Aging_Component'] +
        0.20 * result_df['Time_Component'] +
        0.15 * result_df['Trend_Component']
    )

    # Scale to 0-100 range
    result_df['Collection_Priority_Score'] = result_df['Collection_Priority_Score'] * 100

    # Assign priority categories
    result_df['Collection_Priority'] = pd.cut(
        result_df['Collection_Priority_Score'],
        bins=[0, 25, 50, 75, 100],
        labels=['Low', 'Medium', 'High', 'Critical']
    )

    return result_df


def identify_root_causes(df, snapshot_date_col='Snapshot_Date',
                       vendor_id_col='Vendor ID',
                       aging_beyond_90_col='Aging_Beyond_90',
                       features=None, min_samples=50):
    """
    Identify root causes of aging issues using decision trees.

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        snapshot_date_col (str): Column containing snapshot date
        vendor_id_col (str): Column containing vendor identifier
        aging_beyond_90_col (str): Column containing aging beyond 90 days
        features (list): List of features to consider as potential causes
        min_samples (int): Minimum number of samples needed for analysis

    Returns:
        dict: Root cause analysis results
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Define default features if not provided
    if features is None:
        features = [
            'country_id', 'SalesPersonID', 'AccountManager_ID',
            'Days_Since_Payment'
        ]

    # Ensure we have the aging column
    if aging_beyond_90_col not in result_df.columns:
        # Calculate it if needed
        aging_beyond_90_cols = [col for col in result_df.columns
                              if '91' in col or '120' in col or '180' in col or '360' in col or 'Above_361' in col]
        result_df[aging_beyond_90_col] = result_df[aging_beyond_90_cols].sum(axis=1)

    # Use only the latest snapshot for each vendor
    latest_df = result_df.sort_values(snapshot_date_col).groupby(vendor_id_col).last().reset_index()

    # Check if we have enough samples
    if len(latest_df) < min_samples:
        return {
            'error': f'Not enough samples for reliable analysis. Need at least {min_samples}, got {len(latest_df)}.'
        }

    # Create the target variable: whether vendor has significant aging issues
    with np.errstate(divide='ignore', invalid='ignore'):
        latest_df['Has_Aging_Issues'] = (
            (latest_df[aging_beyond_90_col] / latest_df['Balance Outstanding'].abs()) > 0.3
        ).astype(int)

    # Prepare features and target
    X = latest_df[features].fillna(-1)  # Replace missing values with -1
    y = latest_df['Has_Aging_Issues']

    # Apply decision tree for root cause analysis
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=max(5, min_samples // 20))
    tree.fit(X, y)

    # Extract feature importance
    importance = {feature: importance for feature, importance in zip(features, tree.feature_importances_)}

    # Get tree rules
    tree_rules = export_text(tree, feature_names=features)

    # Identify key factors from the tree
    key_factors = []
    for feature, importance_score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        if importance_score > 0.05:  # Only include significant factors
            key_factors.append({
                'factor': feature,
                'importance': importance_score,
                'description': get_factor_description(latest_df, feature, aging_beyond_90_col)
            })

    return {
        'key_factors': key_factors,
        'feature_importance': importance,
        'tree_rules': tree_rules
    }


def get_factor_description(df, factor, aging_col):
    """
    Generate a natural language description of how a factor relates to aging.

    Args:
        df (pandas.DataFrame): Data to analyze
        factor (str): Factor column name
        aging_col (str): Aging column name

    Returns:
        str: Description of the factor's impact
    """
    descriptions = {
        'country_id': "Geographic location",
        'SalesPersonID': "Sales representative",
        'AccountManager_ID': "Account manager",
        'Days_Since_Payment': "Time since last payment"
    }

    # Special case for days since payment
    if factor == 'Days_Since_Payment':
        # Find the threshold where aging significantly increases
        thresholds = [30, 60, 90, 120, 180]
        best_threshold = None
        max_difference = 0

        for threshold in thresholds:
            group1 = df[df[factor] <= threshold][aging_col].mean()
            group2 = df[df[factor] > threshold][aging_col].mean()
            difference = group2 - group1

            if difference > max_difference:
                max_difference = difference
                best_threshold = threshold

        if best_threshold:
            return f"Vendors who haven't paid in over {best_threshold} days have significantly higher aging"

    # Default description
    base_desc = descriptions.get(factor, factor)
    return f"{base_desc} is associated with differing aging patterns"


def create_action_recommendations(df, priority_threshold=75):
    """
    Create tailored action recommendations based on vendor priority and characteristics.
    
    Args:
        df (pandas.DataFrame): Dataframe with collection priority scores
        priority_threshold (float): Threshold for high priority vendors
        
    Returns:
        pandas.DataFrame: Dataframe with recommended actions
    """
    # Create a copy to work with
    result_df = df.copy()
    
    # Initialize recommendation columns
    result_df['Recommended_Action'] = ""
    result_df['Action_Urgency'] = ""
    result_df['Follow_Up_Frequency'] = ""
    result_df['Specific_Instructions'] = ""
    
    # High priority vendors (critical)
    high_priority = result_df['Collection_Priority_Score'] >= priority_threshold
    result_df.loc[high_priority, 'Action_Urgency'] = "Immediate"
    result_df.loc[high_priority, 'Follow_Up_Frequency'] = "Weekly"
    
    # Determine specific actions based on aging and days since payment
    
    # 1. Long overdue with high aging ratio
    long_overdue_high_aging = (
        (result_df['Days_Since_Payment'] > 90) & 
        (result_df['Aging_Ratio'] > 0.5) &
        high_priority
    )
    result_df.loc[long_overdue_high_aging, 'Recommended_Action'] = "Escalate to management"
    result_df.loc[long_overdue_high_aging, 'Specific_Instructions'] = (
        "Contact senior management; consider credit hold; "
        "request payment plan or security; involve legal if no response"
    )
    
    # 2. High aging but recent payment
    high_aging_recent_payment = (
        (result_df['Days_Since_Payment'] <= 90) & 
        (result_df['Aging_Ratio'] > 0.5) &
        high_priority
    )
    result_df.loc[high_aging_recent_payment, 'Recommended_Action'] = "Structured payment plan"
    result_df.loc[high_aging_recent_payment, 'Specific_Instructions'] = (
        "Acknowledge recent payment; request commitment on payment plan for remaining balance; "
        "consider partial credit hold until aging ratio improves"
    )
    
    # 3. Medium priority vendors
    medium_priority = (
        (result_df['Collection_Priority_Score'] >= 50) & 
        (result_df['Collection_Priority_Score'] < priority_threshold)
    )
    result_df.loc[medium_priority, 'Action_Urgency'] = "This week"
    result_df.loc[medium_priority, 'Follow_Up_Frequency'] = "Bi-weekly"
    result_df.loc[medium_priority, 'Recommended_Action'] = "Proactive contact"
    result_df.loc[medium_priority, 'Specific_Instructions'] = (
        "Review account status with customer; send aging statement; "
        "identify any invoice disputes; request payment timeline"
    )
    
    # 4. Low priority vendors
    low_priority = result_df['Collection_Priority_Score'] < 50
    result_df.loc[low_priority, 'Action_Urgency'] = "Routine"
    result_df.loc[low_priority, 'Follow_Up_Frequency'] = "Monthly"
    result_df.loc[low_priority, 'Recommended_Action'] = "Monitor"
    result_df.loc[low_priority, 'Specific_Instructions'] = (
        "Regular account review; send automated statements; "
        "watch for changes in payment patterns"
    )
    
    return result_df


# =====================================================
# COMPREHENSIVE ANALYSIS CLASS
# =====================================================

class VendorAgingAnalytics:
    """
    A comprehensive analytics engine for vendor aging data that handles multiple snapshots
    and provides descriptive, diagnostic, and prescriptive insights.
    """
    
    def __init__(self, output_dir='vendor_aging_report'):
        """
        Initialize the analytics engine.
        
        Args:
            output_dir (str): Directory to save analysis outputs and visualizations
        """
        self.output_dir = output_dir
        self.df = None
        self.prepared_df = None
        self.metrics_df = None
        self.trends_df = None
        self.payment_df = None
        self.anomalies_df = None
        self.cohort_df = None
        self.cohort_profiles = None
        self.priority_df = None
        self.recommendations_df = None
        self.today = datetime.now().strftime('%Y-%m-%d')
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_data(self, file_path):
        """
        Load vendor aging data from a CSV file.
        
        Args:
            file_path (str): Path to the vendor aging CSV file
            
        Returns:
            self: For method chaining
        """
        self.df = pd.read_csv(file_path)
        return self
    
    def prepare_data(self):
        """
        Prepare the data for analysis by applying the data preparation functions.
        
        Returns:
            self: For method chaining
        """
        print("Preparing data for analysis...")
        self.prepared_df = prepare_time_series_data(self.df)
        self.metrics_df = calculate_aging_metrics_over_time(self.prepared_df)
        self.trends_df = calculate_aging_trends(self.metrics_df)
        self.payment_df = vendor_payment_history_analysis(self.trends_df)
        return self
    
    def run_descriptive_analytics(self):
        """
        Generate descriptive analytics about the current state of aging.
        
        Returns:
            dict: Descriptive analytics results
        """
        print("Generating descriptive analytics...")
        
        # Ensure data is prepared
        if self.payment_df is None:
            self.prepare_data()
        
        # Get the latest snapshot date for current metrics
        latest_date = self.payment_df['Snapshot_Date'].max()
        latest_data = self.payment_df[self.payment_df['Snapshot_Date'] == latest_date]
        
        # Calculate overall aging metrics
        total_balance = latest_data['Balance Outstanding'].sum()
        total_aging_beyond_90 = latest_data['Aging_Beyond_90'].sum()
        avg_pct_aging_beyond_90 = latest_data['Pct_Aging_Beyond_90'].mean()
        
        # Calculate distribution by aging bucket
        aging_cols = [
            'Future_Aging', 'Aging_0_30', 'Aging_31_60', 'Aging_61_90',
            'Aging_91_120', 'Aging_121_180', 'Aging_181_360', 'Above_361_Aging'
        ]
        
        aging_distribution = {col: latest_data[col].sum() for col in aging_cols}
        aging_distribution_pct = {col: value / total_balance * 100 for col, value in aging_distribution.items()}
        
        # Calculate country performance
        country_performance = (
            latest_data.groupby('country_name')
            .agg({
                'Balance Outstanding': 'sum',
                'Aging_Beyond_90': 'sum',
                'Vendor ID': 'nunique'
            })
            .rename(columns={'Vendor ID': 'vendor_count'})
            .reset_index()
        )
        
        # Store results
        results = {
            'snapshot_date': latest_date.strftime('%Y-%m-%d') if isinstance(latest_date, pd.Timestamp) else str(latest_date),
            'total_balance': float(total_balance),
            'total_aging_beyond_90': float(total_aging_beyond_90),
            'pct_aging_beyond_90': float(total_aging_beyond_90 / total_balance * 100),
            'aging_distribution': aging_distribution,
            'aging_distribution_pct': aging_distribution_pct,
            'country_performance': country_performance.to_dict('records'),
            'run_date': self.today
        }
        
        # Save to file
        with open(f"{self.output_dir}/descriptive_analytics.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualization
        self._generate_descriptive_visualizations(latest_date, latest_data, aging_cols, country_performance)
        
        return results
    
    def _generate_descriptive_visualizations(self, latest_date, latest_data, aging_cols, country_performance):
        """
        Generate visualizations for descriptive analytics.
        
        Args:
            latest_date: Date of the latest snapshot
            latest_data: DataFrame filtered to the latest snapshot
            aging_cols: List of aging bucket columns
            country_performance: DataFrame with country performance metrics
        """
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Aging Bucket Distribution
        plt.subplot(2, 2, 1)
        aging_labels = ['Future', '0-30', '31-60', '61-90', '91-120', '121-180', '181-360', '361+']
        aging_values = [latest_data[col].sum() for col in aging_cols]
        plt.pie(aging_values, labels=aging_labels, autopct='%1.1f%%', startangle=90,
               colors=sns.color_palette("YlOrRd", len(aging_cols)))
        plt.axis('equal')
        plt.title(f'Aging Distribution as of {latest_date.strftime("%Y-%m-%d")}')
        
        # Plot 2: Top Countries by Aging Beyond 90
        plt.subplot(2, 2, 2)
        country_perf_sorted = country_performance.sort_values('Aging_Beyond_90', ascending=False).head(5)
        country_perf_sorted.plot(kind='bar', x='country_name', y='Aging_Beyond_90', ax=plt.gca(), color='coral')
        plt.title('Top 5 Countries by Aging Beyond 90')
        plt.xlabel('Country')
        plt.ylabel('Amount ($)')
        plt.xticks(rotation=45, ha='right')
        
        # Plot 3: Aging Trend Over Time
        plt.subplot(2, 2, 3)
        aging_by_snapshot = self.payment_df.groupby('Snapshot_Date')[['Balance Outstanding', 'Aging_Beyond_90']].sum()
        aging_by_snapshot.plot(ax=plt.gca())
        plt.title('Balance and Aging Beyond 90 Over Time')
        plt.xlabel('Snapshot Date')
        plt.ylabel('Amount ($)')
        plt.grid(True)
        
        # Plot 4: Aging by Trend Category
        plt.subplot(2, 2, 4)
        trend_counts = latest_data['Aging_Trend_Category'].value_counts().sort_index()
        colors = ['green', 'lightgreen', 'gray', 'orange', 'red']
        trend_counts.plot(kind='bar', color=colors)
        plt.title('Vendors by Aging Trend Category')
        plt.xlabel('Trend Category')
        plt.ylabel('Number of Vendors')
        plt.grid(axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/descriptive_analytics.png", dpi=300)
        plt.close()
    
    def run_diagnostic_analytics(self):
        """
        Generate diagnostic analytics to identify patterns, cohorts, and anomalies.
        
        Returns:
            dict: Diagnostic analytics results
        """
        print("Running diagnostic analytics...")
        
        # Ensure data is prepared
        if self.payment_df is None:
            self.prepare_data()
        
        # Run cohort analysis
        self.cohort_df, self.cohort_profiles = create_vendor_cohorts(self.payment_df)
        
        # Run anomaly detection
        self.anomalies_df = detect_multivariate_anomalies(self.cohort_df)
        
        # Get the latest snapshot date
        latest_date = self.anomalies_df['Snapshot_Date'].max()
        latest_data = self.anomalies_df[self.anomalies_df['Snapshot_Date'] == latest_date]
        
        # Count anomalies
        num_anomalies = latest_data['Multivariate_Anomaly'].sum()
        pct_anomalies = num_anomalies / len(latest_data) * 100
        
        # Create results dictionary
        results = {
            'snapshot_date': latest_date.strftime('%Y-%m-%d') if isinstance(latest_date, pd.Timestamp) else str(latest_date),
            'cohort_analysis': {
                'num_cohorts': len(self.cohort_profiles),
                'cohort_profiles': {str(k): v for k, v in self.cohort_profiles.items()}
            },
            'anomaly_detection': {
                'num_anomalies': int(num_anomalies),
                'pct_anomalies': float(pct_anomalies),
                'top_anomalies': latest_data[latest_data['Multivariate_Anomaly']]
                    .sort_values('Anomaly_Score', ascending=False)
                    .head(10)[['Vendor ID', 'Vendor', 'Balance Outstanding', 'Aging_Beyond_90', 'Anomaly_Score', 'Anomaly_Reason']]
                    .to_dict('records')
            },
            'run_date': self.today
        }
        
        # Save to file
        with open(f"{self.output_dir}/diagnostic_analytics.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate visualizations
        self._generate_cohort_visualization()
        self._generate_anomaly_visualization(latest_date, latest_data)
        
        return results
    
    def _generate_cohort_visualization(self):
        """Generate cohort analysis visualization."""
        plt.figure(figsize=(14, 8))
        
        # Get the latest snapshot
        latest_date = self.cohort_df['Snapshot_Date'].max()
        latest_cohort = self.cohort_df[self.cohort_df['Snapshot_Date'] == latest_date]
        
        # Plot cohorts in 2D space
        scatter = plt.scatter(
            latest_cohort['Pct_Aging_Beyond_90'], 
            latest_cohort['Aging_Beyond_90_Trend'], 
            c=latest_cohort['Cohort'], 
            s=latest_cohort['Balance Outstanding'] / 1000,  # Size proportional to balance
            alpha=0.6,
            cmap='viridis'
        )
        
        # Add cluster centers
        for cohort_id, profile in self.cohort_profiles.items():
            plt.scatter(
                profile['avg_pct_aging_beyond_90'],
                profile['avg_trend'],
                s=300,
                c=[cohort_id],
                cmap='viridis',
                marker='X',
                edgecolors='black',
                linewidth=2
            )
            plt.annotate(
                f"Cohort {cohort_id+1}",
                (profile['avg_pct_aging_beyond_90'], profile['avg_trend']),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontweight='bold'
            )
        
        plt.colorbar(scatter, label='Cohort')
        plt.title('Vendor Cohorts: % Aging Beyond 90 Days vs. Aging Trend')
        plt.xlabel('Percentage of Aging Beyond 90 Days')
        plt.ylabel('Aging Beyond 90 Days Trend (%)')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        plt.axvline(x=30, color='red', linestyle='--', alpha=0.3)
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/cohort_analysis.png", dpi=300)
        plt.close()
    
    def _generate_anomaly_visualization(self, latest_date, latest_data):
        """Generate anomaly detection visualization."""
        plt.figure(figsize=(14, 6))
        
        # Use two important features for visualization
        x_feature = 'Balance Outstanding'
        y_feature = 'Aging_Beyond_90'
        
        # Create scatter plot
        plt.scatter(
            latest_data[x_feature], 
            latest_data[y_feature],
            c=latest_data['Multivariate_Anomaly'].map({True: 'red', False: 'blue'}),
            alpha=0.6,
            s=100
        )
        
        # Add labels for anomalies
        for _, row in latest_data[latest_data['Multivariate_Anomaly']].iterrows():
            plt.annotate(
                f"Vendor {row['Vendor ID']}",
                (row[x_feature], row[y_feature]),
                textcoords="offset points",
                xytext=(5, 5),
                ha='left'
            )
        
        plt.title(f'Multivariate Anomalies - {latest_date.strftime("%Y-%m-%d")}')
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.legend(['Normal', 'Anomaly'])
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/anomaly_detection.png", dpi=300)
        plt.close()
    
    def run_prescriptive_analytics(self):
        """
        Generate prescriptive analytics with recommended actions.
        
        Returns:
            dict: Prescriptive analytics results
        """
        print("Generating prescriptive recommendations...")
        
        # Ensure previous analyses are complete
        if self.anomalies_df is None:
            self.run_diagnostic_analytics()
        
        # Calculate collection priorities
        self.priority_df = calculate_collection_priority_score(self.anomalies_df)
        
        # Create action recommendations
        self.recommendations_df = create_action_recommendations(self.priority_df)
        
        # Get priority distribution
        priority_counts = self.recommendations_df['Collection_Priority'].value_counts()
        action_counts = self.recommendations_df['Recommended_Action'].value_counts()
        
        # Extract critical vendors
        critical_vendors = self.recommendations_df[
            self.recommendations_df['Collection_Priority'] == 'Critical'
        ].sort_values('Collection_Priority_Score', ascending=False)
        
        # Create results dictionary
        results = {
            'priority_distribution': priority_counts.to_dict(),
            'action_distribution': action_counts.to_dict(),
            'critical_vendors': critical_vendors[
                ['Vendor ID', 'Vendor', 'Balance Outstanding', 'Aging_Beyond_90', 
                 'Aging_Ratio', 'Days_Since_Payment', 'Collection_Priority_Score', 
                 'Recommended_Action', 'Action_Urgency']
            ].head(20).to_dict('records'),
            'run_date': self.today
        }
        
        # Save to file
        with open(f"{self.output_dir}/prescriptive_analytics.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate visualization
        self._generate_priority_visualization()
        
        # Save recommendations to CSV
        self.recommendations_df.to_csv(f"{self.output_dir}/vendor_recommendations.csv", index=False)
        
        return results
    
    def _generate_priority_visualization(self):
        """Generate priority distribution visualization."""
        plt.figure(figsize=(14, 6))
        
        # Priority counts
        priority_counts = self.recommendations_df['Collection_Priority'].value_counts().sort_index()
        ax = priority_counts.plot(kind='bar', color=['green', 'yellow', 'orange', 'red'])
        plt.title('Distribution of Collection Priorities')
        plt.xlabel('Priority')
        plt.ylabel('Number of Vendors')
        plt.grid(axis='y')
        
        # Add count labels on bars
        for i, v in enumerate(priority_counts):
            ax.text(i, v + 5, str(v), ha='center')
        
        plt.savefig(f"{self.output_dir}/priority_distribution.png", dpi=300)
        plt.close()
    
    def run_all_analytics(self):
        """
        Run all analytics processes and generate comprehensive results.
        
        Returns:
            dict: Combined analytics results
        """
        print("Running comprehensive vendor aging analysis...")
        
        # Start with data preparation
        if self.df is None:
            raise ValueError("No data loaded. Please call load_data() first.")
        
        self.prepare_data()
        
        # Run all analyses
        descriptive = self.run_descriptive_analytics()
        diagnostic = self.run_diagnostic_analytics()
        prescriptive = self.run_prescriptive_analytics()
        
        # Create executive summary
        latest_date = self.recommendations_df['Snapshot_Date'].max()
        num_vendors = self.recommendations_df['Vendor ID'].nunique()
        total_balance = self.recommendations_df['Balance Outstanding'].sum()
        total_aging_beyond_90 = self.recommendations_df['Aging_Beyond_90'].sum()
        
        exec_summary = {
            "report_date": self.today,
            "data_as_of": latest_date.strftime("%Y-%m-%d") if isinstance(latest_date, pd.Timestamp) else str(latest_date),
            "total_vendors": int(num_vendors),
            "total_balance": float(total_balance),
            "total_aging_beyond_90": float(total_aging_beyond_90),
            "pct_aging_beyond_90": float(total_aging_beyond_90 / total_balance * 100),
            "num_critical_priority": int(self.recommendations_df[self.recommendations_df['Collection_Priority'] == 'Critical'].shape[0]),
            "critical_balance": float(self.recommendations_df[self.recommendations_df['Collection_Priority'] == 'Critical']['Balance Outstanding'].sum()),
            "anomalies_detected": int(self.anomalies_df['Multivariate_Anomaly'].sum()),
            "cohort_summary": [
                {
                    "cohort_id": cohort_id,
                    "label": profile['label'],
                    "size": profile['size'],
                    "avg_pct_aging_beyond_90": float(profile['avg_pct_aging_beyond_90'])
                }
                for cohort_id, profile in self.cohort_profiles.items()
            ],
            "key_actions": {
                action: int(count) for action, count in self.recommendations_df['Recommended_Action'].value_counts().items()
            }
        }
        
        # Save executive summary
        with open(f"{self.output_dir}/executive_summary.json", 'w') as f:
            json.dump(exec_summary, f, indent=2)
        
        # Combine all results
        all_results = {
            'executive_summary': exec_summary,
            'descriptive': descriptive,
            'diagnostic': diagnostic,
            'prescriptive': prescriptive
        }
        
        print(f"Analysis complete! All results saved to '{self.output_dir}' directory.")
        return all_results
