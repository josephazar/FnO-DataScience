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
from sklearn.tree import DecisionTreeClassifier
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
                              value_cols=None, zscore_threshold=3.0):
    """
    Detect anomalies in vendor data based on time series patterns.

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        snapshot_date_col (str): Column containing snapshot date
        vendor_id_col (str): Column containing vendor identifier
        value_cols (list): List of columns to check for anomalies
        zscore_threshold (float): Z-score threshold for flagging anomalies

    Returns:
        pandas.DataFrame: DataFrame with time-based anomaly flags
    """
    # Define default value columns if not specified
    if value_cols is None:
        value_cols = ['Balance Outstanding', 'Aging_Beyond_90', 'Pct_Aging_Beyond_90']

    # Ensure we have time-based metrics
    if 'Aging_Beyond_90' not in df.columns:
        df = calculate_aging_metrics_over_time(df)

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Initialize anomaly columns
    for col in value_cols:
        result_df[f'{col}_Anomaly'] = False
        result_df[f'{col}_Zscore'] = np.nan

    result_df['Time_Anomaly'] = False
    result_df['Anomaly_Details'] = ''

    # Detect anomalies for each vendor over time
    for vendor_id, group in result_df.groupby(vendor_id_col):
        if len(group) >= 3:  # Need at least 3 data points for meaningful time series
            vendor_data = group.sort_values(snapshot_date_col)

            for col in value_cols:
                if col in vendor_data.columns:
                    # Calculate z-scores for this vendor's time series
                    mean_val = vendor_data[col].mean()
                    std_val = vendor_data[col].std()

                    if std_val > 0:  # Avoid division by zero
                        zscores = np.abs((vendor_data[col] - mean_val) / std_val)

                        # Flag anomalies
                        anomalies = zscores > zscore_threshold
                        result_df.loc[vendor_data.index, f'{col}_Anomaly'] = anomalies
                        result_df.loc[vendor_data.index, f'{col}_Zscore'] = zscores

            # Combine anomalies across columns
            for idx, row in vendor_data.iterrows():
                anomaly_cols = [col for col in value_cols
                              if f'{col}_Anomaly' in row.index and row[f'{col}_Anomaly']]

                if anomaly_cols:
                    result_df.loc[idx, 'Time_Anomaly'] = True

                    # Create detailed description
                    details = []
                    for col in anomaly_cols:
                        direction = "increase" if row[col] > mean_val else "decrease"
                        z_val = row[f'{col}_Zscore']
                        details.append(f"Unusual {direction} in {col} (z={z_val:.2f})")

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

    # Analyze each vendor with sufficient data
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
    from sklearn.cluster import KMeans
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
        pandas.DataFrame: Comparison results
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
# PRACTICAL ANALYTICS FUNCTIONS FOR DECISION MAKING
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
    from sklearn.tree import DecisionTreeClassifier, export_text

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


def identify_seasonal_anomalies(df, snapshot_date_col='Snapshot_Date', vendor_id_col='Vendor ID',
                               value_col='Balance Outstanding', expected_seasonality=12,
                               threshold=2.0):
    """
    Identify vendors with anomalous behavior compared to their seasonal pattern.

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        snapshot_date_col (str): Column containing snapshot date
        vendor_id_col (str): Column containing vendor identifier
        value_col (str): Column to analyze for seasonal anomalies
        expected_seasonality (int): Expected seasonality period (12 for annual)
        threshold (float): Threshold for flagging seasonal anomalies

    Returns:
        pandas.DataFrame: DataFrame with seasonal anomaly flags
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Add seasonal anomaly columns
    result_df['Seasonal_Anomaly'] = False
    result_df['Seasonal_Deviation'] = np.nan

    # Process each vendor with enough data
    for vendor_id, vendor_df in result_df.groupby(vendor_id_col):
        # Need enough data for seasonal analysis
        if len(vendor_df) < expected_seasonality * 2:
            continue

        # Sort by snapshot date
        vendor_data = vendor_df.sort_values(snapshot_date_col)

        # Extract the month for each snapshot
        vendor_data['Month'] = pd.to_datetime(vendor_data[snapshot_date_col]).dt.month

        # Calculate average value by month (seasonal pattern)
        monthly_avg = vendor_data.groupby('Month')[value_col].mean()
        monthly_std = vendor_data.groupby('Month')[value_col].std()

        # Flag anomalies by comparing each snapshot to the monthly average
        for idx, row in vendor_data.iterrows():
            month = row['Month']
            expected_value = monthly_avg[month]
            month_std = monthly_std[month]

            if month_std > 0:  # Avoid division by zero
                deviation = abs(row[value_col] - expected_value) / month_std

                # Flag as anomaly if deviation exceeds threshold
                if deviation > threshold:
                    result_df.loc[idx, 'Seasonal_Anomaly'] = True
                    result_df.loc[idx, 'Seasonal_Deviation'] = deviation

    return result_df


def detect_payment_pattern_anomalies(df, snapshot_date_col='Snapshot_Date', vendor_id_col='Vendor ID',
                                 payment_date_col='LP Date', payment_amount_col='Vendor LP Amount'):
    """
    Detect anomalies in vendor payment patterns across snapshots.

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        snapshot_date_col (str): Column containing snapshot date
        vendor_id_col (str): Column containing vendor identifier
        payment_date_col (str): Column containing last payment date
        payment_amount_col (str): Column containing last payment amount

    Returns:
        pandas.DataFrame: DataFrame with payment pattern anomaly flags
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Ensure date columns are datetime
    if payment_date_col in result_df.columns and not pd.api.types.is_datetime64_any_dtype(result_df[payment_date_col]):
        result_df[payment_date_col] = pd.to_datetime(result_df[payment_date_col], errors='coerce')

    if snapshot_date_col in result_df.columns and not pd.api.types.is_datetime64_any_dtype(result_df[snapshot_date_col]):
        result_df[snapshot_date_col] = pd.to_datetime(result_df[snapshot_date_col], errors='coerce')

    # Initialize anomaly columns
    result_df['Payment_Pattern_Anomaly'] = False
    result_df['Payment_Pattern_Reason'] = ""

    # Process each vendor separately
    for vendor_id, vendor_df in result_df.groupby(vendor_id_col):
        # Need enough snapshots for payment pattern analysis
        if len(vendor_df) < 3:
            continue

        # Sort by snapshot date
        vendor_data = vendor_df.sort_values(snapshot_date_col)

        # Track payment dates across snapshots
        payment_dates = []
        payment_amounts = []

        for idx, row in vendor_data.iterrows():
            payment_date = row[payment_date_col]
            payment_amount = row[payment_amount_col]

            # Check if this is a new payment date
            if payment_date is not pd.NaT and (
                len(payment_dates) == 0 or payment_date != payment_dates[-1]
            ):
                payment_dates.append(payment_date)
                payment_amounts.append(payment_amount)

        # Analyze payment patterns if we have multiple payments
        if len(payment_dates) >= 3:
            # Calculate intervals between payments
            intervals = [(payment_dates[i] - payment_dates[i-1]).days for i in range(1, len(payment_dates))]
            avg_interval = sum(intervals) / len(intervals)
            std_interval = np.std(intervals) if len(intervals) > 1 else 0

            # Calculate typical payment amount
            avg_amount = sum(payment_amounts) / len(payment_amounts)
            std_amount = np.std(payment_amounts) if len(payment_amounts) > 1 else 0

            # Check for pattern anomalies in each snapshot
            for idx, row in vendor_data.iterrows():
                anomaly_reasons = []

                # 1. Check for unusually long time since last payment
                days_since_payment = (row[snapshot_date_col] - row[payment_date_col]).days if row[payment_date_col] is not pd.NaT else None

                if days_since_payment is not None and days_since_payment > avg_interval + 2 * std_interval:
                    anomaly_reasons.append(f"Unusual gap since last payment ({days_since_payment} days vs avg {avg_interval:.0f})")

                # 2. Check for unusually small/large payment amount
                if std_amount > 0 and abs(row[payment_amount_col] - avg_amount) > 2 * std_amount:
                    direction = "small" if row[payment_amount_col] < avg_amount else "large"
                    anomaly_reasons.append(f"Unusually {direction} payment amount (${row[payment_amount_col]:.2f} vs avg ${avg_amount:.2f})")

                # Update the dataframe
                if anomaly_reasons:
                    result_df.loc[idx, 'Payment_Pattern_Anomaly'] = True
                    result_df.loc[idx, 'Payment_Pattern_Reason'] = "; ".join(anomaly_reasons)

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

    print("Detecting pattern anomalies...")
    pattern_df = detect_pattern_anomalies(result_df)

    print("Detecting velocity anomalies...")
    velocity_df = detect_velocity_anomalies(result_df)

    print("Detecting cohort anomalies...")
    cohort_df = detect_cohort_anomalies(result_df)

    # Combine results into a single dataframe
    result_df['Multivariate_Anomaly'] = multivariate_df['Multivariate_Anomaly']
    result_df['Multivariate_Score'] = multivariate_df['Anomaly_Score']
    result_df['Multivariate_Reason'] = multivariate_df['Anomaly_Reason'] if 'Anomaly_Reason' in multivariate_df.columns else ""

    result_df['Pattern_Anomaly'] = pattern_df['Pattern_Anomaly']
    result_df['Pattern_Score'] = pattern_df['Pattern_Zscore']

    result_df['Velocity_Anomaly'] = velocity_df['Velocity_Anomaly']
    result_df['Velocity_Score'] = velocity_df['Velocity_Anomaly_Score']
    result_df['Velocity_Reason'] = velocity_df['Velocity_Anomaly_Reason']

    result_df['Cohort_Anomaly'] = cohort_df['Cohort_Anomaly']
    result_df['Cohort_Score'] = cohort_df['Cohort_Zscore']

    # Create a combined anomaly flag and score
    result_df['Any_Anomaly'] = (
        result_df['Multivariate_Anomaly'] |
        result_df['Pattern_Anomaly'] |
        result_df['Velocity_Anomaly'] |
        result_df['Cohort_Anomaly']
    )

    # Create a combined anomaly score (average of available scores)
    score_columns = [col for col in ['Multivariate_Score', 'Pattern_Score', 'Velocity_Score', 'Cohort_Score']
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

        if row.get('Pattern_Anomaly'):
            reasons.append("Unusual pattern change")

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
        pandas.DataFrame: DataFrame with anomaly pattern segments
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
        pattern_count = sum(vendor_df.get('Pattern_Anomaly', 0))
        velocity_count = sum(vendor_df.get('Velocity_Anomaly', 0))
        cohort_count = sum(vendor_df.get('Cohort_Anomaly', 0))
        any_count = sum(vendor_df.get('Any_Anomaly', 0))

        # Calculate frequencies
        vendor_stats = {
            'Vendor_ID': vendor_id,
            'Total_Snapshots': total_snapshots,
            'Multivariate_Frequency': multivariate_count / total_snapshots,
            'Pattern_Frequency': pattern_count / total_snapshots,
            'Velocity_Frequency': velocity_count / total_snapshots,
            'Cohort_Frequency': cohort_count / total_snapshots,
            'Any_Anomaly_Frequency': any_count / total_snapshots,
            'Max_Anomaly_Score': vendor_df.get('Combined_Anomaly_Score', 0).max() if 'Combined_Anomaly_Score' in vendor_df else 0
        }

        vendor_anomaly_stats.append(vendor_stats)

    # Create dataframe with vendor anomaly statistics
    anomaly_stats_df = pd.DataFrame(vendor_anomaly_stats)

    # Apply clustering to segment vendors
    if len(anomaly_stats_df) >= n_clusters:
        # Features for clustering
        features = [
            'Multivariate_Frequency', 'Pattern_Frequency',
            'Velocity_Frequency', 'Cohort_Frequency',
            'Any_Anomaly_Frequency', 'Max_Anomaly_Score'
        ]

        # Prepare feature matrix
        X = anomaly_stats_df[features].fillna(0)

        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply K-means clustering
        from sklearn.cluster import KMeans
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
            anomaly_types = ['Multivariate', 'Pattern', 'Velocity', 'Cohort']
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


def detect_change_points(df, snapshot_date_col='Snapshot_Date', vendor_id_col='Vendor ID',
                       value_col='Balance Outstanding', min_snapshots=8):
    """
    Detect points of significant change in vendor time series.

    Args:
        df (pandas.DataFrame): Prepared vendor aging data
        snapshot_date_col (str): Column containing snapshot date
        vendor_id_col (str): Column containing vendor identifier
        value_col (str): Column to analyze for change points
        min_snapshots (int): Minimum number of snapshots required for analysis

    Returns:
        dict: Change points by vendor
    """
    try:
        # Try to import the ruptures package for change point detection
        import ruptures as rpt
        has_ruptures = True
    except ImportError:
        print("ruptures package not available - using simplified change point detection")
        has_ruptures = False

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Results dictionary
    change_points = {}

    # Process each vendor with enough data
    for vendor_id, vendor_df in result_df.groupby(vendor_id_col):
        # Sort by snapshot date
        vendor_data = vendor_df.sort_values(snapshot_date_col)

        # Skip if not enough data points
        if len(vendor_data) < min_snapshots:
            continue

        # Extract the time series
        values = vendor_data[value_col].values
        dates = vendor_data[snapshot_date_col].values

        if has_ruptures:
            # Use ruptures for optimal change point detection
            algo = rpt.Pelt(model="rbf").fit(values.reshape(-1, 1))
            result = algo.predict(pen=2)  # Penalty parameter

            # Convert indices to dates
            change_point_dates = [dates[idx-1] for idx in result[:-1]]  # Exclude the last point (end of series)

            change_points[vendor_id] = {
                'dates': [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in change_point_dates],
                'indices': result[:-1],
                'values_before': [values[idx-2] for idx in result[:-1]],
                'values_after': [values[idx] for idx in result[:-1]]
            }
        else:
            # Simplified detection based on standard deviation
            diffs = np.abs(np.diff(values))
            threshold = np.mean(diffs) + 2 * np.std(diffs)

            # Find points exceeding threshold
            cp_indices = np.where(diffs > threshold)[0]
            cp_indices = [idx + 1 for idx in cp_indices]  # Convert to 1-based indices

            # Convert indices to dates
            change_point_dates = [dates[idx] for idx in cp_indices if idx < len(dates)]

            change_points[vendor_id] = {
                'dates': [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in change_point_dates],
                'indices': cp_indices,
                'values_before': [values[idx-1] for idx in cp_indices if idx < len(values)],
                'values_after': [values[idx] for idx in cp_indices if idx < len(values)]
            }

    return change_points



def export_anomaly_detection_results(df, output_dir, vendor_id_col='Vendor ID',
                                     snapshot_date_col='Snapshot_Date', file_format='both'):
    """
    Export the results of anomaly detection analysis to CSV and/or JSON files.

    Args:
        df (pandas.DataFrame): DataFrame containing anomaly detection results
        output_dir (str): Directory to save output files
        vendor_id_col (str): Column containing vendor identifier
        snapshot_date_col (str): Column containing snapshot date
        file_format (str): Export format ('csv', 'json', or 'both')

    Returns:
        dict: Paths to exported files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Current date for filenames
    today = datetime.now().strftime('%Y-%m-%d')

    # Get the latest snapshot date
    latest_snapshot_date = df[snapshot_date_col].max()
    latest_snapshot_str = latest_snapshot_date.strftime('%Y-%m-%d') if hasattr(latest_snapshot_date, 'strftime') else str(latest_snapshot_date)

    # Filter to the latest snapshot for current anomalies
    latest_df = df[df[snapshot_date_col] == latest_snapshot_date]

    # Output files
    output_files = {}

    # ===== 1. Export summary statistics =====
    summary_stats = {
        'generated_date': today,
        'snapshot_date': latest_snapshot_str,
        'total_vendors': latest_df[vendor_id_col].nunique(),
        'total_records': len(latest_df),
        'anomaly_counts': {
            'multivariate_anomalies': int(latest_df['Multivariate_Anomaly'].sum()) if 'Multivariate_Anomaly' in latest_df.columns else 0,
            'pattern_anomalies': int(latest_df['Pattern_Anomaly'].sum()) if 'Pattern_Anomaly' in latest_df.columns else 0,
            'velocity_anomalies': int(latest_df['Velocity_Anomaly'].sum()) if 'Velocity_Anomaly' in latest_df.columns else 0,
            'cohort_anomalies': int(latest_df['Cohort_Anomaly'].sum()) if 'Cohort_Anomaly' in latest_df.columns else 0,
            'payment_pattern_anomalies': int(latest_df['Payment_Pattern_Anomaly'].sum()) if 'Payment_Pattern_Anomaly' in latest_df.columns else 0,
            'seasonal_anomalies': int(latest_df['Seasonal_Anomaly'].sum()) if 'Seasonal_Anomaly' in latest_df.columns else 0,
            'any_anomaly': int(latest_df['Any_Anomaly'].sum()) if 'Any_Anomaly' in latest_df.columns else 0
        }
    }

    # Add percentage of vendors with anomalies
    total_vendors = latest_df[vendor_id_col].nunique()
    if total_vendors > 0:
        summary_stats['anomaly_percentages'] = {
            key: (count / total_vendors * 100)
            for key, count in summary_stats['anomaly_counts'].items()
        }

    # Export summary statistics
    summary_file = os.path.join(output_dir, f'anomaly_summary_{today}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)

    output_files['summary'] = summary_file

    # ===== 2. Export detailed anomaly data =====
    # Create a simplified version with key anomaly information
    anomaly_columns = [
        vendor_id_col, 'Vendor', 'Balance Outstanding', 'Aging_Beyond_90',
        'Multivariate_Anomaly', 'Multivariate_Score', 'Multivariate_Reason',
        'Pattern_Anomaly', 'Pattern_Score', 'Pattern_Change',
        'Velocity_Anomaly', 'Velocity_Score', 'Velocity_Reason',
        'Cohort_Anomaly', 'Cohort_Score', 'Cohort_Deviation',
        'Payment_Pattern_Anomaly', 'Payment_Pattern_Reason',
        'Seasonal_Anomaly', 'Seasonal_Deviation',
        'Any_Anomaly', 'Combined_Anomaly_Score', 'Anomaly_Details',
        'Anomaly_Cluster', 'Anomaly_Cluster_Label'
    ]

    # Filter columns that exist in the dataframe
    export_columns = [col for col in anomaly_columns if col in latest_df.columns]

    # Add key financial and metadata columns if they exist
    if 'SalesPerson_Name' in latest_df.columns:
        export_columns.append('SalesPerson_Name')
    if 'AccountManager_Name' in latest_df.columns:
        export_columns.append('AccountManager_Name')
    if 'country_name' in latest_df.columns:
        export_columns.append('country_name')

    # Create export dataframe with selected columns
    export_df = latest_df[export_columns].copy()

    # Export to CSV
    if file_format in ['csv', 'both']:
        csv_file = os.path.join(output_dir, f'vendor_anomalies_{today}.csv')
        export_df.to_csv(csv_file, index=False)
        output_files['csv'] = csv_file

    # Export to JSON
    if file_format in ['json', 'both']:
        # Export all anomalies
        all_anomalies_file = os.path.join(output_dir, f'vendor_anomalies_{today}.json')
        export_df.to_json(all_anomalies_file, orient='records', indent=2)
        output_files['json_all'] = all_anomalies_file

        # Export only records with anomalies
        anomaly_records = export_df[export_df['Any_Anomaly'] == True] if 'Any_Anomaly' in export_df.columns else pd.DataFrame()

        if not anomaly_records.empty:
            anomaly_only_file = os.path.join(output_dir, f'vendor_anomalies_only_{today}.json')
            anomaly_records.to_json(anomaly_only_file, orient='records', indent=2)
            output_files['json_anomalies'] = anomaly_only_file

    # ===== 3. Export aggregated anomaly data for visualization =====

    # 3.1 Anomalies by country
    if 'country_name' in latest_df.columns:
        country_anomalies = (
            latest_df.groupby('country_name')
            .agg({
                vendor_id_col: 'nunique',
                'Any_Anomaly': 'sum' if 'Any_Anomaly' in latest_df.columns else 'size',
                'Combined_Anomaly_Score': 'mean' if 'Combined_Anomaly_Score' in latest_df.columns else 'size',
                'Balance Outstanding': 'sum',
                'Aging_Beyond_90': 'sum' if 'Aging_Beyond_90' in latest_df.columns else 'size'
            })
            .reset_index()
            .rename(columns={vendor_id_col: 'vendor_count', 'Any_Anomaly': 'anomaly_count'})
        )

        if 'anomaly_count' in country_anomalies.columns and 'vendor_count' in country_anomalies.columns:
            country_anomalies['anomaly_percentage'] = country_anomalies['anomaly_count'] / country_anomalies['vendor_count'] * 100

        country_file = os.path.join(output_dir, f'country_anomalies_{today}.json')
        country_anomalies.to_json(country_file, orient='records', indent=2)
        output_files['country_anomalies'] = country_file

    # 3.2 Anomalies by sales person
    if 'SalesPerson_Name' in latest_df.columns:
        sales_anomalies = (
            latest_df.groupby('SalesPerson_Name')
            .agg({
                vendor_id_col: 'nunique',
                'Any_Anomaly': 'sum' if 'Any_Anomaly' in latest_df.columns else 'size',
                'Combined_Anomaly_Score': 'mean' if 'Combined_Anomaly_Score' in latest_df.columns else 'size',
                'Balance Outstanding': 'sum',
                'Aging_Beyond_90': 'sum' if 'Aging_Beyond_90' in latest_df.columns else 'size'
            })
            .reset_index()
            .rename(columns={vendor_id_col: 'vendor_count', 'Any_Anomaly': 'anomaly_count'})
            .sort_values('anomaly_count', ascending=False)
        )

        if 'anomaly_count' in sales_anomalies.columns and 'vendor_count' in sales_anomalies.columns:
            sales_anomalies['anomaly_percentage'] = sales_anomalies['anomaly_count'] / sales_anomalies['vendor_count'] * 100

        sales_file = os.path.join(output_dir, f'salesperson_anomalies_{today}.json')
        sales_anomalies.to_json(sales_file, orient='records', indent=2)
        output_files['sales_anomalies'] = sales_file

    # 3.3 Anomalies by type
    if 'Any_Anomaly' in latest_df.columns:
        anomaly_columns = [
            'Multivariate_Anomaly', 'Pattern_Anomaly', 'Velocity_Anomaly',
            'Cohort_Anomaly', 'Payment_Pattern_Anomaly', 'Seasonal_Anomaly'
        ]

        anomaly_types = []
        for col in anomaly_columns:
            if col in latest_df.columns:
                count = int(latest_df[col].sum())
                pct = count / len(latest_df) * 100
                anomaly_types.append({
                    'type': col.replace('_Anomaly', ''),
                    'count': count,
                    'percentage': pct
                })

        types_file = os.path.join(output_dir, f'anomaly_types_{today}.json')
        with open(types_file, 'w') as f:
            json.dump(anomaly_types, f, indent=2)

        output_files['anomaly_types'] = types_file

    # 3.4 Export cluster profiles if available
    if 'Anomaly_Cluster' in latest_df.columns:
        clusters = latest_df['Anomaly_Cluster'].dropna().unique()

        cluster_stats = []
        for cluster in clusters:
            cluster_df = latest_df[latest_df['Anomaly_Cluster'] == cluster]

            if 'Anomaly_Cluster_Label' in cluster_df.columns:
                label = cluster_df['Anomaly_Cluster_Label'].iloc[0]
            else:
                label = f"Cluster {int(cluster)}"

            stats = {
                'cluster': int(cluster),
                'label': label,
                'vendor_count': cluster_df[vendor_id_col].nunique(),
                'record_count': len(cluster_df),
                'avg_anomaly_score': float(cluster_df['Combined_Anomaly_Score'].mean()) if 'Combined_Anomaly_Score' in cluster_df.columns else None,
                'total_balance': float(cluster_df['Balance Outstanding'].sum()),
                'total_aging_beyond_90': float(cluster_df['Aging_Beyond_90'].sum()) if 'Aging_Beyond_90' in cluster_df.columns else None
            }

            cluster_stats.append(stats)

        cluster_file = os.path.join(output_dir, f'anomaly_clusters_{today}.json')
        with open(cluster_file, 'w') as f:
            json.dump(cluster_stats, f, indent=2)

        output_files['cluster_stats'] = cluster_file

    # ===== 4. Export trend analysis =====
    # Create a file showing how anomalies have evolved over time

    # Group by snapshot date
    snapshot_trends = (
        df.groupby(snapshot_date_col)
        .agg({
            vendor_id_col: 'nunique',
            'Any_Anomaly': 'sum' if 'Any_Anomaly' in df.columns else 'size',
            'Combined_Anomaly_Score': 'mean' if 'Combined_Anomaly_Score' in df.columns else 'size',
            'Balance Outstanding': 'sum',
            'Aging_Beyond_90': 'sum' if 'Aging_Beyond_90' in df.columns else 'size'
        })
        .reset_index()
    )

    # Convert snapshot date to string for JSON serialization
    if hasattr(snapshot_trends[snapshot_date_col].iloc[0], 'strftime'):
        snapshot_trends[snapshot_date_col] = snapshot_trends[snapshot_date_col].dt.strftime('%Y-%m-%d')
    else:
        snapshot_trends[snapshot_date_col] = snapshot_trends[snapshot_date_col].astype(str)

    # Calculate anomaly rate if possible
    if 'Any_Anomaly' in snapshot_trends.columns and vendor_id_col in snapshot_trends.columns:
        snapshot_trends['anomaly_rate'] = snapshot_trends['Any_Anomaly'] / snapshot_trends[vendor_id_col] * 100

    trends_file = os.path.join(output_dir, f'anomaly_trends_{today}.json')
    snapshot_trends.to_json(trends_file, orient='records', indent=2)
    output_files['trends'] = trends_file

    # ===== 5. Export top anomalies for dashboard =====
    # Create a simplified list of top vendors with anomalies for dashboard display

    if 'Any_Anomaly' in latest_df.columns:
        # Get vendors with any anomaly
        anomaly_vendors = latest_df[latest_df['Any_Anomaly'] == True]

        if not anomaly_vendors.empty and 'Combined_Anomaly_Score' in anomaly_vendors.columns:
            # Sort by anomaly score
            top_anomalies = (
                anomaly_vendors
                .sort_values('Combined_Anomaly_Score', ascending=False)
                .head(20)
            )

            # Create simplified records for dashboard
            dashboard_records = []
            for _, vendor in top_anomalies.iterrows():
                record = {
                    'vendor_id': int(vendor[vendor_id_col]) if pd.notna(vendor[vendor_id_col]) else None,
                    'vendor_name': vendor['Vendor'] if 'Vendor' in vendor else f"Vendor {vendor[vendor_id_col]}",
                    'balance': float(vendor['Balance Outstanding']),
                    'aging_beyond_90': float(vendor['Aging_Beyond_90']) if 'Aging_Beyond_90' in vendor else None,
                    'anomaly_score': float(vendor['Combined_Anomaly_Score']),
                    'anomaly_details': vendor['Anomaly_Details'] if 'Anomaly_Details' in vendor else None,
                    'anomaly_types': []
                }

                # Add anomaly types
                for col in ['Multivariate_Anomaly', 'Pattern_Anomaly', 'Velocity_Anomaly',
                           'Cohort_Anomaly', 'Payment_Pattern_Anomaly', 'Seasonal_Anomaly']:
                    if col in vendor and vendor[col]:
                        record['anomaly_types'].append(col.replace('_Anomaly', ''))

                dashboard_records.append(record)

            dashboard_file = os.path.join(output_dir, f'dashboard_anomalies_{today}.json')
            with open(dashboard_file, 'w') as f:
                json.dump(dashboard_records, f, indent=2)

            output_files['dashboard'] = dashboard_file

    print(f"Anomaly detection results exported to {output_dir}")
    return output_files


class VendorAgingSnapshotAnalytics:
    """
    Analytics engine for vendor aging data that properly handles multiple snapshots.
    """

    def __init__(self, data_path, output_dir):
        """
        Initialize the analytics engine.

        Args:
            data_path: Path to the vendor aging CSV file
            output_dir: Directory to save output JSON/CSV files
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.prepared_df = None
        self.today = datetime.now().strftime('%Y-%m-%d')

        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load the vendor aging data and prepare it for time series analysis"""
        # Load raw data
        self.df = pd.read_csv(self.data_path)

        # Prepare data for time series analysis
        self.prepared_df = prepare_time_series_data(self.df)

        # Calculate key metrics
        self.prepared_df = calculate_aging_metrics_over_time(self.prepared_df)
        self.prepared_df = calculate_aging_trends(self.prepared_df)
        self.prepared_df = vendor_payment_history_analysis(self.prepared_df)

        return self

    def generate_descriptive_analytics(self):
        """
        Generate descriptive analytics that properly account for snapshots
        """
        if self.prepared_df is None:
            self.load_data()

        results = {}

        # Get the latest snapshot date
        latest_snapshot_date = self.prepared_df['Snapshot_Date'].max()

        # Filter to latest snapshot for current metrics
        latest_df = self.prepared_df[self.prepared_df['Snapshot_Date'] == latest_snapshot_date]

        # Get the previous snapshot (1 month ago) for comparison
        snapshot_dates = sorted(self.prepared_df['Snapshot_Date'].unique())
        if len(snapshot_dates) > 1:
            previous_snapshot_date = snapshot_dates[-2]
            previous_df = self.prepared_df[self.prepared_df['Snapshot_Date'] == previous_snapshot_date]
        else:
            previous_df = latest_df  # Use the same if no previous snapshot

        # 1.1 Daily Aging Summary (for latest snapshot)
        aging_summary = {
            'snapshot_date': latest_snapshot_date.strftime('%Y-%m-%d') if isinstance(latest_snapshot_date, pd.Timestamp) else str(latest_snapshot_date),
            'total_balance': latest_df['Balance Outstanding'].sum(),
            'total_vendors': latest_df['Vendor'].nunique(),
            'aging_distribution': {
                'Future': latest_df['Future_Aging'].sum(),
                '0-30': latest_df['Aging_0_30'].sum(),
                '31-60': latest_df['Aging_31_60'].sum(),
                '61-90': latest_df['Aging_61_90'].sum(),
                '91-120': latest_df['Aging_91_120'].sum(),
                '121-180': latest_df['Aging_121_180'].sum(),
                '181-360': latest_df['Aging_181_360'].sum(),
                '361+': latest_df['Above_361_Aging'].sum()
            },
            'total_aged_beyond_90': latest_df['Aging_Beyond_90'].sum(),
            'change_from_previous': {
                'total_balance': latest_df['Balance Outstanding'].sum() - previous_df['Balance Outstanding'].sum(),
                'total_aged_beyond_90': latest_df['Aging_Beyond_90'].sum() - previous_df['Aging_Beyond_90'].sum()
            },
            'run_date': self.today
        }
        results['aging_summary'] = aging_summary

        # 1.2 Vendor Risk Scorecard (based on latest snapshot)
        # Calculate collection priority score
        latest_df = calculate_collection_priority_score(latest_df, latest_snapshot_only=True)

        # Identify high risk vendors
        high_risk_vendors = (
            latest_df[latest_df['Collection_Priority'] == 'Critical']
            .sort_values('Collection_Priority_Score', ascending=False)
            .head(50)[
                ['Vendor', 'Vendor ID', 'Balance Outstanding', 'Aging_Beyond_90',
                 'Collection_Priority_Score', 'Collection_Priority']
            ]
            .to_dict('records')
        )
        results['high_risk_vendors'] = high_risk_vendors

        # 1.3 Country Performance (latest snapshot)
        country_performance = (
            latest_df.groupby('country_name')
            .agg({
                'Balance Outstanding': 'sum',
                'Aging_Beyond_90': 'sum',
                'Vendor ID': 'nunique'
            })
            .reset_index()
            .rename(columns={'Vendor ID': 'vendor_count'})
            .to_dict('records')
        )
        results['country_performance'] = country_performance

        # 1.4 Sales Team Performance (latest snapshot)
        sales_performance = (
            latest_df.groupby(['SalesPerson_Name', 'AccountManager_Name'])
            .agg({
                'Balance Outstanding': 'sum',
                'Aging_Beyond_90': 'sum',
                'Vendor ID': 'nunique',
                'Collection_Priority_Score': 'mean'
            })
            .reset_index()
            .rename(columns={'Vendor ID': 'vendor_count'})
            .sort_values('Collection_Priority_Score', ascending=False)
            .to_dict('records')
        )
        results['sales_performance'] = sales_performance

        # Save outputs
        with open(f"{self.output_dir}/descriptive_analytics_{self.today}.json", 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def generate_diagnostic_analytics(self):
        """
        Generate diagnostic analytics that identify anomalies and patterns across snapshots
        """
        if self.prepared_df is None:
            self.load_data()

        results = {}

        # 2.1 Anomaly Detection
        anomaly_df = detect_time_based_anomalies(self.prepared_df)

        # Get the most recent anomalies
        latest_snapshot_date = self.prepared_df['Snapshot_Date'].max()
        latest_anomalies = anomaly_df[
            (anomaly_df['Snapshot_Date'] == latest_snapshot_date) &
            (anomaly_df['Time_Anomaly'])
        ]

        anomalies = (
            latest_anomalies[
                ['Vendor', 'Vendor ID', 'Balance Outstanding', 'Aging_Beyond_90',
                 'Balance_MoM_Change', 'Aging_Beyond_90_MoM_Change', 'Anomaly_Details']
            ]
            .sort_values('Aging_Beyond_90', ascending=False)
            .head(30)
            .to_dict('records')
        )

        results['anomalies'] = anomalies

        # 2.2 Root Cause Analysis
        # Only use the latest snapshot for each vendor
        latest_df = (
            self.prepared_df
            .sort_values('Snapshot_Date')
            .groupby('Vendor ID')
            .last()
            .reset_index()
        )

        root_causes = identify_root_causes(latest_df)
        results['root_cause_factors'] = root_causes.get('key_factors', [])

        # 2.3 Vendor Cohort Analysis
        cohort_df, cohort_profiles = create_vendor_cohorts(self.prepared_df)

        # Get latest snapshot cohort distributions
        latest_cohort_df = (
            cohort_df[cohort_df['Snapshot_Date'] == latest_snapshot_date]
            .groupby('Cohort_Label')
            .agg({
                'Vendor ID': 'nunique',
                'Balance Outstanding': 'sum',
                'Aging_Beyond_90': 'sum',
                'Pct_Aging_Beyond_90': 'mean'
            })
            .reset_index()
            .rename(columns={'Vendor ID': 'vendor_count'})
        )

        results['cohort_analysis'] = {
            'cohort_profiles': {k: v for k, v in cohort_profiles.items()},
            'cohort_distribution': latest_cohort_df.to_dict('records')
        }

        # 2.4 Snapshot Comparison (latest vs. previous month)
        snapshot_dates = sorted(self.prepared_df['Snapshot_Date'].unique())
        if len(snapshot_dates) > 1:
            comparison = snapshot_comparison_analysis(
                self.prepared_df,
                base_date=snapshot_dates[-2],  # Previous month
                compare_date=snapshot_dates[-1]  # Latest month
            )

            # Calculate overall changes
            overall_changes = {
                'balance_change': comparison['Balance Outstanding_Change'].sum(),
                'balance_pct_change': (
                    comparison['Balance Outstanding_Change'].sum() /
                    comparison['Balance Outstanding_Base'].abs().sum() * 100
                ),
                'aging_beyond_90_change': comparison['Aging_Beyond_90_Change'].sum(),
                'aging_beyond_90_pct_change': (
                    comparison['Aging_Beyond_90_Change'].sum() /
                    comparison['Aging_Beyond_90_Base'].abs().sum() * 100
                ),
                'improved_vendors': sum(comparison['Aging_Beyond_90_Direction'] == 'Decreased'),
                'worsened_vendors': sum(comparison['Aging_Beyond_90_Direction'] == 'Increased'),
                'unchanged_vendors': sum(comparison['Aging_Beyond_90_Direction'] == 'Unchanged')
            }

            results['snapshot_comparison'] = {
                'base_date': snapshot_dates[-2].strftime('%Y-%m-%d') if isinstance(snapshot_dates[-2], pd.Timestamp) else str(snapshot_dates[-2]),
                'compare_date': snapshot_dates[-1].strftime('%Y-%m-%d') if isinstance(snapshot_dates[-1], pd.Timestamp) else str(snapshot_dates[-1]),
                'overall_changes': overall_changes,
                'top_improved': comparison[comparison['Aging_Beyond_90_Direction'] == 'Decreased']
                    .sort_values('Aging_Beyond_90_Change')
                    .head(10)[['Vendor ID', 'Balance Outstanding_Change', 'Aging_Beyond_90_Change', 'Aging_Beyond_90_Pct_Change']]
                    .to_dict('records'),
                'top_worsened': comparison[comparison['Aging_Beyond_90_Direction'] == 'Increased']
                    .sort_values('Aging_Beyond_90_Change', ascending=False)
                    .head(10)[['Vendor ID', 'Balance Outstanding_Change', 'Aging_Beyond_90_Change', 'Aging_Beyond_90_Pct_Change']]
                    .to_dict('records')
            }

        # Save outputs
        with open(f"{self.output_dir}/diagnostic_analytics_{self.today}.json", 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def generate_prescriptive_analytics(self):
        """
        Generate prescriptive analytics to guide decision making
        """
        if self.prepared_df is None:
            self.load_data()

        results = {}

        # Get the latest snapshot date
        latest_snapshot_date = self.prepared_df['Snapshot_Date'].max()

        # Filter to latest snapshot for current recommendations
        latest_df = self.prepared_df[self.prepared_df['Snapshot_Date'] == latest_snapshot_date]

        # 3.1 Collection Priority Queue
        # Calculate collection priority if not already done
        if 'Collection_Priority_Score' not in latest_df.columns:
            latest_df = calculate_collection_priority_score(latest_df, latest_snapshot_only=True)

        collection_priorities = (
            latest_df
            .sort_values('Collection_Priority_Score', ascending=False)
            .head(50)[
                ['Vendor', 'Vendor ID', 'Balance Outstanding', 'Aging_Beyond_90',
                 'Collection_Priority_Score', 'Collection_Priority',
                 'SalesPerson_Name', 'AccountManager_Name']
            ]
            .to_dict('records')
        )

        results['collection_priorities'] = collection_priorities

        # 3.2 Credit Risk Management Recommendations
        # Identify vendors with concerning trends
        credit_recommendations = []

        # Check if we have at least two snapshots
        snapshot_dates = sorted(self.prepared_df['Snapshot_Date'].unique())
        if len(snapshot_dates) >= 2:
            # Get vendors with worsening aging trends
            trend_df = self.prepared_df[
                (self.prepared_df['Snapshot_Date'] == latest_snapshot_date) &
                (self.prepared_df['Aging_Beyond_90_Trend'] > 10)  # 10% or more increase in aging
            ]

            for _, vendor in (
                trend_df.sort_values('Aging_Beyond_90_Trend', ascending=False)
                .head(30)
                .iterrows()
            ):
                action = "Reduce credit limit" if vendor['Pct_Aging_Beyond_90'] > 50 else "Review credit terms"
                credit_recommendations.append({
                    'vendor': vendor['Vendor'],
                    'vendor_id': vendor['Vendor ID'],
                    'current_balance': vendor['Balance Outstanding'],
                    'aging_beyond_90': vendor['Aging_Beyond_90'],
                    'aging_trend': vendor['Aging_Beyond_90_Trend'],
                    'recommended_action': action,
                    'urgency': 'High' if vendor['Aging_Beyond_90_Trend'] > 20 else 'Medium'
                })

        results['credit_recommendations'] = credit_recommendations

        # 3.3 Cash Flow Forecasting
        # Waterfall analysis for top vendors
        top_vendor_ids = (
            latest_df
            .sort_values('Balance Outstanding', ascending=False)
            .head(20)['Vendor ID']
            .tolist()
        )

        waterfall_analysis = aging_waterfall_analysis(
            self.prepared_df[self.prepared_df['Vendor ID'].isin(top_vendor_ids)]
        )

        # Summarize transitions for forecasting
        aging_buckets = [
            'Future_Aging', 'Aging_0_30', 'Aging_31_60', 'Aging_61_90',
            'Aging_91_120', 'Aging_121_180', 'Aging_181_360', 'Above_361_Aging'
        ]

        # Simple forecast based on historical transitions
        # (In a real implementation, you would use proper time series forecasting)
        current_totals = {bucket: latest_df[bucket].sum() for bucket in aging_buckets}

        # Calculate average transitions from waterfall analysis
        # This is a very simplified approach
        transition_rates = {}
        for i in range(len(aging_buckets) - 1):
            from_bucket = aging_buckets[i]
            to_bucket = aging_buckets[i+1]
            transition_key = f'{from_bucket}_to_{to_bucket}'

            transition_sum = 0
            transition_count = 0

            for vendor_id, analysis in waterfall_analysis.items():
                for transition in analysis.get('transitions', []):
                    if transition_key in transition.get('implied_transitions', {}):
                        transition_sum += transition['implied_transitions'][transition_key]
                        transition_count += 1

            if transition_count > 0:
                transition_rates[transition_key] = transition_sum / transition_count
            else:
                transition_rates[transition_key] = 0

        # Simple 30/60/90-day forecast
        forecast_30d = current_totals.copy()
        forecast_60d = current_totals.copy()
        forecast_90d = current_totals.copy()

        # Apply transitions to forecast
        # (This is highly simplified - a real implementation would use proper forecasting methods)
        for i in range(len(aging_buckets) - 1):
            from_bucket = aging_buckets[i]
            to_bucket = aging_buckets[i+1]
            transition_key = f'{from_bucket}_to_{to_bucket}'

            # For 30-day forecast
            forecast_30d[to_bucket] += forecast_30d[from_bucket] * 0.7  # Assume 70% moves to next bucket
            forecast_30d[from_bucket] *= 0.3  # 30% stays in current bucket

            # For 60-day forecast (apply transition twice)
            forecast_60d[to_bucket] += forecast_60d[from_bucket] * 0.7
            forecast_60d[from_bucket] *= 0.3

            # For 90-day forecast (apply transition three times)
            forecast_90d[to_bucket] += forecast_90d[from_bucket] * 0.7
            forecast_90d[from_bucket] *= 0.3

        results['cash_flow_forecast'] = {
            'current': current_totals,
            'forecast_30d': forecast_30d,
            'forecast_60d': forecast_60d,
            'forecast_90d': forecast_90d,
            'transition_rates': transition_rates
        }

        # 3.4 Relationship Management Actions
        # Generate tailored recommendations based on vendor patterns
        relationship_actions = []

        # Use the collection priorities as a basis
        for vendor in collection_priorities[:20]:
            vendor_id = vendor['Vendor ID']
            vendor_data = latest_df[latest_df['Vendor ID'] == vendor_id]

            if not vendor_data.empty:
                # Determine recommended action based on aging pattern
                pct_aging_beyond_90 = (
                    vendor_data['Aging_Beyond_90'].values[0] /
                    vendor_data['Balance Outstanding'].abs().values[0]
                ) if vendor_data['Balance Outstanding'].abs().values[0] > 0 else 0

                days_since_payment = vendor_data['Days_Since_Payment'].values[0] if 'Days_Since_Payment' in vendor_data.columns else 0

                if pct_aging_beyond_90 > 0.7 and days_since_payment > 90:
                    action = "Immediate contact by account manager; escalate to management"
                    frequency = "Weekly follow-up"
                elif pct_aging_beyond_90 > 0.5:
                    action = "Schedule payment plan meeting; discuss credit terms"
                    frequency = "Bi-weekly follow-up"
                elif pct_aging_beyond_90 > 0.3:
                    action = "Contact to review outstanding invoices; request payment timeline"
                    frequency = "Monthly follow-up"
                else:
                    action = "Routine relationship check-in; verify invoice status"
                    frequency = "Quarterly review"

                relationship_actions.append({
                    'vendor': vendor['Vendor'],
                    'vendor_id': vendor['Vendor ID'],
                    'recommended_action': action,
                    'contact_frequency': frequency,
                    'assigned_to': vendor['AccountManager_Name'],
                    'support_from': vendor['SalesPerson_Name'],
                    'aging_beyond_90': vendor['Aging_Beyond_90'],
                    'balance_outstanding': vendor['Balance Outstanding']
                })

        results['relationship_actions'] = relationship_actions

        # Save outputs
        with open(f"{self.output_dir}/prescriptive_analytics_{self.today}.json", 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def run_all_analytics(self):
        """
        Run all analytics processes and generate outputs
        """
        self.load_data()
        descriptive = self.generate_descriptive_analytics()
        diagnostic = self.generate_diagnostic_analytics()
        prescriptive = self.generate_prescriptive_analytics()

        # Combine all results into a single comprehensive file
        all_results = {
            'descriptive': descriptive,
            'diagnostic': diagnostic,
            'prescriptive': prescriptive,
            'generated_at': self.today,
            'snapshot_dates': [
                d.strftime('%Y-%m-%d') if isinstance(d, pd.Timestamp) else str(d)
                for d in sorted(self.prepared_df['Snapshot_Date'].unique())
            ]
        }

        with open(f"{self.output_dir}/all_analytics_{self.today}.json", 'w') as f:
            json.dump(all_results, f, indent=2)

        return all_results