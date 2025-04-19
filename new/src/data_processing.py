# src/data_processing.py
import pandas as pd
import numpy as np
import os
import re

def load_data(file_path='data/merged_leaderboards.csv'):
    """Load and prepare the leaderboard data"""
    try:
        df = pd.read_csv(file_path)

        # Replace empty strings with NaN
        df = df.replace('', np.nan)

        # Ensure numeric columns are properly typed
        numeric_cols = [
            'Global Average', 'Reasoning Average', 'Coding Average',
            'Mathematics Average', 'Data Analysis Average',
            'Language Average', 'IF Average',
            'Arena Rank (No Style Control)', 'Arena Rank (With Style Control)',
            'Arena Score', '# of Votes'
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if '95% Confidence Interval' in df.columns:
             df['95% Confidence Interval'] = df['95% Confidence Interval'].astype(str)

        if 'Global Average' in df.columns:
            df = df.sort_values('Global Average', ascending=False, na_position='last')

        # Create Model Name (prioritize LiveBench) - Changed 'Primary Model Name' to 'Model Name'
        if 'Model Name (LiveBench)' in df.columns and 'Model Name (Arena)' in df.columns:
             df['Model Name'] = df['Model Name (LiveBench)'].fillna(df['Model Name (Arena)'])
        elif 'Model Name (LiveBench)' in df.columns:
             df['Model Name'] = df['Model Name (LiveBench)']
        elif 'Model Name (Arena)' in df.columns:
             df['Model Name'] = df['Model Name (Arena)']
        else:
             df['Model Name'] = 'Unknown Model' # Fallback

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def get_column_groups():
    """Define logical column groupings for the UI, including Model Mapping"""
    # Changed model_identifier to 'Model Name' to match user's working version
    model_identifier = "Model Name"

    column_groups = {
        "Main Metrics": [
            model_identifier, "Organization", "Global Average", "Reasoning Average",
            "Coding Average", "Mathematics Average", "Data Analysis Average",
            "Language Average", "IF Average"
        ],
        "Model Details": [
            model_identifier, "Organization", "Model License",
            "Model Knowledge Cutoff", "Model Link (LiveBench)"
        ],
        "Community Stats": [ # MODIFIED: Removed Model Name (Arena) and Model Link (Arena)
            model_identifier, "Organization",
            "Arena Rank (No Style Control)", "Arena Rank (With Style Control)",
            "Arena Score", "95% Confidence Interval", "# of Votes"
            # Removed: "Model Name (Arena)", "Model Link (Arena)"
        ],
        "Model Mapping": [
            "Model Name (LiveBench)", "Model Name (Arena)",
            "Model Link (LiveBench)", "Model Link (Arena)"
        ],
         # All displayable columns for card view (uses internal names where needed)
         "All Displayable": [
             model_identifier, "Organization", "Global Average", "Reasoning Average",
             "Coding Average", "Mathematics Average", "Data Analysis Average",
             "Language Average", "IF Average", "Model License",
             "Model Knowledge Cutoff", "Model Link (LiveBench)", "Model Link (Arena)",
             "Arena Rank (No Style Control)", "Arena Rank (With Style Control)",
             "Model Name (Arena)", "Arena Score", "95% Confidence Interval", "# of Votes"
        ]
    }
    return column_groups

def filter_data(df, search, min_global, organization):
    """Filter the dataframe based on user inputs using Model Name"""
    filtered_df = df.copy()

    if min_global > 0 and 'Global Average' in filtered_df.columns:
         filtered_df = filtered_df[filtered_df['Global Average'] >= min_global]

    if organization and organization != "All" and 'Organization' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Organization'] == organization]

    if search and not search.isspace() and search != "":
        search_text = re.escape(search.lower())
        search_filter = pd.Series([False] * len(filtered_df), index=filtered_df.index)

        # Search Model Name, Organization, and original Arena/Livebench names
        # Updated to use 'Model Name' as the primary search field
        if 'Model Name' in filtered_df.columns:
             search_filter |= filtered_df['Model Name'].str.lower().str.contains(search_text, na=False)
        if 'Organization' in filtered_df.columns:
             search_filter |= filtered_df['Organization'].str.lower().str.contains(search_text, na=False)
        if 'Model Name (LiveBench)' in filtered_df.columns:
             search_filter |= filtered_df['Model Name (LiveBench)'].str.lower().str.contains(search_text, na=False)
        if 'Model Name (Arena)' in filtered_df.columns:
             search_filter |= filtered_df['Model Name (Arena)'].str.lower().str.contains(search_text, na=False)

        filtered_df = filtered_df[search_filter]

    return filtered_df

def get_organization_list(df):
    """Get sorted list of unique organizations for dropdown"""
    if 'Organization' not in df.columns:
        return ["All"]
    orgs = df['Organization'].dropna().unique().tolist()
    orgs = [str(org) for org in orgs if org]
    orgs.sort()
    return ["All"] + orgs

def get_top_models(df, column='Global Average', n=15):
    """Get top N models by a specific metric, handling sort direction."""
    if column not in df.columns or df.empty:
        return pd.DataFrame()

    # Determine sort order: ascending for ranks, descending otherwise
    ascending_sort = False # Default: Higher is better
    if 'Rank' in column:
        ascending_sort = True # Lower rank is better

    # Sort and handle potential NaNs in the sorting column
    return df.sort_values(column, ascending=ascending_sort, na_position='last').head(n)
