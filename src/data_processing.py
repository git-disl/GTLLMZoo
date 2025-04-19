# src/data_processing.py
import pandas as pd
import numpy as np
import os
import re

def load_data(file_path='data/merged_leaderboards.csv'):
    """Load and prepare the leaderboard data"""
    try:
        df = pd.read_csv(file_path) # [cite: 1]

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

        # Drop duplicates based on the primary model name to avoid issues if names clash
        # Keep the first occurrence (which should be the highest ranked if sorted by Global Average)
        if 'Model Name' in df.columns:
            df = df.drop_duplicates(subset=['Model Name'], keep='first')


        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def get_column_groups():
    """Define logical column groupings for the UI, including Model Mapping"""
    # Changed model_identifier to 'Model Name' to match user's working version
    model_identifier = "Model Name" # This should be the unified name column

    column_groups = {
        "Main Metrics": [
            model_identifier, "Organization", "Global Average", "Reasoning Average",
            "Coding Average", "Mathematics Average", "Data Analysis Average",
            "Language Average", "IF Average"
        ],
        "Model Details": [
            model_identifier, "Organization", "Model License",
            "Model Knowledge Cutoff", "Model Link (LiveBench)", "Model Link (Arena)" # Added Arena Link here too
        ],
        "Community Stats": [
            model_identifier, "Organization",
            "Arena Rank (No Style Control)", "Arena Rank (With Style Control)",
            "Arena Score", "95% Confidence Interval", "# of Votes"
            # Removed: "Model Name (Arena)", "Model Link (Arena)" as they are in Model Details/Mapping
        ],
        "Model Mapping": [
            # MODIFIED: Added the main model_identifier here
            model_identifier,
            "Model Name (LiveBench)", "Model Name (Arena)",
            "Model Link (LiveBench)", "Model Link (Arena)"
        ],
         # All displayable columns for card view (uses internal names where needed)
         # Ensure this list is comprehensive for the card function
         "All Displayable": [
             model_identifier, "Organization", "Global Average", "Reasoning Average",
             "Coding Average", "Mathematics Average", "Data Analysis Average",
             "Language Average", "IF Average", "Model License",
             "Model Knowledge Cutoff", "Model Name (LiveBench)", "Model Link (LiveBench)", # Added LiveBench Name
             "Model Name (Arena)", "Model Link (Arena)", # Added Arena Name/Link
             "Arena Rank (No Style Control)", "Arena Rank (With Style Control)",
             "Arena Score", "95% Confidence Interval", "# of Votes"
        ]
    }
    return column_groups

def filter_data(df, search, min_global, organization):
    """Filter the dataframe based on user inputs using Model Name"""
    if df is None or df.empty:
        return pd.DataFrame()

    filtered_df = df.copy()

    # Apply minimum global average filter
    if 'Global Average' in filtered_df.columns and min_global > 0:
         # Ensure comparison is valid even if column has NaNs
         filtered_df = filtered_df[filtered_df['Global Average'].ge(min_global)]

    # Apply organization filter
    if organization and organization != "All" and 'Organization' in filtered_df.columns:
        # Ensure comparison handles potential NaNs in the column
        filtered_df = filtered_df[filtered_df['Organization'].eq(organization) & filtered_df['Organization'].notna()]

    # Apply search filter across relevant text fields
    if search and search.strip(): # Ensure search is not empty or just whitespace
        search_text = re.escape(search.strip().lower()) # Use strip() and escape regex chars
        # Initialize filter mask
        search_filter = pd.Series([False] * len(filtered_df), index=filtered_df.index)

        # List of columns to search within
        search_cols = [
            'Model Name', 'Organization', 'Model Name (LiveBench)', 'Model Name (Arena)'
        ]

        # Apply search to existing columns only, handling potential NaNs within columns
        for col in search_cols:
            if col in filtered_df.columns:
                # Convert column to string, handle NaN, convert to lower, then check contains
                 search_filter |= filtered_df[col].astype(str).str.lower().str.contains(search_text, na=False)

        filtered_df = filtered_df[search_filter]

    return filtered_df


def get_organization_list(df):
    """Get sorted list of unique organizations for dropdown"""
    if df is None or 'Organization' not in df.columns:
        return ["All"]
    # Drop NaNs, get unique, convert to string, sort, and prepend "All"
    orgs = df['Organization'].dropna().unique().tolist()
    orgs = sorted([str(org) for org in orgs if org]) # Ensure sorting works on strings
    return ["All"] + orgs

def get_top_models(df, column='Global Average', n=15):
    """Get top N models by a specific metric, handling sort direction."""
    if df is None or df.empty or column not in df.columns:
        return pd.DataFrame()

    # Determine sort order: ascending for ranks, descending otherwise
    ascending_sort = 'Rank' in column # True if 'Rank' is in the column name

    # Sort, handle NaNs in the sorting column, and take top N
    # na_position='last' ensures models without a score/rank are at the bottom
    return df.sort_values(column, ascending=ascending_sort, na_position='last').head(n)