# src/ui.py
import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data_processing import (
    load_data, filter_data, get_organization_list,
    get_column_groups, get_top_models
)
import numpy as np # Needed for scaling check

# --- Global Setup ---
try:
    df_global = load_data()
    if df_global.empty:
        print("Warning: Dataframe is empty after loading.")
        # Add Arena Score default if data loading creates an empty frame
        df_global = pd.DataFrame(columns=['Model Name', 'Organization', 'Global Average', 'Arena Score'])

    column_groups_global = get_column_groups()
    # Consistent model identifier from data_processing
    # Safely get the identifier, default to 'Model Name' if group is missing/empty
    model_identifier_col = column_groups_global.get("Main Metrics", ["Model Name"])[0] if column_groups_global.get("Main Metrics") else "Model Name"


    # Define base radar metrics
    base_radar_metrics = [
        "Reasoning Average", "Coding Average", "Mathematics Average",
        "Data Analysis Average", "Language Average", "IF Average"
    ]
    # Add Arena Score if available in the data
    if 'Arena Score' in df_global.columns:
         base_radar_metrics.append('Arena Score')

    # Filter to metrics actually present
    radar_metrics = [m for m in base_radar_metrics if m in df_global.columns]

    # Calculate global min/max for Arena Score scaling (handle NaNs and potential lack of column)
    global_arena_min = np.nan
    global_arena_max = np.nan
    if 'Arena Score' in df_global.columns:
        # Ensure the column exists before trying to access it
        if not df_global['Arena Score'].isna().all(): # Check if there are any non-NaN values
            global_arena_min = df_global['Arena Score'].min(skipna=True)
            global_arena_max = df_global['Arena Score'].max(skipna=True)
            # Handle case where min/max might still be NaN (e.g., column exists but all values are NaN)
            if pd.isna(global_arena_min) or pd.isna(global_arena_max):
                 global_arena_min = 0 # Default scale if calculation fails
                 global_arena_max = 1 # Avoid division by zero later
            elif global_arena_max == global_arena_min:
                 # Avoid division by zero if all scores are the same
                 global_arena_max += 1 # Or set a default range like 0-100? Adjust as needed.
        else:
            # Column exists but all values are NaN
            global_arena_min = 0
            global_arena_max = 1
    else:
         # Arena Score column doesn't exist
         global_arena_min = 0
         global_arena_max = 1


except Exception as e:
    print(f"Critical error during global setup: {e}")
    df_global = pd.DataFrame({'Error': [f'Failed to load data: {e}']})
    column_groups_global = {}
    model_identifier_col = 'Error'
    radar_metrics = []
    global_arena_min, global_arena_max = 0, 1 # Default values on error

# --- Helper Functions ---
def get_valid_columns(df, column_list):
    if df is None or column_list is None: return []
    # Return only columns that exist in the dataframe
    return [col for col in column_list if col in df.columns]

# --- UI Component Creation Functions ---
# create_model_card remains the same as previous version
def create_model_card(model_data):
    """Create a detailed card view for a single model"""
    if not isinstance(model_data, dict): return "<p>Error: Invalid model data format.</p>"

    # Use the globally defined identifier
    model_name = model_data.get(model_identifier_col, 'Unknown Model')

    # Helper to safely get data, handling potential NaNs or missing keys
    def get_data(key, default='N/A'):
        val = model_data.get(key)
        # Check for NaN specifically with pandas, otherwise just check for None
        return default if pd.isna(val) else val

    # Determine the primary link, prioritizing LiveBench then Arena
    model_link_livebench = get_data('Model Link (LiveBench)', default=None)
    model_link_arena = get_data('Model Link (Arena)', default=None)
    # Ensure we don't use 'N/A' as a link
    model_link = None
    if model_link_livebench and model_link_livebench != 'N/A':
        model_link = model_link_livebench
    elif model_link_arena and model_link_arena != 'N/A':
        model_link = model_link_arena


    # Build the card HTML structure
    card_html = f"""
    <div class="model-card">
        <h3>{model_name}</h3>
        <h4>by {get_data('Organization')}</h4>
        <div class="metrics">
            <div class="metric-group">
                <h5>Performance Metrics</h5>
                <table class="metric-table">
                    <tr><td>Global Average:</td><td><b>{get_data('Global Average', 'N/A')}</b></td></tr>
                    <tr><td>Reasoning:</td><td>{get_data('Reasoning Average', 'N/A')}</td></tr>
                    <tr><td>Coding:</td><td>{get_data('Coding Average', 'N/A')}</td></tr>
                    <tr><td>Mathematics:</td><td>{get_data('Mathematics Average', 'N/A')}</td></tr>
                    <tr><td>Data Analysis:</td><td>{get_data('Data Analysis Average', 'N/A')}</td></tr>
                    <tr><td>Language:</td><td>{get_data('Language Average', 'N/A')}</td></tr>
                    <tr><td>IF Average:</td><td>{get_data('IF Average', 'N/A')}</td></tr>
                </table>
            </div>
            <div class="metric-group">
                <h5>Community Data (Arena)</h5>
                <table class="metric-table">
                    <tr><td>Rank (No Style Ctrl):</td><td>{get_data('Arena Rank (No Style Control)', 'N/A')}</td></tr>
                    <tr><td>Rank (Style Ctrl):</td><td>{get_data('Arena Rank (With Style Control)', 'N/A')}</td></tr>
                    <tr><td>Arena Score:</td><td>{get_data('Arena Score', 'N/A')}</td></tr>
                    <tr><td>Confidence Interval:</td><td>{get_data('95% Confidence Interval', 'N/A')}</td></tr>
                    <tr><td># of Votes:</td><td>{get_data('# of Votes', 'N/A')}</td></tr>
                </table>
            </div>
        </div>
        <div class="model-details">
            <h5>Model Information</h5>
            <table class="detail-table">
                <tr><td>License:</td><td>{get_data('Model License', 'N/A')}</td></tr>
                <tr><td>Knowledge Cutoff:</td><td>{get_data('Model Knowledge Cutoff', 'N/A')}</td></tr>
                {f"<tr><td>LiveBench Name:</td><td>{get_data('Model Name (LiveBench)', 'N/A')}</td></tr>" if get_data('Model Name (LiveBench)', default=None) else ""}
                {f"<tr><td>Arena Name:</td><td>{get_data('Model Name (Arena)', 'N/A')}</td></tr>" if get_data('Model Name (Arena)', default=None) else ""}
                {f"<tr><td>LiveBench Link:</td><td><a href='{model_link_livebench}' target='_blank'>{model_link_livebench}</a></td></tr>" if model_link_livebench and model_link_livebench != 'N/A' else ""}
                {f"<tr><td>Arena Link:</td><td><a href='{model_link_arena}' target='_blank'>{model_link_arena}</a></td></tr>" if model_link_arena and model_link_arena != 'N/A' else ""}
            </table>
        </div>
    """
    # Add the "Learn More" button only if a valid link was found
    if model_link:
        card_html += f"""<div class="model-link"><a href="{model_link}" target="_blank" rel="noopener noreferrer">Learn More</a></div>"""

    card_html += """</div>""" # Close model-card div
    return card_html


# create_comparison_chart remains the same as previous version
def create_comparison_chart(df, metric):
    """Create a bar chart comparing models on a specific metric"""
    if df is None or df.empty or metric not in df.columns:
        return go.Figure().update_layout(title=f"No data available for {metric}", xaxis_title="Model", yaxis_title=metric)

    # Use get_top_models which handles sorting correctly (ascending for ranks)
    top_df = get_top_models(df, metric, n=15)

    if top_df.empty:
       return go.Figure().update_layout(title=f"No models found for {metric}", xaxis_title="Model", yaxis_title=metric)

    # Determine category order based on whether it's a rank
    category_order = 'total ascending' if 'Rank' in metric else 'total descending'

    # --- Define columns needed for hover info ---
    # Must include the x-axis identifier and any other desired fields
    custom_data_potential_cols = [
        model_identifier_col, # Index 0 (Model Name)
        'Organization',       # Index 1
        'Arena Score',        # Index 2
        'Arena Rank (No Style Control)', # Index 3
        'Global Average'      # Index 4 (Example: Add Global Average)
    ]
    # Filter to columns that actually exist in top_df
    valid_custom_data_cols = [col for col in custom_data_potential_cols if col in top_df.columns]

    fig = px.bar(
        top_df,
        x=model_identifier_col,
        y=metric,
        color='Organization' if 'Organization' in top_df.columns else None,
        title=f'Top 15 Models by {metric}',
        labels={model_identifier_col: "Model Name", "Organization": "Organization", metric: metric}, # Dynamic Y label
        custom_data=valid_custom_data_cols, # Pass only existing columns
        height=500
    )

    fig.update_layout(
        xaxis_title="Model Name",
        yaxis_title=metric,
        xaxis={'categoryorder': category_order},
        plot_bgcolor='rgba(240, 240, 240, 0.8)',
        margin=dict(l=40, r=20, t=60, b=120), # Adjusted bottom margin for tilted labels
    )
    fig.update_xaxes(tickangle=45) # Tilt labels for better readability

    # --- Build Hovertemplate using custom_data indices ---
    template_parts = []
    # Helper to safely add parts to template
    def add_template_part(col_name, display_name, formatting=""):
        if col_name in valid_custom_data_cols:
            try:
                idx = valid_custom_data_cols.index(col_name)
                template_parts.append(f"<b>{display_name}:</b> %{{customdata[{idx}]{formatting}}}<br>")
            except (ValueError, IndexError): pass # Should not happen if check passes

    # Add parts based on valid columns
    add_template_part(model_identifier_col, "Model")
    add_template_part('Organization', "Organization")
    template_parts.append(f"<b>{metric}:</b> %{{y:,.2f}}<br>") # Always show the main metric from y-axis

    # Add other metrics if they exist and are not the main metric being plotted
    if 'Global Average' in valid_custom_data_cols and 'Global Average' != metric:
        add_template_part('Global Average', "Global Avg", ":,.2f")
    if 'Arena Score' in valid_custom_data_cols and 'Arena Score' != metric:
         add_template_part('Arena Score', "Arena Score", ":,.0f")
    if 'Arena Rank (No Style Control)' in valid_custom_data_cols and 'Arena Rank (No Style Control)' != metric:
         add_template_part('Arena Rank (No Style Control)', "Arena Rank", ":,.0f")

    template_parts.append("<extra></extra>") # Remove trace info
    hovertemplate = "".join(template_parts)

    # Apply the hovertemplate
    fig.update_traces(hovertemplate=hovertemplate)

    return fig


# create_radar_chart remains the same as previous version
def create_radar_chart(df, model_names, metrics):
    """Create a radar chart comparing multiple models across metrics, including scaled Arena Score"""
    # Input validation
    if df is None or df.empty or not model_names or not metrics:
         return go.Figure().update_layout(title="Select models and ensure metrics exist", polar=dict(radialaxis=dict(visible=False)))

    # Ensure metrics requested are valid and exist in the dataframe
    valid_metrics = [m for m in metrics if m in df.columns]
    if not valid_metrics:
         return go.Figure().update_layout(title="None of the selected metrics exist in the data", polar=dict(radialaxis=dict(visible=False)))

    # Filter data for selected models
    # Use .loc to avoid SettingWithCopyWarning if modifications are made later
    filtered_df = df.loc[df[model_identifier_col].isin(model_names)].copy()
    if filtered_df.empty:
        return go.Figure().update_layout(title="Selected models not found in data", polar=dict(radialaxis=dict(visible=False)))

    fig = go.Figure()
    radar_plot_metrics = [] # Final list of metric labels for the chart axes

    # Process each selected model
    processed_models_count = 0 # Keep track of models successfully added
    for _, row in filtered_df.iterrows():
        model_values = []
        current_radar_metrics = [] # Track metrics used for this model row

        for metric in valid_metrics: # Use only valid metrics
            metric_label = metric # Default label
            value = 0 # Default value if missing or invalid

            if metric == 'Arena Score':
                metric_label = 'Arena Score (Scaled)' # Use distinct label
                if 'Arena Score' in row and pd.notna(row['Arena Score']):
                    score = row['Arena Score']
                    # Scale score using global min/max (check for division by zero)
                    if global_arena_max > global_arena_min:
                        scaled_score = ((score - global_arena_min) / (global_arena_max - global_arena_min)) * 100
                        value = np.clip(scaled_score, 0, 100) # Clip to 0-100 range
                    else:
                        value = 50 # Neutral value if scaling is not possible
                # else: value remains 0 (default for missing Arena Score)
            elif metric in row and pd.notna(row[metric]):
                # Standard metric handling
                numeric_val = pd.to_numeric(row[metric], errors='coerce')
                if pd.notna(numeric_val):
                    value = np.clip(numeric_val, 0, 100) # Clip other metrics to 0-100
                # else: value remains 0 (default for non-numeric or NaN)

            # Append value and the corresponding label
            model_values.append(value)
            current_radar_metrics.append(metric_label)

        # Use the metrics from the first processed model as the canonical set for theta
        if not radar_plot_metrics:
            radar_plot_metrics = current_radar_metrics

        # Add trace for this model if data was processed and matches expected metrics
        if model_values and len(model_values) == len(radar_plot_metrics):
             r_values = model_values + [model_values[0]] # Close the polygon
             theta_values = radar_plot_metrics + [radar_plot_metrics[0]] # Close theta polygon

             fig.add_trace(go.Scatterpolar(
                 r=r_values,
                 theta=theta_values,
                 fill='toself',
                 name=row[model_identifier_col], # Model name for the legend
                 hoverinfo='text', # Use custom hover text
                 # Define hover text for each point on the radar
                 text=[f"{metric_label}: {val:.2f}" for metric_label, val in zip(radar_plot_metrics, model_values)] + [f"{radar_plot_metrics[0]}: {model_values[0]:.2f}"] # Text for hover
             ))
             processed_models_count += 1


    # Final figure layout adjustments
    if processed_models_count == 0: # If no traces were added
        return go.Figure().update_layout(title="No data to display for selected models/metrics", polar=dict(radialaxis=dict(visible=False)))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100] # Fixed range due to clipping/scaling
            )
        ),
        showlegend=True,
        legend_title_text='Model Name',
        title="Model Comparison Across Metrics",
        height=600
    )

    return fig


# --- Main UI Function ---

def create_leaderboard_ui():
    """Create the main Gradio UI"""
    css = """
    /* General Layout & Styling */
    .gradio-container { max-width: 95% !important; margin: 0 auto !important;}
    .container { max-width: none; margin: 0 auto; padding: 0 15px; }
    .header { text-align: center; margin-bottom: 1rem; } /* Reduced margin */
    .intro-text { max-width: 800px; margin: 0 auto 2rem auto; padding: 15px; background-color: #f8f9fa; border-radius: 8px; text-align: left; font-size: 0.95em; line-height: 1.6; border: 1px solid #e9ecef; }
    .intro-text h4 { margin-top: 0; margin-bottom: 10px; color: #4a69bd; }
    .intro-text p { margin-bottom: 10px; }
    .intro-text ul { margin-left: 20px; margin-bottom: 10px; }
    .intro-text a { color: #3b5998; text-decoration: none; }
    .intro-text a:hover { text-decoration: underline; }


    /* Filter Container */
    .filter-container { background-color: #f5f7fa; border-radius: 10px; padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }

    /* Model Card Styling */
    .model-card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; border: 1px solid #eee; }
    .model-card h3 { margin-top: 0; color: #333; }
    .model-card h4 { margin-bottom: 15px; color: #555; font-weight: normal; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 15px 0; }
    .metric-group h5 { margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 5px; color: #4a69bd; font-size: 1em; }
    .metric-table, .detail-table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
    .metric-table td, .detail-table td { padding: 5px 0; vertical-align: top; }
    .metric-table td:first-child { color: #555; width: 60%; }
    .metric-table td:last-child { font-weight: bold; text-align: right; }
    .detail-table td:first-child { color: #555; width: 30%;}
    .detail-table td:last-child { width: 70%; word-wrap: break-word; } /* Allow long details to wrap */
    .model-details h5 { margin-top: 20px; margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 5px; color: #4a69bd; font-size: 1em; }
    .model-link { margin-top: 15px; text-align: right; }
    .model-link a { display: inline-block; background: #4a69bd; color: white !important; padding: 8px 16px; border-radius: 4px; text-decoration: none; font-weight: bold; font-size: 0.9em; transition: background-color 0.2s ease; }
    .model-link a:hover { background: #3b549a; }

    /* DataFrame Table Styling */
    .gradio-dataframe { overflow-x: auto; } /* Add horizontal scroll if needed */
    .gradio-dataframe table.dataframe { table-layout: auto; width: 100%; border-collapse: collapse; border: 1px solid #ddd; font-size: 0.9em; }
    .gradio-dataframe table.dataframe th, .gradio-dataframe table.dataframe td { white-space: normal; word-wrap: break-word; padding: 8px 10px; text-align: left; vertical-align: top; }
    .gradio-dataframe table.dataframe th { background-color: #f2f2f2; border-bottom: 2px solid #ddd; font-weight: bold; }
    .gradio-dataframe table.dataframe td { border-bottom: 1px solid #eee; }
    .gradio-dataframe table.dataframe tr:nth-child(even) { background-color: #f9f9f9; }
    /* REMOVED: .gradio-dataframe table.dataframe tr:hover { background-color: #e9f7ef; } */ /* Removed hover effect */

    /* Attempt to hide the multi-cell selection visual */
    .gradio-dataframe table.dataframe td::selection {
        background-color: transparent; /* Make selection background invisible */
        color: inherit; /* Keep text color the same */
    }
     .gradio-dataframe table.dataframe td::-moz-selection { /* Firefox */
        background-color: transparent;
        color: inherit;
    }


    /* Plot Styling */
    .gradio-plot { min-height: 400px; padding-top: 10px; }

    /* Text Selection Control */
    /* Allow selection everywhere by default */
    body, .gradio-container {
         user-select: text;
         -webkit-user-select: text;
         -ms-user-select: text;
    }
    /* Ensure dataframes and model cards specifically allow text selection */
    /* This allows copying text content */
    .gradio-dataframe, .model-card {
         user-select: text !important;
        -webkit-user-select: text !important;
        -ms-user-select: text !important;
    }
    """

    with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue)) as app:
        # --- Header and Introduction ---
        gr.HTML("""<div class="header"><h1>LLM Leaderboard Explorer</h1><p>Interactive visualization of merged leaderboards data</p></div>""")

        # --- ADDED: Introduction Section ---
        gr.Markdown("""
        <div class="intro-text">
        <h4>Welcome!</h4>
        <p>This application provides an interactive view of combined data from two leading LLM evaluation platforms:</p>
        <ul>
            <li><a href="https://livebench.ai/#/" target="_blank">LiveBench</a>: ​LiveBench is a dynamic benchmark, featuring monthly updated, contamination-free tasks with objective scoring across diverse domains.</li>
            <li><a href="https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard" target="_blank">LMSYS Chatbot Arena</a>: Uses crowd-sourced human preferences (Elo ratings) to rank models based on conversation quality.</li>
        </ul>
        <h4>Key Data Points:</h4>
        <ul>
            <li><b>Performance Metrics (LiveBench):</b> Includes 'Global Average' score and specific capability scores like 'Reasoning', 'Coding', 'Mathematics', etc. Higher is generally better.</li>
            <li><b>Community Stats (LMSYS):</b> Features 'Arena Score' (Elo rating) and corresponding Ranks. Higher scores/lower ranks are better.</li>
            <li><b>Model Details:</b> Provides information like Organization, License, and Knowledge Cutoff date.</li>
        </ul>
        <h4>How to Use This App:</h4>
        <ul>
            <li><b>Filter & Search:</b> Use the controls above the tabs to search for models or filter by organization and minimum 'Global Average' score.</li>
            <li><b>Explore Tabs:</b> View different slices of the data (Performance, Details, Community Stats, Mapping).</li>
            <li><b>View Model Card:</b> Click on any row in the tables (except in the Visualizations tab) to see a detailed card with all metrics for that model.</li>
            <li><b>Visualize & Compare:</b> Use the 'Visualizations' tab to compare top models on specific metrics (Bar Chart) or compare selected models across multiple dimensions (Radar Chart).</li>
        </ul>
        </div>
        """, elem_classes="intro-text-container") # Added elem_classes for potential container styling if needed


        # --- Filter Components ---
        with gr.Row(elem_classes="filter-container"):
            search_input = gr.Textbox(label="Search Models/Org", placeholder="Search...", show_label=False, scale=2)
            org_choices = get_organization_list(df_global) # Get org list once
            org_dropdown = gr.Dropdown(choices=org_choices, label="Organization", value="All", show_label=False, scale=1, min_width=160)
            max_slider_val = 100
            if 'Global Average' in df_global.columns and not df_global['Global Average'].isna().all():
                 # Calculate max based on data, ensuring it's at least 100
                 max_slider_val = max(100, df_global['Global Average'].max(skipna=True))
            min_score_slider = gr.Slider(minimum=0, maximum=max_slider_val, value=0, label="Min Global Avg", step=1, scale=2, min_width=200)

        # --- Tabbed Interface ---
        with gr.Tabs() as tabs:
            # Get column sets safely, handling potential missing groups or empty lists
            main_metrics_cols = get_valid_columns(df_global, column_groups_global.get("Main Metrics", []))
            model_details_cols = get_valid_columns(df_global, column_groups_global.get("Model Details", []))
            community_stats_cols = get_valid_columns(df_global, column_groups_global.get("Community Stats", []))
            model_mapping_cols = get_valid_columns(df_global, column_groups_global.get("Model Mapping", [])) # Now includes 'Model Name'

            # Tab 1: Performance Metrics
            with gr.TabItem("Performance Metrics"):
                main_metrics_table = gr.DataFrame(
                    interactive=True, # Enable selection
                    wrap=True,
                    elem_classes="gradio-dataframe"
                )
                gr.Markdown("Click a row for details.")
                main_metrics_card_output = gr.HTML() # Output for this tab's card

            # Tab 2: Model Details
            with gr.TabItem("Model Details"):
                model_details_table = gr.DataFrame(
                    interactive=True, # Enable selection
                    wrap=True,
                    elem_classes="gradio-dataframe"
                )
                gr.Markdown("Click a row for details.")
                model_details_card_output = gr.HTML() # Output for this tab's card

            # Tab 3: Community Stats
            with gr.TabItem("Community Stats"):
                community_stats_table = gr.DataFrame(
                    interactive=True, # Enable selection
                    wrap=True,
                    elem_classes="gradio-dataframe"
                )
                gr.Markdown("Click a row for details.")
                community_stats_card_output = gr.HTML() # Output for this tab's card

            # Tab 4: Model Mapping
            with gr.TabItem("Model Mapping"):
                 model_mapping_table = gr.DataFrame(
                     interactive=True, # Enable selection
                     wrap=True,
                     elem_classes="gradio-dataframe"
                 )
                 gr.Markdown("Click a row for details.")
                 model_mapping_card_output = gr.HTML() # Output for this tab's card

            # Tab 5: Visualizations (No selection needed here)
            with gr.TabItem("Visualizations"):
                gr.Markdown("### Compare Top Models by Metric")
                with gr.Row():
                    # Define potential metrics for the bar chart
                    perf_metrics = ["Global Average", "Reasoning Average", "Coding Average", "Mathematics Average", "Data Analysis Average", "Language Average", "IF Average"]
                    arena_metrics = ["Arena Score", "Arena Rank (No Style Control)", "Arena Rank (With Style Control)", "# of Votes"]
                    # Combine and filter based on actual columns in the dataframe
                    available_bar_metrics = perf_metrics + [m for m in arena_metrics if m in df_global.columns]
                    valid_bar_metrics = get_valid_columns(df_global, available_bar_metrics)
                    # Set default value safely
                    default_bar_metric = valid_bar_metrics[0] if valid_bar_metrics else None
                    metric_dropdown = gr.Dropdown(choices=valid_bar_metrics, label="Select Metric for Bar Chart", value=default_bar_metric)
                bar_chart = gr.Plot(label="Top 15 Model Comparison", elem_classes="gradio-plot")

                gr.Markdown("### Radar Chart Comparison")
                with gr.Row():
                     # Get model choices from the identifier column
                     model_choices = []
                     if model_identifier_col in df_global.columns and not df_global[model_identifier_col].isna().all():
                         model_choices = sorted(df_global[model_identifier_col].dropna().unique().tolist())

                     # Determine default models for radar chart (e.g., top 3 by Global Average)
                     default_radar_models = []
                     if 'Global Average' in df_global.columns and not df_global.empty and model_identifier_col in df_global.columns:
                         # Ensure we don't select more models than exist and handle potential NaNs
                         n_top = min(3, len(df_global.dropna(subset=['Global Average', model_identifier_col])))
                         if n_top > 0:
                             default_radar_models = df_global.nlargest(n_top, 'Global Average')[model_identifier_col].tolist()


                     models_multiselect = gr.Dropdown(choices=model_choices, label="Select Models for Radar Chart (up to 5)", multiselect=True, max_choices=5, value=default_radar_models)
                radar_chart = gr.Plot(label="Radar Comparison", elem_classes="gradio-plot")

        # --- Event Handling --- (Keep this section as it was in the previous correct version) ---

        # Function to update all tables and plots based on filters
        def update_on_filter(search, min_global, organization):
            filtered_df = filter_data(df_global, search, min_global, organization)

            # Prepare data for each table (handle empty filtered results)
            def get_table_data(df, cols):
                 # Ensure cols is a list even if None/empty
                 valid_cols = [c for c in cols if c in df.columns] if cols else []
                 return df[valid_cols].fillna("N/A") if not df.empty and valid_cols else pd.DataFrame(columns=valid_cols)


            main_metrics_data = get_table_data(filtered_df, main_metrics_cols)
            model_details_data = get_table_data(filtered_df, model_details_cols)
            community_stats_data = get_table_data(filtered_df, community_stats_cols)
            model_mapping_data = get_table_data(filtered_df, model_mapping_cols) # Will now include 'Model Name'

            # Update model choices for radar dropdown based on filtered results
            models_list = []
            if not filtered_df.empty and model_identifier_col in filtered_df.columns and not filtered_df[model_identifier_col].isna().all():
                models_list = sorted(filtered_df[model_identifier_col].dropna().unique().tolist())


            # Update bar chart
            current_bar_metric = metric_dropdown.value if hasattr(metric_dropdown, 'value') and metric_dropdown.value in valid_bar_metrics else default_bar_metric
            bar_plot_update = create_comparison_chart(filtered_df, current_bar_metric)

            # Update radar chart multiselect choices and value
            current_radar_models = models_multiselect.value if hasattr(models_multiselect, 'value') else []
            # Keep selected models if they are still in the filtered list, otherwise clear/reset
            valid_radar_selection = [m for m in current_radar_models if m in models_list][:5] # Limit to 5
            radar_multiselect_update = gr.update(choices=models_list, value=valid_radar_selection)

            # Update radar plot itself
            # Decide whether radar compares within filtered set or globally
            # Using df_global to compare selected models' overall profile
            radar_plot_update = create_radar_chart(df_global, valid_radar_selection, radar_metrics)

            # Clear all model card outputs
            clear_card = gr.update(value="")

            return (
                main_metrics_data, model_details_data, community_stats_data, model_mapping_data, # Tables
                radar_multiselect_update, # Radar dropdown update
                bar_plot_update, radar_plot_update, # Plots
                clear_card, clear_card, clear_card, clear_card # Clear all card outputs
            )

        # Function to display model details card
        def show_model_details_card(evt: gr.SelectData, table_data: pd.DataFrame):
            # Basic validation of event data and table data
            if evt is None or not hasattr(evt, 'index') or not isinstance(evt.index, (list, tuple)) or not evt.index:
                 # No valid selection index
                 return gr.update(value="")
            if table_data is None or table_data.empty:
                 # Table data is missing or empty
                 return gr.update(value="<p>Error: Table data is not available.</p>")

            # Check if evt.index is within bounds
            if not (0 <= evt.index[0] < len(table_data)):
                 print(f"Error: Row index {evt.index[0]} out of bounds for table length {len(table_data)}")
                 return gr.update(value="<p>Error: Selected row index is out of bounds.</p>")

            row_idx = evt.index[0] # Get the selected row index

            try:
                # Check if the identifier column exists in the *displayed* table data
                if model_identifier_col not in table_data.columns:
                     print(f"Error: Model identifier column '{model_identifier_col}' not found in the clicked table's data. Columns are: {table_data.columns.tolist()}")
                     return gr.update(value=f"<p>Error: Cannot identify model from this table (missing '{model_identifier_col}').</p>")

                # Get the model identifier from the selected row of the *displayed* table data
                selected_model_identifier = table_data.iloc[row_idx][model_identifier_col]

                # Check if the identifier is valid (e.g., not NaN or placeholder)
                if pd.isna(selected_model_identifier) or selected_model_identifier == "N/A":
                     print(f"Warning: Invalid model identifier '{selected_model_identifier}' selected at index {row_idx}.")
                     return gr.update(value="<p>Cannot display details for this entry (invalid identifier).</p>")


                # Find the full data row in the *original* global dataframe
                # This ensures we have all columns needed for the card, even if not displayed in the clicked table
                full_model_data_row = df_global[df_global[model_identifier_col] == selected_model_identifier]

                if full_model_data_row.empty:
                    print(f"Error: Could not find full details for model identifier '{selected_model_identifier}' in global data.")
                    return gr.update(value=f"<p>Error: Could not find full details for {selected_model_identifier}.</p>")

                # Convert the first found row to a dictionary (handle potential duplicates, take first)
                full_data_dict = full_model_data_row.iloc[0].to_dict()

                # Create the HTML card
                card_html = create_model_card(full_data_dict)
                return gr.update(value=card_html)

            except KeyError as e:
                 # This might happen if iloc fails or the identifier column name is wrong somehow
                 print(f"KeyError during model card generation: {e}. Identifier column: '{model_identifier_col}', Table columns: {table_data.columns.tolist()}")
                 return gr.update(value=f"<p>Error: A data key ('{e}') was not found while generating the card.</p>")
            except Exception as e:
                 # Catch any other unexpected errors
                 import traceback
                 print(f"Unexpected error generating model card: {e}\n{traceback.format_exc()}")
                 return gr.update(value=f"<p>An unexpected error occurred: {e}</p>")


        # Function to update bar chart when metric selection changes
        def update_bar_chart_on_metric_change(metric, search, min_global, organization):
             # Refilter data based on current filter settings
             filtered_df = filter_data(df_global, search, min_global, organization)
             chart = create_comparison_chart(filtered_df, metric)
             return chart

        # Function to update radar chart when model selection changes
        def update_radar_chart_on_selection(selected_models, search, min_global, organization):
             if not selected_models:
                 # Return an empty chart with a message if no models are selected
                 return go.Figure().update_layout(title="Select models to compare", polar=dict(radialaxis=dict(visible=False)))
             # Use df_global for radar chart to compare overall model profiles
             chart = create_radar_chart(df_global, selected_models, radar_metrics)
             return chart


        # --- Connect Components ---
        filter_inputs = [search_input, min_score_slider, org_dropdown]
        # Outputs need to include all tables, the radar multiselect, both plots, and all card outputs
        filter_outputs = [
             main_metrics_table, model_details_table, community_stats_table, model_mapping_table, # Tables
             models_multiselect, # Radar dropdown
             bar_chart, radar_chart, # Plots
             main_metrics_card_output, model_details_card_output, community_stats_card_output, model_mapping_card_output # Card outputs
        ]

        # Trigger update when any filter changes
        search_input.submit(update_on_filter, inputs=filter_inputs, outputs=filter_outputs)
        min_score_slider.release(update_on_filter, inputs=filter_inputs, outputs=filter_outputs)
        org_dropdown.change(update_on_filter, inputs=filter_inputs, outputs=filter_outputs)

        # Connect table selections to their respective card outputs
        # Pass the table component itself as the second input (its value)
        main_metrics_table.select(show_model_details_card, inputs=[main_metrics_table], outputs=[main_metrics_card_output])
        model_details_table.select(show_model_details_card, inputs=[model_details_table], outputs=[model_details_card_output])
        community_stats_table.select(show_model_details_card, inputs=[community_stats_table], outputs=[community_stats_card_output])
        model_mapping_table.select(show_model_details_card, inputs=[model_mapping_table], outputs=[model_mapping_card_output]) # This should now work


        # Connect visualization dropdowns to their update functions
        metric_dropdown.change(
            update_bar_chart_on_metric_change,
            inputs=[metric_dropdown] + filter_inputs, # Pass filters for context
            outputs=bar_chart
        )
        models_multiselect.change(
            update_radar_chart_on_selection,
            inputs=[models_multiselect] + filter_inputs, # Pass filters for context
            outputs=radar_chart
        )

        # --- Initial Load ---
        def initial_load():
            # Prepare initial data for all tables (use fillna)
            initial_main_metrics = df_global[main_metrics_cols].fillna("N/A") if main_metrics_cols else pd.DataFrame(columns=main_metrics_cols)
            initial_model_details = df_global[model_details_cols].fillna("N/A") if model_details_cols else pd.DataFrame(columns=model_details_cols)
            initial_community_stats = df_global[community_stats_cols].fillna("N/A") if community_stats_cols else pd.DataFrame(columns=community_stats_cols)
            initial_model_mapping = df_global[model_mapping_cols].fillna("N/A") if model_mapping_cols else pd.DataFrame(columns=model_mapping_cols) # Will include 'Model Name'

            # Initial bar chart
            bar_plot = create_comparison_chart(df_global, default_bar_metric)

             # Get initial choices for the multiselect dropdown
            initial_model_choices = []
            if model_identifier_col in df_global.columns and not df_global[model_identifier_col].isna().all():
                initial_model_choices = sorted(df_global[model_identifier_col].dropna().unique().tolist())


            # Initial radar chart (using default model selection determined earlier)
            radar_plot = create_radar_chart(df_global, default_radar_models, radar_metrics)

            # Return initial state for all components that need loading
            # Ensure order matches load_outputs
            return (
                initial_main_metrics, initial_model_details, initial_community_stats, initial_model_mapping, # Tables
                bar_plot, # Bar Chart
                gr.update(value=default_radar_models, choices=initial_model_choices), # Update radar dropdown value AND choices
                radar_plot # Radar Chart
            )

        # Define outputs for the load function - ensure order matches initial_load return tuple
        load_outputs = [
             main_metrics_table, model_details_table, community_stats_table, model_mapping_table, # Tables
             bar_chart, # Bar Chart
             models_multiselect, # Radar dropdown (to set initial value and choices)
             radar_chart # Radar Chart
        ]
        app.load(initial_load, inputs=[], outputs=load_outputs)

    return app

# Optional: If running this file directly for testing
# Ensure df_global is valid before launching
if __name__ == "__main__":
    if df_global is not None and not df_global.empty and 'Error' not in df_global.columns:
         ui = create_leaderboard_ui()
         ui.launch(debug=True) # Use debug=True for easier troubleshooting
    else:
         # Provide more context if loading failed
         if df_global is not None and 'Error' in df_global.columns:
              print(f"Failed to load data: {df_global['Error'].iloc[0]}")
         else:
              print("Dataframe is None or empty. Cannot launch UI.")