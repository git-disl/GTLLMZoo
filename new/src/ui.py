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
        df_global = pd.DataFrame(columns=['Model Name', 'Organization', 'Global Average', 'Arena Score']) # Add Arena Score default

    column_groups_global = get_column_groups()
    model_identifier_col = "Model Name"

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
        global_arena_min = df_global['Arena Score'].min(skipna=True)
        global_arena_max = df_global['Arena Score'].max(skipna=True)
        # Handle case where min/max might still be NaN (e.g., all values are NaN)
        if pd.isna(global_arena_min) or pd.isna(global_arena_max):
             global_arena_min = 0 # Default scale if calculation fails
             global_arena_max = 1 # Avoid division by zero later
        elif global_arena_max == global_arena_min:
             # Avoid division by zero if all scores are the same
             global_arena_max += 1


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
    return [col for col in column_list if col in df.columns]

# --- UI Component Creation Functions ---
# create_model_card remains the same as previous version
def create_model_card(model_data):
    """Create a detailed card view for a single model"""
    if not isinstance(model_data, dict): return "<p>Error: Invalid model data.</p>"
    model_name = model_data.get(model_identifier_col, 'Unknown Model')
    def get_data(key, default='N/A'):
        val = model_data.get(key); return default if pd.isna(val) else val
    model_link = get_data('Model Link (LiveBench)', default=None) or get_data('Model Link (Arena)', default=None)
    card_html = f"""<div class="model-card"><h3>{model_name}</h3><h4>by {get_data('Organization')}</h4><div class="metrics"><div class="metric-group"><h5>Performance Metrics</h5><table class="metric-table"><tr><td>Global Average:</td><td><b>{get_data('Global Average')}</b></td></tr><tr><td>Reasoning:</td><td>{get_data('Reasoning Average')}</td></tr><tr><td>Coding:</td><td>{get_data('Coding Average')}</td></tr><tr><td>Mathematics:</td><td>{get_data('Mathematics Average')}</td></tr><tr><td>Data Analysis:</td><td>{get_data('Data Analysis Average')}</td></tr><tr><td>Language:</td><td>{get_data('Language Average')}</td></tr><tr><td>IF Average:</td><td>{get_data('IF Average')}</td></tr></table></div><div class="metric-group"><h5>Community Data (Arena)</h5><table class="metric-table"><tr><td>Rank (No Style Ctrl):</td><td>{get_data('Arena Rank (No Style Control)')}</td></tr><tr><td>Rank (Style Ctrl):</td><td>{get_data('Arena Rank (With Style Control)')}</td></tr><tr><td>Arena Score:</td><td>{get_data('Arena Score')}</td></tr><tr><td>Confidence Interval:</td><td>{get_data('95% Confidence Interval')}</td></tr><tr><td># of Votes:</td><td>{get_data('# of Votes')}</td></tr></table></div></div><div class="model-details"><h5>Model Information</h5><table class="detail-table"><tr><td>License:</td><td>{get_data('Model License')}</td></tr><tr><td>Knowledge Cutoff:</td><td>{get_data('Model Knowledge Cutoff')}</td></tr>{f"<tr><td>LiveBench Link:</td><td>{get_data('Model Link (LiveBench)')}</td></tr>" if get_data('Model Link (LiveBench)') != 'N/A' else ""}{f"<tr><td>Arena Link:</td><td>{get_data('Model Link (Arena)')}</td></tr>" if get_data('Model Link (Arena)') != 'N/A' else ""}</table>"""
    if model_link: card_html += f"""<div class="model-link"><a href="{model_link}" target="_blank" rel="noopener noreferrer">Learn More</a></div>"""
    card_html += """</div></div>"""
    return card_html


# MODIFIED: create_comparison_chart to handle ranks and update hover text
def create_comparison_chart(df, metric):
    """Create a bar chart comparing models on a specific metric"""
    if df.empty or metric not in df.columns:
        return go.Figure().update_layout(title=f"No data available for {metric}", xaxis_title="Model", yaxis_title=metric)

    # Use get_top_models which handles sorting correctly
    top_df = get_top_models(df, metric, n=15)

    if top_df.empty:
       return go.Figure().update_layout(title=f"No models found for {metric}", xaxis_title="Model", yaxis_title=metric)

    category_order = 'total descending'
    if 'Rank' in metric:
        category_order = 'total ascending'

    # --- Define columns needed for hover info ---
    # Must include the x-axis identifier and any other desired fields
    # Order matters for hovertemplate indexing!
    custom_data_potential_cols = [
        model_identifier_col, # Index 0 (Model Name)
        'Organization',       # Index 1
        'Arena Score',        # Index 2
        'Arena Rank (No Style Control)' # Index 3
        # Add more here if needed, maintaining order consistency
    ]
    # Filter to columns that actually exist in top_df
    valid_custom_data_cols = [col for col in custom_data_potential_cols if col in top_df.columns]

    fig = px.bar(
        top_df, # Pass the filtered, sorted dataframe
        x=model_identifier_col,
        y=metric,
        color='Organization' if 'Organization' in top_df.columns else None,
        title=f'Top 15 Models by {metric}',
        labels={model_identifier_col: "Model Name", "Organization": "Organization"}, # Axis/legend labels
        # Pass the specific columns needed for the hovertemplate via custom_data.
        # Plotly Express will align this data row-by-row with the bars.
        custom_data=valid_custom_data_cols,
        height=500
    )

    fig.update_layout(
        xaxis_title="Model Name",
        yaxis_title=metric,
        xaxis={'categoryorder': category_order}, # Ensure bars are ordered as expected
        plot_bgcolor='rgba(240, 240, 240, 0.8)',
        margin=dict(l=40, r=20, t=60, b=120),
    )
    fig.update_xaxes(tickangle=45)

    # --- Build Hovertemplate using custom_data indices ---
    # Indices must match the order in `valid_custom_data_cols`
    template_parts = []

    # Check if each column exists in our final list and add its part to the template
    try: # Model Name (Index 0)
        model_idx = valid_custom_data_cols.index(model_identifier_col)
        template_parts.append(f"<b>Model Name:</b> %{{customdata[{model_idx}]}}<br>")
    except (ValueError, IndexError): pass

    try: # Organization
        org_idx = valid_custom_data_cols.index('Organization')
        template_parts.append(f"<b>Organization:</b> %{{customdata[{org_idx}]}}<br>")
    except (ValueError, IndexError): pass

    # Main Metric (always show using %{y})
    template_parts.append(f"<b>{metric}:</b> %{{y:,.2f}}<br>")

    # Arena Score (if available and not the main metric)
    arena_score_col = 'Arena Score'
    if arena_score_col in valid_custom_data_cols and arena_score_col != metric:
        try:
             score_idx = valid_custom_data_cols.index(arena_score_col)
             # Apply formatting for the number within the template
             template_parts.append(f"<b>{arena_score_col}:</b> %{{customdata[{score_idx}]:,.0f}}<br>")
        except (ValueError, IndexError): pass

    # Arena Rank (if available and not the main metric)
    arena_rank_col = 'Arena Rank (No Style Control)'
    if arena_rank_col in valid_custom_data_cols and arena_rank_col != metric:
         try:
             rank_idx = valid_custom_data_cols.index(arena_rank_col)
             # Apply formatting for the number within the template
             template_parts.append(f"<b>{arena_rank_col}:</b> %{{customdata[{rank_idx}]:,.0f}}<br>")
         except (ValueError, IndexError): pass

    template_parts.append("<extra></extra>") # Remove trace info
    hovertemplate = "".join(template_parts)

    # Apply the hovertemplate to the traces generated by px.bar
    # We don't need to set customdata here again, as it was passed in px.bar
    fig.update_traces(hovertemplate=hovertemplate)

    return fig


# MODIFIED: create_radar_chart to include scaled Arena Score
def create_radar_chart(df, model_names, metrics):
    """Create a radar chart comparing multiple models across metrics, including scaled Arena Score"""
    if df.empty or not model_names or not metrics:
         return go.Figure().update_layout(title="Select models and ensure metrics exist", polar=dict(radialaxis=dict(visible=False)))

    filtered_df = df[df[model_identifier_col].isin(model_names)].copy()

    if filtered_df.empty:
        return go.Figure().update_layout(title="Selected models not found in data", polar=dict(radialaxis=dict(visible=False)))

    fig = go.Figure()

    # Prepare metrics list and values for radar chart
    radar_plot_metrics = [] # Labels for the axes
    all_model_values = [] # List of lists (one per model)

    for _, row in filtered_df.iterrows():
        model_values = []
        current_radar_metrics = [] # Track metrics used for this specific model (handles missing data)

        for metric in metrics:
            # Handle Arena Score scaling
            if metric == 'Arena Score':
                if 'Arena Score' in row and pd.notna(row['Arena Score']):
                    score = row['Arena Score']
                    # Scale score using global min/max (ensure max != min)
                    if global_arena_max > global_arena_min:
                        scaled_score = ((score - global_arena_min) / (global_arena_max - global_arena_min)) * 100
                        # Clip values to be within 0-100 range after scaling
                        model_values.append(np.clip(scaled_score, 0, 100))
                        current_radar_metrics.append('Arena Score (Scaled)') # Use a distinct name for the axis label
                    else: # Handle edge case where min=max or data insufficient
                        model_values.append(50) # Assign a neutral value
                        current_radar_metrics.append('Arena Score (Scaled)')
                else:
                    model_values.append(0) # Assign 0 if Arena Score is NaN
                    current_radar_metrics.append('Arena Score (Scaled)')
            # Handle other numeric metrics
            elif metric in row and pd.notna(row[metric]):
                 value = pd.to_numeric(row[metric], errors='coerce')
                 model_values.append(np.clip(value, 0, 100) if pd.notna(value) else 0) # Clip standard metrics too
                 current_radar_metrics.append(metric)
            else:
                 model_values.append(0) # Default for missing metrics
                 current_radar_metrics.append(metric)

        # Use the metrics found for the first model as the canonical list for theta
        if not radar_plot_metrics:
            radar_plot_metrics = current_radar_metrics

        # Ensure all models have the same number of points (pad with 0 if a metric was missing entirely)
        # This scenario is less likely if radar_metrics is pre-filtered globally
        if len(model_values) == len(radar_plot_metrics):
             all_model_values.append(model_values)


    # Add traces if we have valid data
    if radar_plot_metrics and all_model_values:
        theta = radar_plot_metrics + [radar_plot_metrics[0]] # Close the polygon

        for i, model_vals in enumerate(all_model_values):
            if len(model_vals) == len(radar_plot_metrics): # Check length consistency
                r_values = model_vals + [model_vals[0]] # Close the polygon
                fig.add_trace(go.Scatterpolar(
                    r=r_values,
                    theta=theta,
                    fill='toself',
                    name=filtered_df.iloc[i][model_identifier_col] # Get model name corresponding to the row index
                ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100] # Radial axis is 0-100 due to clipping/scaling
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
    /* Keep CSS */
    .gradio-container { max-width: 95% !important; margin: 0 auto !important;}
    .container { max-width: none; margin: 0 auto; padding: 0 15px; }
    .header { text-align: center; margin-bottom: 2rem; }
    .filter-container { background-color: #f5f7fa; border-radius: 10px; padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .model-card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; border: 1px solid #eee; }
    .model-card h3 { margin-top: 0; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 15px 0; }
    .metric-group h5 { margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 5px; }
    .metric-table, .detail-table { width: 100%; border-collapse: collapse; }
    .metric-table td, .detail-table td { padding: 4px 0; font-size: 0.9em; }
    .metric-table td:first-child { color: #555; width: 60%; }
    .metric-table td:last-child { font-weight: bold; text-align: right; }
    .detail-table td:first-child { color: #555; width: 30%;}
    .detail-table td:last-child { width: 70%;}
    .model-link { margin-top: 15px; }
    .model-link a { display: inline-block; background: #4a69bd; color: white !important; padding: 8px 16px; border-radius: 4px; text-decoration: none; font-weight: bold; font-size: 0.9em; }
    .model-link a:hover { background: #3b549a; }
    .gradio-dataframe table.dataframe { table-layout: auto; width: 100%; border-collapse: collapse; border: 1px solid #ddd; font-size: 0.9em; }
    .gradio-dataframe table.dataframe th, .gradio-dataframe table.dataframe td { white-space: normal; word-wrap: break-word; padding: 8px; text-align: left; vertical-align: top; }
    .gradio-dataframe table.dataframe th { background-color: #f2f2f2; border-bottom: 2px solid #ddd; font-weight: bold; }
    .gradio-dataframe table.dataframe td { border-bottom: 1px solid #eee; }
    .gradio-dataframe table.dataframe tr:nth-child(even) { background-color: #f9f9f9; }
    .gradio-dataframe table.dataframe tr:hover { background-color: #e9f7ef; }
    .gradio-plot { min-height: 400px; }
    /* Add CSS to make text unselectable *except* for the dataframes */
    body, .gradio-container, .header, .filter-container, .tabitem > p, .tabitem > div:not(.gradio-dataframe) {
         user-select: none; /* Disable text selection for non-dataframe elements */
         -webkit-user-select: none; /* Safari */
         -ms-user-select: none; /* IE 10+ */
    }
    .gradio-dataframe, .model-card {
         user-select: text !important; /* Ensure text selection IS enabled for dataframes and model cards */
        -webkit-user-select: text !important;
        -ms-user-select: text !important;
    }
    """ # Added CSS for selectability

    with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue)) as app:
        gr.HTML("""<div class="header"><h1>LLM Leaderboard Explorer</h1><p>Interactive visualization of merged leaderboards data</p></div>""")

        with gr.Row(elem_classes="filter-container"):
            # Filters remain the same
            with gr.Column(scale=2): search_input = gr.Textbox(label="Search Models/Org", placeholder="Search...", show_label=False)
            with gr.Column(scale=1, min_width=160): org_dropdown = gr.Dropdown(choices=get_organization_list(df_global), label="Organization", value="All", show_label=False)
            with gr.Column(scale=2, min_width=200):
                 max_score = 100
                 if 'Global Average' in df_global.columns and not df_global['Global Average'].isna().all(): max_score = max(100, df_global['Global Average'].max(skipna=True))
                 min_score_slider = gr.Slider(minimum=0, maximum=max_score, value=0, label="Min Global Avg", step=1)

        with gr.Tabs() as tabs:
            main_metrics_cols = get_valid_columns(df_global, column_groups_global.get("Main Metrics", []))
            model_details_cols = get_valid_columns(df_global, column_groups_global.get("Model Details", []))
            community_stats_cols = get_valid_columns(df_global, column_groups_global.get("Community Stats", []))
            model_mapping_cols = get_valid_columns(df_global, column_groups_global.get("Model Mapping", []))

            with gr.TabItem("Performance Metrics"):
                # Keep interactive=True for row selection -> model card
                main_metrics_table = gr.DataFrame(
                    interactive=False, wrap=True, elem_classes="gradio-dataframe"
                )
                gr.Markdown("Click a row for details.")
                model_card_output = gr.HTML()

            with gr.TabItem("Model Details"):
                # Set interactive=False - no selection needed, ensures read-only
                model_details_table = gr.DataFrame(
                    interactive=False, wrap=True, elem_classes="gradio-dataframe"
                )

            with gr.TabItem("Community Stats"):
                # Set interactive=False - no selection needed, ensures read-only
                community_stats_table = gr.DataFrame(
                    interactive=False, wrap=True, elem_classes="gradio-dataframe"
                )

            with gr.TabItem("Model Mapping"):
                 # Set interactive=False - no selection needed, ensures read-only
                 model_mapping_table = gr.DataFrame(
                     interactive=False, wrap=True, elem_classes="gradio-dataframe"
                 )

            with gr.TabItem("Visualizations"):
                # Visualizations setup remains the same
                gr.Markdown("### Compare Top Models")
                with gr.Row():
                    perf_metrics = ["Global Average", "Reasoning Average", "Coding Average", "Mathematics Average", "Data Analysis Average", "Language Average", "IF Average"]
                    arena_metrics = ["Arena Score", "Arena Rank (No Style Control)", "Arena Rank (With Style Control)"]
                    available_metrics = perf_metrics + [m for m in arena_metrics if m in df_global.columns]
                    valid_bar_metrics = get_valid_columns(df_global, available_metrics)
                    metric_dropdown = gr.Dropdown(choices=valid_bar_metrics, label="Select Metric for Bar Chart", value=valid_bar_metrics[0] if valid_bar_metrics else None)
                bar_chart = gr.Plot(label="Top 15 Model Comparison", elem_classes="gradio-plot")
                gr.Markdown("### Radar Chart Comparison")
                with gr.Row():
                     model_choices = sorted(df_global[model_identifier_col].dropna().unique().tolist()) if model_identifier_col in df_global.columns else []
                     default_radar_models = []
                     if 'Global Average' in df_global.columns and not df_global.empty: default_radar_models = df_global.nlargest(3, 'Global Average')[model_identifier_col].tolist()
                     models_multiselect = gr.Dropdown(choices=model_choices, label="Select Models for Radar Chart (up to 5)", multiselect=True, max_choices=5, value=default_radar_models)
                radar_chart = gr.Plot(label="Radar Comparison", elem_classes="gradio-plot")

        # --- Event Handling ---
        # [ Event handling functions (update_on_filter, show_model_details_card, etc.) remain the same ]
        def update_on_filter(search, min_global, organization):
            filtered_df = filter_data(df_global, search, min_global, organization)
            main_metrics_data = filtered_df[main_metrics_cols].fillna("N/A") if not filtered_df.empty else pd.DataFrame(columns=main_metrics_cols)
            model_details_data = filtered_df[model_details_cols].fillna("N/A") if not filtered_df.empty else pd.DataFrame(columns=model_details_cols)
            community_stats_data = filtered_df[community_stats_cols].fillna("N/A") if not filtered_df.empty else pd.DataFrame(columns=community_stats_cols)
            model_mapping_data = filtered_df[model_mapping_cols].fillna("N/A") if not filtered_df.empty else pd.DataFrame(columns=model_mapping_cols)
            models_list = sorted(filtered_df[model_identifier_col].dropna().unique().tolist()) if not filtered_df.empty and model_identifier_col in filtered_df.columns else []
            current_bar_metric = metric_dropdown.value if hasattr(metric_dropdown, 'value') and metric_dropdown.value in valid_bar_metrics else (valid_bar_metrics[0] if valid_bar_metrics else None)
            bar_plot_update = create_comparison_chart(filtered_df, current_bar_metric)
            model_card_clear = gr.update(value="")
            current_radar_models = models_multiselect.value if hasattr(models_multiselect, 'value') else []
            valid_radar_selection = [m for m in current_radar_models if m in models_list][:5]
            return (gr.update(value=main_metrics_data), gr.update(value=model_details_data), gr.update(value=community_stats_data), gr.update(value=model_mapping_data), gr.update(choices=models_list, value=valid_radar_selection), bar_plot_update, model_card_clear)

        def show_model_details_card(evt: gr.SelectData, current_main_table_data: pd.DataFrame):
            if evt is None or not hasattr(evt, 'index') or not isinstance(evt.index, (list, tuple)) or not evt.index: return gr.update(value="")
            row_idx = evt.index[0]
            if current_main_table_data is None or current_main_table_data.empty or row_idx >= len(current_main_table_data): return gr.update(value="")
            try:
                selected_model_identifier = current_main_table_data.iloc[row_idx][model_identifier_col]
                full_model_data_row = df_global[df_global[model_identifier_col] == selected_model_identifier]
                if full_model_data_row.empty: return gr.update(value=f"<p>Error: Could not find details for {selected_model_identifier}</p>")
                full_data_dict = full_model_data_row.iloc[0].to_dict()
                card_html = create_model_card(full_data_dict)
                return gr.update(value=card_html)
            except KeyError: return gr.update(value=f"<p>Error: Cannot find model identifier column ('{model_identifier_col}') in table.</p>")
            except Exception as e: print(f"Error generating model card: {e}"); return gr.update(value=f"<p>Error generating card: {e}</p>")

        def update_bar_chart_on_metric_change(metric, search, min_global, organization):
             filtered_df = filter_data(df_global, search, min_global, organization)
             chart = create_comparison_chart(filtered_df, metric)
             return chart

        def update_radar_chart_on_selection(selected_models):
             if not selected_models: return go.Figure().update_layout(title="Select models to compare", polar=dict(radialaxis=dict(visible=False)))
             chart = create_radar_chart(df_global, selected_models, radar_metrics)
             return chart


        # --- Connect Components ---
        # [ Connections remain the same ]
        filter_inputs = [search_input, min_score_slider, org_dropdown]
        filter_outputs = [main_metrics_table, model_details_table, community_stats_table, model_mapping_table, models_multiselect, bar_chart, model_card_output]
        search_input.submit(update_on_filter, inputs=filter_inputs, outputs=filter_outputs)
        min_score_slider.release(update_on_filter, inputs=filter_inputs, outputs=filter_outputs)
        org_dropdown.change(update_on_filter, inputs=filter_inputs, outputs=filter_outputs)
        main_metrics_table.select(show_model_details_card, inputs=[main_metrics_table], outputs=[model_card_output])
        metric_dropdown.change(update_bar_chart_on_metric_change, inputs=[metric_dropdown] + filter_inputs, outputs=bar_chart)
        models_multiselect.change(update_radar_chart_on_selection, inputs=[models_multiselect], outputs=radar_chart)


        # --- Initial Load ---
        # [ Initial load function remains the same ]
        def initial_load():
            initial_main_metrics = df_global[main_metrics_cols].fillna("N/A")
            initial_model_details = df_global[model_details_cols].fillna("N/A")
            initial_community_stats = df_global[community_stats_cols].fillna("N/A")
            initial_model_mapping = df_global[model_mapping_cols].fillna("N/A")
            initial_bar_metric = valid_bar_metrics[0] if valid_bar_metrics else None
            bar_plot = create_comparison_chart(df_global, initial_bar_metric)
            initial_radar_models = []
            if 'Global Average' in df_global.columns and not df_global.empty: initial_radar_models = df_global.nlargest(3, 'Global Average')[model_identifier_col].tolist()
            radar_plot = create_radar_chart(df_global, initial_radar_models, radar_metrics)
            return (initial_main_metrics, initial_model_details, initial_community_stats, initial_model_mapping, bar_plot, gr.update(value=initial_radar_models), radar_plot)

        # [ Load outputs remain the same ]
        load_outputs = [ main_metrics_table, model_details_table, community_stats_table, model_mapping_table, bar_chart, models_multiselect, radar_chart ]
        app.load(initial_load, inputs=[], outputs=load_outputs)

    return app

# Optional: If running this file directly for testing
# if __name__ == "__main__":
#     if df_global is not None and not df_global.empty and 'Error' not in df_global.columns:
#          ui = create_leaderboard_ui()
#          ui.launch(debug=True)
#     else:
#          print("Failed to load data or data is empty. Cannot launch UI.")