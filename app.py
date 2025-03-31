import gradio as gr
import json
import pandas as pd
from src.control import create_dataset_checkbox_callback, create_callback
from src.leaderboard import get_df
from ui.assets import custom_css
from src.lb_utils import extract_model_info

# LLM_COLUMN_TO_DATATYPE ={
#     # llm
#     "name": "markdown",
#     "urls": "str",
#     "num_param": "number",
#     "context_window": "number",
#     "backbone": "str",
#     "license": "str",
#     "easy_case": "str",
#     "medium_case": "str",
#     "hard_case": "str",
#     "paper_link": "str",
#     "pretrained_datasets": "str",
#     "languages": "str",
#     "post_train_techniques": "str",
#     "post_train_datasets": "str",
#     "release_date": "date",
#     "arena_rank": "number",
#     "arena_elo": "number",
#     "arena_votes": "number",
#     "pretraining_cost": "number",
#     "post_training_cost": "number"
# }

Merged_LLM_COLUMN_TO_DATATYPE ={"Model_name": "markdown"}

DATASET_COLUMN_TO_DATATYPE ={
    # dataset
    "name": "markdown",
    "urls": "str",
    "license": "str",
    "token_size": "number",
    "storage_size": "number",
}

DATASET_PRIMARY_COLUMNS = [
    # dataset
    "name",
    "urls",
    "license",
    "token_size",
    "storage_size",
]

LLM_BASIC_INFO = ['Model_name', 'Average_open_llm_score ⬆️', '#Params (B)', 'Hub ❤️', 'Model_link', 'Model_experiment_details', 'Model sha']
LLM_BENCHMARK_DSET = ['ARC', 'HellaSwag', 'MMLU', 'TruthfulQA', 'Winogrande', 'GSM8K']
LLM_BINARY_COLUMNS = ['Not Available on HF', 'Merged', 'MoE', 'Flagged', 'No Efficiency Data', 'No Safety Data', 'No Arena Data']
LLM_CATEGORY = ['Type', 'Architecture', 'Precision', 'Hub License', 'Weight type']
LLM_SAFETY = ['Average_llm_safety_score ⬆️', 'Non-toxicity', 'Non-Stereotype', 'AdvGLUE++', 'OoD', 'Adv Demo', 'Privacy', 'Ethics', 'Fairness']
LLM_PERF = ['Experiment 🧪', 'Prefill (s)', 'Decode (tokens/s)', 'Memory (MB)', 'Energy (tokens/kWh)']
LLM_ARENA = ['chatbot_arena_rank ⬆️', 'Arena Elo', 'chatbot_arena_votes', 'Knowledge Cutoff']

LLM_PRIMARY_COLUMNS = LLM_BASIC_INFO[:4]
LLM_PRIMARY_COLUMNS.extend(LLM_BENCHMARK_DSET)

llm_df = get_df("merged_llm")
dset_df = get_df("dataset")
llm_headers = llm_df.columns.tolist()

LLM_TYPE = llm_df['Type'].unique().tolist()
LLM_ARCHITECTURE = llm_df['Architecture'].unique().tolist()
LLM_PRECISION = llm_df['Precision'].unique().tolist()
LLM_LICENSE = llm_df['Hub License'].unique().tolist()
LLM_WEIGHT_TYPE = llm_df['Weight type'].unique().tolist()

llm_df = llm_df[LLM_PRIMARY_COLUMNS]
# Sort by 'Name' and then descending 'Score'
df_sorted = llm_df.sort_values(by=['Model_name', 'Average_open_llm_score ⬆️'], ascending=[False, False])
llm_df = df_sorted.drop_duplicates(subset='Model_name', keep='first')
llm_df = llm_df.sort_values(by=['Hub ❤️'], ascending=[False])
def create_leaderboard_table(df, type: str):
    if type == "llm":
        # create dataframe
        leaderboard_table = gr.DataFrame(
            value=df,
            datatype=list(Merged_LLM_COLUMN_TO_DATATYPE.values()),
            headers=llm_headers,
            elem_id="llm-leaderboard", # for future usage
            # wrap=True,
            # column_widths=column_widths,
            interactive=False,
            )
        return leaderboard_table
    elif type == "dataset":
        # create checkboxes
        with gr.Row():
            columns_checkboxes = gr.CheckboxGroup(
                label="Select the columns to display",
                value=DATASET_PRIMARY_COLUMNS, # PRIMARY_COLUMNS
                choices=list(DATASET_COLUMN_TO_DATATYPE.keys()),
                # info="☑️ Select the columns to display",
                elem_id="dset-columns-checkboxes",
            )
        leaderboard_table = gr.DataFrame(
            value=df[DATASET_PRIMARY_COLUMNS],
            datatype=list(DATASET_COLUMN_TO_DATATYPE.values()),
            headers=list(DATASET_COLUMN_TO_DATATYPE.keys()),
            elem_id="dset-leaderboard" # for future usage
            )
        return columns_checkboxes, leaderboard_table

def export_csv(df):
    # Apply the function to the 'Model' column
    df['model_info'] = df['Model_name'].apply(extract_model_info)

    # Access the extracted data (assuming 'model_info' is the new column name)
    df['Model_link'] = df['model_info'].apply(lambda x: x.get('Model_link', ''))
    df['Model_name'] = df['model_info'].apply(lambda x: x.get('Model_name', ''))
    df = df.drop('model_info', axis=1)
    name = "leaderboard_output.csv"
    df.to_csv(name)
    return gr.File(value=name, visible=True)

demo = gr.Blocks(css=custom_css)
with demo:
    gr.Label("GTLLMZoo")

    with gr.Tabs(elem_classes="tabs"):
        with gr.TabItem("LLMs", id=0):
            # create search bar
            with gr.Row():
                search_bar = gr.Textbox(
                    label="Search",
                    info="🔍 Search for a model name",
                    elem_id="search-bar",
                )

            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Control Panel 🎛️", open=False, elem_id="control-panel"):
                        with gr.Row():
                            type_checkboxes = gr.CheckboxGroup(
                                label="Model Types",
                                value=LLM_TYPE, # PRIMARY_COLUMNS
                                choices=LLM_TYPE,
                                # info="☑️ Select the columns to display",
                                elem_id="llm-columns-checkboxes",
                            )

                        with gr.Row():
                            precision_checkboxes = gr.CheckboxGroup(
                                label="Precision",
                                value=LLM_PRECISION, # PRIMARY_COLUMNS
                                choices=LLM_PRECISION,
                                # info="☑️ Select the columns to display",
                                elem_id="llm-columns-checkboxes",
                            )

                        with gr.Row():
                            weight_checkboxes = gr.CheckboxGroup(
                                label="Weight type",
                                value=LLM_WEIGHT_TYPE, # PRIMARY_COLUMNS
                                choices=LLM_WEIGHT_TYPE,
                                # info="☑️ Select the columns to display",
                                elem_id="llm-columns-checkboxes",
                            )

                        with gr.Accordion("Model Architecure", open=False, elem_id="control-panel"):
                            architecture_checkboxes = gr.CheckboxGroup(
                                label="",
                                value=LLM_ARCHITECTURE, # PRIMARY_COLUMNS
                                choices=LLM_ARCHITECTURE,
                                # info="☑️ Select the columns to display",
                                elem_id="llm-columns-checkboxes",
                            )

                        with gr.Accordion("License", open=False, elem_id="control-panel"):
                            license_checkboxes = gr.CheckboxGroup(
                                label="",
                                value=LLM_LICENSE, # PRIMARY_COLUMNS
                                choices=LLM_LICENSE,
                                # info="☑️ Select the columns to display",
                                elem_id="llm-columns-checkboxes",
                            )

                        # create hide columns checkboxes
                        with gr.Accordion("Hide Models", open=False, elem_id="control-panel"):
                            hide_checkboxes = gr.CheckboxGroup(
                                label="",
                                value=[], # PRIMARY_COLUMNS
                                choices=LLM_BINARY_COLUMNS,
                                # info="☑️ Select the columns to display",
                                elem_id="llm-columns-checkboxes",
                            )
                        with gr.Row():
                            filter_button = gr.Button(
                                value="Filter 🚀",
                                elem_id="filter-button",
                                elem_classes="boxed-option",
                            )

                with gr.Column():
                    with gr.Accordion("Columns 📊", open=False, elem_id="column-selector"):
                        # create Basic Information checkboxes
                        with gr.Row():
                            basic_info_checkboxes = gr.CheckboxGroup(
                                label="Basic Information",
                                value=LLM_BASIC_INFO[:4], # PRIMARY_COLUMNS
                                choices=LLM_BASIC_INFO,
                                # info="☑️ Select the columns to display",
                                elem_id="llm-columns-checkboxes",
                            )

                        # create dataset columns checkboxes
                        with gr.Row():
                            benchmark_checkboxes = gr.CheckboxGroup(
                                label="Benchmark datasets",
                                value=LLM_BENCHMARK_DSET, # PRIMARY_COLUMNS
                                choices=LLM_BENCHMARK_DSET,
                                # info="☑️ Select the columns to display",
                                elem_id="llm-columns-checkboxes",
                            )

                        # create efficiency columns checkboxes
                        with gr.Row():
                            efficiency_checkboxes = gr.CheckboxGroup(
                                label="Efficiency Metrics",
                                value=[], # PRIMARY_COLUMNS
                                choices=LLM_PERF,
                                # info="☑️ Select the columns to display",
                                elem_id="llm-columns-checkboxes",
                            )

                        # create safety columns checkboxes
                        with gr.Row():
                            safety_checkboxes = gr.CheckboxGroup(
                                label="Safety Metrics",
                                value=[], # PRIMARY_COLUMNS
                                choices=LLM_SAFETY,
                                # info="☑️ Select the columns to display",
                                elem_id="llm-columns-checkboxes",
                            )

                        # create chatbot arena columns checkboxes
                        with gr.Row():
                            arena_checkboxes = gr.CheckboxGroup(
                                label="Chatbot Arena Metrics",
                                value=[], # PRIMARY_COLUMNS
                                choices=LLM_ARENA,
                                # info="☑️ Select the columns to display",
                                elem_id="llm-columns-checkboxes",
                            )

            leaderboard_table = create_leaderboard_table(llm_df, "llm")
            with gr.Row():
                export_button = gr.Button("Export Table Above")                    
                csv = gr.File(interactive=False, visible=False)
            # callback
            create_callback(
                search_bar,     
                basic_info_checkboxes,
                benchmark_checkboxes,
                efficiency_checkboxes,
                safety_checkboxes,
                arena_checkboxes,

                filter_button,
                type_checkboxes,
                precision_checkboxes,
                weight_checkboxes,
                architecture_checkboxes,
                license_checkboxes,
                hide_checkboxes,

                leaderboard_table,
            )

            # export button, now export full leaderboard
            export_button.click(export_csv, leaderboard_table, csv)

        with gr.TabItem("Datasets", id=1):

            columns_checkboxes, leaderboard_table = create_leaderboard_table(dset_df, "dataset")

            # control callback
            create_dataset_checkbox_callback(
                # input
                columns_checkboxes,
                # outputs
                leaderboard_table,
            )
        
demo.launch(share=True)