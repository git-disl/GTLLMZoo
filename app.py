import gradio as gr
import json
import pandas as pd
from control import create_dataset_checkbox_callback, create_llm_checkbox_callback
from leaderboard import get_df
LLM_COLUMN_TO_DATATYPE ={
    # llm
    "name": "markdown",
    "urls": "str",
    "num_param": "number",
    "context_window": "number",
    "backbone": "str",
    "license": "str",
    "easy_case": "str",
    "medium_case": "str",
    "hard_case": "str",
    "paper_link": "str",
    "pretrained_datasets": "str",
    "languages": "str",
    "post_train_techniques": "str",
    "post_train_datasets": "str",
    "release_date": "date",
    "arena_rank": "number",
    "arena_elo": "number",
    "arena_votes": "number",
    "pretraining_cost": "number",
    "post_training_cost": "number"
}

DATASET_COLUMN_TO_DATATYPE ={
    # dataset
    "name": "markdown",
    "urls": "str",
    "license": "str",
    "token_size": "number",
    "storage_size": "number",
}

LLM_PRIMARY_COLUMNS = [
    "name",
    "num_param",
    "context_window",
    "backbone",
    "license",
]

DATASET_PRIMARY_COLUMNS = [
    # dataset
    "name",
    "urls",
    "license",
    "token_size",
    "storage_size",
]

# def model_hyperlink(link, model_name):
#     return f'<a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}</a>'

# def process_model(model_name):
#     link = f"https://huggingface.co/{model_name}"
#     return model_hyperlink(link, model_name)

# def get_df(type: str):
#     if type == "llm":
#         # read json files
#         with open('llm.json', 'r') as json_file:
#             llm_data = json.load(json_file)
#         # if data is not a list, put it into a list (temporary measure)
#         if not isinstance(llm_data, list):
#             llm_data = [llm_data]
#         # transform into DataFrame for better visualization
#         llm_df = pd.DataFrame(llm_data)
#         # transform name+link to markdown
#         llm_df["name"] = llm_df["name"].apply(process_model)
#     elif type == "dataset":
#         with open('dset.json', 'r') as json_file:
#             dset_data = json.load(json_file)
#         if not isinstance(dset_data, list):
#             dset_data = [dset_data]
#         dset_df = pd.DataFrame(dset_data)
#         dset_df["name"] = dset_df["name"].apply(process_model)

# # read json files
# with open('llm.json', 'r') as json_file:
#     llm_data = json.load(json_file)
# with open('dset.json', 'r') as json_file:
#     dset_data = json.load(json_file)

# # if data is not a list, put it into a list (temporary measure)
# if not isinstance(llm_data, list):
#     llm_data = [llm_data]
# if not isinstance(dset_data, list):
#     dset_data = [dset_data]

# # transform into DataFrame for better visualization
# llm_df = pd.DataFrame(llm_data)
# dset_df = pd.DataFrame(dset_data)
# llm_df["name"] = llm_df["name"].apply(process_model)
# dset_df["name"] = dset_df["name"].apply(process_model)
llm_df = get_df("llm")
dset_df = get_df("dataset")

def create_leaderboard_table(df, type: str):
    if type == "llm":
        # create checkboxes
        with gr.Row():
            columns_checkboxes = gr.CheckboxGroup(
                label="Columns 📊",
                value=LLM_PRIMARY_COLUMNS, # PRIMARY_COLUMNS
                choices=list(LLM_COLUMN_TO_DATATYPE.keys()),
                info="☑️ Select the columns to display",
                elem_id="llm-columns-checkboxes",
            )
        # create dataframe
        leaderboard_table = gr.DataFrame(
            value=df[LLM_PRIMARY_COLUMNS],
            datatype=list(LLM_COLUMN_TO_DATATYPE.values()),
            headers=list(LLM_COLUMN_TO_DATATYPE.keys()),
            elem_id="llm-leaderboard" # for future usage
            )
        return columns_checkboxes, leaderboard_table
    elif type == "dataset":
        # create checkboxes
        with gr.Row():
            columns_checkboxes = gr.CheckboxGroup(
                label="Columns 📊",
                value=DATASET_PRIMARY_COLUMNS, # PRIMARY_COLUMNS
                choices=list(DATASET_COLUMN_TO_DATATYPE.keys()),
                info="☑️ Select the columns to display",
                elem_id="dset-columns-checkboxes",
            )
        leaderboard_table = gr.DataFrame(
            value=df[DATASET_PRIMARY_COLUMNS],
            datatype=list(DATASET_COLUMN_TO_DATATYPE.values()),
            headers=list(DATASET_COLUMN_TO_DATATYPE.keys()),
            elem_id="dset-leaderboard" # for future usage
            )
        return columns_checkboxes, leaderboard_table
demo = gr.Blocks()
with demo:
    gr.Label("GTLLMZoo")

    with gr.Tabs(elem_classes="tabs"):
        with gr.TabItem("LLMs", id=0):
            # # create checkboxes
            # with gr.Row():
            #     columns_checkboxes = gr.CheckboxGroup(
            #         label="Columns 📊",
            #         value=LLM_PRIMARY_COLUMNS, # PRIMARY_COLUMNS
            #         choices=list(LLM_COLUMN_TO_DATATYPE.keys()),
            #         info="☑️ Select the columns to display",
            #         elem_id="llm-columns-checkboxes",
            #     )
            # # create dataframe
            # gr.DataFrame(
            #     value=llm_df,
            #     datatype=list(LLM_COLUMN_TO_DATATYPE.values()),
            #     headers=list(LLM_COLUMN_TO_DATATYPE.keys()),
            #     elem_id="llm-leaderboard" # for future usage
            #     ),

            columns_checkboxes, leaderboard_table = create_leaderboard_table(llm_df, "llm")

            # control callback
            create_llm_checkbox_callback(
                # input
                columns_checkboxes,
                # outputs
                leaderboard_table,
            )

        with gr.TabItem("Datasets", id=1):

            # # create checkboxes
            # with gr.Row():
            #     columns_checkboxes = gr.CheckboxGroup(
            #         label="Columns 📊",
            #         value=DATASET_PRIMARY_COLUMNS, # PRIMARY_COLUMNS
            #         choices=list(DATASET_COLUMN_TO_DATATYPE.keys()),
            #         info="☑️ Select the columns to display",
            #         elem_id="dset-columns-checkboxes",
            #     )

            # gr.DataFrame(
            #     value=dset_df,
            #     datatype=list(DATASET_COLUMN_TO_DATATYPE.values()),
            #     headers=list(DATASET_COLUMN_TO_DATATYPE.keys()),
            #     elem_id="dset-leaderboard" # for future usage
            #     ),

            columns_checkboxes, leaderboard_table = create_leaderboard_table(dset_df, "dataset")

            # control callback
            create_dataset_checkbox_callback(
                # input
                columns_checkboxes,
                # outputs
                leaderboard_table,
            )
        
demo.launch()