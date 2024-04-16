import gradio as gr
import json
import pandas as pd

# read json files
with open('llm.json', 'r') as json_file:
    llm_data = json.load(json_file)
with open('dset.json', 'r') as json_file:
    dset_data = json.load(json_file)

# if data is not a list, put it into a list (temporary measure)
if not isinstance(llm_data, list):
    llm_data = [llm_data]
if not isinstance(dset_data, list):
    dset_data = [dset_data]

# transform into DataFrame for better visualization
llm_df = pd.DataFrame(llm_data)
dset_df = pd.DataFrame(dset_data)

demo = gr.Blocks()
with demo:
    gr.Label("GTLLMZoo")
    with gr.Tabs(elem_classes="tabs"):
        with gr.TabItem("LLMs", id=0):
            gr.DataFrame(llm_df)
        with gr.TabItem("Datasets", id=1):
            gr.DataFrame(dset_df)
        
demo.launch()