import pandas as pd
import json

def model_hyperlink(link, model_name):
    return f'<a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}</a>'

def process_llm(model_name):
    link = f"https://huggingface.co/{model_name}"
    return model_hyperlink(link, model_name)

def process_dset(model_name, url):
    return model_hyperlink(url, model_name)

def get_df(type: str):
    if type == "llm":
        # read json files
        with open('llm.json', 'r') as json_file:
            llm_data = json.load(json_file)
        # if data is not a list, put it into a list (temporary measure)
        if not isinstance(llm_data, list):
            llm_data = [llm_data]
        # transform into DataFrame for better visualization
        llm_df = pd.DataFrame(llm_data)
        # transform name+link to markdown
        llm_df["name"] = llm_df["name"].apply(process_llm)
        return llm_df
    elif type == "dataset":
        with open('data/dset.json', 'r') as json_file:
            dset_data = json.load(json_file)
        if not isinstance(dset_data, list):
            dset_data = [dset_data]
        dset_df = pd.DataFrame(dset_data)
        dset_df['name'] = dset_df.apply(lambda row: model_hyperlink(row['urls'], row['name']), axis=1)
        return dset_df
    elif type == "merged_llm":
        # read csv file
        llm_df = pd.read_csv('data/merged.csv')
        # transform name+link to markdown
        llm_df['Model_name'] = llm_df.apply(
            lambda row: row['Model_name'].apply(process_llm) if pd.isna(row['Model_link']) else model_hyperlink(row['Model_link'], row['Model_name']),
            axis=1
        )
        return llm_df