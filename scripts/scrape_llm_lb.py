import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
from gradio_client import Client

def get_json_format_data(url:str=""):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser') # response.content is in raw bytes
    script_elements = soup.find_all('script')
    # [:31], [-10:] are <script>window.gradio_config = and ;</script>, using [31:-10] to seperate json data
    json_format_data = json.loads(str(script_elements[1])[31:-10])
    # print(str(script_elements[1])[31:-10])
    # print(str(script_elements[1])[:31])
    # print("-"*100)
    # print(str(script_elements[1])[-10:])
    # exit()
    return json_format_data


def get_datas(data):
    data = data['components'] # list
    result_list = []
    headers_len = 0
    component_index = 0 # 0-indexed list index
    for i in range(len(data)):
        if not data[i]['type'] in {'dataframe', 'leaderboard'}:
            continue
        headers = data[i]['props']['value']['headers'] # list of strings
        if len(headers) > headers_len:
            headers_len = len(headers)
            component_index = i
    # print("element: ", component_index)
    # exit()
    # we found the dataframe with the max #columns, i.e. the most data

    headers = data[component_index]['props']['value']['headers']
    # print(headers)
    # exit()
    data = data[component_index]['props']['value']['data']
    assert len(headers) == len(data[0])
    for i in range(len(data)):
        results = data[i]
        results_json = {}
        for idx, header in enumerate(headers):
            if header == 'Model' or header == 'Model 🤗':
                name_regex = r">(.*?)<" # searches for text between > and < characters
                match = re.search(name_regex, results[idx])
                # Extract the name if found, otherwise return an empty string
                if match:
                    results_json['Model_name_for_merge'] = match.group(1)
                html_content = results[idx]
                soup = BeautifulSoup(html_content, 'html.parser')
                links = soup.find_all('a')
                if len(links) >= 2: # for open llm lb
                    model_link = links[0].get('href')
                    model_experiment_details = links[-1].get('href')
                elif len(links) == 1:
                    model_link = links[0].get('href') # for llm safety lb and llm perf lb
                    model_experiment_details = ''
                else:
                    model_link = ''
                    model_experiment_details = ''
                if model_link: results_json['Model_link'] = model_link
                if model_experiment_details: results_json['Model_experiment_details'] = model_experiment_details
            else:
                results_json[header] = results[idx]
        result_list.append(results_json)
    
    return result_list

def extract_model_info(model_html):
  """
  Extracts model_link and model_experiment_details from HTML content in a 'Model' column.

  Args:
      model_html (str): HTML content from the 'Model' column.

  Returns:
      dict: A dictionary containing 'model_link' and 'model_experiment_details' if found, otherwise empty strings.
  """
  results_json = {}
  if not pd.isna(model_html):  # Check if value is not missing
    soup = BeautifulSoup(model_html, 'html.parser')
    links = soup.find_all('a')
    if len(links) == 2:
      results_json['model_link'] = links[0].get('href')
      results_json['model_experiment_details'] = links[1].get('href')
  return results_json

def scrape(name:str="open_llm_lb"):
    if name == "open_llm_lb":
        url = 'https://huggingfaceh4-open-llm-leaderboard.hf.space/' # for open llm lb
    elif name == "llm_perf_lb":
        url = 'https://optimum-llm-perf-leaderboard.hf.space/' # for llm perf lb
    elif name == "llm_safety_lb":
        url = "https://ai-secure-llm-trustworthy-leaderboard.hf.space/" # for llm safety lb
    elif name == "chatbot_arena": # chatbot arena, we can use notebook and raw comparison data to get more direct ratings
        url = "https://lmsys-chatbot-arena-leaderboard.hf.space/" 
    else:
        url = ""
        print("No matched leaderboard!")
        exit()

    data = get_json_format_data(url)
    finished_models = get_datas(data)
    df = pd.DataFrame(finished_models)

    if name == "open_llm_lb":
        df = df.drop('fullname', axis=1)

        # backup plan: Felixz's lb mirror
        # client = Client("https://felixz-open-llm-leaderboard.hf.space/")
        # dict_data = client.predict("","", api_name='/predict')
        # # print(dict_data.keys())
        # df = pd.DataFrame(dict_data['data'], columns=dict_data['headers'])

        # # Apply the function to the 'Model' column
        # df['model_info'] = df['Model'].apply(extract_model_info)

        # # Access the extracted data (assuming 'model_info' is the new column name)
        # df['Model_link'] = df['model_info'].apply(lambda x: x.get('model_link', ''))
        # df['Model_experiment_details'] = df['model_info'].apply(lambda x: x.get('model_experiment_details', ''))

        # # Drop the 'model_info' column if not needed (optional)
        # df = df.drop('model_info', axis=1)
        # df = df.drop('Model', axis=1)
        # # df.rename(columns={'model_name_for_query': 'Model_name_for_merge'}, inplace=True)
    # else:
    #     data = get_json_format_data(url)
    #     finished_models = get_datas(data)
    #     df = pd.DataFrame(finished_models)

    if name == "chatbot_arena":
        hf_prefix = "https://huggingface.co/"
        df['Model_name_for_merge'] = df.apply(lambda row: row["Model_link"].split(hf_prefix)[-1] if hf_prefix in row["Model_link"] else row["Model_name_for_merge"], axis=1)

        # print(set(df['License']))
        license_map = {"CC-BY-NC-SA-4.0": "cc-by-nc-sa-4.0", "Apache-2.0": "apache-2.0", "Apache 2.0": "apache-2.0", "Llama 2 Community": "llama2", "MIT": "mit", "CC-BY-NC-4.0": "cc-by-nc-4.0", "Gemma license": "gemma", "Llama 3 Community": "llama3"}
        df["Hub License"] = df["License"].map(license_map).fillna(df["License"])
        df = df.drop('License', axis=1)

    if 'model_name_for_query' in df.columns:
        if "Model_name_for_merge" in df.columns:
            df = df.drop('model_name_for_query', axis=1)
        else:
            df = df.rename(columns={'model_name_for_query': 'Model_name_for_merge'})

    df.to_csv(f"{name}.csv", index=False)
    print(f"{name} data exported to CSV")

if __name__ == "__main__":
    scrape("open_llm_lb")
    # scrape("llm_perf_lb")
    # scrape("llm_safety_lb")
    # scrape("chatbot_arena")