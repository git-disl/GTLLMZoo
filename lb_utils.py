from bs4 import BeautifulSoup
import re
import pandas as pd

def extract_model_info(model_html):
  """
  Extracts model_link and model_name from HTML content in a column.

  Args:
      model_html (str): HTML content from the column.

  Returns:
      dict: A dictionary containing 'model_link' and 'model_name'.
  """
  results_json = {}
  if not pd.isna(model_html):  # Check if value is not missing
    soup = BeautifulSoup(model_html, 'html.parser')
    links = soup.find_all('a')
    if len(links) == 1:
        results_json['Model_link'] = links[0].get('href')
        name_regex = r">(.*?)<" # searches for text between > and < characters
        match = re.search(name_regex, model_html)
        # Extract the name if found, otherwise return an empty string
        if match:
            results_json['Model_name'] = match.group(1)
  return results_json