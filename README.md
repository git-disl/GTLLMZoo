# LLM Leaderboard Explorer

An interactive dashboard for exploring and visualizing merged data from LLM leaderboards, built with Gradio. **Check it out our deployed HuggingFace Space: [Link](https://huggingface.co/spaces/RaccoonOnion/gt-llm-zoo)**
YouTube Video about GTLLMZoo available here: [Link](https://www.youtube.com/watch?v=bCO5Sl74g1Y)

## 📊 Overview

This application provides an interactive interface to view, filter, and compare Large Language Models (LLMs) based on aggregated data from prominent leaderboard sources:

* **LiveBench:** Features performance metrics like Global Average, Reasoning, Coding, Mathematics, Data Analysis, Language, and Instruction Following scores.
* **LMSYS Chatbot Arena:** Includes community-based Elo ratings (Arena Score), rankings, and voting data.

The dashboard allows users to easily navigate and compare models across various metrics and categories.

## ✨ Features

* **Interactive Data Tables:** View LLM data organized into tabs:
    * **Performance Metrics:** Core benchmark scores from LiveBench.
    * **Model Details:** Information like Organization, License, Knowledge Cutoff, and links.
    * **Community Stats:** Data from the Chatbot Arena Leaderboard (Ranks, Score, Votes).
    * **Model Mapping:** Shows the unified model name alongside original names from LiveBench and Arena.
* **Filtering:** Dynamically filter the displayed models by:
    * Search term (searches Model Name and Organization).
    * Organization.
    * Minimum Global Average score.
* **Detailed Model Card:** Click on any row in the data tables to view a comprehensive card summarizing all metrics for that specific model.
* **Visualizations Tab:**
    * **Bar Chart:** Compare the top 15 models based on a user-selected metric (e.g., Global Average, Arena Score, Coding Average).
    * **Radar Chart:** Select multiple models (up to 5) to compare their performance profile across key metrics (Reasoning, Coding, Math, Data Analysis, Language, IF Average, and scaled Arena Score).

## 💾 Data

The application uses a pre-merged CSV file (`data/merged_leaderboards.csv`) containing data aggregated from the sources mentioned above.

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* pip (Python package installer)

### Installation

1.  **Clone the repository (Optional):**
    ```bash
    # If you have the code in a git repository
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
    If you just have the files, navigate to the project directory in your terminal.

2.  **Install Dependencies:**
    Create a `requirements.txt` file with the following content:
    ```text
    gradio==4.9.0
    pandas
    plotly
    numpy
    ```
    Then, install the requirements:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

To run the application locally:

```bash
python app.py
```

The application will typically be available at http://127.0.0.1:7860 in your web browser.

## 📁 Project Structure

```
GTLLMZoo2
├─ app.py                  # Main Gradio application entry point
├─ requirements.txt        # Python dependencies
├─ data
│  └─ merged_leaderboards.csv # Merged leaderboard data
└─ src
   ├─ data_processing.py  # Data loading and filtering logic
   └─ ui.py               # Gradio UI definition and logic

```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request if you have improvements or bug fixes.

## 📄 License

MIT License
