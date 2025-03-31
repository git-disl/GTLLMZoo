# GTLLMZoo 🦙

A comprehensive framework for aggregating, comparing, and evaluating Large Language Models (LLMs) through benchmark performance data from multiple sources.

![GTLLMZoo Banner](https://placehold.co/600x200/EEE/31343C?text=GTLLMZoo)

## 📋 Overview

GTLLMZoo provides a unified platform for comparing LLMs across multiple dimensions including performance, efficiency, and safety. The framework aggregates data from various benchmark sources to enable researchers, developers, and decision-makers to make informed selections based on their specific requirements.

Key features:
- **Unified Benchmarks**: Combines data from Open LLM Leaderboard, LLM Safety Leaderboard, LLM Performance Leaderboard, and Chatbot Arena
- **Interactive UI**: Intuitive filtering and selection interface built with Gradio
- **Comprehensive Metrics**: Compare models across performance, safety, efficiency, and user preference metrics
- **Customizable Views**: Select specific metrics and model attributes for focused comparison

## 🚀 Getting Started

### Prerequisites

- Python >= 3.9
- gradio==4.9.0
- Pandas
- Beautiful Soup (for data scraping)

### Installation

```bash
git clone https://github.com/git-disl/GTLLMZoo.git
cd GTLLMZoo
pip install -r requirements.txt
```

### Running the Application

To run the application locally:

```bash
python app.py
```

For development with hot reloading:

```bash
gradio app.py
```

## 🔍 Features

### LLM Comparison Tab

Compare LLMs based on:
- **Basic Information**: Model name, parameter count, hub popularity
- **Benchmark Performance**: Scores on ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K
- **Efficiency Metrics**: Prefill time, decode speed, memory usage, energy efficiency
- **Safety Metrics**: Non-toxicity, non-stereotype, fairness, ethics
- **Arena Performance**: Chatbot arena ranking, ELO scores, user votes

### Control Panel

Filter models by:
- Model type
- Architecture
- Precision
- License
- Weight type

### Data Export

Export filtered data to CSV for further analysis.

## 📊 Data Sources

GTLLMZoo aggregates data from:
- [Open LLM Leaderboard](https://huggingfaceh4-open-llm-leaderboard.hf.space/)
- [LLM Safety Leaderboard](https://ai-secure-llm-trustworthy-leaderboard.hf.space/)
- [LLM Performance Leaderboard](https://optimum-llm-perf-leaderboard.hf.space/)
- [Chatbot Arena Leaderboard](https://lmsys-chatbot-arena-leaderboard.hf.space/)

## 🏗️ Project Structure

- `app.py`: Main Gradio UI application
- `leaderboard.py`: Functions to load and process leaderboard data
- `control.py`: UI control callbacks and filtering functions
- `data_structures.py`: Data structure definitions for LLMs and datasets
- `utils.py`: Utility functions and enum classes
- `scrape_llm_lb.py`: Scripts to scrape latest leaderboard data
- `merge.py`: Functions to merge data from different sources
- `assets.py`: Custom CSS and UI assets

## 💾 Data Files

- `llm.json`: LLM metadata
- `dset.json`: Dataset information
- `merged.csv`: Merged data from all leaderboards

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📬 Contact

Project Link: [https://github.com/git-disl/GTLLMZoo](https://github.com/git-disl/GTLLMZoo)

## 🙏 Acknowledgements

- HuggingFace for hosting the original leaderboards
- All benchmark creators and maintainers
- The open-source LLM community
