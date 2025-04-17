# ui.py
import gradio as gr
from data_handler import load_leaderboard_data
import logging

def create_leaderboard_ui():
    """
    Creates the Gradio interface for displaying the leaderboard.

    Returns:
        gradio.Blocks: The Gradio UI instance.
    """
    logging.info("Creating Gradio UI...")
    df = load_leaderboard_data()

    with gr.Blocks(theme=gr.themes.Soft(), title="Model Leaderboard") as demo:
        gr.Markdown("# Merged Model Leaderboard")

        if df is not None:
            gr.Markdown("Displaying data from `data/merged_leaderboards.csv`")
            gr.DataFrame(
                value=df,
                wrap=True, # Wrap text in cells
                interactive=False # Make table read-only
            )
            logging.info("DataFrame component created successfully.")
        else:
            gr.Markdown("## Error")
            gr.Markdown(
                "Could not load or process the leaderboard data. "
                "Please check the logs and ensure `data/merged_leaderboards.csv` exists and is valid."
            )
            logging.error("DataFrame component could not be created due to data loading issues.")

    return demo

if __name__ == '__main__':
    # For testing purposes when running this file directly
    print("Testing UI creation (will not launch)...")
    ui_instance = create_leaderboard_ui()
    print("UI instance created (run app.py to launch).")
    # ui_instance.launch() # You could launch here for isolated UI testing