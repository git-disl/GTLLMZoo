# app.py
import gradio as gr
from ui import create_leaderboard_ui
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    logging.info("Starting leaderboard application...")
    leaderboard_app = create_leaderboard_ui()

    logging.info("Launching Gradio interface...")
    # You can add server_name="0.0.0.0" to allow access from other devices on your network
    leaderboard_app.launch()
    logging.info("Gradio interface stopped.")