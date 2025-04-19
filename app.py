import os
import sys

# Add project root to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ui import create_leaderboard_ui

if __name__ == "__main__":
    # Create and launch the UI
    app = create_leaderboard_ui()
    app.launch()