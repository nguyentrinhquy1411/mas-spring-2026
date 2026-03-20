import os
import sys

# Add the src directory to the python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from house_price_prediction.webapp.main import app
