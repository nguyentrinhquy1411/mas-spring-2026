import sys
import os

# Root directory pathing for the python builder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from house_price_prediction.webapp.main import app
