import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from house_price_prediction.webapp.main import app

# Ensure FastAPI object is exposed as 'app'

