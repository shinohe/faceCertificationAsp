import os
from flask import Blueprint
currentDir = os.path.dirname(os.path.abspath(__file__))
app = Blueprint("data", __name__,
    static_url_path='/image', static_folder=currentDir+os.sep+'..'+os.sep+'image'
)