"""
This file for loggging every request to Google Drive
            1. Saves original image + face crop as JPG files
            2. Saves the report as a text file
            3. Appends one row to log.csv with all session metadata
"""
import os
import io
import csv
import json
import uuid
import tempfile
from datetime  import datetime
from dataclasses import dataclass
from typing    import Optional, Dict, Any
import cv2
import numpy as np

#______________*______________*______________
# Session Data Class
#____________________________________________
@dataclass
class SessionLog:
