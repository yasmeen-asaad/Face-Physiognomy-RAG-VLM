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
    session_id : str
    timestamp : str
    email : str
    status : str # success/ invalid_face/error
    latency_sec : float
    face_bbox : str    = ""
    regions_ok : int    = 0 # how many face parts were successfully extracted
    report_preview : str    = "" # report 
    error_msg : str    = "" # if something failed

#______________*______________*______________
# Google drive logger class
#____________________________________________
class DriveLogger:
    LOG_FILENAME = "log.csv"
    CSV_COLUMNS = ["session_id", "timestamp", "email", "status",
                   "latency_sec", "face_bbox", "regions_ok",
                   "report_preview", "error_msg"]
    def __init__(self):
        pass
	    
    def _build_service(self, sa_json_str: str): 
        pass
	    
    def log_session():
        pass
	    
    def _create_folder():
        pass
	    
    def _upload_text():
        pass
	    
    def _append_to_csv():
        pass
