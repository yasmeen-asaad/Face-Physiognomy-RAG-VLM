"""
This file for loggging every request to my Google Drive
   For each seesion:-
   1. Save Json file with RAG output + final report for debugging saved on my drive in Json folder
   2. Save the input image with session_id on a separately folder on my drive 
   In order to control the drive storage, set MAX_LOGGED_SESSIONS 
   I'm using saving seprate json files for each session for easier debugging & to store nested pipeline output 
   and it will be easier to load all files later for analysis
Driver Structure:
  📁 Physiognomy Logs/          ← GDRIVE_FOLDER_ID
  ├── 📁 logs/
  │   ├── session_abc123.json
  │   ├── session_def456.json
  │   └── ...
  └── 📁 images/
      ├── session_abc123.jpg
      ├── session_def456.jpg
      └── ...
  REQUIRED SECRETS (HuggingFace Space → Settings → Secrets):
  GDRIVE_FOLDER_ID = your Drive folder ID
  GDRIVE_SERVICE_ACCOUNT = { entire service account JSON }
"""
import os
import io
import json
import uuid
from datetime  import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import cv2
import numpy as np

#______________*______________*______________
# Configuration
#____________________________________________
# Only log the first N sessions to Drive.
# After this limit, the app keeps working, logging is just skipped.
MAX_LOGGED_SESSIONS = 150
#______________*______________*______________*______________
# Session Data Class
#__________________________________________________________
@dataclass
class SessionLog:
	"""
	Complete record of one pipeline run, Stored as a single JSON file per session.
	Designed for VLM + RAG pipeline debugging:
      - descriptions    : full VLM output per region
      - retrieval_results: queries, passages, scores per region
      - report_text     : final generated report
      - error_msg       : what went wrong (if anything)
	"""
    session_id: str
    timestamp: str
    status: str # success/ invalid_face/error/ ay moshkela
    latency_sec: float = 0.0
    #__ optional data 
    #image_file_id : Optional[str] = None  # Drive file ID of uploaded image
    descriptions : Optional[Dict] = None  # VLM output per region
    retrieval_results : Optional[Dict] = None  # RAG results per region
    report_text : Optional[str] = None  # final report
    error_msg : Optional[str] = None  # error details if failed
    pipeline_stats : Dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> Dict:
		return asdict(self)

#____________________________________________
# Session ID Generator
#____________________________________________
def new_session_id():
	"""Generate a short unique session ID e.g. 'a3f8b2c1'"""
	return uuid.uuid4().hex[:8]

#______________*______________*______________*______________
# Google Drive Logger Class
#___________________________________________________________
class DriveLogger:
	"""
	Logs pipeline sessions to Google Drive.
	Each session produces:
	   1.logs/session_<id>.json --> full pipeline data for debugging
	   2.images/session_<id>.jpg --> uploaded image for visual review
	"""
#____________________________________________________________    
    def __init__(self):
		"""Initialize Google Drive API client from service account credentials."""
		
		self.folder_id = os.environ.get("GDRIVE_FOLDER_ID")
		sa_json = os.environ.get("GDRIVE_SERVICE_ACCOUNT")
		if not self.folder_id or not sa_json:
			raise ValueError("Google Drive credentials not found.\n")
		self.service = self._build_service(sa_json)
		self._logs_folder_id = self._get_or_create_folder("logs")
		self._images_folder_id = self._get_or_create_folder("images")
		self._session_count = self._count_existing_sessions()
		print(f"DriveLogger ready — {self._session_count} sessions logged so far")
#____________________________________________________________    
    def _build_service(self, sa_json_str): 
		""" Build Google Drive API client from service from service account JSON string. 
		sa_json_str: secret json in string format contains google account info
		"""
		sa_info = json.loads(sa_json_str)
		# needed credentilas to deal only with drive 
		credentials = service_account.Credentials.from_service_account_info(sa_info, scopes=["https://www.googleapis.com/auth/drive"])
		return build("drive", "v3", credentials=credentials) # build clint with that crediential     
#____________________________________________________________    
    def _get_or_create_folder(self, name):
        """ Get folder ID if it exists, create it if not.
		    Return folder ID, due to driver api use forder ids not paths
		"""
		results = self.service.files().list(q = (f"name='{name}' and "
												 f"'{self.folder_id}' in parents and "
												 f"mimeType='application/vnd.google-apps.folder' and "
												 f"trashed=false"),
											fields = "files(id)").execute().get("files", [])
		if results:
			return results[0]["id"]
		# else create it !
		metadata = {"name" : name,"mimeType": "application/vnd.google-apps.folder","parents" : [self.folder_id]}
		folder = self.service.files().create(body=metadata, fields="id").execute()
		return folder["id"]
#____________________________________________________________
    def _count_existing_sessions(self):
		"""Count how many session JSON files already exist in logs/."""
		results = self.service.files().list(q=(f"'{self._logs_folder_id}' in parents and "
										   f"mimeType='application/json' and "
										   f"trashed=false"),
										fields = "files(id)").execute().get("files", [])
		return len(results)
#____________________________________________________________
    def log_session(self, session:SessionLog, original_img : Optional[np.ndarray] = None)-> bool:
		""" Save session JSON + uploaded image to Google Drive.
		    Returns True if logged, False if skipped (limit reached).
		"""
		# check if we reached to the logging limit 
		if self._session_count >= MAX_LOGGED_SESSIONS: 
			print(f"Logging limit reached: {MAX_LOGGED_SESSIONS}")
			return False
		try:
			if original_img is not None:
				# upload the image 
				image_file_id = self._upload_image(img=original_img,filename = f"session_{session.session_id}.jpg")
                #session.image_file_id = image_file_id #get image name
				# upload the json
                self._upload_json(data=session.to_dict(),filename=f"session_{session.session_id}.json")
                self._session_count += 1
                print(f"Session {session.session_id} logged "f"({self._session_count}/{MAX_LOGGED_SESSIONS})")
                return True
		except Exception as e:
			print(f"Drive logging failed for {session.session_id}: {e}")
			return False
#____________________________________________________________
    def _upload_image(self, img:np.ndarray, filename:str):
		""" Upload user photo to Drive. 
		    Return the Drive file ID (stored in session JSON for traceability)
		"""
		from googleapiclient.http import MediaIoBaseUpload
		# Convert image from numpy to jpg (compress)
		success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
		if not success:
			raise ValueError(f"Failed to encode image: {filename}")
			
		img_bytes = io.BytesIO(buffer.tobytes()) # Create in-memory file object
		metadata  = {"name": filename, "parents": [self._images_folder_id]}
		media = MediaIoBaseUpload(img_bytes, mimetype="image/jpeg")	
		file = self.service.files().create(body=metadata, media_body=media, fields="id").execute()
        #return file["id"]
#____________________________________________________________
    def _upload_json(self, data:Dict, filename:str):
		""" Serialize dict --> JSON bytes --> upload to logs/ folder."""
		from googleapiclient.http import MediaIoBaseUpload
		json_bytes = io.BytesIO(json.dumps(data, indent=2, default=str).encode("utf-8"))
		metadata = {"name": filename, "parents": [self._logs_folder_id]}
		media = MediaIoBaseUpload(json_bytes, mimetype="application/json")
		self.service.files().create(body=metadata, media_body=media, fields="id").execute()
#=============================================================
