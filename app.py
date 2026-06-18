"""
 Face Physiognomy Project UI using Gradio for depolying the project on HuggingFace
  1. Main entry point for the hugging face space
  2. Build Gardio UI, to wire all the pipline component together together 
  3. Handel Email gating, logging, and error display 
 _________________________________________________________________________________
 APP workflow
 1. Load the all models at the start
 2. Save the logs in my own google driver [Cropped_pic, queries, result, user_email]
     FLOW:
      1. Validate email
      2. Validate image (face detection)
      3. Extract face parts
      4. Describe each part (VLM)
      5. Search book (RAG)
      6. Generate report
      7. Send email to user
      8. Log to Drive
 _________________________________________________________________________________
 The user workflow of using the app:
     1. Upload Front Photo (a must)     2. Upload Profile Photo (an optional)
                         3. Recieve the model result by Email (a must)
_________________________________________________________________________________
"""
import os
import sys
import time
#import json
import tempfile
from datetime import datetime

import cv2
#import numpy as np
import gradio as gr

# Add src to the path 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from face_detector import FaceDetectorValidator
from face_part_extractor import FacePartExtractor, PaddingConfig
from face_describer import FaceDescriptor
from rag_retriever import PhysiognomyRetriever
from report_generator import ReportGenerator
from email_sender import EmailSender
from drive_logger import DriveLogger, SessionLog, new_session_id

# Load pipline components (once at the startup)
# Note: loading models takes (10-30 sec) 
# So, if the user will use the app more than one request, i want all models to load once, not per each user request!!

INDEX_DIR = os.environ.get("RAG_INDEX_DIR", "/app/rag_index")
detector  = FaceDetectorValidator()
describer = FaceDescriptor()
retriever = PhysiognomyRetriever(index_path  = f"{INDEX_DIR}/index.faiss",chunks_path = f"{INDEX_DIR}/chunks.pkl")
generator = ReportGenerator()
#_________________________________________________________________________
# Check my email & driver 
#_________________________________________________________________________
try:
    email_sender = EmailSender()
    print("Email sender is ready")
except Exception as e:
    email_sender = None
    print(f" Email sender is not configured: {e}")

try:
    drive_logger = DriveLogger()
    print("Driver Logger is ready")
except Exception as e:
      drive_logger = None
      print(f" Drive logger is not configured: {e}")
#_________________________________________________________________________
# Email Validation
#_________________________________________________________________________
def is_valid_email(email):
  import re
  pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
  return bool(re.match(pattern, email.strip()))
#_________________________________________________________________________
# Main Pipeline Function
#_________________________________________________________________________
def analyze_face(email, front_image, profile_image=None):
  """
  Input: email (string) : user email address
         front_image (np.ndarray) : user fron image in RGP: , 
         profile_image (np.ndarray) : optional 
  Output (string) :  physiognomy report
  """
  session_id = new_session_id()
  start_time = time.time()
  session_log = SessionLog(session_id=session_id,
                          timestamp=datetime.utcnow().isoformat(),
                          email=email.strip() if email else "unknown",
                          status = "started",
                          latency_sec = 0.0)

  #__1) Email validation check_______________
  if not email.strip():
    return ("Please enter your email address first.")
  if not is_valid_email(email):
    return ("Please enter a valid email address!")
  if front_image is None:
    return ("Please upload an image.")
  #__2) Convert image to BGR _______________
  image_bgr = cv2.cvtColor(front_image, cv2.COLOR_RGB2BGR)
  
  #__3) Face Detection & Validation _______________
  detection = detector.process(image_bgr)
  if not detection.is_valid:
    session_log.status = detection.status.value
    session_log.latency_sec = time.time() - start_time
    session_log.error_msg = detection.message
    # Log the faild try with the orginal image!
    if drive_logger:
      drive_logger.log_session(session = session_log, original_img = image_bgr)
    return ( f"{detection.message}")
    
  #__4) Face Parts Extraction _______________
  extractor = FacePartExtractor(face_crop = detection.face_crop,
                                crop_landmarks = detection.crop_landmarks,
                                padding = PaddingConfig())
  all_parts = extractor.extract_all_parts(include_ears=False)
  parts_ok  = len(all_parts.valid_parts())
  session_log.regions_ok = parts_ok
  ############ ضيفي شيك بوينت هنا انه الاجزائ تمام اصلا قبل مروح لل ف ل م عشان اذا مش تمام بعدين هيدرب 
  ############ واعمل بيها لوج 
  
  #__5) VLM Description _______________
  descriptions = describer.describe_all_parts(all_parts, delay_between_calls=2)
  
  #__6) RAG Retrieval _______________
  all_evidence = retriever.search_all_parts(descriptions, top_k=3)

  #__7) Generate Report _______________
  report = generator.generate(all_evidence)
  if not report.success:
    session_log.status = "report_failed"
    session_log.latency_sec = time.time() - start_time
    session_log.error_msg = report.error
    if drive_logger:
      drive_logger.log_session(session = session_log,
                               original_img = image_bgr,
                               face_crop = detection.face_crop)
    return (f"Report generation failed: {report.error}")

  #__8) Send Email with the Report _______________
  email_status = "email_not_configured"
  if email_sender:
    # Save face crop to temp file for email attachment
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
      cv2.imwrite(tmp.name, detection.face_crop)
      tmp_path = tmp.name
    email_result = email_sender.send_report(to_email = email.strip(),
                                            report_text = report.report_text,
                                            session_id = session_id,
                                            face_image_path = tmp_path)
    os.unlink(tmp_path)   # clean up temp file
    email_status = "email_sent" if email_result.success else "email_failed"
    
  #__9) Log sucess try to the Drive _______________
  latency = time.time() - start_time
  session_log.status = "success"
  session_log.latency_sec = round(latency, 2)
  session_log.face_bbox = str(detection.face_bbox)
  session_log.report_preview = report.report_text
  if drive_logger:
    drive_logger.log_session(session = session_log,
                             original_img = image_bgr,
                             face_crop = detection.face_crop,
                             report_text  = report.report_text)
    

  #__10) Output for Gradio UI _______________
  status_msg = (f"Analysis is completed in {latency:.1f}s\n"
                f"Session: {session_id}\n"
                f"{'Report is sent to your email' if email_status == 'email_sent' else ' Email delivery pending'}")
  return status_msg, report.report_text

#_________________________________________________________________________
# Gradio UI 
#_________________________________________________________________________
def build_ui():
    """
    Build the Gradio interface.

    LAYOUT:
      ┌────────────────────────────────────────┐
      │        Face Physiognomy Analyzer       │
      │  Enter email + Upload photo to Analyze  │
      ├──────────────────┬─────────────────────┤
      │  Email input     │                     │
      │  Image upload    │   Report output     │
      │  [Analyze btn]   │   Status output     │
      └──────────────────┴─────────────────────┘
    """
    with gr.Blocks(title="Face Physiognomy Analyzer") as demo:
      # ── Header
      gr.Markdown("""
      # 🔍 Face Physiognomy Analyzer
      Analyze facial features based on the principles of physiognomy.

**How it works:**
1. Enter your email address
2. Upload a clear, neutral-expression frontal face photo
3. Click **Analyze** — your report will appear here and be sent to your email

**Photo requirements:** Frontal view · Neutral expression · Good lighting
        """)

        # ── Inputs 
        with gr.Row():
            with gr.Column(scale=1):
                email_input = gr.Textbox(label = "Your Email Address",
                                         placeholder = "your@email.com",
                                         info = "Required as your report will be sent here")
              
                image_input = gr.Image(label = "Upload Face Photo",
                                       type = "numpy", # Gradio returns RGB numpy array
                                       height = 350)
              
                analyze_btn = gr.Button("Analyze Face",
                                        variant = "primary",
                                        size = "lg")

            # ── Outputs 
            with gr.Column(scale=1):
                status_output = gr.Textbox(label = "Status",
                                           lines = 3,
                                           interactive = False,)
                report_output = gr.Textbox(label = "Physiognomy Report",
                                           lines = 20,
                                           interactive = False)

        # ── Footer 
        gr.Markdown("""
---
*This tool is for educational and entertainment purposes only.*
*Based on "Amazing Face Reading" by Mac Fulfer.*
        """)

        # ── Wire button to function ───────────────────────────
        analyze_btn.click(
          fn = analyze_face,
          inputs  = [email_input, image_input],
          outputs = [status_output, report_output])

    return demo

#_________________________________________________________________________
#  Launch
#_________________________________________________________________________

if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
  

    

  



