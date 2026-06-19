"""
 Face Physiognomy Project UI using Gradio for depolying the project on HuggingFace
  1. Main entry point for the hugging face space
  2. Build Gardio UI, to wire all the pipline component together together 
Will use every thing ijn global to load models once( not to load the models per each request)
 _________________________________________________________________________________
 APP workflow
 1. Load the all models at the start
 2. Save the logs in my own google driver [Cropped_pic, queries, result]
     FLOW:
      1. Validate image (face detection)
      2. Extract face parts
      3. Describe each part (VLM)
      4. Search book (RAG)
      5. Generate report
      6. Log to Drive
 _________________________________________________________________________________
 The user workflow of using the app:
     1. Upload Front Photo (a must)     2. Upload Profile Photo (an optional)
                         3. Display the model result 
_________________________________________________________________________________
"""
import os
import sys
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import gradio as gr

# Add src to the path 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from face_detector import FaceDetectorValidator
from face_part_extractor import FacePartExtractor, PaddingConfig
from face_describer import FaceDescriptor
from rag_retriever import PhysiognomyRetriever
from report_generator import ReportGenerator
from drive_logger import DriveLogger, SessionLog, new_session_id

# Load pipline components (once at the startup)
# Note: loading models takes (10-30 sec) 
# So, if the user will use the app more than one request, i want all models to load once, not per each user request!!


from huggingface_hub import hf_hub_download, login
HF_TOKEN = os.environ.get("HF_TOKEN")
os.makedirs("RAG_INDEX_DIR", exist_ok=True)
hf_hub_download(
    repo_id="YasmeenAsaad/RAG_INDEX_DIR",
    filename="index.faiss",          # Change if your filename is different
    repo_type="dataset",
    local_dir="RAG_INDEX_DIR",
    token=HF_TOKEN,
    local_dir_use_symlinks=False
)

hf_hub_download(
    repo_id="YasmeenAsaad/RAG_INDEX_DIR",
    filename="chunks.pkl",          # Change if your filename is different
    repo_type="dataset",
    local_dir="RAG_INDEX_DIR",
    token=HF_TOKEN,
    local_dir_use_symlinks=False
)

INDEX_DIR = os.path.join(os.path.dirname(__file__), "RAG_INDEX_DIR")
api_key = os.environ.get("GEMINI_API_KEY")
model_name="gemini-flash-lite-latest"
detector  = FaceDetectorValidator()
describer = FaceDescriptor(api_key=api_key, model_name=model_name) 
retriever = PhysiognomyRetriever(index_path = f"{INDEX_DIR}/index.faiss", chunks_path = f"{INDEX_DIR}/chunks.pkl")

generator = ReportGenerator(api_key=api_key, model_name=model_name) 
#_________________________________________________________________________
# Check my driver 
#_________________________________________________________________________
try:
    drive_logger = DriveLogger()
    print("driver logger is loaded")
except Exception as e: 
    drive_logger = None
    print(f"driver logger is disabled: {e}")

#_________________________________________________________________________
# Main Pipeline Function
#_________________________________________________________________________
def analyze_face(front_image, profile_image=None):
  """
  Input: front_image (np.ndarray) : user fron image in RGP: , 
         profile_image (np.ndarray) : optional 
  Output (string) :  physiognomy report
  """
  session_id = new_session_id()
  t0 = time.time()
  session = SessionLog(session_id=session_id, timestamp=datetime.now(timezone.utc).isoformat(), status="started")

  if front_image is None:
    return ("Please upload an image.")
  # Convert image to BGR 
  image_bgr = cv2.cvtColor(front_image, cv2.COLOR_RGB2BGR)
  
  # Face Detection & Validation 
  detection = detector.process(image_bgr)
  session.pipeline_stats["face_detection_sec"] = round(time.time() - t0, 2)
 
  if not detection.is_valid:
    session.status = detection.status.value
    session.error_msg = detection.message
    session.latency_sec = round(time.time() - t0, 2)
    _log(session, original_img=image_bgr)
    return f" {detection.message}"
  t1 = time.time()
  # Face Parts Extraction 
  extractor = FacePartExtractor(face_crop = detection.face_crop,crop_landmarks = detection.crop_landmarks,padding = PaddingConfig())
  all_parts = extractor.extract_all_parts(include_ears=False)
  session.pipeline_stats["face_parts_sec"] = round(time.time() - t1, 2)
  parts_ok  = len(all_parts.valid_parts()) 
  session.pipeline_stats["regions_extracted"] = parts_ok
  session.pipeline_stats["face_bbox"] = str(detection.face_bbox)
 
  ############ ضيفي شيك بوينت هنا انه الاجزائ تمام اصلا قبل مروح لل ف ل م عشان اذا مش تمام بعدين هيدرب 
  ############ واعمل بيها لوج 
  if parts_ok < 4:
   session.status = "bad_extraction"
   session.error_msg = f"Only {parts_ok} regions extracted"
   _log(session, original_img=image_bgr)
   return ("Face extraction quality is too low.\nPlease upload a clearer image.")
  t2 = time.time()
  # VLM Description
  descriptions = describer.describe_all_parts(all_parts, delay_between_calls=2)
  session.descriptions = {region: desc.features_json for region, desc in descriptions.items() if desc.success}
  session.pipeline_stats["descriptions_ok"] = len(session.descriptions)
  session.pipeline_stats["vlm_sec"] = round(time.time() - t2, 2)
 
  t3 = time.time()
  # RAG Retrieval
  all_evidence = retriever.search_all_parts(descriptions, top_k=3)
  session.retrieval_results = _format_retrieval_for_log(all_evidence)
  session.pipeline_stats["regions_retrieved"] = len(all_evidence)
  session.pipeline_stats["retrieval_sec"] = round(time.time() - t3, 2)

  t4 = time.time()
  # Generate Report 
  report = generator.generate(all_evidence)
  if not report.success:
    session.status = "report_failed"
    session.latency_sec = time.time() - t0
    session.error_msg = report.error
    _log(session, original_img=image_bgr)
    return f"Report generation failed: {report.error}"
  session.pipeline_stats["report_sec"] = round(time.time() - t4, 2)
  # Log sucess try to the Drive 
  latency = time.time() - t0
  session.status = "success"
  session.latency_sec = latency   #total latency
  session.report_text = report.report_text
  session.pipeline_stats["total_tokens"] = report.tokens_used
  _log(session, original_img=image_bgr)
    
  # Output for Gradio UI _______________
  status_msg = (f"Analysis completed in {latency:.1f}s\n"
                f"Session: {session_id}\n"
                f"Regions: {parts_ok} extracted\n")

  return status_msg, report.report_text

#_________________________________________________________________________
# Format RAG Results to be bretty for Session Log
#_________________________________________________________________________

def _format_retrieval_for_log(all_evidence):
  """ Prepare RAG results for JSON logging. """
  log_ready = {}
  for region, evidence in all_evidence.items():
   log_ready[region] = {"query": evidence.get("query", ""),
                        "results": [{"rank": r.get("rank"), "score": r.get("score"),
                                     "page": r.get("page"), "chapter": r.get("chapter"),
                                     "content": r.get("content", "")[:500]  # truncate long passages
                                    }
                                    for r in evidence.get("results", [])
                                   ]
                       }
  return log_ready
#_________________________________________________________________________
# Log Function
#_________________________________________________________________________
def _log(session: SessionLog, original_img=None):
  """ Log session to Drive, silently skips if logger not configured. """
  if drive_logger is None:
    return
  try:
    drive_logger.log_session(session, original_img=original_img)
  except Exception as e:
    print(f"Logging error (non-fatal): {e}")
#_________________________________________________________________________
# Gradio UI 
#_________________________________________________________________________
def build_ui():
    """
    Build the Gradio interface.

    LAYOUT:
      ┌──────────────────────────────────────────────────────┐
      │               Face Physiognomy Analyzer              │
      │            Upload Front photo to Analyze             |
      |            Upload Profile photot (optional)          |
      ├─────────────────────────────┬────────────────────────┤
      │  Upload Front photo         │ Status output          │
      │  Upload Profile photot (op) │ Report (Full Text)     │
      |   Analyze botton]           |                        │
      └─────────────────────────────┴────────────────────────┘
    """
    with gr.Blocks(title="Face Physiognomy Analyzer") as demo:
     # Header 
      gr.Markdown("""
      # 🔍 Face Physiognomy Analyzer
      Analyze facial features based on the principles of physiognomy.

**How it works:**
1. Upload a clear, neutral-expression frontal face photo
2. Upload a clear, neutral-expression profile face photo (optional)
2. Click **Analyze**, your report will appear on the right

**Photo requirements:** Frontal view · Neutral expression · Good lighting
        """)

        # Inputs 
        with gr.Row():
            with gr.Column(scale=1): # left column
                image_input = gr.Image(label = "Upload Front Face Photo",
                                       type = "numpy", # Gradio returns RGB numpy array
                                       height = 350)
                image_input_profile = gr.Image(label = "Upload Profile Face Photo (optional)",
                                               type = "numpy", # Gradio returns RGB numpy array
                                               height = 350) 
              
                analyze_btn = gr.Button("Analyze Face", variant = "primary",size = "lg")

            # Outputs: in the right column
            with gr.Column(scale=1):
                status_output = gr.Textbox(label="Status", lines=3, interactive=False)
                report_output = gr.Textbox(label="Physiognomy Report", lines=20, interactive=False)

        # Footer 
        gr.Markdown("""---*This tool is for educational and entertainment purposes only.* *Based on "Amazing Face Reading" book by Mac Fulfer.*""")

        # Link the button to the function
        analyze_btn.click(fn=analyze_face, inputs=[image_input, image_input_profile],outputs=[status_output, report_output])
    return demo
#_________________________________________________________________________
#  Launch
#_________________________________________________________________________

if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
  

    

  



