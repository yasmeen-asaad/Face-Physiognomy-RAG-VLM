import os
import re
import json
import base64
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import cv2
import numpy as np
# =========*==================*===============*=============*======
# Feature Map, What to describe per region
# =========*==================*===============*=============*======
FEATURES_MAP =  {
     "nose": ["size", "shape", "ridge", "width", "tip_angle", "nose_tip_size_shape", "nostrils_size_shape"],
     "eyes": ["spacing", "angle", "depth", "eye_puffs_presence", "eye_puffs_severity", "eyelashes_presence", ""eyelashes_density"", "eye_color", "iris_size", "corner_indents", "eyelid_top", "eyelid_bottom"],
     "eyebrows": ["shape", "position", "thickness", "color", "type"],
     "forehead": ["shape", "height", "width", "lines"],
     "mouth": ["size", "angle", "lips_size", "teeth", "smile_type"],
     "jaw_chin": ["jaw_shape", "jaw_width", "chin_shape", "chin_projection", "chin_size", "cheek_fullness", "dimples", "clefts"],
     "ears": ["size", "cups_ridges", "placement", "height"],
     "face_overview": ["face_shape", "face_type", "head_type", "face_color", "overall_skin_tone", "ear_eyebrow_combinations", "chin_eyebrow_combinations", "profile_type", "face_lines", "facial_hair"]
        
              }
# =========*==================*===============*=============*======
# How result should look like -Result Data Class -
# =========*==================*===============*=============*======
#dataclass
class DescriptionResult:
     """
     The output of each one face region should look like the following:
         1. region: The face part region 
         2. features_json: the structured JSON from the VLM, where keys = feature names, values = observations
         3. raw_response: The full raw text from the API for debugging
         4. success: True if json was paresed successfully 
         5. error: error message if somrthing bad happend 
         6. tokens_used: count API tokens, to monitor free trier usage 
     """
     region: str
     features_json: Dict[str, Any] | None = None
     raw_response: str = ""
     success: bool = False
     error: str | None = None 
     tokens_used: int = 0
     
# =========*==================*===============*=============*======
#  Face Describer Class
# =========*==================*===============*=============*======
import google.generativeai as genai
from dotenv import load_dotenv
import os

class FaceDescriptor:
     """
    Sends face-part crops to Gemini Flash and gets structured
    visual descriptions back as JSON.
     """
     
     def __init__(self, model_name="gemini-1.5-flash", need_key=True):
          # Facial part confidence
          self.confidence_score = 0.5
          # Load Vllm
          self.model_name = model_name
          self.max_retries = 3
          self.retry_delays = 5 # seconds between retries
          if need_key:
               api_key = os.getenv("GOOGLE_API_KEY")
               if not api_key:
                    raise ValueError("No model API key found.")
               genai.configure(api_key=api_key)
               self.model = genai.GenerativeModel(self.MODEL)
               self.genai = genai
          print(f"Model {model_name} is loaded successfully")
#____________________________________________
# Prepare Image for the model
#____________________________________________
     def _prepare_img(self, img_bgr: np.ndarray):
          """
          1. Encode image to base64 string, to pass it throw HTTP API
             (base64: is a text representation of binary data)
             1.1 Convert BGR --> RGP, to be compatible with img compression "JPEG"
             1.2 Convert numpy --> JPEG bytes, to be smaller to pass through http 
                 (JPEG is smaller, hence fast api call it keep 95% detailes  of the orogonal image)
             1.3 Convert JPEG bytes --> base64 string (for the sake of hhttp )
          2. Return Dict 
          """
          # 1. Convert to RGP 
          img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

          # 2. Convert to JPEG
          encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
          success, buffer = cv2.imencode(".jpg", img_rgb, encode_params)
          if not success:
            raise ValueError("Failed to encode image to JPEG")
               
          # 3. Convert to base64
          img_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
          # output dict
          return {"mime_type": "image/jpeg",
                 "data" : img_base64}

#____________________________________________
# Build the Prompt
#____________________________________________
    def get_prompt(self, face_part:str):

        features = self.features_map[part_name]
        feature_list = "\n".join([f"- {feature}" for feature in features])
        schema_example = {f: {"value": "...", "confidence": 0.0, "description": "..."} for f in features}
        schema_str = json.dumps(schema_example, indent=2)
        prompt = f"""You are a facial morphology analyzer. Your task is to analyze facial features with precise visual observations.
Analyze ONLY the {region.replace('_', ' ')} visible in the image.
Describe the following features:{features_list} 
STRICT RULES:
1. Use ONLY what you can directly observe in the image.
2. Do NOT infer personality traits.
3. Do NOT infer emotions or mood.
4. Do NOT infer character or intelligence.
5. If a feature cannot be clearly determined, set value to null.
6. confidence: 0.0 (not sure) to 1.0 (very sure)
7. description: one short sentence describing what you see visually.
8. Return VALID JSON ONLY — no markdown, no explanation, no preamble.

Required output format (JSON keys must exactly match feature names above):
{schema_str}"""
    return prompt
#____________________________________________
# Main Method
#____________________________________________
    def describe_part(self, part_name:str, part_img : np.ndarray):
         """ 
         Send one face-part image to Gemini and get JSON description.
         Returns: DescriptionResult with .features_json if successful
         """
         if part_name not in FEATURES_MAP:
              return DescriptionResult(region  = part_name, success = False,
                                       error = f"Unknown region: '{part_name}'")
         try:
              image_data = self._prepare_image(part_img)
              prompt = self._build_prompt(part_name)
         except Exception as e:
              return DescriptionResult(region=part_name, success=False,
                                       error=f"Image preparation failed: {e}")

         # Handling calling model with limits ex: 5 requests/min.
         # Add delay in order to handle not to hit the limit of free tier
         last_error = None
         for attempt in range(self.MAX_RETRIES):
              try:
                   response = self.model.generate_content(
                        contents=[
                             # Gemini accepts a list of: text + image parts
                             {"mime_type": image_data["mime_type"],
                             "data" : image_data["data"]},
                             prompt])
                   response_text = response.text
                   tokens_used = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
                   features_json = self._parse_json(raw_text)
                   return DescriptionResult(region = part_name,
                                            features_json = features_json,
                                            raw_response = raw_text,
                                            success = True,
                                            tokens_used = tokens_used)
              except Exception as e:
                   last_error = str(e)
                   if attempt < self.MAX_RETRIES - 1:
                    print(f" Attempt {attempt+1} failed: {e}. "
                          f"Retrying in {self.RETRY_DELAY_S}s...")
                          time.sleep(self.RETRY_DELAY_S)

         return DescriptionResult(region = part_name, success = False,
                                  error = f"All {self.MAX_RETRIES} attempts failed: {last_error}")

#____________________________________________
# Describe All Parts
#____________________________________________
    def describe_all_parts(self, all_parts, delay_between_calls = 2, ):
         """
         all_parts: AllPartsResult from FacePartExtractor
         delay_between_calls: delay in seconds 
                              ( Importanr due to not to exceed the free tier ex. 15 requests/minute.
         output: Dict mapping region_name --> DescriptionResult 
         """
         results = {}
         parts_dict = all_parts.valid_parts()
         total = len(parts_dict)
         for i, (region_name, part_result) in enumerate(parts_dict.items(), 1):
              # Skip regions not in FEATURES_MAP
              if region_name not in FEATURES_MAP:
                   print(f"[{i}/{total}] {region_name} is skipped as it's not in the feature map)")
                   continue
               # Skip facial parts with low confidence score ex ears in version 1 of the app
              if part_result.confidence_score < self.confidence_score:
                   print(f"[{i}/{total}] {region_name} is skipped as it has low confidence: {part_result.confidence_score}")
                   continue
              print(f"[{i}/{total}] Describing: {region_name}......")
              result = self.describe_part(region_name, part_result.image)
              if result.success:
                   print(f"success, ({result.tokens_used} tokens)")
              else:
                   print(f"failed {result.error}")
              results[region_name] = result
              # Pause between calls to respect rate limit
              if i < total:
                   time.sleep(delay_between_calls)
         return results
#____________________________________________
# JSON Parser Function 
#____________________________________________ 
    def _parse_json(self, raw_text: str):
         """
         Parsing model's response to JSON,
         MAke sure the output is a valid JSon 
         1. Try to pasre directly, if no 
         2. Strip famous llms markdown in json files ```, if no 
         3. Using regex to find JSON-like content, if no 
         4. Else Rasie error !!!!
         """
         # 1. Try Direct Parsing 
         try:
              return json.loads(raw_text.strip())
         except json.JSONDecodeError:
              pass
              
         # 2. Try Strip 
         cleaned = re.sub(r"```(?:json)?\s*", "", raw_text)
         cleaned = cleaned.replace("```", "").strip()
         try:
              return json.loads(cleaned)
         except json.JSONDecodeError:
              pass
              
         # 3. Try regex 
         match = re.search(r"\{.*\}", cleaned, re.DOTALL)
         if match:
              try:
                   return json.loads(match.group())
               except json.JSONDecodeError:
                    pass
         # 4. ugh! raise error
         raise ValueError(f"Could not parse JSON from response:\n{raw_text[:300]}")

              

         
         
     











