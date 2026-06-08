"""
=============================================================
  Face Physiognomy Project — Part 1: Face Detection & Validation
=============================================================

What this module does:
  1. Load an image (file path or numpy array)
  2. Detect whether a human face exists            → invalid if no face
  3. Reject images with multiple faces             → invalid (ambiguous)
  4. Check for facial expressions                  → invalid if expressive
  5. Crop and return the face region (ROI)         → ready for segmentation

Libraries used:
  - mediapipe  : face detection + face mesh (for expression check)
  - opencv     : image loading, drawing, saving
  - numpy      : array math

Install:
  pip install mediapipe opencv-python numpy
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─────────────────────────────────────────────
#  Enums & Data Classes
# ─────────────────────────────────────────────

class ValidationStatus(Enum):
    """Every possible outcome from the validation pipeline."""
    VALID              = "valid"
    NO_FACE_DETECTED   = "no_face_detected"
    MULTIPLE_FACES     = "multiple_faces"
    EXPRESSION_DETECTED = "expression_detected"
    IMAGE_TOO_SMALL    = "image_too_small"
    LOW_CONFIDENCE     = "low_confidence"


@dataclass
class FaceValidationResult:
    """
    Container for everything the pipeline returns.

    Attributes:
        status          : VALID or a specific error reason
        is_valid        : True only when status == VALID
        message         : Human-readable explanation
        face_crop       : Cropped face image (numpy array) — None if invalid
        face_bbox       : (x, y, w, h) bounding box in original image pixels
        detection_score : Confidence score from MediaPipe (0.0 – 1.0)
        expression_scores : Dict of expression metrics used for rejection
        annotated_image : Original image with landmarks/bbox drawn (for debug)
    """
    status            : ValidationStatus
    is_valid          : bool
    message           : str
    face_crop         : Optional[np.ndarray] = None
    face_bbox         : Optional[Tuple[int,int,int,int]] = None
    detection_score   : float = 0.0
    expression_scores : Optional[dict] = None
    annotated_image   : Optional[np.ndarray] = None


# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────

class Config:
    """
    Central place for all thresholds.
    Tune these if you get too many false rejections.
    """

    # MediaPipe face detection confidence
    DETECTION_CONFIDENCE    : float = 0.7   # minimum to accept a detection

    # MediaPipe face mesh confidence
    MESH_DETECTION_CONF     : float = 0.5
    MESH_TRACKING_CONF      : float = 0.5

    # Minimum face size (pixels) — too small = low quality
    MIN_FACE_SIZE_PX        : int   = 80

    # Padding added around the face crop (fraction of face size)
    CROP_PADDING_RATIO      : float = 0.25

    # ── Expression thresholds ──────────────────────────────────────
    # These are ratios between landmark distances.
    # Experiment with them on your dataset!

    # Mouth openness: distance between upper/lower lip landmarks
    # relative to face height. > threshold → mouth open (smile/surprise)
    MOUTH_OPEN_THRESHOLD    : float = 0.04

    # Mouth width stretch: mouth width relative to face width.
    # > threshold → wide smile
    MOUTH_STRETCH_THRESHOLD : float = 0.52

    # Eyebrow raise: how far the brow landmark is from the eye landmark
    # relative to face height. > threshold → raised brows (surprise/anger)
    EYEBROW_RAISE_THRESHOLD : float = 0.075

    # Eye squeeze: Eye aspect ratio. < threshold → squinting/winking
    # EAR = (vertical distances) / (2 * horizontal distance)
    EYE_SQUEEZE_THRESHOLD   : float = 0.18

    # Head tilt: angle of the line between both eyes.
    # > threshold degrees → tilted head
    HEAD_TILT_THRESHOLD_DEG : float = 12.0


# ─────────────────────────────────────────────
#  MediaPipe Landmark Indices
# ─────────────────────────────────────────────
# MediaPipe Face Mesh has 468 landmarks.
# We only need a small subset for expression checks.

class Landmarks:
  """Key landmark indices from the MediaPipe 468-point face mesh."""
  # Mouth
  UPPER_LIP_TOP    = 13    # inner upper lip center
  LOWER_LIP_BOTTOM = 14    # inner lower lip center
  MOUTH_LEFT       = 61    # left mouth corner
  MOUTH_RIGHT      = 291   # right mouth corner

  # Eyes (right eye from viewer's perspective)
  RIGHT_EYE_TOP    = 159
  RIGHT_EYE_BOTTOM = 145
  RIGHT_EYE_LEFT   = 133
  RIGHT_EYE_RIGHT  = 33

  # Eyes (left eye from viewer's perspective)
  LEFT_EYE_TOP     = 386
  LEFT_EYE_BOTTOM  = 374
  LEFT_EYE_LEFT    = 362
  LEFT_EYE_RIGHT   = 263

  # Eyebrows
  RIGHT_BROW_CENTER = 105   # right eyebrow midpoint
  LEFT_BROW_CENTER  = 334   # left  eyebrow midpoint

  # Face boundary for scale reference
  CHIN_BOTTOM      = 152
  FOREHEAD_TOP     = 10
  FACE_LEFT        = 234
  FACE_RIGHT       = 454


# ─────────────────────────────────────────────
#  Main Detector Class
# ─────────────────────────────────────────────

class FaceDetectorValidator:
  """
  Full face detection + validation pipeline using MediaPipe Tasks API.
  [REFACTORED FOR MEDIAPIPE TASKS API - to be PYTHON 3.12 COMPATIBLE]

  Usage:
      detector = FaceDetectorValidator()
      result   = detector.process("photo.jpg")

      if result.is_valid:
         cv2.imwrite("face_crop.jpg", result.face_crop)
      else:
            print(result.message)
  """

  def __init__(self, config: Config = None):
      self.config = config or Config()
      self._init_mediapipe()

  
  def _init_mediapipe(self):
        """Initialize MediaPipe models (loaded once, reused for every image)."""
        import os
        import urllib.request

        model_filename = "face_landmarker.task"
        if not os.path.exists(model_filename):
            print("Downloading face_landmarker.task topology map...")
            model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(model_url, model_filename)

        base_options = python.BaseOptions(model_asset_path=model_filename)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=2, # to check if more than 2 faces --> Reject
            min_face_detection_confidence=self.config.DETECTION_CONFIDENCE,
            min_face_presence_confidence=self.config.MESH_DETECTION_CONF
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        print(" MediaPipe FaceLandmarker Task Engine fully initialized.")

    # ──────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────

  def process(self, image_input) -> FaceValidationResult:
      """
      Main entry point.
      Args:
         image_input: file path (str) OR numpy BGR image array
      Returns:
         FaceValidationResult — check .is_valid before using .face_crop
      """
      # Step 1: Load image
      image_bgr = self._load_image(image_input)
      if image_bgr is None:
        return self._fail(ValidationStatus.IMAGE_TOO_SMALL, "Could not load image — check the file path.")

      h, w = image_bgr.shape[:2]
      # Step 2: Convert to RGB (MediaPipe expects RGB)
      if isinstance(image_input, str):
        mp_image = mp.Image.create_from_file(image_input)
      else:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
          
      # Step 3: Detect faces
      detection_result = self.landmarker.detect(mp_image)
      if not detection_result.face_landmarks:
        return self._fail(ValidationStatus.NO_FACE_DETECTED, "No human face detected.")
            
      num_detected_faces = len(detection_result.face_landmarks)
      if num_detected_faces > 1:
        return self._fail(ValidationStatus.MULTIPLE_FACES, f"{num_detected_faces} faces detected. Exactly one face required.")
        
      # Step 4: Getting bounding box from landmarkes 
      landmarks = detection_result.face_landmarks[0]
      x_coords = [int(lm.x * w) for lm in landmarks]
      y_coords = [int(lm.y * h) for lm in landmarks]
      x, y = min(x_coords), min(y_coords)
      bw, bh = max(x_coords) - x, max(y_coords) - y
      bbox = (x, y, bw, bh)

      confidence   = 1.0

      # Step 5: Check minimum face size
      if bw < self.config.MIN_FACE_SIZE_PX or bh < self.config.MIN_FACE_SIZE_PX:
        return self._fail(ValidationStatus.IMAGE_TOO_SMALL, f"Face region too small ({bw}×{bh} px). Minimum required: {self.config.MIN_FACE_SIZE_PX} px.")
          
      # Step 6: Geometric Expression Analysis Check
      expr_result = self._check_expression(landmarks, w, h)
      if expr_result["has_expression"]:
        return self._fail(ValidationStatus.EXPRESSION_DETECTED, f"Expression rejected: {expr_result['reason']}.", expression_scores=expr_result["scores"])
          
      # 7. Success State Pipeline -> All checks passed → crop and return
      face_crop = self._crop_face(image_bgr, bbox)
      annotated = self._draw_debug(image_bgr.copy(), bbox, confidence, landmarks, w, h)
      return FaceValidationResult(status=ValidationStatus.VALID,
                                  is_valid=True,
                                  message="Valid face detected.",
                                  face_crop=face_crop,
                                  face_bbox=bbox,
                                  detection_score=confidence,
                                  expression_scores=expr_result["scores"],
                                  annotated_image=annotated
                                  )

    # ──────────────────────────────────────────────────────────────
    #  Step 5: Expression Check
    # ──────────────────────────────────────────────────────────────

  def _check_expression(self, landmarks, img_w: int, img_h: int) -> dict:
      """
      Analyse face mesh landmarks to detect non-neutral expressions.
      Strategy: compute geometric ratios between landmark distances.
      All distances are normalized by face height so they're scale-invariant.
      Returns:
      {
      "has_expression": bool,
      "reason": str or None,
      "scores": dict of measured values
      }
      """
      cfg = self.config
      lm  = landmarks   # shorthand

  def px(idx):
        """Convert normalized landmark to pixel coordinates."""
        return np.array([lm[idx].x * img_w, lm[idx].y * img_h])

      # ── Reference: face height for normalization ───────────────
      face_height = np.linalg.norm(px(Landmarks.FOREHEAD_TOP) - px(Landmarks.CHIN_BOTTOM))
      face_width  = np.linalg.norm(px(Landmarks.FACE_LEFT) - px(Landmarks.FACE_RIGHT))

      if face_height < 1:
        face_height = 1   # avoid division by zero

      # ─────────────────────────────────────────────────────────
      #  Metric 1: Mouth openness
      #  Vertical distance between inner upper and lower lip
      # ─────────────────────────────────────────────────────────
      mouth_open = np.linalg.norm(px(Landmarks.UPPER_LIP_TOP) - px(Landmarks.LOWER_LIP_BOTTOM)) / face_height

      # ─────────────────────────────────────────────────────────
      #  Metric 2: Mouth stretch (smile width)
      #  Width of mouth relative to face width
      # ─────────────────────────────────────────────────────────
      mouth_width = np.linalg.norm(px(Landmarks.MOUTH_LEFT) - px(Landmarks.MOUTH_RIGHT))
      mouth_stretch = mouth_width / face_width if face_width > 1 else 0

      # ─────────────────────────────────────────────────────────
      #  Metric 3: Eyebrow raise
      #  Distance from brow to eye, normalized by face height
      # ─────────────────────────────────────────────────────────
      right_brow_raise = np.linalg.norm(px(Landmarks.RIGHT_BROW_CENTER) - px(Landmarks.RIGHT_EYE_TOP)) / face_height

      left_brow_raise  = np.linalg.norm(px(Landmarks.LEFT_BROW_CENTER) - px(Landmarks.LEFT_EYE_TOP)) / face_height

      brow_raise = max(right_brow_raise, left_brow_raise)

      # ─────────────────────────────────────────────────────────
      #  Metric 4: Eye Aspect Ratio (EAR)
      #  EAR = (eye_height) / (eye_width)
      #  Low EAR → squinting or winking
      # ─────────────────────────────────────────────────────────
  def eye_aspect_ratio(top, bottom, left, right):
        eye_h = np.linalg.norm(px(top) - px(bottom))
        eye_w = np.linalg.norm(px(left) - px(right))
        return eye_h / eye_w if eye_w > 0 else 1.0

      right_ear = eye_aspect_ratio(Landmarks.RIGHT_EYE_TOP, Landmarks.RIGHT_EYE_BOTTOM, Landmarks.RIGHT_EYE_LEFT, Landmarks.RIGHT_EYE_RIGHT)
      left_ear = eye_aspect_ratio(Landmarks.LEFT_EYE_TOP, Landmarks.LEFT_EYE_BOTTOM, Landmarks.LEFT_EYE_LEFT, Landmarks.LEFT_EYE_RIGHT)
      min_ear = min(right_ear, left_ear)

      # ─────────────────────────────────────────────────────────
      #  Metric 5: Head tilt
      #  Angle of the line connecting left and right eye corners
      # ─────────────────────────────────────────────────────────
      eye_l = px(Landmarks.LEFT_EYE_RIGHT)    # inner corner of left eye
      eye_r = px(Landmarks.RIGHT_EYE_LEFT)    # inner corner of right eye
      delta = eye_r - eye_l
      tilt_deg = abs(np.degrees(np.arctan2(delta[1], delta[0])))

      # ── Collect all scores ────────────────────────────────────
      scores = {
        "mouth_open"    : round(float(mouth_open),    4),
        "mouth_stretch" : round(float(mouth_stretch), 4),
        "brow_raise"    : round(float(brow_raise),    4),
        "min_eye_EAR"   : round(float(min_ear),       4),
        "head_tilt_deg" : round(float(tilt_deg),      2),
      }

      # ── Evaluate thresholds ───────────────────────────────────
      reasons = []

      if mouth_open > cfg.MOUTH_OPEN_THRESHOLD:
        reasons.append(f"mouth open (score {scores['mouth_open']:.3f} >" f" threshold {cfg.MOUTH_OPEN_THRESHOLD})")

      if mouth_stretch > cfg.MOUTH_STRETCH_THRESHOLD:
        reasons.append(f"wide smile (score {scores['mouth_stretch']:.3f} >" f" threshold {cfg.MOUTH_STRETCH_THRESHOLD})")

      if brow_raise > cfg.EYEBROW_RAISE_THRESHOLD:
        reasons.append(f"raised eyebrows (score {scores['brow_raise']:.3f} > " f"threshold {cfg.EYEBROW_RAISE_THRESHOLD})")

      if min_ear < cfg.EYE_SQUEEZE_THRESHOLD:
        reasons.append(f"eye squint/wink (EAR {scores['min_eye_EAR']:.3f} < "f"threshold {cfg.EYE_SQUEEZE_THRESHOLD})")

      if tilt_deg > cfg.HEAD_TILT_THRESHOLD_DEG:
        reasons.append(f"head tilt {scores['head_tilt_deg']}° > "f"threshold {cfg.HEAD_TILT_THRESHOLD_DEG}°")

      return {
            "has_expression": len(reasons) > 0,
            "reason"        : " | ".join(reasons) if reasons else None,
            "scores"        : scores}

    # ──────────────────────────────────────────────────────────────
    #  Step 6: Crop Face
    # ──────────────────────────────────────────────────────────────

  def _crop_face(self, image_bgr: np.ndarray, bbox: Tuple[int,int,int,int]) -> np.ndarray:
      """
      Crop the face with padding around it.
      Padding prevents cutting off chin or forehead edges.
      """
      x, y, bw, bh = bbox
      h, w = image_bgr.shape[:2]

      pad_x = int(bw * self.config.CROP_PADDING_RATIO)
      pad_y = int(bh * self.config.CROP_PADDING_RATIO)

      x1 = max(0, x - pad_x)
      y1 = max(0, y - pad_y)
      x2 = min(w, x + bw + pad_x)
      y2 = min(h, y + bh + pad_y)

      return image_bgr[y1:y2, x1:x2]

    # ──────────────────────────────────────────────────────────────
    #  Drawing / Debug
    # ──────────────────────────────────────────────────────────────

  def _draw_detection(self, image: np.ndarray, bbox, confidence) -> np.ndarray:
      """Draw bounding box and confidence score on the image."""
      x, y, bw, bh = bbox
      color = (0, 200, 0)   # green
      cv2.rectangle(image, (x, y), (x+bw, y+bh), color, 2)
      label = f"Face  {confidence:.0%}"
      cv2.putText(image, label, (x, max(y-8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
      return image

  def _draw_debug(self, image: np.ndarray, bbox, confidence, landmarks, img_w, img_h) -> np.ndarray:
        """Draw bbox + key landmarks used for expression checks."""
      image = self._draw_detection(image, bbox, confidence)

        # Draw key landmarks as small circles
      key_indices = [Landmarks.UPPER_LIP_TOP, Landmarks.LOWER_LIP_BOTTOM,
                     Landmarks.MOUTH_LEFT, Landmarks.MOUTH_RIGHT,
                     Landmarks.RIGHT_EYE_TOP, Landmarks.RIGHT_EYE_BOTTOM,
                     Landmarks.LEFT_EYE_TOP,  Landmarks.LEFT_EYE_BOTTOM,
                     Landmarks.RIGHT_BROW_CENTER, Landmarks.LEFT_BROW_CENTER,
                     Landmarks.CHIN_BOTTOM, 
                     Landmarks.FOREHEAD_TOP,
                     ]
      for idx in key_indices:
        lm = landmarks[idx]
        px = int(lm.x * img_w)
        py = int(lm.y * img_h)
        cv2.circle(image, (px, py), 3, (0, 255, 255), -1)

      return image

    # ──────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────

  def _load_image(self, image_input) -> Optional[np.ndarray]:
      """Accept a file path string or a numpy array."""
      if isinstance(image_input, str):
        img = cv2.imread(image_input)
        return img   # None if file not found
      elif isinstance(image_input, np.ndarray):
        return image_input.copy()
      return None

  def _fail(self, status: ValidationStatus, message: str, expression_scores=None) -> FaceValidationResult:
      return FaceValidationResult(status=status,
                                  is_valid=False,
                                  message=message,
                                  expression_scores=expression_scores
                                  )

  def __del__(self):
        """Securely release MediaPipe Tasks API Engine resources from memory."""
        if hasattr(self, 'landmarker'):
          try:
            self.landmarker.close()
            print("MediaPipe Landmarker closed and RAM allocated resources released successfully.")
          except Exception:
            pass
