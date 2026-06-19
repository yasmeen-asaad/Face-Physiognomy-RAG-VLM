"""
=============================================================
  Face Physiognomy Project — Part 1: Face Detection & Validation
=============================================================
  [UPDATED: Uses MediaPipe Tasks API for Python 3.12 compatibility]
  [UPDATED: Returns crop_landmarks — landmarks remapped to crop pixels]

What this module does:
  1. Load an image (file path or numpy array)
  2. Detect whether a human face exists            → invalid if no face
  3. Reject images with multiple faces             → invalid (ambiguous)
  4. Check for facial expressions                  → invalid if expressive
  5. Crop and return the face region (ROI)         → ready for segmentation

  NEW in this version:
  6. Returns crop_landmarks: List of (x, y) pixel tuples
     coordinates are relative to the face_crop, NOT the original image.
     FacePartExtractor can use them directly — no remapping needed.

Install:
  pip install mediapipe opencv-python numpy
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =============================================================
#  Enums & Data Classes
# =============================================================

class ValidationStatus(Enum):
    VALID               = "valid"
    NO_FACE_DETECTED    = "no_face_detected"
    MULTIPLE_FACES      = "multiple_faces"
    EXPRESSION_DETECTED = "expression_detected"
    IMAGE_TOO_SMALL     = "image_too_small"
    LOW_CONFIDENCE      = "low_confidence"


@dataclass
class FaceValidationResult:
    """
    Container for everything the pipeline returns.

    Key fields for Part 2 (FacePartExtractor):
        face_crop       : cropped face image (numpy BGR array)
        face_bbox       : (x, y, w, h) of the crop in the ORIGINAL image
        crop_landmarks  : List of (x, y) pixel tuples
                          !! coordinates are relative to face_crop !!
                          index N → pixel position of landmark N in the crop
                          Pass this directly to FacePartExtractor.
    """
    status            : ValidationStatus
    is_valid          : bool
    message           : str
    face_crop         : Optional[np.ndarray]              = None
    face_bbox         : Optional[Tuple[int,int,int,int]]  = None
    detection_score   : float                             = 0.0
    expression_scores : Optional[dict]                    = None
    annotated_image   : Optional[np.ndarray]              = None

    # NEW — landmarks in crop-pixel space, ready for FacePartExtractor
    crop_landmarks    : Optional[List[Tuple[int,int]]]    = None
    raw_landmarks     : Optional[list]                  = None


# =============================================================
#  Configuration
# =============================================================

class Config:
    """Thresholds. Tune these if you get false rejections."""

    DETECTION_CONFIDENCE    : float = 0.5
    MESH_DETECTION_CONF     : float = 0.5
    MESH_TRACKING_CONF      : float = 0.5
    MIN_FACE_SIZE_PX        : int   = 50
    CROP_PADDING_RATIO      : float = 0.25

    # Expression thresholds — ratios relative to face height/width
    MOUTH_OPEN_THRESHOLD    : float = 0.09
    MOUTH_STRETCH_THRESHOLD : float = 0.65
    EYEBROW_RAISE_THRESHOLD : float = 0.17
    EYE_SQUEEZE_THRESHOLD   : float = 0.12
    HEAD_TILT_THRESHOLD_DEG : float = 12.0


# =============================================================
#  Landmark Indices  (MediaPipe FaceMesh — 468 points)
# =============================================================

class Landmarks:
  """Key landmark indices for expression checks."""
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


# =============================================================
#  Main Detector Class
# =============================================================

class FaceDetectorValidator:
    """
    Full face detection + validation pipeline using MediaPipe Tasks API.
    Uses MediaPipe Tasks API (compatible with Python 3.12 / Kaggle).

    Usage:
        detector = FaceDetectorValidator()
        result   = detector.process("photo.jpg")

        if result.is_valid:
            # result.face_crop       → pass to FacePartExtractor
            # result.crop_landmarks  → pass to FacePartExtractor
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self._init_mediapipe()

    def _init_mediapipe(self):
        """Download model if needed, then initialize the FaceLandmarker."""
        import os, urllib.request

        model_filename = "face_landmarker.task"
        if not os.path.exists(model_filename):
            print("Downloading face_landmarker.task ...")
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
            urllib.request.urlretrieve(url, model_filename)

        base_options = python.BaseOptions(model_asset_path=model_filename)
        options = vision.FaceLandmarkerOptions(
            base_options                  = base_options,
            running_mode                  = vision.RunningMode.IMAGE,
            num_faces                     = 2,   # detect up to 2 → reject if >1
            min_face_detection_confidence = self.config.DETECTION_CONFIDENCE,
            min_face_presence_confidence  = self.config.MESH_DETECTION_CONF,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        print("MediaPipe FaceLandmarker initialized.")

    # ----------------------------------------------------------
    #  Public API
    # ----------------------------------------------------------

    def process(self, image_input, check_expression=False) -> FaceValidationResult:
        """
        Run the full validation pipeline.

        Args:
            image_input : file path (str) OR numpy BGR array

        Returns:
            FaceValidationResult
              → check .is_valid first
              → if True, use .face_crop and .crop_landmarks
        """
        # Step 1: Load image
        image_bgr = self._load_image(image_input)
        if image_bgr is None:
            return self._fail(ValidationStatus.IMAGE_TOO_SMALL, "Could not load image — check the file path.")

        full_h, full_w = image_bgr.shape[:2]

        # Step 2: Build MediaPipe image object
        if isinstance(image_input, str):
            mp_image = mp.Image.create_from_file(image_input)
        else:
            mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        # Step 3: Run landmark detection
        detection_result = self.landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return self._fail(ValidationStatus.NO_FACE_DETECTED, "No human face detected.")

        if len(detection_result.face_landmarks) > 1:
            return self._fail(ValidationStatus.MULTIPLE_FACES, f"{len(detection_result.face_landmarks)} faces detected. " "Exactly one face required.")

        # Step 4: Compute face bounding box from landmark extents
        raw_landmarks = detection_result.face_landmarks[0]
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")
      
        for lm in raw_landmarks:
          x = int(lm.x * full_w) 
          y = int(lm.y * full_h)

          min_x = min(min_x, x)
          max_x = max(max_x, x)

          min_y = min(min_y, y)
          max_y = max(max_y, y)

        fbw = max_x - min_x
        fbh = max_y - min_y
        face_bbox = (int(min_x), int(min_y), fbw, fbh)

        # Step 5: Minimum face size check
        if fbw < self.config.MIN_FACE_SIZE_PX or fbh < self.config.MIN_FACE_SIZE_PX:
            return self._fail(ValidationStatus.IMAGE_TOO_SMALL, f"Face too small ({fbw}×{fbh} px)." 
                              f"Minimum: {self.config.MIN_FACE_SIZE_PX} px.")

        # Step 6: Expression check (uses raw_landmarks + full image dimensions)
        if check_expression:
          expr = self._check_expression(raw_landmarks, full_w, full_h)
          expression_scores = expr["scores"]
          if expr["has_expression"]:
            return self._fail(ValidationStatus.EXPRESSION_DETECTED, f"Expression detected: {expr['reason']}. "
                              "Please use a neutral, relaxed face.", expression_scores=expression_scores)
        else:
          expression_scores  = None
 
        # Step 7: Crop face
        face_crop = self._crop_face(image_bgr, face_bbox)
        crop_h, crop_w = face_crop.shape[:2]

        # ── Step 8: Remap landmarks → crop-pixel coordinates ──────
        #
        #  Why here (in Part 1) instead of Part 2?
        #  ─────────────────────────────────────────
        #  MediaPipe gives us normalized coords (0.0–1.0) relative
        #  to the FULL original image. After we crop, those coords
        #  are wrong for the crop.
        #
        #  We do the remapping once here, right after cropping,
        #  while we still have , full_h, and face_bbox.
        #  FacePartExtractor then receives clean pixel coords and
        #  needs zero coordinate math of its own.
        #
        #  Formula per landmark:
        #    pixel_x_in_full  = landmark.x * full_w
        #    pixel_x_in_crop  = pixel_x_in_full - crop_x1
        #
        #  crop_x1 / crop_y1 are the top-left of the PADDED crop
        #  (computed inside _crop_face). We recalculate them here
        #  using the same formula so the coords stay in sync.
        # ──────────────────────────────────────────────────────────

        # Recalculate the padded crop origin (must match _crop_face exactly)
        pad_x  = int(fbw * self.config.CROP_PADDING_RATIO)
        pad_y  = int(fbh * self.config.CROP_PADDING_RATIO)
        crop_x1 = max(0, min_x - pad_x)
        crop_y1 = max(0, min_y - pad_y)

        crop_landmarks: List[Tuple[int, int]] = []
        for lm in raw_landmarks:
            # 1. Normalized → pixel in full image
            px_full = lm.x * full_w
            py_full = lm.y * full_h

            # 2. Shift by crop origin → pixel in crop
            px_crop = int(px_full) - crop_x1
            py_crop = int(py_full) - crop_y1

            # 3. Clamp to crop bounds (safety)
            px_crop = max(0, min(px_crop, crop_w - 1))
            py_crop = max(0, min(py_crop, crop_h - 1))

            crop_landmarks.append((px_crop, py_crop))

        # Step 9: Draw debug image
        annotated = self._draw_debug(image_bgr.copy(), face_bbox, 1.0, raw_landmarks, full_w, full_h)

        return FaceValidationResult(status = ValidationStatus.VALID,
                                    is_valid = True,
                                    message = "Valid face detected. Ready for physiognomy analysis.",
                                    face_crop = face_crop,
                                    face_bbox = face_bbox,
                                    detection_score = 1.0,
                                    expression_scores = expression_scores,
                                    annotated_image = annotated,
                                    crop_landmarks = crop_landmarks,   # ← NEW: ready for Part 2
                                    #raw_landmarks  = raw_landmarks
                                   )

    # ----------------------------------------------------------
    #  Expression Check
    # ----------------------------------------------------------

    def _check_expression(self, landmarks, img_w: int, img_h: int) -> dict:
        """
        5 geometric metrics to detect non-neutral expressions.
        All distances normalized by face height → scale-invariant.
        """
        cfg = self.config

        def px(idx):
            return np.array([landmarks[idx].x * img_w, landmarks[idx].y * img_h])

        face_height = np.linalg.norm(px(Landmarks.FOREHEAD_TOP) - px(Landmarks.CHIN_BOTTOM))
        face_width  = np.linalg.norm(px(Landmarks.FACE_LEFT) - px(Landmarks.FACE_RIGHT))
        if face_height < 1: face_height = 1

        mouth_open = np.linalg.norm(px(Landmarks.UPPER_LIP_TOP) - px(Landmarks.LOWER_LIP_BOTTOM)) / face_height
        mouth_width = np.linalg.norm(px(Landmarks.MOUTH_LEFT) - px(Landmarks.MOUTH_RIGHT))
        mouth_stretch = mouth_width / face_width if face_width > 1 else 0

        r_brow = np.linalg.norm(px(Landmarks.RIGHT_BROW_CENTER) - px(Landmarks.RIGHT_EYE_TOP)) / face_height
        l_brow = np.linalg.norm(px(Landmarks.LEFT_BROW_CENTER) - px(Landmarks.LEFT_EYE_TOP)) / face_height
        brow_raise = max(r_brow, l_brow)

        def ear(top, bot, left, right):
            h = np.linalg.norm(px(top) - px(bot))
            w = np.linalg.norm(px(left) - px(right))
            return h / w if w > 0 else 1.0

        min_ear = min(
            ear(Landmarks.RIGHT_EYE_TOP, Landmarks.RIGHT_EYE_BOTTOM,
                Landmarks.RIGHT_EYE_LEFT, Landmarks.RIGHT_EYE_RIGHT),
            ear(Landmarks.LEFT_EYE_TOP,  Landmarks.LEFT_EYE_BOTTOM,
                Landmarks.LEFT_EYE_LEFT, Landmarks.LEFT_EYE_RIGHT)
            )

        eye_l    = px(Landmarks.LEFT_EYE_RIGHT)
        eye_r    = px(Landmarks.RIGHT_EYE_LEFT)
        delta    = eye_r - eye_l
        tilt_deg = np.degrees(np.arctan(abs(delta[1]) / abs(delta[0])) if abs(delta[0]) > 0 else 0)

        scores = {
            "mouth_open"    : round(float(mouth_open),    4),
            "mouth_stretch" : round(float(mouth_stretch), 4),
            "brow_raise"    : round(float(brow_raise),    4),
            "min_eye_EAR"   : round(float(min_ear),       4),
            "head_tilt_deg" : round(float(tilt_deg),      2),
        }

        reasons = []
        if mouth_open > cfg.MOUTH_OPEN_THRESHOLD:
            reasons.append(f"mouth open ({scores['mouth_open']:.3f} > {cfg.MOUTH_OPEN_THRESHOLD})")
        if mouth_stretch > cfg.MOUTH_STRETCH_THRESHOLD:
            reasons.append(f"wide smile ({scores['mouth_stretch']:.3f} > {cfg.MOUTH_STRETCH_THRESHOLD})")
        if brow_raise > cfg.EYEBROW_RAISE_THRESHOLD:
            reasons.append(f"raised brows ({scores['brow_raise']:.3f} > {cfg.EYEBROW_RAISE_THRESHOLD})")
        if min_ear < cfg.EYE_SQUEEZE_THRESHOLD:
            reasons.append(f"eye squint (EAR {scores['min_eye_EAR']:.3f} < {cfg.EYE_SQUEEZE_THRESHOLD})")
        if tilt_deg > cfg.HEAD_TILT_THRESHOLD_DEG:
            reasons.append(f"head tilt {scores['head_tilt_deg']}° > {cfg.HEAD_TILT_THRESHOLD_DEG}°")

        return {
            "has_expression" : len(reasons) > 0,
            "reason"         : " | ".join(reasons) if reasons else None,
            "scores"         : scores,
        }

    # ----------------------------------------------------------
    #  Crop Face
    # ----------------------------------------------------------

    def _crop_face(self, image_bgr: np.ndarray,
                   bbox: Tuple[int,int,int,int]) -> np.ndarray:
        """Crop face with padding. Must stay in sync with crop origin in process()."""
        x, y, bw, bh = bbox
        h, w = image_bgr.shape[:2]
        pad_x = int(bw * self.config.CROP_PADDING_RATIO)
        pad_y = int(bh * self.config.CROP_PADDING_RATIO)
        x1 = max(0, x - pad_x);  y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x); y2 = min(h, y + bh + pad_y)
        return image_bgr[y1:y2, x1:x2]

    # ----------------------------------------------------------
    #  Debug Drawing
    # ----------------------------------------------------------

    def _draw_detection(self, image, bbox, confidence):
        x, y, bw, bh = bbox
        color = (0, 200, 0)
        cv2.rectangle(image, (x, y), (x+bw, y+bh), color, 2)
        cv2.putText(image, f"Face {confidence:.0%}",
                    (x, max(y-8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return image

    def _draw_debug(self, image, bbox, confidence, landmarks, img_w, img_h):
        image = self._draw_detection(image, bbox, confidence)
        key_indices = [
            Landmarks.UPPER_LIP_TOP, Landmarks.LOWER_LIP_BOTTOM,
            Landmarks.MOUTH_LEFT, Landmarks.MOUTH_RIGHT,
            Landmarks.RIGHT_EYE_TOP, Landmarks.RIGHT_EYE_BOTTOM,
            Landmarks.LEFT_EYE_TOP,  Landmarks.LEFT_EYE_BOTTOM,
            Landmarks.RIGHT_BROW_CENTER, Landmarks.LEFT_BROW_CENTER,
            Landmarks.CHIN_BOTTOM, Landmarks.FOREHEAD_TOP,
        ]
        for idx in key_indices:
            lm = landmarks[idx]
            cv2.circle(image,
                       (int(lm.x * img_w), int(lm.y * img_h)),
                       3, (0, 255, 255), -1)
        return image

    # ----------------------------------------------------------
    #  Helpers
    # ----------------------------------------------------------

    def _load_image(self, image_input):
        if isinstance(image_input, str):
            return cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            return image_input.copy()
        return None

    def _fail(self, status, message, expression_scores=None):
        return FaceValidationResult(
            status=status, is_valid=False,
            message=message, expression_scores=expression_scores
        )

    def __del__(self):
        if hasattr(self, 'landmarker'):
            try:
                self.landmarker.close()
            except Exception:
                pass
