"""
=============================================================
  Face Physiognomy Project — Face Section Builder
=============================================================

WHAT THIS FILE DOES:
  Takes the front face crop + landmarks (required)
  and optionally a profile image + side label.

  Divides both into 3 sections using landmark boundaries,
  then combines each front section with its corresponding
  profile section (if available) into one image separated
  by a thick red horizontal line.

  Also calculates geometric measurements per section
  to pass as context to the VLM prompts.

SECTIONS:
  Section 1 — Upper  : forehead + eyes + nose bridge
  Section 2 — Middle : eyebrows + eyes + nose + cheeks
  Section 3 — Lower  : nose tip + mouth + jaw + chin

  Overlap between sections is intentional — gives the VLM
  enough context to understand proportions.

BOUNDARIES (all in face_crop pixel space):

  Horizontal boundaries (y axis):
    Section 1 top    : landmark[10].y  − 30% of face_height
                       (clamped to 0)
    Section 1 bottom : landmark[168].y (nose bridge)

    Section 2 top    : landmark[70].y  (right eyebrow top)
                       or landmark[300].y (left eyebrow top) — whichever is higher
    Section 2 bottom : landmark[17].y  (mouth bottom / chin top area)

    Section 3 top    : landmark[4].y   (nose tip)
    Section 3 bottom : landmark[152].y + 10% face_height
                       (clamped to crop height)

  Vertical boundaries (x axis) — same for all 3 sections:
    Left  : landmark[234].x − 20% of face_width
             (clamped to 0)
    Right : landmark[454].x + 20% of face_width
             (clamped to crop width)

PROFILE MERGE:
  If profile image is available:
    combined = vstack([front_section, RED_LINE, profile_section])
    The red line (8px thick) visually separates front from profile.

  If no profile:
    combined = front_section only

OUTPUT:
  {
    "section_1": np.ndarray,   # combined image (front [+ profile])
    "section_2": np.ndarray,
    "section_3": np.ndarray,
    "measurements": {          # geometric measurements from landmarks
      "section_1": {...},
      "section_2": {...},
      "section_3": {...},
    }
  }
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List


# =============================================================
#  Red separator line
# =============================================================

RED_LINE_THICKNESS = 8    # pixels
RED_COLOR          = (0, 0, 255)   # BGR


def _make_red_line(width: int) -> np.ndarray:
    """Create a solid red horizontal separator line."""
    line = np.zeros((RED_LINE_THICKNESS, width, 3), dtype=np.uint8)
    line[:] = RED_COLOR
    return line


# =============================================================
#  Landmark helper
# =============================================================

def _lm(crop_landmarks: List[Tuple[int, int]], idx: int) -> Tuple[int, int]:
    """Get (x, y) for landmark index. Safe — returns (0,0) if out of range."""
    if idx < len(crop_landmarks):
        return crop_landmarks[idx]
    return (0, 0)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


# =============================================================
#  Section Boundaries
# =============================================================

def compute_section_boundaries(
    crop_landmarks : List[Tuple[int, int]],
    crop_h         : int,
    crop_w         : int,
) -> Dict:
    """
    Compute pixel boundaries for all 3 sections.

    Returns dict with keys: x_left, x_right, s1_top, s1_bot,
                                              s2_top, s2_bot,
                                              s3_top, s3_bot
    """
    # ── Reference measurements ────────────────────────────────
    lm10  = _lm(crop_landmarks, 10)    # top of forehead
    lm152 = _lm(crop_landmarks, 152)   # chin bottom
    lm234 = _lm(crop_landmarks, 234)   # left jaw hinge
    lm454 = _lm(crop_landmarks, 454)   # right jaw hinge

    face_height = max(lm152[1] - lm10[1], 1)
    face_width  = max(lm454[0] - lm234[0], 1)

    # ── Shared x boundaries (same for all sections) ───────────
    x_left  = _clamp(lm234[0] - int(face_width * 0.20), 0, crop_w - 1)
    x_right = _clamp(lm454[0] + int(face_width * 0.20), 0, crop_w - 1)

    # ── Section 1: Upper (forehead → nose bridge) ─────────────
    lm168 = _lm(crop_landmarks, 168)   # nose bridge top

    s1_top = _clamp(lm10[1] - int(face_height * 0.30), 0, crop_h - 1)
    s1_bot = _clamp(lm168[1], 0, crop_h - 1)

    # ── Section 2: Middle (eyebrows → mouth bottom) ───────────
    lm70  = _lm(crop_landmarks, 70)    # right eyebrow top
    lm300 = _lm(crop_landmarks, 300)   # left eyebrow top
    lm17  = _lm(crop_landmarks, 17)    # mouth bottom

    # Use the higher of the two eyebrow tops (smaller y = higher in image)
    s2_top = _clamp(min(lm70[1], lm300[1]), 0, crop_h - 1)
    s2_bot = _clamp(lm17[1], 0, crop_h - 1)

    # ── Section 3: Lower (nose tip → chin) ────────────────────
    lm4   = _lm(crop_landmarks, 4)     # nose tip

    s3_top = _clamp(lm4[1], 0, crop_h - 1)
    s3_bot = _clamp(lm152[1] + int(face_height * 0.10), 0, crop_h - 1)

    return {
        "x_left" : x_left,
        "x_right": x_right,
        "s1_top" : s1_top,  "s1_bot": s1_bot,
        "s2_top" : s2_top,  "s2_bot": s2_bot,
        "s3_top" : s3_top,  "s3_bot": s3_bot,
        "face_height": face_height,
        "face_width" : face_width,
    }


# =============================================================
#  Crop Section from Image
# =============================================================

def _crop_section(
    image   : np.ndarray,
    y_top   : int,
    y_bot   : int,
    x_left  : int,
    x_right : int,
) -> np.ndarray:
    """Crop a rectangular section from image. Returns empty array if invalid."""
    if y_bot <= y_top or x_right <= x_left:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return image[y_top:y_bot, x_left:x_right].copy()


# =============================================================
#  Geometric Measurements
# =============================================================

def _dist(p1: Tuple[int,int], p2: Tuple[int,int]) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def compute_measurements(
    crop_landmarks : List[Tuple[int, int]],
    bounds         : Dict,
) -> Dict:
    """
    Calculate geometric ratios from landmarks.
    These are passed to the VLM prompts as ground-truth measurements
    so the model does not need to guess proportions.

    All ratios are relative to face_height or face_width
    to be scale-invariant.
    """
    fh = bounds["face_height"]
    fw = bounds["face_width"]

    def lm(idx):
        return _lm(crop_landmarks, idx)

    # ── Section 1 measurements ────────────────────────────────
    # Eye spacing: distance between inner eye corners / eye width
    r_inner  = lm(133);  l_inner  = lm(362)
    r_outer  = lm(33);   l_outer  = lm(263)
    r_top    = lm(159);  r_bot    = lm(145)
    l_top    = lm(386);  l_bot    = lm(374)

    between_eyes = _dist(r_inner, l_inner)
    r_eye_width  = _dist(r_inner, r_outer)
    eye_spacing_ratio = round(between_eyes / r_eye_width, 2) if r_eye_width > 0 else 0

    # Eye angle: outer corner y vs inner corner y (per eye)
    def eye_angle(inner, outer):
        dy = outer[1] - inner[1]   # positive = outer is lower
        if dy < -3:   return "upward"
        if dy >  3:   return "downward"
        return "level"

    # Forehead proportion: forehead height / face height
    lm10  = lm(10);  lm70  = lm(70);  lm300 = lm(300)
    forehead_h = abs(min(lm70[1], lm300[1]) - lm10[1])
    forehead_ratio = round(forehead_h / fh, 2) if fh > 0 else 0

    # Iris size approximation: eye opening height / eye width
    r_eye_h   = _dist(r_top, r_bot)
    iris_ratio = round(r_eye_h / r_eye_width, 2) if r_eye_width > 0 else 0

    section_1 = {
        "eye_spacing"        : eye_spacing_ratio,
        "eye_spacing_label"  : "wide" if eye_spacing_ratio > 1.1 else ("close" if eye_spacing_ratio < 0.9 else "average"),
        "right_eye_angle"    : eye_angle(r_inner, r_outer),
        "left_eye_angle"     : eye_angle(l_inner, l_outer),
        "iris_size_ratio"    : iris_ratio,
        "iris_size_label"    : "large" if iris_ratio > 0.35 else "small",
        "forehead_height_ratio": forehead_ratio,
        "forehead_size"      : "high" if forehead_ratio > 0.35 else ("low" if forehead_ratio < 0.22 else "medium"),
    }

    # ── Section 2 measurements ────────────────────────────────
    # Nose width ratio
    l_nostril = lm(129);  r_nostril = lm(358)
    nose_width = _dist(l_nostril, r_nostril)
    nose_width_ratio = round(nose_width / fw, 2) if fw > 0 else 0

    # Nose length ratio (bridge to tip)
    lm168 = lm(168);  lm4 = lm(4)
    nose_length = _dist(lm168, lm4)
    nose_length_ratio = round(nose_length / fh, 2) if fh > 0 else 0

    # Eye white showing: compare iris center y to eye center y
    r_eye_center_y = (r_top[1] + r_bot[1]) / 2
    r_iris_y       = lm(159)[1]   # approximate iris center
    white_offset   = r_iris_y - r_eye_center_y

    if white_offset > 4:
        white_showing = "white_above_iris"
    elif white_offset < -4:
        white_showing = "white_below_iris"
    else:
        white_showing = "moderate"

    section_2 = {
        "nose_width_ratio"  : nose_width_ratio,
        "nose_width_label"  : "wide" if nose_width_ratio > 0.38 else ("thin" if nose_width_ratio < 0.25 else "average"),
        "nose_large"        : nose_width_ratio > 0.38 or nose_length_ratio > 0.38,
        "nose_length_ratio" : nose_length_ratio,
        "nose_length_label" : "long" if nose_length_ratio > 0.38 else ("short" if nose_length_ratio < 0.25 else "average"),
        "eye_white_showing" : white_showing,
    }

    # ── Section 3 measurements ────────────────────────────────
    # Mouth width ratio
    m_left  = lm(61);   m_right = lm(291)
    mouth_width = _dist(m_left, m_right)
    mouth_width_ratio = round(mouth_width / fw, 2) if fw > 0 else 0

    # Mouth angle: center of mouth vs corners
    m_center  = lm(13)
    corners_avg_y = (m_left[1] + m_right[1]) / 2
    dy = m_center[1] - corners_avg_y
    if dy < -3:   mouth_angle = "turns_up"
    elif dy > 3:  mouth_angle = "turns_down"
    else:         mouth_angle = "straight"

    # Jaw width ratio
    lm234 = lm(234);  lm454 = lm(454)
    jaw_width = _dist(lm234, lm454)
    jaw_width_ratio = round(jaw_width / fw, 2) if fw > 0 else 0

    # Chin length ratio (mouth bottom to chin tip)
    lm17  = lm(17);   lm152 = lm(152)
    chin_length = _dist(lm17, lm152)
    chin_length_ratio = round(chin_length / fh, 2) if fh > 0 else 0

    # Chin width ratio
    lm136 = lm(136);  lm365 = lm(365)
    chin_width = _dist(lm136, lm365)
    chin_width_ratio = round(chin_width / fw, 2) if fw > 0 else 0

    # Facial sections dominance (top/middle/bottom)
    lm10_y  = lm(10)[1]
    lm70_y  = min(lm(70)[1], lm(300)[1])
    lm4_y   = lm(4)[1]
    lm152_y = lm(152)[1]

    top_h    = max(lm70_y  - lm10_y,  1)
    mid_h    = max(lm4_y   - lm70_y,  1)
    bot_h    = max(lm152_y - lm4_y,   1)
    total_h  = top_h + mid_h + bot_h

    section_3 = {
        "mouth_width_ratio" : mouth_width_ratio,
        "mouth_size_label"  : "large" if mouth_width_ratio > 0.48 else ("small" if mouth_width_ratio < 0.32 else "average"),
        "mouth_angle"       : mouth_angle,
        "jaw_width_ratio"   : jaw_width_ratio,
        "jaw_size_label"    : "wide" if jaw_width_ratio > 0.95 else ("narrow" if jaw_width_ratio < 0.75 else "average"),
        "chin_length_ratio" : chin_length_ratio,
        "chin_long"         : chin_length_ratio > 0.22,
        "chin_width_ratio"  : chin_width_ratio,
        "chin_width_label"  : "broad" if chin_width_ratio > 0.55 else ("small" if chin_width_ratio < 0.35 else "average"),
        "facial_sections"   : {
            "top_ratio"    : round(top_h  / total_h, 2),
            "middle_ratio" : round(mid_h  / total_h, 2),
            "bottom_ratio" : round(bot_h  / total_h, 2),
            "dominant"     : "top" if top_h > mid_h and top_h > bot_h
                             else ("middle" if mid_h > bot_h else "bottom"),
        }
    }

    return {
        "section_1": section_1,
        "section_2": section_2,
        "section_3": section_3,
    }


# =============================================================
#  Profile Section Splitter
# =============================================================

def split_profile_into_sections(
    profile_img    : np.ndarray,
    front_bounds   : Dict,
    front_crop_h   : int,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Split profile image into 3 sections using the same proportional
    boundaries as the front image.

    WHY proportional?
      The profile image has no landmarks (MediaPipe is unreliable
      on profile faces). So we use the same vertical ratios from
      the front face to estimate where the sections are.

    The profile image is resized to the same height as the front
    crop first, then split using the same y ratios.
    """
    if profile_img is None:
        return {"section_1": None, "section_2": None, "section_3": None}

    ph, pw = profile_img.shape[:2]

    # Resize profile to match front crop height
    scale  = front_crop_h / ph if ph > 0 else 1.0
    new_w  = int(pw * scale)
    resized = cv2.resize(profile_img, (new_w, front_crop_h))

    fh = front_bounds["face_height"]
    if fh <= 0:
        return {"section_1": None, "section_2": None, "section_3": None}

    # Use same y ratios as front sections
    s1_top = front_bounds["s1_top"]
    s1_bot = front_bounds["s1_bot"]
    s2_top = front_bounds["s2_top"]
    s2_bot = front_bounds["s2_bot"]
    s3_top = front_bounds["s3_top"]
    s3_bot = front_bounds["s3_bot"]

    def safe_crop(img, y1, y2):
        h = img.shape[0]
        y1 = _clamp(y1, 0, h - 1)
        y2 = _clamp(y2, 0, h)
        if y2 <= y1:
            return None
        return img[y1:y2, :].copy()

    return {
        "section_1": safe_crop(resized, s1_top, s1_bot),
        "section_2": safe_crop(resized, s2_top, s2_bot),
        "section_3": safe_crop(resized, s3_top, s3_bot),
    }


# =============================================================
#  Combine Front + Profile
# =============================================================

TARGET_WIDTH = 512   # resize all sections to this width before merging

def _resize_to_width(img: np.ndarray, width: int) -> np.ndarray:
    """Resize image to target width, keep aspect ratio."""
    if img is None or img.size == 0:
        return np.zeros((1, width, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    if w == 0:
        return np.zeros((1, width, 3), dtype=np.uint8)
    scale  = width / w
    new_h  = max(1, int(h * scale))
    return cv2.resize(img, (width, new_h))


def combine_sections(
    front_section   : np.ndarray,
    profile_section : Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Combine front section (top) and profile section (bottom)
    with a thick red horizontal line between them.

    If no profile: return front section only.

    Both sections are resized to TARGET_WIDTH before combining
    so the final image has consistent dimensions.
    """
    front_resized = _resize_to_width(front_section, TARGET_WIDTH)

    if profile_section is None or profile_section.size == 0:
        return front_resized

    profile_resized = _resize_to_width(profile_section, TARGET_WIDTH)
    red_line        = _make_red_line(TARGET_WIDTH)

    combined = np.vstack([front_resized, red_line, profile_resized])
    return combined


# =============================================================
#  Main Builder Class
# =============================================================

@dataclass
class SectionBuilderResult:
    """
    Output of FaceSectionBuilder.build().

    sections     : {"section_1": img, "section_2": img, "section_3": img}
    measurements : {"section_1": {...}, "section_2": {...}, "section_3": {...}}
    has_profile  : True if profile image was provided and used
    """
    sections     : Dict[str, np.ndarray]
    measurements : Dict[str, Dict]
    has_profile  : bool


class FaceSectionBuilder:
    """
    Divides front (and optionally profile) face into 3 sections
    for VLM analysis.

    Usage:
        builder = FaceSectionBuilder(
            face_crop      = result.face_crop,
            crop_landmarks = result.crop_landmarks,
            profile_img    = profile_numpy_array,   # or None
            profile_side   = "left",                # or "right" or None
        )
        result = builder.build()

        # Pass to FaceDescriber:
        section_1_img = result.sections["section_1"]
        measurements  = result.measurements
    """

    def __init__(
        self,
        face_crop      : np.ndarray,
        crop_landmarks : List[Tuple[int, int]],
        profile_img    : Optional[np.ndarray] = None,
        profile_side   : Optional[str]        = None,   # "left" or "right"
    ):
        self.face_crop      = face_crop
        self.crop_landmarks = crop_landmarks
        self.profile_img    = profile_img
        self.profile_side   = profile_side

        self.crop_h, self.crop_w = face_crop.shape[:2]

    def build(self) -> SectionBuilderResult:
        """
        Run the full section building pipeline.

        Steps:
          1. Compute section boundaries from landmarks
          2. Crop 3 front sections
          3. Split profile into 3 sections (if available)
          4. Combine front + profile per section
          5. Compute geometric measurements
        """
        # Step 1: Compute boundaries
        bounds = compute_section_boundaries(
            self.crop_landmarks, self.crop_h, self.crop_w
        )

        # Step 2: Crop front sections
        def crop_front(y_top_key, y_bot_key):
            return _crop_section(
                self.face_crop,
                bounds[y_top_key], bounds[y_bot_key],
                bounds["x_left"],  bounds["x_right"],
            )

        front_s1 = crop_front("s1_top", "s1_bot")
        front_s2 = crop_front("s2_top", "s2_bot")
        front_s3 = crop_front("s3_top", "s3_bot")

        # Step 3: Split profile (if available)
        profile_sections = split_profile_into_sections(
            self.profile_img, bounds, self.crop_h
        )

        # Step 4: Combine
        section_1 = combine_sections(front_s1, profile_sections["section_1"])
        section_2 = combine_sections(front_s2, profile_sections["section_2"])
        section_3 = combine_sections(front_s3, profile_sections["section_3"])

        # Step 5: Measurements
        measurements = compute_measurements(self.crop_landmarks, bounds)

        return SectionBuilderResult(
            sections={
                "section_1": section_1,
                "section_2": section_2,
                "section_3": section_3,
            },
            measurements = measurements,
            has_profile  = self.profile_img is not None,
        )



