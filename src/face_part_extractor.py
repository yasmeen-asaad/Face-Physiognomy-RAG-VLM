"""
=============================================================
  Face Physiognomy Project — Part 2: Face Parts Extraction
=============================================================

INPUT  : face crop (numpy array) + MediaPipe FaceMesh landmarks
         comes directly from FaceDetectorValidator output

OUTPUT : cropped image per facial region, ready for Vision-LLM

Regions extracted:
  - whole_face   (full crop, baseline reference)
  - forehead     (hairline to eyebrow tops)
  - eyebrows     (left + right combined)
  - eyes         (left + right combined, with eyelids)
  - nose         (bridge to nostrils)
  - mouth        (lips + surrounding area)
  - jaw_chin     (jawline + chin)
  - ears         (optional, low confidence)

Why we accept landmarks + crop (not the full image):
  FaceDetectorValidator already ran detection and cropping.
  All landmark coordinates from FaceMesh are normalized (0.0-1.0)
  RELATIVE to the image passed to it - which was the full original image.
  So we need to remap them onto the crop coordinates.
  This class handles that remapping transparently.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple


# =============================================================
#  Data Classes
# =============================================================

@dataclass
class FacePartResult:
    """
    Result for a single facial region.

    Attributes:
        region_name      : human-readable name ("forehead", "eyes", etc.)
        image            : cropped numpy BGR image of this region
        bbox             : (x, y, w, h) in the face-crop coordinate space
        landmarks_used   : list of MediaPipe landmark indices used
        confidence_score : 1.0 = high confidence, 0.4 = low (ears)
        notes            : explanation of any caveats
    """
    region_name      : str
    image            : np.ndarray
    bbox             : Tuple[int, int, int, int]
    landmarks_used   : List[int]
    confidence_score : float = 1.0
    notes            : str   = ""


@dataclass
class AllPartsResult:
    """
    Container returned by extract_all_parts().
    Each field is a FacePartResult (or None if extraction failed).
    """
    whole_face : Optional[FacePartResult] = None
    forehead   : Optional[FacePartResult] = None
    eyebrows   : Optional[FacePartResult] = None
    eyes       : Optional[FacePartResult] = None
    nose       : Optional[FacePartResult] = None
    mouth      : Optional[FacePartResult] = None
    jaw_chin   : Optional[FacePartResult] = None
    ears       : Optional[FacePartResult] = None

    def to_dict(self) -> Dict[str, Optional[FacePartResult]]:
        """Convert to plain dict - useful for iterating or serialization."""
        return {
            "whole_face" : self.whole_face,
            "forehead"   : self.forehead,
            "eyebrows"   : self.eyebrows,
            "eyes"       : self.eyes,
            "nose"       : self.nose,
            "mouth"      : self.mouth,
            "jaw_chin"   : self.jaw_chin,
            "ears"       : self.ears,
        }

    def valid_parts(self) -> Dict[str, FacePartResult]:
        """Return only the parts that were successfully extracted."""
        return {k: v for k, v in self.to_dict().items() if v is not None}


# =============================================================
#  Landmark Index Reference
# =============================================================
#
#  MediaPipe Face Mesh = 468 landmarks (+ 10 iris with refine=True)
#  All coordinates normalized: x,y in [0.0, 1.0], origin = TOP-LEFT
#
#  Reference map:
#  https://developers.google.com/mediapipe/solutions/vision/face_landmarker
#
#  Key principle: pick BOUNDARY landmarks (outermost points of a region)
#  then compute bounding box around them + padding.

class LM:
    """Landmark indices organized by facial region."""

    # ── Whole face silhouette ──────────────────────────────────────
    FACE_SILHOUETTE = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
    ]

    # ── Forehead ──────────────────────────────────────────────────
    # TOP boundary: upper silhouette (hairline area)
    FOREHEAD_TOP    = [10, 109, 67, 103, 54, 21, 162, 127]
    # BOTTOM boundary: top edge of eyebrows
    FOREHEAD_BOTTOM = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]

    # ── Eyebrows ──────────────────────────────────────────────────
    # Right eyebrow (viewer's perspective): upper + lower edge
    RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    # Left eyebrow: upper + lower edge
    LEFT_EYEBROW  = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]

    # ── Eyes ──────────────────────────────────────────────────────
    # Right eye: full eyelid boundary (upper + lower lid)
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155,
                 133, 173, 157, 158, 159, 160, 161, 246]
    # Left eye: full eyelid boundary
    LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249,
                 263, 466, 388, 387, 386, 385, 384, 398]

    # ── Nose ──────────────────────────────────────────────────────
    # Bridge (top) to nostrils (bottom) + ala (wings)
    NOSE_ALL = [
        168, 6, 197, 195, 5, 4,        # centerline: bridge -> tip
        209, 49, 64, 98, 97,            # left nostril wing
        429, 279, 294, 327, 326,        # right nostril wing
        2, 240, 99, 219, 218, 237       # nostril base
    ]

    # ── Mouth ─────────────────────────────────────────────────────
    # Outer lip contour
    MOUTH_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                   375, 321, 405, 314, 17, 84, 181, 91, 146]
    # Inner lip contour (the opening)
    MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                   324, 318, 402, 317, 14, 87, 178, 88, 95]

    # ── Jaw & Chin ────────────────────────────────────────────────
    # Lower silhouette from ear to ear through chin
    JAW_LINE = [
        172, 136, 150, 149, 176, 148, 152,  # chin center
        377, 400, 378, 379, 365, 397, 288,  # right jaw
        58, 132, 93, 234                    # left jaw
    ]

    # ── Ears (estimated only) ────────────────────────────────────
    # MediaPipe does NOT model ears - these are the closest landmarks
    EAR_LEFT_ANCHOR  = 234   # left jaw hinge area
    EAR_RIGHT_ANCHOR = 454   # right jaw hinge area
    TEMPLE_LEFT      = 127
    TEMPLE_RIGHT     = 356


# =============================================================
#  Padding Configuration
# =============================================================

@dataclass
class PaddingConfig:
    """
    Padding added around each region (fraction of bounding box size).
    Tune these if crops cut off feature edges.
    """
    whole_face : float = 0.05
    forehead   : float = 0.15   # extra top padding for hairline
    eyebrows   : float = 0.40   # brows are thin - need generous padding
    eyes       : float = 0.35   # include full eyelids
    nose       : float = 0.20
    mouth      : float = 0.25   # includes philtrum + chin area
    jaw_chin   : float = 0.15
    ears       : float = 0.30


# =============================================================
#  Main Extractor Class
# =============================================================

class FacePartExtractor:
    """
    Extracts cropped image regions for each facial part.

    COORDINATE SYSTEM NOTE:
      MediaPipe landmarks are normalized to the image passed to
      FaceMesh.process() - the FULL original image before cropping.

      We receive:
        face_crop     : the already-cropped face image
        landmarks     : normalized to the FULL original image
        face_bbox     : (x, y, w, h) of the crop in the full image
        full_img_size : (height, width) of the full original image

      Remapping formula for landmark -> crop pixel:
        px_in_crop = (landmark.x * full_w) - crop_x
        py_in_crop = (landmark.y * full_h) - crop_y

    Usage:
        extractor = FacePartExtractor(
            face_crop     = result.face_crop,
            landmarks     = mesh_result.multi_face_landmarks[0].landmark,
            face_bbox     = result.face_bbox,
            full_img_size = (original_h, original_w)
        )
        all_parts = extractor.extract_all_parts()

        # Access a specific part:
        nose_img = all_parts.nose.image
        cv2.imwrite("nose.jpg", nose_img)
    """

    def __init__(self, 
                 face_crop : np.ndarray,
                 landmarks, # mediapipe landmark list
                 face_bbox : Tuple[int,int,int,int], # (x,y,w,h) in full image
                 full_img_size : Tuple[int,int],         # (height, width)
                 padding : PaddingConfig = None):
        self.face_crop      = face_crop
        self.landmarks      = landmarks
        self.face_bbox      = face_bbox
        self.full_h, self.full_w = full_img_size
        self.crop_h, self.crop_w = face_crop.shape[:2]
        self.padding        = padding or PaddingConfig()

    # ----------------------------------------------------------
    #  Coordinate Remapping
    # ----------------------------------------------------------

    def _lm_to_crop_px(self, idx: int) -> Tuple[int, int]:
        """
        Convert landmark index -> pixel in face_crop.

        Why this works:
          landmark.x * full_w  = absolute pixel x in the full image
          - crop_x              = shift to crop-local coordinates
          Clamp to [0, crop_size-1] to handle edge cases.
        """
        lm     = self.landmarks[idx]
        crop_x = self.face_bbox[0]
        crop_y = self.face_bbox[1]

        px = int(lm.x * self.full_w) - crop_x
        py = int(lm.y * self.full_h) - crop_y

        px = max(0, min(px, self.crop_w - 1))
        py = max(0, min(py, self.crop_h - 1))
        return px, py

    def _landmarks_to_points(self, indices: List[int]) -> List[Tuple[int,int]]:
        """Convert a list of landmark indices -> list of (x,y) pixel points."""
        return [self._lm_to_crop_px(i) for i in indices]

    def _points_to_bbox(self, points: List[Tuple[int,int]]) -> Tuple[int,int,int,int]:
        """Axis-aligned bounding box from a list of (x,y) points -> (x,y,w,h)."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x  = min(xs);  y = min(ys)
        w  = max(xs) - x
        h  = max(ys) - y
        return x, y, max(w, 1), max(h, 1)

    def _crop_with_padding(self, bbox : Tuple[int,int,int,int], pad_frac: float) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
        """
        Crop from face_crop with padding, safely clamped.

        Returns (cropped_image, padded_bbox).
        The padded_bbox is relative to face_crop (not the full image).
        """
        x, y, w, h = bbox
        pad_x = int(w * pad_frac)
        pad_y = int(h * pad_frac)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(self.crop_w, x + w + pad_x)
        y2 = min(self.crop_h, y + h + pad_y)

        return self.face_crop[y1:y2, x1:x2], (x1, y1, x2-x1, y2-y1)

    def _make_result(
        self,
        region_name : str,
        indices     : List[int],
        pad_frac    : float,
        confidence  : float = 1.0,
        notes       : str   = ""
    ) -> Optional[FacePartResult]:
        """
        Generic pipeline: indices -> points -> bbox -> padded crop -> result.
        Returns None if the resulting crop is empty.
        """
        points            = self._landmarks_to_points(indices)
        tight_bbox        = self._points_to_bbox(points)
        cropped, pad_bbox = self._crop_with_padding(tight_bbox, pad_frac)

        if cropped.size == 0:
            return None

        return FacePartResult(
            region_name      = region_name,
            image            = cropped,
            bbox             = pad_bbox,
            landmarks_used   = indices,
            confidence_score = confidence,
            notes            = notes
        )

    # ----------------------------------------------------------
    #  Extraction Methods
    # ----------------------------------------------------------

    def extract_whole_face(self) -> Optional[FacePartResult]:
        """
        Full face crop as baseline reference.

        Landmarks: full silhouette (36 points around the face oval).
        Used by LLM as the context image before analysing individual parts.
        """
        return self._make_result(
            region_name = "whole_face",
            indices     = LM.FACE_SILHOUETTE,
            pad_frac    = self.padding.whole_face,
            notes       = "Full face reference. Sent to LLM first for overall context."
        )

    def extract_forehead(self) -> Optional[FacePartResult]:
        """
        Forehead: hairline to top of eyebrows.

        Landmark strategy:
          TOP row    -> upper silhouette (lm 10 = top-center, + lateral points)
          BOTTOM row -> top edge of both eyebrows
          Combining these two rows creates a horizontal band.

        Caveat: MediaPipe does NOT model the hairline explicitly.
          Landmark 10 is the highest point modeled on the forehead.
          Extra upward padding (pad_frac=0.15) estimates the hairline.
          For bald subjects, this may include more scalp than intended.
        """
        return self._make_result(
            region_name = "forehead",
            indices     = LM.FOREHEAD_TOP + LM.FOREHEAD_BOTTOM,
            pad_frac    = self.padding.forehead,
            notes       = (
                "Top boundary from silhouette (lm 10). "
                "Hairline not in FaceMesh - padding adds estimated hairline region."
            )
        )

    def extract_eyebrows(self) -> Optional[FacePartResult]:
        """
        Both eyebrows in one crop.

        Landmark strategy:
          Upper + lower edge landmarks for each brow.
          Both brows combined -> wide horizontal strip.

        Why combine both? Physiognomy compares left/right brow symmetry.
        The LLM sees both brows and can comment on shape, arch, thickness,
        and asymmetry in a single description.

        Generous padding (0.40) because brows are thin structures -
        a tight crop would lose the surrounding skin context.
        """
        return self._make_result(
            region_name = "eyebrows",
            indices     = LM.RIGHT_EYEBROW + LM.LEFT_EYEBROW,
            pad_frac    = self.padding.eyebrows,
            notes       = "Both brows combined for symmetry analysis."
        )

    def extract_eyes(self) -> Optional[FacePartResult]:
        """
        Both eyes in one crop (including full eyelids).

        Landmark strategy:
          Full eyelid contour for each eye (16 points per eye).
          Includes inner corner (towards nose) and outer corner (towards ear).
          Padding ensures eyelashes, brow shadow, and tear ducts are visible.

        Why combine? LLM can compare size, shape, spacing, and symmetry.
        """
        return self._make_result(
            region_name = "eyes",
            indices     = LM.RIGHT_EYE + LM.LEFT_EYE,
            pad_frac    = self.padding.eyes,
            notes       = "Both eyes + full eyelid boundary. Padding includes eyelashes."
        )

    def extract_nose(self) -> Optional[FacePartResult]:
        """
        Nose: from bridge to nostril base.

        Landmark strategy:
          - BRIDGE centerline: lm 168, 6, 197, 195, 5 (top to bottom)
          - TIP: lm 4
          - ALA (wings): left and right nostril outline
          - BASE: nostril bottom edge
          Together these define the full nose from top to bottom and
          across the full nostril width.

        Physiognomy reads: bridge height, tip shape, nostril size/shape,
        nose width relative to face. All visible in this crop.
        """
        return self._make_result(
            region_name = "nose",
            indices     = LM.NOSE_ALL,
            pad_frac    = self.padding.nose,
            notes       = "Bridge (lm 168) to tip (lm 4) to full nostril width."
        )

    def extract_mouth(self) -> Optional[FacePartResult]:
        """
        Mouth: full lip boundary and surrounding skin.

        Landmark strategy:
          - OUTER lip contour: the visible colored lip edges
          - INNER lip contour: the opening between lips
          Padding adds philtrum (above lips) and chin dimple area (below).

        Physiognomy reads: lip fullness, cupid's bow shape, corner angle,
        philtrum length, and overall mouth width.
        """
        return self._make_result(
            region_name = "mouth",
            indices     = LM.MOUTH_OUTER + LM.MOUTH_INNER,
            pad_frac    = self.padding.mouth,
            notes       = "Outer + inner lip contours. Padding adds philtrum and chin area."
        )

    def extract_jaw_chin(self) -> Optional[FacePartResult]:
        """
        Jaw and chin: full lower face.

        Landmark strategy:
          - JAW_LINE: 17-point silhouette from left jaw hinge to right hinge
          - Includes chin bottom (lm 152 = lowest face point)
          - Includes jaw hinges (lm 234 left, lm 454 right = widest jaw points)
          This covers the complete mandible shape.

        Physiognomy reads: jaw width (dominance), chin shape (willpower),
        jaw angle (determination), overall lower face structure.
        """
        return self._make_result(
            region_name = "jaw_chin",
            indices     = LM.JAW_LINE,
            pad_frac    = self.padding.jaw_chin,
            notes       = "Full mandible: jaw hinge (lm 234/454) to chin tip (lm 152)."
        )

    def extract_ears(self) -> Optional[FacePartResult]:
        """
        Ear region — ESTIMATED, LOW CONFIDENCE (0.4).

        Why low confidence:
          MediaPipe FaceMesh does NOT model ear landmarks at all.
          The model was trained on frontal faces where ears are partially
          visible at best. There are no ear-specific landmark indices.

        What we do instead:
          1. Use jaw-hinge landmarks (lm 234 left, lm 454 right) as anchors
             - these are the closest modeled points to where ears attach
          2. Extend outward laterally by ~12% of face width
          3. Extend vertically by ~18% of face width
          This creates an estimated ear bounding box on each side.

        Current output: a single wide crop spanning both ear sides.
        This is intentionally rough - ear reading is secondary in v1.

        V2 plan: use a dedicated ear detector or side-profile support.
        The low confidence_score (0.4) signals to the LLM pipeline
        to either skip this region or caveat its description.
        """
        crop_x = self.face_bbox[0]
        crop_y = self.face_bbox[1]

        left_px,  left_py  = self._lm_to_crop_px(LM.EAR_LEFT_ANCHOR)
        right_px, right_py = self._lm_to_crop_px(LM.EAR_RIGHT_ANCHOR)

        face_width_in_crop = abs(right_px - left_px)
        ear_half_w = int(face_width_in_crop * 0.12)
        ear_half_h = int(face_width_in_crop * 0.18)

        x1 = max(0, left_px  - ear_half_w)
        x2 = min(self.crop_w, right_px + ear_half_w)
        y1 = max(0, min(left_py, right_py) - ear_half_h)
        y2 = min(self.crop_h, max(left_py, right_py) + ear_half_h)

        if x2 <= x1 or y2 <= y1:
            return None

        cropped = self.face_crop[y1:y2, x1:x2]
        if cropped.size == 0:
            return None

        return FacePartResult(
            region_name      = "ears",
            image            = cropped,
            bbox             = (x1, y1, x2-x1, y2-y1),
            landmarks_used   = [LM.EAR_LEFT_ANCHOR, LM.EAR_RIGHT_ANCHOR,
                                 LM.TEMPLE_LEFT,     LM.TEMPLE_RIGHT],
            confidence_score = 0.4,
            notes            = (
                "LOW CONFIDENCE: No ear landmarks in FaceMesh. "
                "Position estimated from jaw-hinge (lm 234/454). "
                "Replace with dedicated ear detector in v2."
            )
        )

    # ----------------------------------------------------------
    #  Convenience Method
    # ----------------------------------------------------------

    def extract_all_parts(self, include_ears: bool = True) -> AllPartsResult:
        """
        Run all extractors and return AllPartsResult.

        Args:
            include_ears : False to skip ear estimation

        Returns:
            AllPartsResult - use .to_dict() to iterate all parts,
                             or .valid_parts() to skip None results.

        Example:
            parts = extractor.extract_all_parts()
            for name, part in parts.valid_parts().items():
                print(f"{name}: {part.image.shape}, conf={part.confidence_score}")
                cv2.imwrite(f"{name}.jpg", part.image)
        """
        return AllPartsResult(
            whole_face = self.extract_whole_face(),
            forehead   = self.extract_forehead(),
            eyebrows   = self.extract_eyebrows(),
            eyes       = self.extract_eyes(),
            nose       = self.extract_nose(),
            mouth      = self.extract_mouth(),
            jaw_chin   = self.extract_jaw_chin(),
            ears       = self.extract_ears() if include_ears else None,
        )
