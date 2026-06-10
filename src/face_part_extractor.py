"""
=============================================================
  Face Physiognomy Project — Part 2: Face Parts Extraction
=============================================================
  [UPDATED: Accepts crop_landmarks from FaceDetectorValidator]
  [No coordinate remapping needed — landmarks already in crop space]

INPUT:
  face_crop      : numpy BGR array  ← result.face_crop
  crop_landmarks : List[(x,y)]      ← result.crop_landmarks
                   index N = pixel position of landmark N in the crop

OUTPUT:
  AllPartsResult — one FacePartResult per facial region,
                   each containing image crop + bbox + metadata
                   ready for Vision-LLM description.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


# =============================================================
#  Data Classes
# =============================================================

@dataclass
class FacePartResult:
    """
    Result for one facial region.

    Fields:
        region_name      : "forehead", "eyes", etc.
        image            : cropped numpy BGR image
        bbox             : (x, y, w, h) in face_crop pixel space
        landmarks_used   : MediaPipe landmark indices that defined this region
        confidence_score : 1.0 = reliable, 0.4 = estimated (ears)
        notes            : caveats or design decisions
    """
    region_name      : str
    image            : np.ndarray
    bbox             : Tuple[int, int, int, int]
    landmarks_used   : List[int]
    confidence_score : float = 1.0
    notes            : str   = ""


@dataclass
class AllPartsResult:
    """Holds one FacePartResult per region. None = extraction failed."""
    whole_face : Optional[FacePartResult] = None
    forehead   : Optional[FacePartResult] = None
    eyebrows   : Optional[FacePartResult] = None
    eyes       : Optional[FacePartResult] = None
    nose       : Optional[FacePartResult] = None
    mouth      : Optional[FacePartResult] = None
    jaw_chin   : Optional[FacePartResult] = None
    ears       : Optional[FacePartResult] = None

    def to_dict(self) -> Dict[str, Optional[FacePartResult]]:
        return {
            "whole_face": self.whole_face, "forehead" : self.forehead,
            "eyebrows"  : self.eyebrows,  "eyes"     : self.eyes,
            "nose"      : self.nose,       "mouth"    : self.mouth,
            "jaw_chin"  : self.jaw_chin,   "ears"     : self.ears,
        }

    def valid_parts(self) -> Dict[str, FacePartResult]:
        """Return only successfully extracted parts."""
        return {k: v for k, v in self.to_dict().items() if v is not None}


# =============================================================
#  Landmark Index Groups  (MediaPipe FaceMesh — 468 points)
# =============================================================
#
#  Each group lists the BOUNDARY landmarks of a region.
#  We compute the bounding box of all points in the group,
#  then add padding to get the final crop.

class LM:
    # ── Whole face ────────────────────────────────────────────────
    FACE_SILHOUETTE = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ]

    # ── Forehead ──────────────────────────────────────────────────
    # TOP row:    upper silhouette landmarks (highest modeled points)
    # BOTTOM row: top edge of both eyebrows
    # Together they form a horizontal band across the forehead.
    FOREHEAD_TOP    = [10, 109, 67, 103, 54, 21, 162, 127]
    FOREHEAD_BOTTOM = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]

    # ── Eyebrows ──────────────────────────────────────────────────
    RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    LEFT_EYEBROW  = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]

    # ── Eyes ──────────────────────────────────────────────────────
    # Full eyelid contour (upper lid + lower lid) for each eye
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155,
                 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249,
                 263, 466, 388, 387, 386, 385, 384, 398]

    # ── Nose ──────────────────────────────────────────────────────
    # Bridge centerline + ala (wings) + base
    NOSE_ALL = [
        168, 6, 197, 195, 5, 4,
        209, 49, 64, 98, 97,
        429, 279, 294, 327, 326,
        2, 240, 99, 219, 218, 237
    ]

    # ── Mouth ─────────────────────────────────────────────────────
    MOUTH_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                   375, 321, 405, 314, 17, 84, 181, 91, 146]
    MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                   324, 318, 402, 317, 14, 87, 178, 88, 95]

    # ── Jaw & Chin ────────────────────────────────────────────────
    # Full lower silhouette: jaw hinge → chin tip → jaw hinge
    JAW_LINE = [
        172, 136, 150, 149, 176, 148, 152,   # chin
        377, 400, 378, 379, 365, 397, 288,   # right jaw
        58, 132, 93, 234                     # left jaw
    ]

    # ── Ear estimation anchors (no real ear landmarks in FaceMesh) ─
    EAR_LEFT_ANCHOR  = 234
    EAR_RIGHT_ANCHOR = 454
    TEMPLE_LEFT      = 127
    TEMPLE_RIGHT     = 356


# =============================================================
#  Padding Configuration
# =============================================================

@dataclass
class PaddingConfig:
    """
    Padding around each region (fraction of bounding-box size).
    Increase a value if crops cut off feature edges.
    """
    whole_face : float = 0.05
    forehead   : float = 0.15   # extra upward to reach hairline
    eyebrows   : float = 0.40   # brows are thin — generous padding needed
    eyes       : float = 0.35   # include full eyelids + lashes
    nose       : float = 0.20
    mouth      : float = 0.25   # adds philtrum above + chin area below
    jaw_chin   : float = 0.15
    ears       : float = 0.30


# =============================================================
#  Main Extractor Class
# =============================================================

class FacePartExtractor:
    """
    Extracts one image crop per facial region.

    Accepts crop_landmarks from FaceDetectorValidator — these are already
    in face_crop pixel space, so NO coordinate remapping is needed here.

    Usage:
        # After running FaceDetectorValidator:
        extractor = FacePartExtractor(
            face_crop      = result.face_crop,
            crop_landmarks = result.crop_landmarks,
        )
        all_parts = extractor.extract_all_parts()

        # Iterate results:
        for name, part in all_parts.valid_parts().items():
            print(name, part.image.shape, part.confidence_score)
            cv2.imwrite(f"{name}.jpg", part.image)
    """

    def __init__(self, face_crop : np.ndarray, crop_landmarks : List[Tuple[int, int]],   # ← directly from result
                 padding : PaddingConfig = None):
                  self.face_crop = face_crop
                  self.lm = crop_landmarks # List[(x,y)] in crop pixels
                  self.crop_h, self.crop_w = face_crop.shape[:2]
                  self.padding = padding or PaddingConfig()

    # ----------------------------------------------------------
    #  Core Geometry Helpers
    # ----------------------------------------------------------

    def _pts(self, indices: List[int]) -> List[Tuple[int,int]]:
        """
        Look up pixel coordinates for a list of landmark indices.
        self.lm[i] is already (x, y) in crop-pixel space — direct lookup.
        """
        return [self.lm[i] for i in indices]

    def _bbox_from_pts(self, pts: List[Tuple[int,int]]) -> Tuple[int,int,int,int]:
        """Tight axis-aligned bounding box from a list of (x,y) points → (x,y,w,h)."""
        xs = [p[0] for p in pts];  ys = [p[1] for p in pts]
        x  = min(xs);  y = min(ys)
        return x, y, max(max(xs) - x, 1), max(max(ys) - y, 1)

    def _crop(self, bbox: Tuple[int,int,int,int],
              pad_frac: float) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
        """
        Expand bbox by pad_frac, clamp to image bounds, return crop + padded bbox.
        Safe against any landmark that lands exactly on the image edge.
        """
        x, y, w, h = bbox
        px = int(w * pad_frac);  py = int(h * pad_frac)
        x1 = max(0, x - px);    y1 = max(0, y - py)
        x2 = min(self.crop_w, x + w + px)
        y2 = min(self.crop_h, y + h + py)
        return self.face_crop[y1:y2, x1:x2], (x1, y1, x2-x1, y2-y1)

    def _make(self, region: str, indices: List[int], pad: float,
              conf: float = 1.0, notes: str = "") -> Optional[FacePartResult]:
        """
        Generic pipeline: indices → points → bbox → padded crop → result.
        Returns None if the crop is empty (safety net).
        """
        pts             = self._pts(indices)
        tight           = self._bbox_from_pts(pts)
        cropped, padded = self._crop(tight, pad)
        if cropped.size == 0:
            return None
        return FacePartResult(
            region_name      = region,
            image            = cropped,
            bbox             = padded,
            landmarks_used   = indices,
            confidence_score = conf,
            notes            = notes,
        )

    # ----------------------------------------------------------
    #  Extraction Methods
    # ----------------------------------------------------------

    def extract_whole_face(self) -> Optional[FacePartResult]:
        """
        Full face crop — baseline reference sent to LLM first.
        Uses the 36-point face silhouette for the bounding box.
        """
        return self._make(
            "whole_face", LM.FACE_SILHOUETTE, self.padding.whole_face,
            notes="Full face reference for overall LLM context."
        )

    def extract_forehead(self) -> Optional[FacePartResult]:
        """
        Forehead: from hairline to top of eyebrows.

        TOP boundary    = upper silhouette landmarks (lm 10 = highest point)
        BOTTOM boundary = top edge of both eyebrows
        Extra upward padding compensates for the hairline not being modeled.
        """
        return self._make(
            "forehead",
            LM.FOREHEAD_TOP + LM.FOREHEAD_BOTTOM,
            self.padding.forehead,
            notes="Top boundary = lm 10. Hairline not in FaceMesh — padding extends upward."
        )

    def extract_eyebrows(self) -> Optional[FacePartResult]:
        """
        Both eyebrows in one crop (for symmetry comparison).
        Upper + lower edge landmarks for each brow.
        Generous padding because brows are thin structures.
        """
        return self._make(
            "eyebrows",
            LM.RIGHT_EYEBROW + LM.LEFT_EYEBROW,
            self.padding.eyebrows,
            notes="Both brows combined. Padding needed — brows are thin."
        )

    def extract_eyes(self) -> Optional[FacePartResult]:
        """
        Both eyes in one crop (full eyelid contour, for symmetry).
        16-point eyelid boundary per eye — includes inner/outer corners.
        """
        return self._make(
            "eyes",
            LM.RIGHT_EYE + LM.LEFT_EYE,
            self.padding.eyes,
            notes="Full eyelid contour. Padding includes lashes."
        )

    def extract_nose(self) -> Optional[FacePartResult]:
        """
        Nose: bridge (lm 168) → tip (lm 4) → full nostril width.
        Includes ala (wings) and nostril base for complete nose shape.
        """
        return self._make(
            "nose", LM.NOSE_ALL, self.padding.nose,
            notes="Bridge + tip (lm 4) + ala + nostril base."
        )

    def extract_mouth(self) -> Optional[FacePartResult]:
        """
        Mouth: outer lip contour + inner opening.
        Padding adds philtrum (above) and chin dimple area (below).
        """
        return self._make(
            "mouth",
            LM.MOUTH_OUTER + LM.MOUTH_INNER,
            self.padding.mouth,
            notes="Outer + inner lip contours. Padding adds philtrum and chin."
        )

    def extract_jaw_chin(self) -> Optional[FacePartResult]:
        """
        Full mandible: jaw hinge (lm 234/454) → chin tip (lm 152).
        17-point lower silhouette covering the complete jaw shape.
        """
        return self._make(
            "jaw_chin", LM.JAW_LINE, self.padding.jaw_chin,
            notes="Jaw hinge (lm 234/454) to chin tip (lm 152)."
        )

    def extract_ears(self) -> Optional[FacePartResult]:
        """
        Ear region — LOW CONFIDENCE (0.4).

        FaceMesh has NO ear landmarks. We estimate using jaw-hinge
        landmarks (lm 234/454) and extend laterally.
        Replace with a dedicated ear detector in v2.
        """
        left_x,  left_y  = self.lm[LM.EAR_LEFT_ANCHOR]
        right_x, right_y = self.lm[LM.EAR_RIGHT_ANCHOR]

        face_w   = abs(right_x - left_x)
        half_w   = int(face_w * 0.12)
        half_h   = int(face_w * 0.18)

        x1 = max(0, left_x  - half_w)
        x2 = min(self.crop_w, right_x + half_w)
        y1 = max(0, min(left_y, right_y) - half_h)
        y2 = min(self.crop_h, max(left_y, right_y) + half_h)

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
                "Estimated from jaw-hinge (lm 234/454). "
                "Replace with dedicated detector in v2."
            )
        )

    # ----------------------------------------------------------
    #  Convenience Method
    # ----------------------------------------------------------

    def extract_all_parts(self, include_ears: bool = True) -> AllPartsResult:
        """
        Run all extractors and return AllPartsResult.

        Args:
            include_ears : False to skip the low-confidence ear estimation

        Example:
            parts = extractor.extract_all_parts()
            for name, part in parts.valid_parts().items():
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
