"""
=============================================================
  Face Detector — Test & Demo Script
=============================================================

Run this to test the detector on your own images.

Usage:
  python test_detector.py                         # uses built-in test images
  python test_detector.py --image path/to/photo.jpg
  python test_detector.py --image photo.jpg --save_crops
  python test_detector.py --image photo.jpg --debug     # saves annotated image
"""

import argparse
import sys
import os
import cv2
import numpy as np

# ── Make sure the src module is importable ────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from face_detector import FaceDetectorValidator, Config, ValidationStatus


# ─────────────────────────────────────────────
#  Pretty Print Result
# ─────────────────────────────────────────────

def print_result(result, image_name="image"):
    """Print a nicely formatted validation result to the terminal."""
    SEP   = "─" * 55
    VALID = "\033[92m✓ VALID\033[0m"
    FAIL  = "\033[91m✗ INVALID\033[0m"

    print(f"\n{SEP}")
    print(f"  Image : {image_name}")
    print(SEP)

    status_str = VALID if result.is_valid else FAIL
    print(f"  Status  : {status_str}")
    print(f"  Reason  : {result.message}")

    if result.detection_score > 0:
        print(f"  Confidence : {result.detection_score:.1%}")

    if result.face_bbox:
        x, y, w, h = result.face_bbox
        print(f"  Face bbox  : x={x}, y={y}, w={w}, h={h}")
        if result.face_crop is not None:
            ch, cw = result.face_crop.shape[:2]
            print(f"  Crop size  : {cw}×{ch} px")

    if result.expression_scores:
        print(f"\n  Expression metrics:")
        for k, v in result.expression_scores.items():
            print(f"    {k:<20} {v}")

    print(SEP)


# ─────────────────────────────────────────────
#  Built-in Synthetic Tests (no camera needed)
# ─────────────────────────────────────────────

def create_synthetic_test_image(size=400) -> np.ndarray:
    """
    Creates a plain grey image (no real face).
    Expected result: NO_FACE_DETECTED.
    Useful to confirm the rejection logic works without needing a photo.
    """
    return np.ones((size, size, 3), dtype=np.uint8) * 128


def run_synthetic_tests(detector: FaceDetectorValidator):
    """Run basic sanity checks with synthetic images."""
    print("\n" + "═" * 55)
    print("  SYNTHETIC TESTS  (no camera needed)")
    print("═" * 55)

    # Test: blank image should return NO_FACE_DETECTED
    blank = create_synthetic_test_image()
    result = detector.process(blank)
    print_result(result, "blank grey image")

    expected = ValidationStatus.NO_FACE_DETECTED
    passed   = result.status == expected
    print(f"  TEST {'PASSED ✓' if passed else 'FAILED ✗'} "
          f"(expected {expected.value})")


# ─────────────────────────────────────────────
#  Real Image Test
# ─────────────────────────────────────────────

def run_on_image(image_path: str,
                 save_crops: bool = False,
                 debug: bool = False):
    """
    Run the full pipeline on a real image file.

    Args:
        image_path  : path to JPG/PNG file
        save_crops  : if True, save the cropped face to disk
        debug       : if True, save the annotated image to disk
    """
    print(f"\n[INFO] Loading: {image_path}")

    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)

    # You can customise thresholds here without changing the main code
    config = Config()
    # Example overrides:
    # config.MOUTH_OPEN_THRESHOLD = 0.05    # stricter
    # config.HEAD_TILT_THRESHOLD_DEG = 15   # more lenient

    detector = FaceDetectorValidator(config)
    result   = detector.process(image_path)

    print_result(result, os.path.basename(image_path))

    # ── Save outputs ──────────────────────────────────────────────
    base = os.path.splitext(image_path)[0]

    if result.is_valid and save_crops and result.face_crop is not None:
        crop_path = f"{base}_face_crop.jpg"
        cv2.imwrite(crop_path, result.face_crop)
        print(f"\n[SAVED] Face crop  → {crop_path}")

    if debug and result.annotated_image is not None:
        debug_path = f"{base}_debug.jpg"
        cv2.imwrite(debug_path, result.annotated_image)
        print(f"[SAVED] Debug image → {debug_path}")

    return result


# ─────────────────────────────────────────────
#  Kaggle/Notebook-Friendly Helper
# ─────────────────────────────────────────────

def process_image_for_notebook(image_path: str):
    """
    Use this function directly in a Kaggle notebook:

        from test_detector import process_image_for_notebook
        result = process_image_for_notebook("/kaggle/input/mydata/face.jpg")
        if result.is_valid:
            # pass result.face_crop to the next step (segmentation)
            pass
    """
    detector = FaceDetectorValidator()
    result   = detector.process(image_path)
    print_result(result, image_path)
    return result


# ─────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the Face Detection & Validation module."
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to an image file (JPG or PNG)."
    )
    parser.add_argument(
        "--save_crops", action="store_true",
        help="Save the cropped face to disk if valid."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save annotated image showing landmarks."
    )
    args = parser.parse_args()

    detector = FaceDetectorValidator()

    if args.image:
        run_on_image(args.image, args.save_crops, args.debug)
    else:
        # No image provided → run synthetic tests
        run_synthetic_tests(detector)
        print("\n[TIP] Pass --image path/to/photo.jpg to test a real image.")
