"""
=============================================================
  Face Physiognomy Project — RAG Builder  (v2)
=============================================================

CHANGES FROM v1:
  - extract() now uses detail=1 + line grouping → returns structured lines
  - parse_page_to_chunks() detects Layout A (3-col) vs Layout B (narrative)
  - Each feature becomes one chunk: feature_name + traits + description
  - Embedding prefix includes Region + Chapter for better retrieval
  - Removed unreachable code in get_region_and_chapter()
"""

import os
import json
import pickle
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from collections import Counter

import numpy as np


# =============================================================
#  Chapter / Page Mapping
# =============================================================

CHAPTER_MAPPING: Dict[str, Dict] = {
    "forehead": {
        "pages": (11, 12, 61, 62),
        "chapters": {
            11: "forehead_shapes",
            12: "forehead_shapes",
            61: "lines",
            62: "lines",
        }
    },
    "eyebrows": {
        "pages": (13, 14, 15, 16),
        "chapters": {
            13: "eyebrows_basic_shapes",
            14: "eyebrows_position",
            15: "eyebrows_specific_types",
            16: "eyebrows_specific_types",
        }
    },
    "eyes": {
        "pages": (17, 18, 19, 20, 21, 22, 23, 24, 25, 26),
        "chapters": {
            17: "eyes_spacing",
            18: "eyes_angle",
            19: "eyes_depth",
            20: "eyes_corner_indents_and_eyes_iris_size",
            21: "eyes_pupil_response",
            22: "eyes_stress_signs",
            23: "eyelids_top",
            24: "eyelids_bottom",
            25: "eyelashes",
            26: "eye_puffs",
        }
    },
    "nose": {
        "pages": (27, 28, 29, 30, 31, 32, 33, 34, 35),
        "chapters": {
            27: "nose_size_shape",
            28: "nose_size_shape",
            29: "nose_ridge",
            30: "nose_width",
            31: "nose_tip_angle",
            32: "nose_tip_size_shape",
            33: "nose_tip_size_shape",
            34: "nostrils_size_shape",
            35: "nostrils_size_shape",
        }
    },
    "ears": {
        "pages": (36, 37, 38, 39, 40, 41),
        "chapters": {
            36: "ears_size",
            37: "ears_cups_ridges",
            38: "ears_placement",
            39: "ears_placement",
            40: "ears_height",
            41: "ear_eyebrow_combinations",
        }
    },
    "mouth": {
        "pages": (44, 45, 46, 47, 48, 49),
        "chapters": {
            44: "mouth_size",
            45: "mouth_angle",
            46: "lips_size_shape",
            47: "lips_size_shape",
            48: "teeth",
            49: "smiles",
        }
    },
    "jaw_chin": {
        "pages": (42, 43, 50, 51, 52, 53, 54, 59, 60),
        "chapters": {
            42: "cheeks",
            43: "cheeks",
            50: "jaws",
            51: "chins",
            52: "chins",
            53: "chins",
            54: "chins",
            59: "dimples_clefts",
            60: "dimples_clefts",
        }
    },
    "face_overview": {
        "pages": (41, 55, 56, 57, 58, 63, 64, 65, 66, 67, 68,
                  69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80),
        "chapters": {
            41: "ear_eyebrow_combinations",
            55: "chin_eyebrow_combinations",
            56: "chin_eyebrow_combinations",
            57: "chin_eyebrow_combinations",
            58: "chin_eyebrow_combinations",
            63: "lines",
            64: "lines",
            65: "lines",
            66: "lines",
            67: "facial_hair",
            68: "facial_hair",
            69: "face_shape",
            70: "face_shape",
            71: "face_shape",
            72: "face_types",
            73: "combination_face_types",
            74: "facial_dominance",
            75: "facial_dominance",
            76: "profile_types",
            77: "profile_types",
            78: "profile_combinations",
            79: "head_types",
            80: "head_types",
        }
    },
}


def get_region_and_chapter(page_num: int) -> Tuple[str, str]:
    """Page number → (region, chapter). Falls back to general/introduction."""
    for region, data in CHAPTER_MAPPING.items():
        if page_num in data["pages"]:
            chapter = data["chapters"].get(page_num, region)
            return region, chapter
    return "general", "introduction"


# =============================================================
#  Data Classes
# =============================================================

@dataclass
class TextChunk:
    """One searchable unit from the book."""
    chunk_id     : str
    page         : int
    region       : str
    chapter      : str
    content      : str           # embedded text (feature + traits + description)
    feature_name : str           = ""
    traits       : List[str]     = field(default_factory=list)
    layout       : str           = "structured"
    char_len     : int           = 0

    def __post_init__(self):
        self.char_len = len(self.content)


@dataclass
class BuildStats:
    pdf_path        : str  = ""
    pages_processed : int  = 0
    pages_empty     : int  = 0
    total_chunks    : int  = 0
    embedding_dim   : int  = 0
    region_dist     : dict = field(default_factory=dict)
    chapter_dist    : dict = field(default_factory=dict)
    embed_model     : str  = ""


# =============================================================
#  Page Parser — Layout Detection + Chunk Building
# =============================================================

def _zone(x_center: float, page_width: float) -> str:
    """
    Classify x position into zone.

    Zone thresholds (from EasyOCR output analysis on page 45):
      trait      : x < 25%  → bullet labels (Optimist, Objective)
      description: 25%-70%  → paragraph text
      skip       : x > 70%  → illustrations, page numbers
    """
    ratio = x_center / page_width if page_width else 0
    if ratio < 0.25:
        return "trait"
    if ratio <= 0.70:
        return "description"
    return "skip"


def parse_page_to_chunks(lines: List[Dict], page_width: float) -> List[Dict]:
    """
    Convert grouped OCR lines into structured feature chunks.

    Each line dict has:
      {"y": float, "items": [{"x": float, "text": str}, ...]}

    Returns list of:
      {
        "feature_name": str,
        "traits"      : List[str],
        "description" : str,
        "layout"      : "structured" | "narrative"
      }

    Layout A (structured): pages with trait column + description column.
      Detected by: >= 2 lines where leftmost item x < 25% page width.
      Feature headings: short (<80 chars), starts uppercase, no trait items.

    Layout B (narrative): full-width paragraphs (intros, section openers).
      All non-skip text joined into one chunk.
    """

    # ── Classify each line ────────────────────────────────────
    parsed = []
    for line in lines:
        items    = line.get("items", [])
        if not items:
            continue

        # Sort items left to right
        items    = sorted(items, key=lambda it: it.get("x", 0))
        text     = " ".join(it.get("text", "").strip() for it in items
                            if it.get("text", "").strip())
        if not text:
            continue

        left_x   = items[0].get("x", 0)
        zone     = _zone(left_x, page_width)

        # Check if any item is in trait zone
        has_trait = any(_zone(it.get("x", 0), page_width) == "trait"
                        for it in items)

        parsed.append({
            "text"     : text,
            "zone"     : zone,
            "has_trait": has_trait,
        })

    # ── Detect layout ─────────────────────────────────────────
    trait_lines = sum(1 for p in parsed if p["zone"] == "trait")
    is_layout_a = trait_lines >= 2

    if not is_layout_a:
        # Layout B: join all non-skip text
        description = " ".join(
            p["text"] for p in parsed if p["zone"] != "skip"
        ).strip()
        return [{
            "feature_name": "",
            "traits"      : [],
            "description" : description,
            "layout"      : "narrative",
        }]

    # ── Layout A: find headings ───────────────────────────────
    #
    # Heading criteria:
    #   - zone == "description" (not left-column)
    #   - no trait items on same line
    #   - text < 80 chars
    #   - starts with uppercase letter
    #
    heading_idx = set()
    for i, p in enumerate(parsed):
        text = p["text"]
        if (
            p["zone"] == "description"
            and not p["has_trait"]
            and len(text) < 80
            and text[0].isupper()
        ):
            heading_idx.add(i)

    # Fallback to narrative if no headings found
    if not heading_idx:
        description = " ".join(
            p["text"] for p in parsed if p["zone"] != "skip"
        ).strip()
        return [{
            "feature_name": "",
            "traits"      : [],
            "description" : description,
            "layout"      : "narrative",
        }]

    # ── Build chunks from headings ────────────────────────────
    chunks  = []
    current = None

    def flush():
        if current:
            current["description"] = " ".join(current.pop("desc_parts")).strip()
            chunks.append({k: v for k, v in current.items()})

    for i, p in enumerate(parsed):
        if p["zone"] == "skip":
            continue

        if i in heading_idx:
            flush()
            current = {
                "feature_name": p["text"],
                "traits"      : [],
                "desc_parts"  : [],
                "layout"      : "structured",
            }
            continue

        if current is None:
            continue

        if p["zone"] == "trait":
            current["traits"].append(p["text"])
        elif p["zone"] == "description":
            current["desc_parts"].append(p["text"])

    flush()
    return chunks


# =============================================================
#  RAG Builder
# =============================================================

class RAGBuilder:
    """
    Builds the FAISS RAG index from the physiognomy book PDF.

    Pipeline: extract() → chunk() → embed() → save()
    """

    def __init__(
        self,
        pdf_path         : str,
        output_dir       : str  = "rag_index",
        start_page       : int  = 11,
        end_page         : int  = 81,
        min_chunk_len    : int  = 60,
        paragraph_merge_threshold : int = 80,
        embed_model_name : str  = "all-MiniLM-L6-v2",
        ocr_gpu          : bool = False,
        render_dpi       : int  = 200,
        pdf_page_offset  : int  = 14,   # adjust if book page ≠ PDF index
    ):
        self.pdf_path         = pdf_path
        self.output_dir       = output_dir
        self.start_page       = start_page
        self.end_page         = end_page
        self.min_chunk_len    = min_chunk_len
        self.paragraph_merge_threshold = paragraph_merge_threshold
        self.embed_model_name = embed_model_name
        self.ocr_gpu          = ocr_gpu
        self.render_dpi       = render_dpi
        self.pdf_page_offset  = pdf_page_offset
        self.zoom             = render_dpi / 72

        self.chunks     : List[TextChunk]      = []
        self.embeddings : Optional[np.ndarray] = None
        self.stats      = BuildStats(
            pdf_path    = pdf_path,
            embed_model = embed_model_name,
        )

    # ----------------------------------------------------------
    #  Step 1: Extract  (OCR with structured line grouping)
    # ----------------------------------------------------------

    def extract(self) -> List[Dict]:
        """
        Render each PDF page → EasyOCR (detail=1) → group into lines.

        Returns list of page dicts:
          {
            "page"      : int,        # book page number
            "lines"     : List[Dict], # grouped OCR lines
            "page_width": int,        # pixel width of rendered image
            "content"   : str,        # raw joined text (fallback)
          }
        """
        import fitz
        import easyocr

        print(f"[1/4] Extracting — pages {self.start_page}→{self.end_page} "
              f"at {self.render_dpi} DPI")

        reader = easyocr.Reader(['en'], gpu=self.ocr_gpu, verbose=False)
        print("      EasyOCR ready")

        doc    = fitz.open(self.pdf_path)
        matrix = fitz.Matrix(self.zoom, self.zoom)
        pages  = []

        for book_page in range(self.start_page, self.end_page + 1):
            pdf_idx = book_page - 1 + self.pdf_page_offset

            if pdf_idx >= len(doc):
                print(f"  p{book_page}: PDF index {pdf_idx} out of range — stop")
                break

            # ── Render page as image ──────────────────────────
            pix = doc[pdf_idx].get_pixmap(matrix=matrix, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img.reshape(pix.h, pix.w, pix.n)
            page_width = img.shape[1]

            # ── OCR with bounding boxes ───────────────────────
            #
            # detail=1  → returns (bbox, text, confidence) per detection
            # paragraph=False → raw word/phrase detections, not merged
            # We do our own grouping by y-coordinate below.
            #
            ocr_raw = reader.readtext(img, detail=1, paragraph=False)

            if not ocr_raw:
                self.stats.pages_empty += 1
                print(f"  p{book_page}: empty")
                continue

            # ── Sort by y then x ─────────────────────────────
            ocr_raw = sorted(
                ocr_raw,
                key=lambda r: (int(r[0][0][1]), int(r[0][0][0]))
            )

            # ── Group detections into lines by y proximity ────
            #
            # y_threshold=15px: detections within 15px vertically
            # are considered on the same line.
            # Each line collects items as {"x", "text"} dicts.
            #
            Y_THRESHOLD = 15
            lines: List[Dict] = []

            for bbox, text, conf in ocr_raw:
                text = text.strip()
                if not text:
                    continue
                x = int(bbox[0][0])
                y = int(bbox[0][1])

                placed = False
                for line in lines:
                    if abs(line["y"] - y) < Y_THRESHOLD:
                        line["items"].append({"x": x, "text": text})
                        placed = True
                        break
                if not placed:
                    lines.append({"y": y, "items": [{"x": x, "text": text}]})

            # Sort lines top to bottom
            lines = sorted(lines, key=lambda l: l["y"])

            # Raw text for fallback
            raw_text = " ".join(
                it["text"]
                for line in lines
                for it in sorted(line["items"], key=lambda i: i["x"])
            )

            pages.append({
                "page"      : book_page,
                "lines"     : lines,
                "page_width": page_width,
                "content"   : raw_text,
            })

            n_lines = len(lines)
            print(f"  p{book_page}: {n_lines} lines, {len(raw_text)} chars")

        doc.close()
        self.stats.pages_processed = len(pages)
        print(f"\n  Done: {len(pages)} pages, {self.stats.pages_empty} empty")
        return pages

    # ----------------------------------------------------------
    #  Step 2: Chunk  (structured feature chunks)
    # ----------------------------------------------------------

    def chunk(self, pages: List[Dict]) -> List[TextChunk]:
        print(f"\n[2/4] Chunking text")

        chunks  = []
        counter = 0
        target_min = 400
        target_max = 700
        overlap_min = 80
        overlap_max = 100

        def join_paragraphs(paragraphs: List[str]) -> str:
            return "\n\n".join(paragraphs)

        def count_overlap_paragraphs(paragraphs: List[str]) -> int:
            total = 0
            count = 0
            for para in reversed(paragraphs):
                separator_len = 2 if count else 0
                next_total = total + separator_len + len(para)
                if count and total >= overlap_min and next_total > overlap_max:
                    break
                total = next_total
                count += 1
                if total >= overlap_min:
                    break
            return count

        for page_data in pages:
            book_page       = page_data["page"]
            region, chapter = get_region_and_chapter(book_page)

            paragraphs = [
                para.strip()
                for para in page_data["content"].split("\n\n")
                if para.strip()
            ]

            merged_paragraphs = []
            i = 0
            while i < len(paragraphs):
                current = paragraphs[i]

                while len(current) < self.paragraph_merge_threshold and i + 1 < len(paragraphs):
                    i += 1
                    current += "\n\n" + paragraphs[i]

                merged_paragraphs.append(current)
                i += 1

            paragraphs = merged_paragraphs

            start = 0
            while start < len(paragraphs):
                current = []
                idx = start

                while idx < len(paragraphs):
                    candidate = current + [paragraphs[idx]]
                    candidate_len = len(join_paragraphs(candidate))

                    if current and len(join_paragraphs(current)) >= target_min and candidate_len > target_max:
                        break

                    current = candidate
                    idx += 1

                    if len(join_paragraphs(current)) >= target_min:
                        break

                chunks.append(TextChunk(
                    chunk_id = f"p{book_page}_{counter}",
                    page     = book_page,
                    region   = region,
                    chapter  = chapter,
                    content  = join_paragraphs(current),
                ))
                counter += 1

                if idx >= len(paragraphs):
                    break

                overlap_count = count_overlap_paragraphs(current)
                next_start = idx - overlap_count
                start = next_start if next_start > start else idx

        self.stats.total_chunks = len(chunks)
        self.stats.region_dist  = dict(Counter(c.region  for c in chunks))
        self.stats.chapter_dist = dict(Counter(c.chapter for c in chunks))

        print(f"  Total chunks: {len(chunks)}")
        print(f"  Region distribution:")
        for r, cnt in sorted(self.stats.region_dist.items(), key=lambda x: -x[1]):
            print(f"    {r:<22} {cnt:>4}  {'█' * (cnt // 2)}")

        self.chunks = chunks
        return chunks

    # ----------------------------------------------------------
    #  Step 3: Embed
    # ----------------------------------------------------------

    def embed(self, chunks: List[TextChunk]) -> np.ndarray:
        """
        Convert chunks to vectors.

        Embedding text includes Region + Chapter prefix so FAISS
        retrieval is biased toward the correct facial region.

        "Region: nose Chapter: nose_ridge Wide nose. Traits: ..."
        """
        from sentence_transformers import SentenceTransformer

        print(f"\n[3/4] Embedding {len(chunks)} chunks — {self.embed_model_name}")

        model = SentenceTransformer(self.embed_model_name)
        texts = [
            f"Region: {c.region} Chapter: {c.chapter} {c.content}"
            for c in chunks
        ]

        embeddings = model.encode(
            texts,
            show_progress_bar = True,
            batch_size        = 32,
            convert_to_numpy  = True,
        ).astype("float32")

        self.stats.embedding_dim = embeddings.shape[1]
        print(f"  Shape: {embeddings.shape}")
        self.embeddings = embeddings
        return embeddings

    # ----------------------------------------------------------
    #  Step 4: Save
    # ----------------------------------------------------------

    def save(self, embeddings: np.ndarray):
        """Build FAISS index and save index + chunks + stats."""
        import faiss

        print(f"\n[4/4] Building FAISS index")
        os.makedirs(self.output_dir, exist_ok=True)

        dim   = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        faiss_path  = os.path.join(self.output_dir, "index.faiss")
        chunks_path = os.path.join(self.output_dir, "chunks.pkl")
        info_path   = os.path.join(self.output_dir, "build_info.json")

        faiss.write_index(index, faiss_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        with open(info_path, "w") as f:
            json.dump(asdict(self.stats), f, indent=2)

        print(f"  {index.ntotal} vectors, dim={dim}")
        print(f"  Saved: {faiss_path}")
        print(f"  Saved: {chunks_path}")
        print(f"  Saved: {info_path}")
        print(f"\n✓ RAG index ready → '{self.output_dir}/'")

    # ----------------------------------------------------------
    #  Main Entry Point
    # ----------------------------------------------------------

    def build(self) -> BuildStats:
        """Run full pipeline: extract → chunk → embed → save."""
        print("=" * 55)
        print("  RAG Builder v2")
        print("=" * 55)

        pages      = self.extract()
        chunks     = self.chunk(pages)
        embeddings = self.embed(chunks)
        self.save(embeddings)

        print("\n" + "=" * 55)
        print(f"  Chunks        : {self.stats.total_chunks}")
        print(f"  Embedding dim : {self.stats.embedding_dim}")
        print(f"  Regions       : {len(self.stats.region_dist)}")
        print("=" * 55)

        return self.stats
