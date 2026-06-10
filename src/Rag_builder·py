"""
=============================================================
  Face Physiognomy Project — RAG Builder
=============================================================
  One-time script. Run it once to build the knowledge base
  from the physiognomy book PDF. Saves the index to disk.
  Never needs to run again unless the book changes.

OUTPUT FILES
  rag_index/
  ├── index.faiss      ← the search index (vectors)
  ├── chunks.pkl       ← the original text chunks with metadata
  └── build_info.json  ← stats about the build (for debugging)

Install:
  pip install pymupdf sentence-transformers faiss-cpu
"""

import os
import json
import pickle
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from collections import Counter

import numpy as np


# =============================================================
#  Data Classes
# =============================================================

@dataclass
class TextChunk:
    """
    One searchable piece of the book.

    WHY we store all this metadata:
      When the RAG returns a result, we want to know not just
      WHAT it said, but WHERE in the book it came from (page),
      and WHICH facial region it's about (region).
      The region tag lets us filter results later — e.g. only
      return chunks about the nose when analysing the nose crop.
    """
    chunk_id : str            # unique ID e.g. "p42_3"
    page     : int            # page number in the book (human-readable)
    region   : str            # detected facial region (or "general")
    content  : str            # the actual text
    char_len : int = 0        # character count (set automatically)

    def __post_init__(self):
        self.char_len = len(self.content)


@dataclass
class BuildStats:
    """Stats collected during the build — useful for debugging."""
    pdf_path        : str   = ""
    pages_extracted : int   = 0
    pages_empty     : int   = 0
    total_chunks    : int   = 0
    embedding_dim   : int   = 0
    region_dist     : dict  = field(default_factory=dict)
    embed_model     : str   = ""


# =============================================================
#  Region Keyword Map
# =============================================================
#
# CONCEPT: Keyword-based region tagging
#
# When we extract text from the book, we don't know which facial
# region each paragraph is about. We use keyword matching to tag
# each chunk with a region label.
#
# WHY THIS MATTERS FOR RAG:
#   Later, when we analyse the nose crop, we can filter the search
#   to only return chunks tagged "nose". This makes results more
#   relevant and reduces noise.
#
# LIMITATION: Keyword matching is simple and can be wrong.
#   A paragraph about "the nose affecting personality" might
#   mention "eyes" too. That's fine — we just tag it with the
#   FIRST matching region. Good enough for v1.

REGION_KEYWORDS: Dict[str, List[str]] = {
    "forehead"  : ["forehead", "brow", "frontal lobe", "hairline"],
    "eyebrows"  : ["eyebrow", "brow shape", "brow line"],
    "eyes"      : ["eye ", "eyes", "eyelid", "eyelash", "iris",
                   "pupil", "eye corner", "eye spacing"],
    "nose"      : ["nose", "nostril", "nasal", "nose bridge",
                   "nose tip", "nose ridge", "nose width"],
    "mouth"     : ["mouth", "lip", "lips", "teeth", "smile",
                   "upper lip", "lower lip"],
    "jaw_chin"  : ["jaw", "chin", "mandible", "jawline"],
    "cheeks"    : ["cheek", "cheekbone", "malar"],
    "ears"      : ["ear ", "ears", "earlobe", "ear shape"],
    "whole_face": ["face shape", "face type", "facial structure",
                   "head shape", "facial symmetry", "profile",
                   "oval face", "round face", "square face"],
    "lines"     : ["wrinkle", "line ", "lines", "furrow", "crease"],
    "hair"      : ["hair", "beard", "mustache", "facial hair"],
}


def detect_region(text: str) -> str:
    """
    Detect which facial region a text chunk is about.

    Strategy: check if any keyword from each region appears
    in the text. Return the first match, or "general" if none.

    WHY "general" and not skip?
      Some chunks discuss overall personality or methodology.
      These are still useful context for the LLM, so we keep
      them tagged as "general" rather than throwing them away.
    """
    text_lower = text.lower()
    for region, keywords in REGION_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return region
    return "general"


# =============================================================
#  RAG Builder Class
# =============================================================

class RAGBuilder:
    """
    Builds the RAG knowledge base from the physiognomy book PDF.

    The 4-step pipeline:
      extract() → chunk() → embed() → save()

    All steps called automatically by build().

    Usage:
        builder = RAGBuilder(
            pdf_path   = "/path/to/book.pdf",
            output_dir = "rag_index",
            start_page = 8,    # page 9  (0-indexed)
            end_page   = 127,  # page 128 (0-indexed)
        )
        builder.build()
    """

    def __init__(
        self,
        pdf_path    : str,
        output_dir  : str  = "rag_index",
        start_page  : int  = 8,       # 0-indexed → page 9
        end_page    : int  = 127,     # 0-indexed → page 128
        min_chunk_len    : int  = 80,  # ignore paragraphs shorter than this
        embed_model_name : str  = "all-MiniLM-L6-v2",
    ):
        self.pdf_path         = pdf_path
        self.output_dir       = output_dir
        self.start_page       = start_page
        self.end_page         = end_page
        self.min_chunk_len    = min_chunk_len
        self.embed_model_name = embed_model_name

        self.chunks     : List[TextChunk] = []
        self.embeddings : Optional[np.ndarray] = None
        self.stats      = BuildStats(pdf_path=pdf_path,
                                     embed_model=embed_model_name)

    # ----------------------------------------------------------
    #  Step 1: Extract Text from PDF
    # ----------------------------------------------------------

    def extract(self) -> List[Dict]:
        """
        Extract raw text from each page of the PDF.

        LIBRARY: PyMuPDF (imported as fitz)
          - fitz.open()         → open the PDF
          - doc[i].get_text()   → extract text from page i
          - "text" mode         → plain text, preserves line breaks

        WHY NOT USE PyPDF2 or pdfplumber?
          PyMuPDF (fitz) is faster and handles more PDF types.
          It also preserves text order better for multi-column layouts.

        Returns:
            List of {"page": int, "content": str} dicts
        """
        import fitz   # pymupdf

        print(f"[1/4] Extracting text from {self.pdf_path}")
        print(f"      Pages: {self.start_page+1} → {self.end_page+1}")

        doc   = fitz.open(self.pdf_path)
        pages = []

        for i in range(self.start_page, self.end_page + 1):
            text = doc[i].get_text("text").strip()

            if text:
                # Basic cleaning:
                # Remove lines that are just page numbers or headers
                lines = [
                    line.strip() for line in text.split("\n")
                    if len(line.strip()) > 3          # skip very short lines
                    and not line.strip().isdigit()    # skip page numbers
                ]
                clean_text = "\n".join(lines)
                if clean_text:
                    pages.append({"page": i + 1, "content": clean_text})
                else:
                    self.stats.pages_empty += 1
            else:
                self.stats.pages_empty += 1

        doc.close()
        self.stats.pages_extracted = len(pages)
        print(f"      Extracted: {len(pages)} pages "
              f"({self.stats.pages_empty} empty/skipped)")
        return pages

    # ----------------------------------------------------------
    #  Step 2: Chunk the Text
    # ----------------------------------------------------------

    def chunk(self, pages: List[Dict]) -> List[TextChunk]:
        """
        Split pages into smaller searchable paragraphs (chunks).

        CONCEPT: Why chunk at all?
          The embedding model has a max input length (~512 tokens).
          A full page is too long. We split into paragraphs so:
            1. Each chunk fits the model's context window
            2. Search results are more precise (paragraph vs full page)
            3. We can tag each chunk with a region (nose, eye, etc.)

        CHUNKING STRATEGY: Split by double newline (paragraph break)
          This is the simplest strategy. In v2, you could use
          RecursiveCharacterTextSplitter from LangChain for smarter
          overlap-based chunking.

        Returns:
            List[TextChunk] with chunk_id, page, region, content
        """
        print(f"\n[2/4] Chunking text")

        chunks  = []
        counter = 0   # global chunk counter for unique IDs

        for page_data in pages:
            # Split on paragraph breaks (double newline)
            # This works well for books where paragraphs are
            # separated by blank lines.
            raw_paragraphs = page_data["content"].split("\n\n")

            for para in raw_paragraphs:
                clean = para.strip()

                # Skip chunks that are too short to be meaningful
                # (headers, captions, standalone numbers, etc.)
                if len(clean) < self.min_chunk_len:
                    continue

                region = detect_region(clean)

                chunk = TextChunk(
                    chunk_id = f"p{page_data['page']}_{counter}",
                    page     = page_data["page"],
                    region   = region,
                    content  = clean,
                )
                chunks.append(chunk)
                counter += 1

        # Stats
        self.stats.total_chunks = len(chunks)
        region_dist = Counter(c.region for c in chunks)
        self.stats.region_dist = dict(region_dist)

        print(f"      Total chunks: {len(chunks)}")
        print(f"      Region distribution:")
        for region, count in sorted(region_dist.items(),
                                    key=lambda x: -x[1]):
            bar = "█" * (count // 2)
            print(f"        {region:<15} {count:>4}  {bar}")

        self.chunks = chunks
        return chunks

    # ----------------------------------------------------------
    #  Step 3: Embed the Chunks
    # ----------------------------------------------------------

    def embed(self, chunks: List[TextChunk]) -> np.ndarray:
        """
        Convert each chunk's text into a vector (embedding).

        CONCEPT: What is an embedding?
          An embedding is a list of numbers (e.g. 384 numbers) that
          represents the MEANING of a sentence. Sentences with similar
          meanings will have vectors that are close to each other in
          mathematical space.

          Example:
            "wide nose with flared nostrils"  → [0.2, -0.5, 0.8, ...]
            "broad nasal with open nostrils"  → [0.19, -0.48, 0.79, ...]
            "blue sky on a sunny day"         → [-0.9, 0.3, -0.1, ...]

          The first two are CLOSE (similar meaning).
          The third is FAR (different topic).

        MODEL: all-MiniLM-L6-v2
          - 384-dimensional vectors
          - Fast and accurate for English
          - Free, runs on CPU

        Returns:
            numpy array of shape (num_chunks, 384)
        """
        from sentence_transformers import SentenceTransformer

        print(f"\n[3/4] Embedding {len(chunks)} chunks")
        print(f"      Model: {self.embed_model_name}")

        model = SentenceTransformer(self.embed_model_name)
        texts = [c.content for c in chunks]

        # encode() converts each text to a vector
        # show_progress_bar=True prints a progress bar (useful on Kaggle)
        # batch_size=32 processes 32 texts at a time (memory efficient)
        embeddings = model.encode(
            texts,
            show_progress_bar = True,
            batch_size        = 32,
            convert_to_numpy  = True,
        )
        embeddings = embeddings.astype("float32")

        self.stats.embedding_dim = embeddings.shape[1]
        print(f"      Shape: {embeddings.shape}  "
              f"(chunks × vector_dim)")

        self.embeddings = embeddings
        return embeddings

    # ----------------------------------------------------------
    #  Step 4: Build FAISS Index and Save
    # ----------------------------------------------------------

    def save(self, embeddings: np.ndarray):
        """
        Build a FAISS index from the embeddings and save everything.

        CONCEPT: What is FAISS?
          FAISS = Facebook AI Similarity Search.
          It takes all your vectors and builds an internal structure
          that allows finding the NEAREST vectors to a query vector
          VERY FAST — even with millions of vectors.

          IndexFlatL2 = the simplest index type.
          L2 = Euclidean distance (straight-line distance between vectors).
          "Flat" = no compression, exact search.
          Good for < 100k vectors. For larger datasets use IndexIVFFlat.

        WHAT GETS SAVED:
          index.faiss  → the FAISS index (the vectors + search structure)
          chunks.pkl   → the original TextChunk objects (with text + metadata)
          build_info.json → stats for debugging

        WHY SAVE SEPARATELY?
          FAISS only stores vectors (numbers). It doesn't store the
          original text. So when search returns "vector #42 is the closest",
          we need chunks.pkl to look up what text chunk #42 actually says.
        """
        import faiss

        print(f"\n[4/4] Building FAISS index and saving")
        os.makedirs(self.output_dir, exist_ok=True)

        # Build the index
        dim   = embeddings.shape[1]          # 384 for MiniLM
        index = faiss.IndexFlatL2(dim)       # simple exact-search index
        index.add(embeddings)                # add all vectors

        print(f"      Index: {index.ntotal} vectors, dim={dim}")

        # Save FAISS index
        faiss_path = os.path.join(self.output_dir, "index.faiss")
        faiss.write_index(index, faiss_path)
        print(f"      Saved: {faiss_path}")

        # Save chunks (text + metadata)
        chunks_path = os.path.join(self.output_dir, "chunks.pkl")
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"      Saved: {chunks_path}")

        # Save build info
        info_path = os.path.join(self.output_dir, "build_info.json")
        with open(info_path, "w") as f:
            json.dump(asdict(self.stats), f, indent=2)
        print(f"      Saved: {info_path}")

        print(f"\n✓ RAG index built successfully in '{self.output_dir}/'")
        print(f"  → Upload this folder as a Kaggle Dataset")
        print(f"  → Then import it in your other notebooks")

    # ----------------------------------------------------------
    #  Main Entry Point
    # ----------------------------------------------------------

    def build(self):
        """
        Run the full pipeline: extract → chunk → embed → save.
        Call this once per book. Takes ~2-5 min on Kaggle CPU.
        """
        print("=" * 55)
        print("  RAG Builder — Face Physiognomy Project")
        print("=" * 55)

        pages      = self.extract()
        chunks     = self.chunk(pages)
        embeddings = self.embed(chunks)
        self.save(embeddings)

        print("\n" + "=" * 55)
        print(f"  Done! {self.stats.total_chunks} chunks indexed")
        print(f"  Embedding dim : {self.stats.embedding_dim}")
        print(f"  Regions found : {len(self.stats.region_dist)}")
        print("=" * 55)

        return self.stats


# =============================================================
#  Quick Kaggle Usage  (copy into a Kaggle cell)
# =============================================================
#
# from rag_builder import RAGBuilder
#
# builder = RAGBuilder(
#     pdf_path   = "/kaggle/input/physio-book/book.pdf",
#     output_dir = "/kaggle/working/rag_index",
#     start_page = 8,    # page 9
#     end_page   = 127,  # page 128
# )
# builder.build()
#
# After running:
#   Go to Kaggle → Datasets → New Dataset
#   Upload the /kaggle/working/rag_index/ folder
#   Name it "physio-rag-index"
#   Then import it in your other notebooks as input
