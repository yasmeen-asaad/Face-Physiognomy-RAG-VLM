import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


class PhysiognomyRetriever:

    def __init__(self, index_path: str, chunks_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Load:
          - FAISS index
          - chunks metadata
          - embedding model
        """

        self.index = faiss.read_index(index_path)
        self.model = SentenceTransformer(model_name)
      
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        print(f"Loaded {len(self.chunks)} chunks")

    def search(self, query: str,top_k: int = 5):
        """
        Semantic search over the book.
        """

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        results = []

        for rank, idx in enumerate(indices[0]):

            chunk = self.chunks[idx]

            results.append({"rank": rank + 1,
                            "score": float(distances[0][rank]),
                            "page": chunk.page,
                            "region": chunk.region,
                            "chapter": chunk.chapter,
                            "content": chunk.content
                            }
                          )

        return results
