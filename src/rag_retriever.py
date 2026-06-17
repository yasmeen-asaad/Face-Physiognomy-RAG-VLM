import pickle
import faiss
import numpy as np
import re
from typing import Dict, List, Optional, Any
from sentence_transformers import SentenceTransformer


class PhysiognomyRetriever:
    def __init__(self, index_path: str, chunks_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Load:
          - FAISS index
          - chunks metadata
          - embedding model
        """

        self.index = faiss.read_index(index_path)  # Load FAISS index
        self.model = SentenceTransformer(model_name)  # Load embedding model
      
        with open(chunks_path, "rb") as f: # Load chunks (text + metadata)
            self.chunks = pickle.load(f)
       
        print(f"Loaded {len(self.chunks)} chunks")
        
    def _build_natural_query(self, region, features_json):
        """
        Convert structured JSON features into a natural language query.
        to matchy matchy with book language 
        output: string query in natural language
        """
        if not features_json:
             return region.replace("_", " ")
             
         # 1. Extract only high-confidence features
        parts = []
        for feature_name, feature_data in features_json.items():
            if isinstance(feature_data, dict):
                value = feature_data.get("value")
                confidence = feature_data.get("confidence", 1.0)
            else:
                value = feature_data
                confidence = 1.0
            # Skip null values & low conf
            if value is None or value == "null":
                continue
            if isinstance(confidence, (int, float)) and confidence < 0.5:
                continue
            
            clean_feature = feature_name.replace("_", " ") # clean it for more readability
            parts.append(f"{value} {clean_feature}")
        if not parts:
            return region.replace("_", " ")
            
        # 2. Build a descriptive sentence
        region_clean = region.replace("_", " ") #3. Prepend the region name for context
        if len(parts) == 1:
             return f"{parts[0]} {region_clean}"
        elif len(parts) == 2:
             return f"{parts[0]} {region_clean} with {parts[1]}"
        else:
            rest = " and ".join(parts[1:])
            return f"{parts[0]} {region_clean} with {rest}"
    
    #____________________________________________
    #  Core Search Function
    #____________________________________________
    
    def search(self, query, region=None, chapter=None, top_k: int = 3):
        """
        Search the book using semantic similarity.
        The idea here is to search using region in order to focuse only on the required face part not whole book 
        Note that region & chapter are optional filters
        output:  List of dicts with keys: rank, score, page, region, chapter, content, query_used
        """
        # 1. Convert query to vector/embedding
        query_vector = self.model.encode([query], convert_to_numpy=True).astype("float32")
        
        # 2. FAISS finds the nearest vectors # 3. Optionally filter by region or chapter
        # Note: searching more than top_k to have room for filtering
        search_k  = top_k * 5 if (region or chapter) else top_k 
        distances, indices = self.index.search(query_vector, search_k)

        # 4. collect results with optional filter
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue # FAISS returns -1 for empty slots
            chunk = self.chunks[idx]
            if region and chunk.region != region: # Apply region filter
                continue
            if chapter and chunk.chapter != chapter: # Apply chapter filter
                continue
            score = float(1 / (1 + dist)) # convert distance to score in a scale of [0,1], (small distance = high score)
            results.append({"rank" : len(results) + 1,
                            "score" : round(score, 4),
                            "page" : chunk.page,
                            "region" : chunk.region,
                            "chapter" : chunk.chapter,
                            "content" : chunk.content,
                            "query_used" : query,
                            })
            if len(results) >= top_k:  # 5. Return top_k results with text + metadata
                break; 
        return results
    #____________________________________________
    #  Search from JSON (main method for the pipeline)
    #____________________________________________
    def search_by_json(self, region, features_json, top_k =3):
        """
        Full pipeline: JSON → natural query → search → results.
        Combines build_natural_query + search in one call.
        Note that features_json is the output from FaceDescriber.describe_part()
        output is a Dict with: the used query, results list of matching book passages, the region searched
        """
        query = self._build_natural_query(region, features_json)
        return {"query"  : query,
                "region" : region,
                "results": self.search(query=query, region=region, top_k=top_k)
               }
    #____________________________________________
    # Search All Parts
    #____________________________________________
    def search_all_parts(self, descriptions, top_k):
        """
        Run search_by_json for every face part.
        descriptions is the output of FaceDescriber.describe_all_parts() {region_name: DescriptionResult}
        Output: {region_name: {"query": ..., "region": ..., "results": [...]}}
        """
        all_evidence = {}
        for region_name, desc_result in descriptions.items():
            if not desc_result.success:
                print(f" Skipping {region_name} as the description failed")
                continue
            print(f" Searching: {region_name}")
            evidence = self.search_by_json(region=region_name, features_json=desc_result.features_json, top_k=top_k)
            all_evidence[region_name] = evidence
            print(f" sucess query: '{evidence['query'][:50]}'")
        return all_evidence
        
        

