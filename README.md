# Physiognomy Project

This project explores how Vision Language Models (VLMs) and Retrieval-Augmented Generation (RAG) can be combined to analyze facial morphology and retrieve relevant descriptions from a physiognomy reference book.

Project Pipeline
Face Detection & Landmark Extraction
Face Part Segmentation (Forehead, Eyes, Eyebrows, Nose, Mouth, Jaw/Chin, Ears)
Vision-Language Analysis using Gemini
Structured JSON Feature Extraction
Semantic Retrieval using FAISS and Sentence Transformers
Evidence Collection from the Reference Book
Final Report Generation
Example Workflow
Face Image → Face Parts Extraction → Visual Feature Description → Semantic Search in Knowledge Base → Retrieved Evidence → Final Face Reading Report

Technologies
Python
OpenCV
MediaPipe
Gemini Vision Models
Sentence Transformers
FAISS
EasyOCR
Retrieval-Augmented Generation (RAG)
Disclaimer
This project is intended for educational and research purposes only. The generated descriptions and retrieved interpretations originate from the source reference material and should not be considered scientific, psychological, medical, or personality assessments.
