---
title: Face Physiognomy
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.34.2"
app_file: app.py
pinned: false
---
# Physiognomy Project: Vision-Language Face Analysis with RAG

A Computer Vision and Retrieval-Augmented Generation (RAG) system that extracts facial morphology features using Vision Language Models (VLMs), retrieves relevant evidence from a physiognomy knowledge base, and generates structured face-reading reports.

## Overview

This project explores how Vision Language Models (VLMs) and Retrieval-Augmented Generation (RAG) can be combined to analyze facial morphology and retrieve relevant descriptions from a physiognomy reference book.

The system combines computer vision, semantic search, and large language models to produce structured facial feature descriptions and evidence-based reports.

## Project Pipeline

1. Face Detection & Landmark Extraction
2. Face Part Segmentation
   * Forehead
   * Eyes
   * Eyebrows
   * Nose
   * Mouth
   * Jaw / Chin
   * Ears
3. Vision-Language Analysis using Gemini
4. Structured JSON Feature Extraction
5. Semantic Retrieval using FAISS and Sentence Transformers
6. Evidence Collection from the Knowledge Base
7. Final Report Generation

## Example Workflow

Face Image 
↓
Face Parts Extraction
↓
Visual Feature Description (VLM)
↓
Semantic Search (RAG)
↓
Retrieved Evidence
↓
Final Face Reading Report

## Technologies

* Python
* OpenCV
* MediaPipe
* Google Gemini
* Sentence Transformers
* FAISS
* EasyOCR
* Retrieval-Augmented Generation (RAG)

## Features

* Facial landmark detection
* Face part extraction
* Vision-based feature description
* Knowledge retrieval from a physiognomy reference book
* Semantic search using vector embeddings
* Structured JSON outputs
* Evidence-based report generation

## Note
### System Dependencies

The project uses MediaPipe FaceLandmarker Tasks API, which requires
additional Linux libraries when deployed on Hugging Face Spaces:

- libgl1
- libglib2.0-0
- libgles2

These dependencies are installed through `packages.txt`.
## Data Availability

The original book, extracted text chunks, and generated FAISS index are not included in this repository.

The project uses a copyrighted reference book for educational and research purposes. To respect copyright and intellectual property rights, the source material, processed text chunks, embeddings, and vector database files have been excluded from this public repository.

Users may build their own knowledge base from legally obtained source materials using the provided pipeline.

## Disclaimer

This project is intended for educational and research purposes only.

The generated descriptions and retrieved interpretations originate from the source reference material and should not be considered scientific, psychological, medical, personality, or hiring assessments. ^^

No claims are made regarding the scientific validity of physiognomy.


