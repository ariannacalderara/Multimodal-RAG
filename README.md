# Multimodal RAG 
This repository contains the implementation of a Multimodal Retrieval-Augmented Generation (RAG) system developed for a data science course at WU Vienna. The project addresses the "Image Blind Spot" in traditional RAG architectures by integrating vision models to interpret and index outputs.

## 📌 Project Overview
This system extracts provided visuals inputs and converts them into descriptive text, ensuring that the AI tutor can answer questions based on the entire document, not just the text

## 🏗️ The Multimodal Architecture
The pipeline moves beyond text-only processing by implementing two specialized functions:

- extract_image_chunks: Uses PyMuPDF to inspect PDFs for images. When located, image bytes are extracted and passed to the captioning engine.
- caption_image: Leverages the Moondream vision model to encode visuals into base64 and generate descriptive text focusing on data labels and relationships.

## 🛠️ Tech Stack
- Framework: Streamlit
- Orchestration: Ollama (Local LLM serving)
- Vector Database: ChromaDB
- Models: 
  - Text: tinyllama
  - Vision: moondream
- Embeddings: all-MiniLM-L6-v2
- Parsing: unstructured, PyMuPDF (fitz)

## Setup 

1. Pull the models after installing Ollama:
   
   `ollama pull tinyllama`
   `ollama pull moondream`

2. Create an environment and install all needed dependencies
   
   - `conda create env -n envmultimodal`
   - `conda activate envmultimodal`
   - `pip install "sentence-transformers==2.7.0" "transformers==4.41.0" "numpy==1.26.4"`
   - `pip install streamlit chromadb pymupdf requests`
   - `pip install "unstructured[all-docs]"`
   - `brew install tesseract poppler libmagic`

3. Run the Streamlit app
   
   - `streamlit run multimodal.py`
  


## Results & Testing
The RAG was tested against clinical queries regarding PDAC and XAI metrics.

Key Findings:

- Success: The system successfully identified visual elements (e.g., pipeline stages) that Naive RAG implementations would miss.

- Limitation: Small-scale local models (TinyLlama) exhibit a higher tendency for "hallucination patterns" when dealing with complex attribute contributions.

- Vision Accuracy: While the system captures layout and context, granular reading of chart labels remains a challenge for sub-7B vision models.
   

