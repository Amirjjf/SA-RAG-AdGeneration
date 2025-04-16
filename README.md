# Advertising Content Generator

## Overview
This project implements a pipeline to generate product advertisements based on customer feedback. It is composed of three main stages:

1. **Sentiment Analysis**  
   Classify review sentiment using BERT.

2. **Retrieval-Augmented Generation (RAG)**  
   Build a FAISS-backed knowledge base with Sentence‑Transformers embeddings.

3. **Ad Generation**  
   Use OpenAI’s API to craft persuasive ads that address customer concerns and highlight improvements.

## Requirements
- Python 3.8+  
- Key libraries:
  - pandas, numpy
  - torch, transformers, sentence-transformers
  - faiss-cpu
  - openai
  - scikit-learn, nltk
  - tqdm, pickle
  - matplotlib, seaborn

Install dependencies:
    pip install -r requirements.txt

## Setup
1. **Clone the repository**  
    git clone <repo_url>  
    cd <repo_dir>

2. **Set your OpenAI API key**  
   In `ad_generate.py`, update:  
    openai.api_key = "YOUR_API_KEY"

3. **Download NLTK data**  
    python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

## Usage

1. **Sentiment Analysis**  
    python sentiment_analysis.py \
      --input data/reviews.csv \
      --output data/reviews_labeled.csv

2. **Build RAG Knowledge Base**  
    python rag_build.py \
      --reviews data/reviews_labeled.csv \
      --embeddings embeddings/product_embeddings.npy \
      --metadata embeddings/product_metadata.pkl

3. **Generate Advertisements**  
    python ad_generate.py \
      --reviews data/reviews_labeled.csv \
      --output ads/ \
      --openai-key YOUR_API_KEY

## Configuration
- Adjust model parameters (max token length, FAISS index settings) directly in the scripts.  
- For GPU use, ensure `torch.cuda.is_available()` returns `True`; otherwise, the scripts default to CPU.

## Tips & Resources
- Performance boosts: Consider mixed‑precision or smaller batch sizes for limited GPU memory.  
- Further reading:
  - Hugging Face Transformers documentation  
  - FAISS GitHub repository  
  - Retrieval‑Augmented Generation research paper
