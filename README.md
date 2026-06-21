# CFPB Complaint Analytics RAG Pipeline (Tasks 1 & 2)

This repository contains the data preprocessing, exploratory data analysis (EDA), text chunking, and vector indexing pipeline for a Retrieval-Augmented Generation (RAG) system utilizing the Consumer Financial Protection Bureau (CFPB) complaint dataset.

## Project Description
The objective of this project is to ingest a large-scale financial complaint dataset, filter and clean text narratives, and build a localized semantic search vector database. The system targets four core product categories: **Credit Card, Savings Account, Money Transfer, and Personal Loan**. It implements a stratified sampling methodology to preserve representation across imbalanced classes, divides lengthy consumer logs into manageable context nodes, and encodes them into dense vectors for downstream RAG integration.

---

## Folder Structure
The repository is organized according to standard production layout guidelines:

```text
├── .github/                 
│   └── workflows/
│       └── unittests.yml
├── data/
│   ├── raw/                  
│   └── processed/            
├── notebooks/
│   ├── __init__.py
│   ├── 01_eda_cfpb.ipynb     
│   └── 02_text_chunking.ipynb
├── src/
│   ├── __init__.py
│   ├── initial_eda.py        
│   └── text_chunking.py      
├── vector_store/
│   ├── faiss_index.index     
│   ├── chunks.pkl           
│   └── metadata.pkl  
├── tests/                 
│   └── __init__.py         
├── .gitignore                
├── requirements.txt          
└── README.md                 

```

---

## Data Pipeline Pipeline Summary

### Task 1: EDA & Preprocessing Findings

* **Initial Population:** The raw corpus spans 9,609,797 records, with roughly 69% (6,629,041 records) completely lacking text narratives.
* **Target Refinement:** The pipeline filters out non-target records and empty narratives, producing a consolidated collection of **464,010 clean text entries**.
* **Text Normalization:** Narratives are cast to lowercase, standard institutional boilerplates (e.g., *"i am writing to file a complaint"*) are removed, and masking identifiers (e.g., *"xxxx"*) are scrubbed. The resulting dataset is written directly to `data/processed/filtered_complaints.csv`.

### Task 2: Vector Store Specifications

* **Sampling Matrix:** A proportional Stratified Random Sample pulls **12,000 corporate records**, maintaining proper category balance (Credit Card, Savings Account, Money Transfer, and Personal Loan).
* **Chunking Protocol:** LangChain's `RecursiveCharacterTextSplitter` breaks narratives into windows of **512 characters with a 50-character overlap**, mapping the 12,000 files into **34,605 granular text nodes**.
* **Vector Index:** The chunks are processed via `sentence-transformers/all-MiniLM-L6-v2` into **384-dimensional normalized vectors**, written onto a flat inner-product `FAISS` layout (`IndexFlatIP`) for high-speed Cosine Similarity searching.

---

## Setup and Installation

### 1. Environment Preparation

Ensure you have Python 3.10+ installed. Clone this repository and establish a clean workspace container:

```bash
# Create a local virtual environment
python -m venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

```

### 2. Install Required Dependencies

Run the install command using the project specification ledger:

```bash
pip install -r requirements.txt

```

### 3. Running the Code Pipeline

The pipeline is fully modularized and can be executed via notebooks or direct python execution.

* To run the data exploration and preprocessing stage:

```bash
python src/initial_eda.py

```


* To execute the stratification sampling, token parsing, embedding generation, and vector index persistence stage:

```bash
python src/text_chunking.py

```