import polars as pl
import os
import logging
from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStoreBuilder:
    def __init__(self, cleaned_data_path: str = "../data/processed/filtered_complaints.csv"):
        self.cleaned_data_path = cleaned_data_path
        self.df = None
        self.sample_df = None
        self.chunks = []
        self.embeddings = None
        self.metadata = []
        
    def load_cleaned_data(self):
        if not os.path.exists(self.cleaned_data_path):
            raise FileNotFoundError(f"Cleaned data not found at: {self.cleaned_data_path}")
        
        self.df = pl.read_csv(self.cleaned_data_path)
        logger.info(f"Loaded cleaned dataset with shape: {self.df.shape}")
        return self.df

    def create_stratified_sample(self, target_size: int = 12000):
        categories = self.df["Product_Category"].unique().to_list()
        
        counts = self.df.group_by("Product_Category").len()
        total = counts["len"].sum()
        
        sample_per_cat = {}
        for cat in categories:
            proportion = counts.filter(pl.col("Product_Category") == cat)[0, "len"] / total
            sample_per_cat[cat] = max(100, int(proportion * target_size))  # minimum 100 per category
      
        current_total = sum(sample_per_cat.values())
        if current_total != target_size:
            diff = target_size - current_total
            
            for cat in sorted(sample_per_cat, key=lambda x: sample_per_cat[x], reverse=True):
                sample_per_cat[cat] += diff // len(categories)
                if sum(sample_per_cat.values()) >= target_size:
                    break
        
        sampled_dfs = []
        for cat, n in sample_per_cat.items():
            cat_df = self.df.filter(pl.col("Product_Category") == cat)
            if len(cat_df) > 0:
                sampled = cat_df.sample(n=min(n, len(cat_df)), seed=42)
                sampled_dfs.append(sampled)
        
        self.sample_df = pl.concat(sampled_dfs)
        logger.info(f"Created stratified sample of {len(self.sample_df)} complaints")
        logger.info(f"Distribution:\n{self.sample_df.group_by('Product_Category').len()}")
        return self.sample_df

    def chunk_narratives(self, chunk_size: int = 512, chunk_overlap: int = 50):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.chunks = []
        self.metadata = []
        
        for row in self.sample_df.iter_rows(named=True):
            narrative = row["Consumer complaint narrative"]
            if not narrative or len(narrative.strip()) < 20:
                continue
                
            splits = text_splitter.split_text(narrative)
            
            for chunk in splits:
                self.chunks.append(chunk)
                self.metadata.append({
                    "complaint_id": row.get("Complaint ID") or row.get("complaint_id"),
                    "product_category": row["Product_Category"],
                    "original_narrative_length": len(narrative),
                    "chunk_length": len(chunk)
                })
        
        logger.info(f"Created {len(self.chunks)} text chunks from {len(self.sample_df)} complaints")
        return self.chunks, self.metadata

    def generate_embeddings(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        
        logger.info("Generating embeddings... (this may take a few minutes)")
        self.embeddings = model.encode(
            self.chunks, 
            batch_size=128, 
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        logger.info(f"Generated embeddings with shape: {self.embeddings.shape}")
        return self.embeddings

    def build_and_save_faiss_index(self, vector_store_dir: str = "../vector_store"):
        Path(vector_store_dir).mkdir(parents=True, exist_ok=True)
        
        dimension = self.embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  
        index.add(self.embeddings.astype(np.float32))
        
        faiss.write_index(index, os.path.join(vector_store_dir, "faiss_index.index"))
        
        with open(os.path.join(vector_store_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        
        with open(os.path.join(vector_store_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        
        logger.info(f"Vector store successfully saved to {vector_store_dir}/")
        return vector_store_dir

    def run_pipeline(self, sample_size: int = 12000):
        self.load_cleaned_data()
        self.create_stratified_sample(target_size=sample_size)
        self.chunk_narratives(chunk_size=512, chunk_overlap=50)
        self.generate_embeddings()
        self.build_and_save_faiss_index()
        logger.info("Vector Store Pipeline Completed Successfully!")


