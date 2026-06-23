import polars as pl
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import logging
import ast 

from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, parquet_path: str = "../data/raw/complaint_embeddings.parquet"):
        self.parquet_path = parquet_path
        self.df = None
        self.embedding_model = None
        self.generator = None
        self._load_parquet()

    def _load_parquet(self):
        self.df = pl.read_parquet(self.parquet_path)
        logger.info(f" Loaded pre-built parquet with {len(self.df):,} chunks")

        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        self.generator = pipeline(
            "text-generation",
            model="microsoft/Phi-3-mini-4k-instruct",
            device="cpu",               
            torch_dtype="auto",
            max_new_tokens=700,
            temperature=0.05,      
            do_sample=False,      
            pad_token_id=50256,
            top_p=0.9
        )

    def retrieve(self, question: str, k: int = 5):
        """Retrieve top-k similar chunks using pre-computed embeddings"""
        
        query_embedding = self.embedding_model.encode([question], normalize_embeddings=True)[0]

        embeddings = np.stack(self.df["embedding"].to_numpy())
        similarities = np.dot(embeddings, query_embedding)
        
        top_k_idx = np.argsort(similarities)[-k:][::-1]

        context = self.df["document"][top_k_idx].to_list()
        metadata = self.df["metadata"][top_k_idx].to_list()

        return context, metadata

    def query(self, question: str, k: int = 5) -> Dict:
    
        context, metadata = self.retrieve(question, k)
        
        context_str = "\n\n".join([f"Excerpt {i+1}: {c}" for i, c in enumerate(context)])
        
        prompt = f"""You are a strict financial analyst.
Answer the question **only using the excerpts below**. 
Do not add any numbers, statistics, or general knowledge.
Do not make up information.
If the excerpts do not clearly answer the question, reply exactly: 
"I don't have enough relevant information from the complaints to answer this."

Context:
{context_str}

Question: {question}

Answer:"""

        output = self.generator(prompt, return_full_text=False)
        answer = output[0]['generated_text'].strip()
        
        if "according to" in answer.lower() or "survey" in answer.lower() or "industry" in answer.lower():
            answer = "I don't have enough relevant information from the complaints to answer this."
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_metadata": metadata[:3],
            "num_chunks_used": k
        }



if __name__ == "__main__":
    rag = RAGPipeline()
    
    result = rag.query("What are the most common complaints about Credit Cards?")
    print("Question:", result["question"])
    print("\nAnswer:\n", result["answer"])