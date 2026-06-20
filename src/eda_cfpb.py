import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class InitialEda:
    def __init__(self, filepath: str):
        self.filepath=filepath
        self.df=None

    def load_data(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Data file not found: {self.filepath}")

        self.df=pl.read_csv(self.filepath, null_values=["N/A", "NA", "null", ""])
        return self.df

       

    def data_summary(self):
        print('=' * 75)
        print('The data types/schema of the Dataframe \n')
        print(self.df.schema)
        print('=' * 75)
        print('\n')
        print('The missing data\n')
        print(self.df.null_count().transpose(include_header=True))
        print('=' * 75)

    def analyze_product_distribution(self):
        dist = self.df.group_by("Product").len().sort("len", descending=True)
        print('=' * 75)
        print("\nComplaint Distribution by Product:")
        print(dist)
        print('=' * 75)
        return dist

    def analyze_narrative_lengths(self):
        self.df = self.df.with_columns(
            pl.col("Consumer complaint narrative")
            .str.split(by=" ")
            .list.len()
            .alias("word_count")
        )

        print("\nNarrative Length Statistics:")
        print(self.df.select("word_count").describe())

        plt.figure(figsize=(10, 6))
        sns.histplot(self.df["word_count"].to_numpy(), bins=50, kde=True)
        plt.title("Distribution of Narrative Word Counts")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")
        plt.show()
        
        return self.df.select(["word_count"]).describe()

    def count_narrative_presence(self):
        counts = self.df.select([
            pl.col("Consumer complaint narrative").is_null().sum().alias("missing"),
            pl.col("Consumer complaint narrative").is_not_null().sum().alias("present")
        ])
        print('=' * 75)
        print("\nNarrative Presence Summary:")
        print(counts)
        print('=' * 75)
        return counts
    
    def clean_data(self):
        self.df = self.df.with_columns(
            pl.col("Product").str.to_lowercase().alias("Product_Lower")
        )
        
        self.df = self.df.with_columns(
            pl.when(pl.col("Product_Lower").str.contains(r"credit card|prepaid card"))
            .then(pl.lit("Credit Card"))
           
            .when(pl.col("Product_Lower").str.contains(r"payday loan|title loan|personal loan"))
            .then(pl.lit("Personal Loan"))
            
            .when(pl.col("Product_Lower").str.contains(r"checking|savings"))
            .then(pl.lit("Savings Account"))
            
          
            .when(pl.col("Product_Lower").str.contains(r"money transfer"))
            .then(pl.lit("Money Transfer"))
            
            .otherwise(pl.lit(None))
            .alias("Product_Category")
        )

        self.df = self.df.drop("Product_Lower")
        self.df = self.df.filter(
            pl.col("Product_Category").is_not_null() & 
            pl.col("Consumer complaint narrative").is_not_null()
        )
        
        logger.info(f"Data cleaned. New shape: {self.df.shape}")
        return self.df


    def clean_narratives_text(self):
      
        patterns = [
            r"i am writing to file a complaint",
            r"to whom it may concern",
            r"dear cfpb",
            r"my account number is xxxx",
            r"xxxx", 
            r"please see attached",
            r"i would like to report"
        ]
        
        
        cleaned = pl.col("Consumer complaint narrative").str.to_lowercase()
        
        for p in patterns:
            cleaned = cleaned.str.replace_all(p, "", literal=False)
            
        cleaned = (
            cleaned.str.replace_all(r"[^a-zA-Z0-9\s\.\?!]", "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )
        
        self.df = self.df.with_columns(cleaned.alias("Consumer complaint narrative"))
        
        logger.info("Narratives successfully cleaned and normalized.")
        return self.df
    

    def perform_text_eda(self, stage_name="Data"):
        
        print(f"\n--- NLP EDA: {stage_name} ---")
        
        if "word_count" not in self.df.columns:
            self.df = self.df.with_columns(
                pl.col("Consumer complaint narrative").str.split(" ").list.len().alias("word_count")
            )

        print("\n[Class Balance]\n", self.df.group_by("Product_Category").len().sort("len", descending=True))
        
        unique_words = self.df.select(pl.col("Consumer complaint narrative").str.split(" ").list.explode().n_unique()).item()
        print(f"\nTotal Unique Vocabulary Size: {unique_words}")

        outliers = self.df.filter((pl.col("word_count") < 5) | (pl.col("word_count") > 1000))
        print(f"Potential Outliers (<5 or >1000 words): {outliers.height}")

        
        plt.figure(figsize=(12, 5))
        sns.histplot(self.df["word_count"].to_numpy(), bins=50, kde=True)
        plt.title(f"Distribution of Narrative Lengths ({stage_name})")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")
        plt.show()

    def impute_metadata(self):
        
        sub_product_map = (
            self.df.drop_nulls(subset=["Sub-product"])
            .group_by("Product", "Sub-product")
            .len()
            .sort("len", descending=True)
            .unique("Product", keep="first")
            .select(["Product", "Sub-product"])
        )
        
        self.df = self.df.join(sub_product_map, on="Product", how="left", suffix="_mode")
        self.df = self.df.with_columns(
            pl.col("Sub-product").fill_null(pl.col("Sub-product_mode"))
        ).drop("Sub-product_mode")
        
       
        sub_issue_map = (
            self.df.drop_nulls(subset=["Sub-issue"])
            .group_by("Issue", "Sub-issue")
            .len()
            .sort("len", descending=True)
            .unique("Issue", keep="first")
            .select(["Issue", "Sub-issue"])
        )
        
        self.df = self.df.join(sub_issue_map, on="Issue", how="left", suffix="_mode")
        self.df = self.df.with_columns(
            pl.col("Sub-issue").fill_null(pl.col("Sub-issue_mode"))
        ).drop("Sub-issue_mode")
        
        self.df = self.df.with_columns([
            pl.col("Sub-product").fill_null("Unspecified"),
            pl.col("Sub-issue").fill_null("Unspecified")
        ])
        
        logger.info("Metadata imputed.")
        return self.df

    def prepare_metadata(self):
        self.df = self.df.drop(["Tags", "Consumer disputed?"])
        
        metadata_map = {
            "Company public response": "No public response",
            "State": "Unknown"
        }
        for col, placeholder in metadata_map.items():
            if col in self.df.columns:
                self.df = self.df.with_columns(pl.col(col).fill_null(placeholder))
        
        logger.info("Metadata standardized for RAG indexing.")
        return self.df

    def save_csv(self, filename: str = "filtered_complaints.csv"):
        output_dir = "../data/processed/"
        filepath = os.path.join(output_dir, filename)

        try:
            os.makedirs(output_dir, exist_ok=True)
            self.df.write_csv(filepath)
            logger.info(f"Successfully saved cleaned data to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            raise e