#!/usr/bin/env python3
"""
Simple Product Indexer for LightRAG
A streamlined script to index product data into LightRAG
"""
import os
import asyncio
import argparse
import pandas as pd
from typing import List
from pydantic import BaseModel, Field
import dotenv

# Import LightRAG and required modules
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

# Load environment variables from .env file
dotenv.load_dotenv()

# Set working directory
WORKING_DIR = "./lightrag_data"
os.makedirs(WORKING_DIR, exist_ok=True)

class ProductItem(BaseModel):
    """Product data model for indexing to LightRAG"""
    asin: str = Field(..., description="Product identifier")
    title: str = Field(..., description="Product name/title")
    category: str = Field(..., description="Product category")
    price: float = Field(None, description="Product price if available")
    description: str = Field(None, description="Product description")

def load_csv_products(file_path: str, limit: int = None) -> List[ProductItem]:
    """Load products from a CSV file"""
    # Don't override the input path
    actual_path = file_path
    products = []
    print(f"Loading products from {actual_path}")
    
    try:
        # Check if the file exists and is accessible
        if not os.path.exists(actual_path):
            print(f"Error: File not found at {actual_path}")
            return products
            
        df = pd.read_csv(actual_path)
        for i, row in enumerate(df.itertuples()):
            if limit and i >= limit:
                break
                
            # Extract data using pd.Series row
            try:
                price = None
                if hasattr(row, 'price'):
                    try:
                        price = float(str(row.price).replace('$', '').replace(',', ''))
                    except (ValueError, TypeError):
                        pass
                
                product = ProductItem(
                    asin=str(row.asin) if hasattr(row, 'asin') else f"product_{i}",
                    title=str(row.title) if hasattr(row, 'title') else "Unknown Product",
                    category=str(row.category) if hasattr(row, 'category') else "General",
                    price=price,
                    description=str(row.description) if hasattr(row, 'description') else None
                )
                products.append(product)
            except Exception as e:
                print(f"Error processing row {i}: {e}")
                continue
    except Exception as e:
        print(f"Error loading CSV file: {e}")
    
    print(f"Loaded {len(products)} products")
    return products

async def initialize_rag():
    """Initialize LightRAG with OpenAI embedding and LLM"""
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete
    )
    
    # Initialize storage systems
    await rag.initialize_storages()
    await initialize_pipeline_status()
    
    return rag

async def main():
    """Main async function for indexing products"""
    parser = argparse.ArgumentParser(description="Simple Product Indexer for LightRAG")
    parser.add_argument("--file", type=str, required=True, help="Path to product data CSV file")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of products to index")
    args = parser.parse_args()
    
    # Load products from CSV
    products = load_csv_products(args.file, args.limit)
    
    if not products:
        print("No products loaded. Exiting.")
        return
    
    # Initialize LightRAG
    try:
        rag = await initialize_rag()
        print("LightRAG initialized successfully!")
        
        # Prepare documents for indexing
        documents = []
        document_ids = []
        
        for product in products:
            # Create a formatted document for each product
            content = f"Product: {product.title}\n"
            if product.description:
                content += f"Description: {product.description}\n"
            content += f"Category: {product.category}\n"
            if product.price:
                content += f"Price: ${product.price:.2f}\n"
            
            documents.append(content)
            document_ids.append(product.asin)
        
        # Insert documents into LightRAG
        print(f"Indexing {len(documents)} products...")
        await rag.ainsert(documents, ids=document_ids)
        print("Indexing complete!")
        
        # Test query
        test_query = "Show me product recommendations for kitchen"
        print(f"\nTesting LightRAG with query: '{test_query}'")
        # Perform aquery and print raw LLM response (chunk-level parsing not supported in simple mode)
        response = await rag.aquery(test_query, param=QueryParam(mode="hybrid"))
        print(f"\nLLM response: {response}")
        
        # Clean up
        await rag.finalize_storages()
        print("LightRAG finalized successfully!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
