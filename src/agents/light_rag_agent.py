#!/usr/bin/env python3
import os
import asyncio
from dataclasses import dataclass, field
from typing import List, Any, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_embed, gpt_4o_mini_complete
from lightrag.kg.shared_storage import initialize_pipeline_status

# ---------------------------------------------------------
# 1) Define dependencies for injection
# ---------------------------------------------------------
@dataclass
class AppDeps:
    rag: LightRAG

# ---------------------------------------------------------
# 2) Pydantic models for structured inputs/outputs
# ---------------------------------------------------------
class Product(BaseModel):
    id: str = Field(..., description="Product identifier")
    name: str = Field(..., description="Name of the product")
    category: str = Field(..., description="Product category")
    price: float = Field(..., description="Product price")
    description: str = Field(..., description="Product description")

# ---------------------------------------------------------
# 3) Initialize OpenAI model and PydanticAI Agent
# ---------------------------------------------------------
model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
openai_key = os.getenv("OPENAI_API_KEY")
provider = OpenAIProvider(api_key=openai_key)
llm_model = OpenAIModel(model_name, provider=provider)

agent = Agent[List[Product], AppDeps](
    model=llm_model,
    deps_type=AppDeps,
    result_type=List[Product],
    system_prompt=(
        "You are a product retrieval assistant. "
        "Use the retrieve_products tool to fetch matching products."
    )
)

# ---------------------------------------------------------
# 4) Tool: retrieve_products (async, uses aquery)
# ---------------------------------------------------------
@agent.tool
async def retrieve_products(
    ctx: RunContext[AppDeps],
    intent: str,
    keywords: List[str] = None,
) -> List[Product]:
    """
    Retrieve products by running a hybrid query against LightRAG.

    Args:
        intent: The primary intent label.
        keywords: List of extracted keywords.
    Returns:
        List of Product models.
    """
    # Debug and properly handle keywords to ensure we have a valid search query
    print(f"DEBUG - Intent: {intent}, Keywords type: {type(keywords)}, Keywords: {keywords}")
    
    # Process keywords properly to create a search query
    search_terms = []
    
    # First check if keywords have usable content
    if keywords:
        if isinstance(keywords, list):
            # Extract terms from list keywords
            for kw in keywords:
                if kw and isinstance(kw, str) and len(kw) > 2:
                    search_terms.append(kw)
        elif isinstance(keywords, str) and len(keywords) > 2:
            # Single string keyword
            search_terms.append(keywords)
    
    # Extract search terms from intent if useful
    if intent and isinstance(intent, str) and not any(x in intent.lower() for x in ['retrieve', 'product_search', 'search']):
        search_terms.append(intent)
    
    # Create query text from search terms
    if search_terms:
        query_text = ' '.join(search_terms)
    else:
        # Fallback to a generic search if no good terms found
        query_text = "product"
    
    # Ensure we have a non-empty search string
    if not query_text.strip():
        query_text = "product"
        
    print(f"Searching for products with query: '{query_text}'")
    
    # Use a hybrid search for better results
    result = await ctx.deps.rag.aquery(query_text, param=QueryParam(mode="hybrid"))

    products: List[Product] = []
    chunks = getattr(result, 'chunks', None)
    print(f"Found {len(chunks) if chunks else 0} chunks in search results")
    
    if chunks:
        for chunk in chunks[:5]:
            # Extract metadata consistently
            meta = getattr(chunk, 'metadata', None) or getattr(chunk, 'meta', {}) or {}
            
            # Log what we found
            product_id = meta.get('id', '') or meta.get('asin', '')
            product_name = meta.get('name', '')
            print(f"Processing product: {product_name} (ID: {product_id})")
            products.append(Product(
                id=meta.get('id', ''),
                name=meta.get('name', ''),
                category=meta.get('category', 'General'),
                price=float(meta.get('price', 0.0)),
                description=meta.get('text', getattr(chunk, 'content', '')),
            ))
    # Else: no chunks -> return empty list
    return products

# ---------------------------------------------------------
# 5) Tool: index_csv (sync wrapper)
# ---------------------------------------------------------
@agent.tool
def index_csv(
    ctx: RunContext[AppDeps],
    file_path: str,
    limit: int = 100,
) -> str:
    import sys
    import os
    # Add the project root to the path for relative imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.data_indexing.simple_indexer import load_csv_products
    products = load_csv_products(file_path, limit)
    contents, ids = [], []
    for p in products:
        txt = f"Product: {p.title}\n"
        if p.description:
            txt += f"Description: {p.description}\n"
        txt += f"Category: {p.category}\n"
        if p.price:
            txt += f"Price: ${p.price:.2f}\n"
        contents.append(txt)
        ids.append(p.asin)
    # Directly use the synchronous insert method since we're already in a sync context
    # We can't create nested event loops
    ctx.deps.rag.insert(contents, ids=ids)
    return f"Indexed {len(products)} products"

# ---------------------------------------------------------
# 6) CLI entrypoint
# ---------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser("Product agent CLI")
    parser.add_argument("--index-csv", help="CSV to index via index_csv tool")
    parser.add_argument("--query", help="Text query to retrieve products")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    rag = LightRAG(
        working_dir=os.getenv("LIGHTRAG_DIR", "./lightrag_data"),
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    asyncio.run(rag.initialize_storages())
    asyncio.run(initialize_pipeline_status())
    deps = AppDeps(rag=rag)

    if args.index_csv:
        res = agent.run_sync(
            "index_csv", file_path=args.index_csv, limit=args.limit, deps=deps
        )
        print(res)

    elif args.query:
        tokens = args.query.split()
        res = agent.run_sync(
            "retrieve_products", intent="product_search", keywords=tokens, deps=deps
        )
        for p in res:
            print(p.json())

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
