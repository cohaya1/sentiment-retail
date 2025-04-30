#!/usr/bin/env python3
import asyncio
import os
import sys
import json
from types import SimpleNamespace

# Fix import paths for both direct execution and import
import os
import sys

# Add the project root to sys.path when running as script
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, project_root)

# Now we can import from either way
from src.agents.user_intent_agent import IntentDeps, identify_intent
from src.agents.light_rag_agent import agent as product_agent, AppDeps, LightRAG, openai_embed, gpt_4o_mini_complete, Product
from lightrag.kg.shared_storage import initialize_pipeline_status
from src.agents.sentiment_agent import sentiment_agent, SentimentDeps, SentimentResult
from src.agents.feedback_agent import process_feedback  # pure-Python feedback processor

async def run_retail_recommendation_agent(user_input: str, user_id: str = "anonymous", data_path: str = None) -> dict:
    # 1. Classify user intent
    deps_intent = IntentDeps(rasa_url=os.getenv("RASA_URL", "http://localhost:5005"))
    ctx_intent = SimpleNamespace(deps=deps_intent)
    intent = identify_intent(ctx_intent, user_input)

    # 2. Set up LightRAG and initialize
    rag = LightRAG(
        working_dir=os.getenv("LIGHTRAG_DIR", "./lightrag_data"),
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    deps_prod = AppDeps(rag=rag)
    
    # 2a. Load sample product data directly without going through the agent
    sample_path = "./sample_data/sample_products.csv"
    if os.path.exists(sample_path):
        print(f"Loading sample products from {sample_path}")
        # Import directly 
        sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../..'))
        from src.data_indexing.simple_indexer import load_csv_products
        
        products = load_csv_products(sample_path, limit=10)
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
        
        # Insert directly into RAG using the async method since we're in an async context
        await rag.ainsert(contents, ids=ids)
        print(f"Indexed {len(products)} products from sample data")
    # 2. Simple approach: directly search for products with user's query terms
    print(f"Using intent \"{intent.primary_intent}\" with keywords {intent.keywords}")
    
    # Extract meaningful search terms from the user's query
    # This simple approach uses the user's input directly
    search_terms = [word.lower() for word in user_input.split() 
                  if len(word) > 3 and word.lower() not in ['what', 'when', 'where', 'which', 'with', 'that', 'this', 'good', 'does']]
    
    # Ensure we have a valid search query
    search_query = ' '.join(search_terms) if search_terms else "product"
    print(f"Searching with query: '{search_query}'")
    
    # Following official LightRAG example - simple query with hybrid mode
    from lightrag import QueryParam
    rag_result = await rag.aquery(search_query, param=QueryParam(mode="hybrid"))
    
    # Convert chunks to products
    products = []
    chunks = getattr(rag_result, 'chunks', [])
    print(f"Found {len(chunks)} matching items")
    
    # If no results with hybrid search, try a more general search
    if not chunks:
        print("No results with hybrid search, trying simple text matching")
        # Simple manual search through our indexed products
        with open(sample_path, 'r') as f:
            import csv
            reader = csv.DictReader(f)
            for row in reader:
                # Check if any search term is in title or description
                title = row.get('title', '').lower()
                desc = row.get('description', '').lower()
                if any(term in title or term in desc for term in search_terms):
                    print(f"Found match in CSV: {row.get('title')}")
                    products.append(Product(
                        id=row.get('asin', f"product_{len(products)}"),
                        name=row.get('title', 'Unknown Product'),
                        category=row.get('category', 'General'),
                        price=float(row.get('price', 0.0)),
                        description=row.get('description', '')
                    ))
    else:
        # Process LightRAG chunks into products
        for chunk in chunks[:5]:
            content = getattr(chunk, 'content', '')
            print(f"Processing chunk: {content[:100]}...")
            
            # Parse product details from content
            lines = content.split('\n') if content else []
            product_data = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if 'product' in key:
                        product_data['name'] = value
                    elif 'category' in key:
                        product_data['category'] = value
                    elif 'price' in key:
                        try:
                            product_data['price'] = float(value.replace('$', ''))
                        except:
                            product_data['price'] = 0.0
                    elif 'description' in key:
                        product_data['description'] = value
            
            products.append(Product(
                id=getattr(chunk, 'id', f"product_{len(products)}"),
                name=product_data.get('name', 'Unknown Product'),
                category=product_data.get('category', 'General'),
                price=product_data.get('price', 0.0),
                description=product_data.get('description', content)
            ))
    
    print(f"Final product count: {len(products)}")

    # 3. Analyze sentiment
    if not products:
        # Handle the case when no products are found
        sentiments = []
        print("No products to analyze for sentiment")
    else:
        # Debug product structure
        print(f"Products for sentiment analysis: {len(products)}")
        for i, p in enumerate(products):
            print(f"Product {i+1}: {p.name} | Desc type: {type(p.description).__name__} | Has desc: {bool(p.description)}")
        
        # Extract product descriptions for sentiment analysis
        # Ensure we're getting actual strings for the descriptions
        descriptions = []
        for p in products:
            if hasattr(p, 'description') and p.description:
                # Ensure it's a string and not empty
                desc = str(p.description).strip()
                if desc:
                    descriptions.append(desc)
                    print(f"Added description: {desc[:50]}...")
        
        print(f"Analyzing sentiment for {len(descriptions)} product descriptions")
        
        # Make sure we have at least one description to analyze
        if not descriptions:
            print("Warning: No product descriptions to analyze")
            sentiments = [{
                "id": p.id,
                "name": p.name,
                "label": "NEUTRAL",
                "score": 0.5,
            } for p in products]
        else:
            # Direct approach with PydanticAI structure but ensuring results
            try:
                # Initialize sentiment dependencies with the pipeline
                deps_sent = SentimentDeps()
                print(f"\n### DIRECT SENTIMENT ANALYSIS ###")
                print(f"Processing {len(descriptions)} descriptions with transformers")
                
                # Directly use the sentiment pipeline from the dependencies
                results = []
                sentiment_results = []
                
                # Process each description individually for reliable results
                for i, description in enumerate(descriptions):
                    try:
                        # Use the Amazon-specific predict_sentiment method
                        result = deps_sent.predict_sentiment([description])[0]
                        
                        # Get the prediction (already in POSITIVE/NEGATIVE format)
                        label = result['label']
                        score = float(result['score'])
                        print(f"Product {i+1} sentiment: {label} (score: {score:.2f})")
                        
                        # Instead of creating SentimentResult directly, just use label and score
                        # We'll create proper sentiment objects directly in the next step
                        sentiment_results.append({
                            'label': label,
                            'score': score
                        })
                    except Exception as e:
                        print(f"Error analyzing sentiment for product {i+1}: {str(e)}")
                        # Default neutral sentiment if analysis fails
                        sentiment_results.append(SentimentResult(label="NEUTRAL", score=0.5))
                
                # Create sentiment objects that match our products
                sentiments = []
                for i, (product, sentiment) in enumerate(zip(products, sentiment_results)):
                    # Get label and score from the dictionary
                    label = sentiment['label'] if isinstance(sentiment, dict) else 'NEUTRAL'
                    score = sentiment['score'] if isinstance(sentiment, dict) else 0.5
                    
                    print(f"Final sentiment for {product.name}: {label} (score: {score:.2f})")
                    
                    # Create the sentiment object for the response
                    sentiments.append({
                        "id": product.id,
                        "name": product.name,
                        "label": label,
                        "score": score
                    })
                    
            except Exception as e:
                print(f"Error in sentiment analysis: {str(e)}")
                # Fallback with safe values
                sentiments = [{
                    "id": p.id,
                    "name": p.name,
                    "label": "NEUTRAL",
                    "score": 0.5
                } for p in products]

    # 4. Generate structured feedback synchronously via thread
    fb_result = await asyncio.to_thread(
        lambda: process_feedback([p.model_dump() for p in products], sentiments)
    )
    feedback = fb_result.model_dump()
    # assemble full recommendation payload
    payload = {
        "products": [p.model_dump() for p in products],
        "sentiments": sentiments,
        "feedback": feedback
    }
    return payload

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Retail Recommendation Orchestrator")
    parser.add_argument("--query", type=str, required=True, help="User query for recommendations")
    parser.add_argument("--user-id", type=str, default="anonymous", help="User ID")
    parser.add_argument("--data", type=str, help="Path to product data CSV file")
    args = parser.parse_args()

    result = asyncio.run(run_retail_recommendation_agent(args.query, args.user_id, args.data))
    print(json.dumps(result, indent=2))
