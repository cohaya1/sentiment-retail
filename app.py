#!/usr/bin/env python3
"""
Streamlit App for Retail Recommendation System
"""
import os
import sys
import json
import re
import subprocess
import streamlit as st
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Retail Recommender",
    page_icon="🛍️",
    layout="wide"
)

# Title and description
st.title("🛍️ Retail Recommendation System")
st.markdown("""
This intelligent recommendation system uses PydanticAI multi-agent orchestration to:
1. ✨ **Analyze user intent** with Rasa NLU
2. 🔍 **Retrieve relevant products** with LightRAG hybrid search
3. 💭 **Analyze sentiment** with an Amazon reviews-specific model
4. 📊 **Generate structured feedback** with actionable insights
""")

# Add system information
with st.sidebar:
    st.header("System Information")
    st.markdown("""    
    - **Intent Classification**: Rasa NLU
    - **Product Retrieval**: LightRAG hybrid search engine
    - **Sentiment Analysis**: DistilBERT model fine-tuned on Amazon reviews
    - **Orchestration**: PydanticAI agents with LangGraph
    """)
    
    st.markdown("---")
    st.caption("Built with PydanticAI, PyTorch, and Transformers")

# Query section with example prompts
st.subheader("Ask about products")  
st.markdown("Type your product query or select an example below")

# Example prompts for the user to try
example_queries = [
    "Is this bluetooth speaker any good?",
    "What are some good kitchen appliances?",
    "Are these wireless headphones comfortable?",
    "Tell me about standing desks",
    "Review this smart watch for fitness"
]

# Create tabs for direct query or examples
query_tab, examples_tab = st.tabs(["Your Query", "Example Queries"])

with query_tab:
    # Input form
    with st.form("query_form"):
        query = st.text_area("What product are you looking for?", 
                        height=100,
                        value="Is this bluetooth speaker waterproof and good for outdoor use?")
        col1, col2 = st.columns([3,1])
        with col1:
            sample_data = st.checkbox("Use sample products dataset", value=True, 
                                    help="Uses a curated set of sample products from various categories")
        with col2:
            submitted = st.form_submit_button("🔍 Analyze", use_container_width=True)

with examples_tab:
    st.markdown("Select one of these example queries to try:")
    for i, example in enumerate(example_queries):
        if st.button(f"{example}", key=f"example_{i}", use_container_width=True):
            query = example
            sample_data = True
            submitted = True

if submitted:
    with st.spinner("Processing your query..."):
        # Create a progress container
        progress_container = st.empty()
        progress_container.info("Classifying user intent...")
        
        # Run the recommendation agent using subprocess
        try:
            # Prepare the command
            cmd = [
                "python", 
                "src/agents/orchestrator.py", 
                "--query", f"{query}", 
                "--user-id", "streamlit_user"
            ]
            
            # Add the data path if sample data is selected
            if sample_data:
                cmd.extend(["--data", "./sample_data/sample_products.csv"])
            
            # Execute the command
            st.text("Running command: " + " ".join(cmd))
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Display raw output for debugging
            with st.expander("Debug Output", expanded=False):
                st.code(result.stdout)
            
            # Parse JSON payload using regex to extract the JSON block
            try:
                stdout = result.stdout
                match = re.search(r'({.*})', stdout, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    results = json.loads(json_str)
                else:
                    # Show parse error in debug expander only
                    with st.expander("Parsing Error", expanded=False):
                        st.error("No valid JSON found in the output")
                        st.code(stdout)
                    results = {"products": [], "sentiments": [], "feedback": {}}
            except Exception as e:
                with st.expander("Parsing Error", expanded=False):
                    st.error("Failed to parse JSON from recommendation engine")
                    st.code(str(e))
                    st.code(stdout)
                results = {"products": [], "sentiments": [], "feedback": {}}
            
            # Display the results
            st.success("✅ Analysis complete!")
            
            # Extract products, sentiments, and feedback
            products = results.get("products", [])
            sentiments = results.get("sentiments", [])
            feedback = results.get("feedback", {})
            
            # Display feedback
            if feedback:
                st.header("Overall Feedback")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"### Summary\n{feedback.get('summary', 'No summary available')}")
                    st.markdown(f"### Response\n{feedback.get('response', 'No response available')}")
                
                with col2:
                    st.metric("Sentiment Score", f"{feedback.get('sentiment_score', 0):.2f}")
                    
                    st.subheader("Key Points")
                    for point in feedback.get("key_points", ["No key points available"]):
                        st.markdown(f"- {point}")
                    
                    st.subheader("Actionable Insights")
                    for insight in feedback.get("actionable_insights", ["No insights available"]):
                        st.markdown(f"- {insight}")
            # Display products
            if products:
                st.header(f"Found {len(products)} Products")
                
                # Create a more modern layout for products
                for i, product in enumerate(products):
                    # Get matching sentiment
                    sentiment = next((s for s in sentiments if s.get("id") == product.get("id")), {})
                    
                    # Create a card-like display with better layout
                    with st.container():
                        # Product header with price
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.subheader(product.get("name", "Unknown Product"))
                        with col2:
                            st.markdown(f"<h3 style='text-align: right; color:#1E88E5;'>${product.get('price', 0):.2f}</h3>", unsafe_allow_html=True)
                        
                        # Product info and sentiment
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**Category:** {product.get('category', 'General')}")
                            st.markdown(f"**Product ID:** {product.get('id', 'Unknown')}")
                            st.markdown("### Description")
                            st.markdown(f"{product.get('description', 'No description available')}")
                        
                        with col2:
                            # Display sentiment with improved visualization
                            sentiment_label = sentiment.get("label", "NEUTRAL")
                            sentiment_score = sentiment.get("score", 0.5)
                            
                            # Custom styling based on sentiment
                            st.markdown("### Amazon Sentiment")
                            
                            # Enhanced color coding and icons for sentiment
                            if sentiment_label == "POSITIVE":
                                color = "#28a745"  # green
                                icon = "✅"
                                interpretation = "Positive reviews likely"
                            elif sentiment_label == "NEGATIVE":
                                color = "#dc3545"  # red
                                icon = "❌"
                                interpretation = "Negative reviews likely"
                            else:
                                color = "#6c757d"  # gray
                                icon = "⚪"
                                interpretation = "Mixed reviews"
                            
                            # Show sentiment with styled badge
                            st.markdown(f"<div style='background-color:{color}; padding:10px; border-radius:5px; color:white; text-align:center;'><strong>{icon} {sentiment_label}</strong></div>", unsafe_allow_html=True)
                            
                            # Show confidence score
                            st.progress(sentiment_score)
                            st.metric("Confidence", f"{sentiment_score:.2f}")
                            st.caption(f"Model interpretation: {interpretation}")
                            
                            # Add explanation about the model
                            with st.expander("About this analysis"):
                                st.caption("This sentiment prediction comes from a DistilBERT model fine-tuned specifically on Amazon product reviews, making it highly accurate for e-commerce applications.")
                    
                    # Add divider between products
                    st.divider()
            else:
                st.info("No products found for your query. Try a different search term.")
                
        except subprocess.CalledProcessError as e:
            st.error(f"An error occurred while running the recommendation engine")
            st.code(e.stderr)
            
        except json.JSONDecodeError:
            st.error("Could not parse results from the recommendation engine")
            st.code(result.stdout if 'result' in locals() else "No output available")
            
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.code(str(e))

# Footer with more information
st.markdown("---")
col1, col2, col3 = st.columns([1,2,1])
with col1:
    st.markdown("### Technologies")
    st.markdown("""    
    - PydanticAI
    - LangGraph
    - Transformers
    - LightRAG
    - Streamlit
    """)
with col2:
    st.markdown("### About This Project")
    st.markdown("""
    This retail recommendation system demonstrates advanced multi-agent orchestration using PydanticAI. 
    The system combines intent classification, product retrieval, sentiment analysis, and feedback generation 
    in a coordinated workflow.
    
    The sentiment analysis leverages a DistilBERT model specifically fine-tuned on Amazon reviews for 
    more accurate e-commerce sentiment predictions.
    """)
with col3:
    st.markdown("### Model Details")
    st.markdown("""    
    Sentiment Model: 
    [sohan-ai/sentiment-analysis-model-amazon-reviews](https://huggingface.co/sohan-ai/sentiment-analysis-model-amazon-reviews)
    
    Embedding: OpenAI
    
    LLM: GPT-4o
    """)
