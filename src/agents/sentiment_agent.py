#!/usr/bin/env python3
import os
import argparse
import asyncio
import torch
from dataclasses import dataclass
from typing import List, Any, Dict

from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification
)
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# ---------------------------------------------------------
# 1) Dependency container for injection with Amazon-specific model
# ---------------------------------------------------------
@dataclass
class SentimentDeps:
    # Load Amazon reviews sentiment model and tokenizer
    tokenizer: Any = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model: Any = DistilBertForSequenceClassification.from_pretrained(
        "sohan-ai/sentiment-analysis-model-amazon-reviews"
    )
    
    # Custom predict function instead of using pipeline
    def predict_sentiment(self, texts: List[str]) -> List[Dict]:
        """Process a list of texts and return sentiment predictions"""
        results = []
        
        for text in texts:
            # Tokenize the input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get prediction and confidence
            logits = outputs.logits
            predicted_class = logits.argmax().item()
            
            # Convert logits to probabilities with softmax
            probs = torch.nn.functional.softmax(logits, dim=1)
            confidence = probs[0][predicted_class].item()
            
            # Map to expected format (positive=1, negative=0 in this model)
            label = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
            
            results.append({
                "label": label,
                "score": confidence
            })
        
        return results

# ---------------------------------------------------------
# 2) Pydantic model for structured output
# ---------------------------------------------------------
class SentimentResult(BaseModel):
    label: str = Field(..., description="Sentiment label, e.g. POSITIVE or NEGATIVE")
    score: float = Field(..., description="Confidence score between 0 and 1")

# ---------------------------------------------------------
# 3) Initialize OpenAI Model
# ---------------------------------------------------------
model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
openai_key = os.getenv("OPENAI_API_KEY")
provider = OpenAIProvider(api_key=openai_key)
llm_model = OpenAIModel(model_name, provider=provider)

# ---------------------------------------------------------
# 4) Instantiate PydanticAI Agent
# ---------------------------------------------------------
sentiment_agent: Agent[List[SentimentResult], SentimentDeps] = Agent(
    model=llm_model,
    deps_type=SentimentDeps,
    output_type=List[SentimentResult],  # Changed from result_type to output_type per PydanticAI docs
    system_prompt=(
        "You are a sentiment classification assistant that analyzes product descriptions. "
        "Use the classify_sentiment tool to label the sentiment of each provided text snippet. "
        "Return the results in a structured format."
    ),
)

# ---------------------------------------------------------
# 5) Register the classification tool
# ---------------------------------------------------------
@sentiment_agent.tool
def classify_sentiment(
    ctx: RunContext[SentimentDeps],
    snippets: List[str]
) -> List[SentimentResult]:
    """Classify sentiment of product descriptions using Amazon-specific DistilBERT model"""
    # For debugging - print actual received snippets
    print(f"\n### AMAZON REVIEW SENTIMENT ANALYSIS ###")
    print(f"Received {len(snippets)} snippets for sentiment analysis")
    for i, s in enumerate(snippets):
        print(f"Snippet {i+1}: {type(s).__name__} | Length: {len(str(s))} | Preview: '{str(s)[:50]}...'")
    
    # Ensure all snippets have content - be more defensive with type checking
    valid_snippets = []
    for s in snippets:
        if s is None:
            continue
        snippet_str = str(s).strip() if not isinstance(s, str) else s.strip()
        if snippet_str:
            valid_snippets.append(snippet_str)
            
    if not valid_snippets:
        print("Warning: No valid snippets to analyze after filtering")
        return []  # Return empty list if no valid snippets
    
    print(f"Processing {len(valid_snippets)} valid snippets with Amazon sentiment model")
    
    try:
        # Use our custom predict_sentiment method from dependencies
        raw_results = ctx.deps.predict_sentiment(valid_snippets)
        
        # Convert raw results to SentimentResult objects
        results = []
        for i, raw_result in enumerate(raw_results):
            try:
                # Get prediction details
                label = raw_result['label']  # Already in 'POSITIVE'/'NEGATIVE' format
                score = raw_result['score']
                
                print(f"Product {i+1} result: {label} (confidence: {score:.2f})")
                
                # Create SentimentResult object
                sentiment = SentimentResult(
                    label=label,
                    score=score
                )
                results.append(sentiment)
            except Exception as snippet_error:
                print(f"Error processing result {i+1}: {str(snippet_error)}")
                results.append(SentimentResult(label="NEUTRAL", score=0.5))
        
        print(f"Successfully analyzed {len(results)}/{len(valid_snippets)} snippets")
        return results
    except Exception as e:
        print(f"Fatal error in Amazon sentiment classification: {str(e)}")
        # Fallback with neutral sentiments
        return [SentimentResult(label="NEUTRAL", score=0.5) for _ in valid_snippets]

# ---------------------------------------------------------
# 6) CLI entry point
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Sentiment Agent CLI")
    parser.add_argument(
        "--snippets",
        nargs='+',
        help="List of review snippets to classify"
    )
    args = parser.parse_args()

    deps = SentimentDeps()
    inputs = args.snippets or []

    if inputs:
        # Directly classify sentiments using the pipeline
        raw = deps.sentiment_pipe(inputs)
        for item in raw:
            res = SentimentResult(**item)
            print(res.model_dump_json(indent=2))
    else:
        # No snippets: run demo examples
        demo = ["Great product!", "Terrible experience."]
        print("No snippets provided. Running demo classifications:")
        raw = deps.sentiment_pipe(demo)
        for item in raw:
            res = SentimentResult(**item)
            print(res.model_dump_json(indent=2))

if __name__ == '__main__':
    main()
