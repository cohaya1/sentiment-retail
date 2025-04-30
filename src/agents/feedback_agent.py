#!/usr/bin/env python3
import os
import asyncio
import json
from typing import List
from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIResponsesModel

class FeedbackResult(BaseModel):
    """Schema for structured feedback processing results."""
    summary: str = Field(..., description="Concise summary of the user's feedback.")
    sentiment_score: float = Field(..., description="Overall sentiment score from 0 to 1.")
    key_points: List[str] = Field(..., description="Key points extracted from the feedback.")
    actionable_insights: List[str] = Field(..., description="Suggestions based on the feedback.")
    response: str = Field(..., description="Generated response to the user.")

# Initialize the OpenAI provider and model
provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
model = OpenAIResponsesModel(
    model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
    provider=provider
)

# Build the Feedback Agent
feedback_agent = Agent[
    None,          # no deps_type
    FeedbackResult # structured output
](
    model=model,
    output_type=FeedbackResult,
    system_prompt=(
        "You are a feedback processor. Given pre-computed statistics, key points, and insights, "
        "format the response as a FeedbackResult JSON only."
    ),
)

# --- Tools for function-calling (used internally) ---
@feedback_agent.tool
async def compute_statistics(
    ctx: RunContext,
    sentiments: List[dict]
) -> dict:
    total = len(sentiments)
    avg = sum(s.get('score', 0) for s in sentiments) / max(total, 1)
    pos = sum(1 for s in sentiments if s.get('label', '').upper() == 'POSITIVE')
    neg = sum(1 for s in sentiments if s.get('label', '').upper() == 'NEGATIVE')
    neu = total - pos - neg
    return {
        'average_score': avg,
        'positive_count': pos,
        'negative_count': neg,
        'neutral_count': neu
    }

@feedback_agent.tool
async def extract_key_points(
    ctx: RunContext,
    products: List[dict],
    sentiments: List[dict]
) -> List[str]:
    paired = sorted(
        zip(products, sentiments),
        key=lambda ps: ps[1].get('score', 0),
        reverse=True
    )
    return [
        f"{p.get('name')} had a sentiment score of {s.get('score'):.2f}"
        for p, s in paired[:3]
    ]

@feedback_agent.tool
async def generate_insights(
    ctx: RunContext,
    statistics: dict,
    key_points: List[str]
) -> List[str]:
    insights = []
    avg = statistics.get('average_score', 0)
    if avg < 0.3:
        insights.append("Improve product quality to boost satisfaction.")
    elif avg < 0.7:
        insights.append("Enhance popular features highlighted by users.")
    else:
        insights.append("Maintain current performance; sentiment is high.")
    insights.extend([f"Highlight: {kp}" for kp in key_points])
    return insights

# --- Main processing functions ---
async def process_feedback_async(
    products: List[dict],
    sentiments: List[dict]
) -> FeedbackResult:
    """
    Orchestrate internal tools then call LLM to format FeedbackResult.
    """
    # compute intermediate data
    stats = await compute_statistics(None, sentiments)
    key_points = await extract_key_points(None, products, sentiments)
    insights = await generate_insights(None, stats, key_points)

    # prepare prompt with JSON strings
    context = {
        'statistics': stats,
        'key_points': key_points,
        'insights': insights
    }
    prompt = (
        "Based on the following data, return only the FeedbackResult JSON:\n" +
        json.dumps(context, indent=2)
    )

    result = await feedback_agent.run(prompt)
    return result.output


def process_feedback(
    products: List[dict],
    sentiments: List[dict]
) -> FeedbackResult:
    """Synchronous wrapper."""
    return asyncio.run(process_feedback_async(products, sentiments))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Feedback Agent CLI")
    parser.add_argument("--products", required=True)
    parser.add_argument("--sentiments", required=True)
    args = parser.parse_args()
    prods = json.loads(args.products)
    sents = json.loads(args.sentiments)
    fb = process_feedback(prods, sents)
    print(fb.model_dump_json(indent=2))
