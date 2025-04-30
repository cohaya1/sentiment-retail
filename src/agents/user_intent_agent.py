#!/usr/bin/env python3
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any

import requests
import os
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

# 1) Dependency container
@dataclass
class IntentDeps:
    rasa_url: str = "http://localhost:5005"

# 2) Pydantic model for the final output
class UserIntent(BaseModel):
    primary_intent: str = Field(..., description="Parsed intent label (or 'unknown' on failure)")
    keywords: List[str] = Field(default_factory=list, description="Entity values from Rasa")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata or error info")

# 3) Instantiate the agent with result_type
model_name = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
provider = GroqProvider(api_key=os.getenv("GROQ_API_KEY"))
model = GroqModel(model_name, provider=provider)

intent_agent = Agent(
    model=model,
    deps_type=IntentDeps,
    result_type=UserIntent,
    system_prompt=(
        "You are an intent‐parsing assistant. "
        "Use the identify_intent tool to convert raw user text into a UserIntent object."
    ),
)

# 4) Register the tool, with built‐in error handling
@intent_agent.tool
def identify_intent(
    ctx: RunContext[IntentDeps],
    query: str,
) -> UserIntent:
    try:
        resp = requests.post(
            f"{ctx.deps.rasa_url}/model/parse",
            json={"text": query},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        intent = data.get("intent", {})
        entities = data.get("entities", [])
        # Normalize and map raw Rasa intent to two canonical categories
        raw_intent = intent.get("name", "unknown").lower()
        if any(k in raw_intent for k in ["review", "sentiment"]):
            primary = "sentiment_query"
        elif any(k in raw_intent for k in ["recommend", "provide", "suggest", "find", "search"]):
            primary = "product_search"
        else:
            primary = "product_search"
        # Extract keywords: entity values or fallback from query
        keywords = [e.get("value") for e in entities if e.get("value")]
        if not keywords:
            keywords = [w for w in query.lower().split() if len(w) > 3]
        return UserIntent(
            primary_intent=primary,
            keywords=keywords,
            context={
                "raw_intent": raw_intent,
                "confidence": intent.get("confidence"),
                "entities": entities,
            },
        )
    except requests.exceptions.RequestException as e:
        # Simulate a Rasa response without showing the full error trace
        q = query.lower()
        if any(k in q for k in ["review", "sentiment"]):
            primary = "sentiment_query"
        else:
            primary = "product_search"
        # Extract keywords from text
        keywords = [w for w in q.split() if len(w) > 3]
        return UserIntent(
            primary_intent=primary,
            keywords=keywords,
            context={"fallback_mode": True, "query": query},
        )

# 5) Demo invocation
if __name__ == "__main__":
    deps = IntentDeps(rasa_url="http://localhost:5005")

    # Synchronous
    sync_res = intent_agent.run_sync("Show me reviews for wireless earbuds", deps=deps)
    # use model_dump_json() instead of .json()
    print("Sync result →", sync_res.data.model_dump_json(indent=2))

    # Asynchronous
    async def main():
        async_res = await intent_agent.run("Recommend budget gaming laptops", deps=deps)
        print("Async result →", async_res.data.model_dump_json(indent=2))

    asyncio.run(main())
