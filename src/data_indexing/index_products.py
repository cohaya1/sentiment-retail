#!/usr/bin/env python3
import os
import argparse
import asyncio
from typing import List, Optional, Dict, Any

import pandas as pd
import textract
from tqdm import tqdm
from pydantic import BaseModel, Field
import dotenv

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

dotenv.load_dotenv()

WORKING_DIR = os.getenv("LIGHTRAG_DIR", "./lightrag_data")
os.makedirs(WORKING_DIR, exist_ok=True)


class ProductItem(BaseModel):
    asin: str = Field(..., description="Amazon product ID")
    title: str = Field(..., description="Product name")
    category: str = Field(..., description="Product category")
    price: Optional[float] = Field(None, description="Product price")
    description: Optional[str] = Field(None, description="Product description")

    def to_document(self) -> Dict[str, Any]:
        parts = [f"Product: {self.title}"]
        if self.description:
            parts.append(f"Description: {self.description}")
        parts.append(f"Category: {self.category}")
        if self.price is not None:
            parts.append(f"Price: ${self.price:.2f}")
        content = "\n".join(parts)
        metadata = {
            "id": self.asin,
            "name": self.title,
            "category": self.category,
            "price": self.price or 0.0,
            "text": self.description or f"A {self.category} product",
        }
        return {"id": self.asin, "content": content, "metadata": metadata}


def parse_price(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        import re
        cleaned = re.sub(r"[^\d\.]", "", value)
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def load_jsonl(path: str, limit: Optional[int]) -> List[ProductItem]:
    items = []
    with open(path, "r") as f:
        for i, line in enumerate(tqdm(f, desc="loading jsonl")):
            if limit and i >= limit:
                break
            try:
                data = __import__("json").loads(line)
                desc = data.get("description")
                if isinstance(desc, list):
                    desc = "\n".join(map(str, desc))
                items.append(ProductItem(
                    asin=data.get("asin", f"p{i}"),
                    title=str(data.get("title", "")),
                    category=str(data.get("category", "")),
                    price=parse_price(data.get("price")),
                    description=desc,
                ))
            except Exception:
                continue
    return items


def load_dataframe(path: str, limit: Optional[int]) -> List[ProductItem]:
    df = pd.read_excel(path) if path.lower().endswith(("xls", "xlsx")) else pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    mapping = {
        "asin": ["asin", "id", "sku"],
        "title": ["title", "name", "product"],
        "category": ["category", "type", "department"],
        "price": ["price", "cost", "amount"],
        "description": ["description", "details", "info"],
    }
    items = []
    for i, row in enumerate(df.itertuples(index=False)):
        if limit and i >= limit:
            break
        data = {}
        for field, opts in mapping.items():
            for opt in opts:
                if opt in cols:
                    data[field] = getattr(row, cols[opt])
                    break
        items.append(ProductItem(
            asin=str(data.get("asin", f"p{i}")),
            title=str(data.get("title", "")),
            category=str(data.get("category", "")),
            price=parse_price(data.get("price")),
            description=data.get("description"),
        ))
    return items


def load_doc(path: str, limit: Optional[int]) -> List[ProductItem]:
    text = textract.process(path).decode("utf-8")
    products, block = [], {}
    for line in text.splitlines():
        line = line.strip()
        if not line and "title" in block:
            products.append(ProductItem(
                asin=block.get("asin", f"doc_{len(products)}"),
                title=block.get("title", ""),
                category=block.get("category", ""),
                price=parse_price(block.get("price")),
                description=block.get("description"),
            ))
            block = {}
            if limit and len(products) >= limit:
                break
            continue
        for key in ("title", "category", "price", "description"):
            if key in line.lower() and ":" in line:
                block[key] = line.split(":", 1)[1].strip()
                break
        else:
            if "description" in block:
                block["description"] += " " + line
    if "title" in block:
        products.append(ProductItem(
            asin=block.get("asin", f"doc_{len(products)}"),
            title=block.get("title", ""),
            category=block.get("category", ""),
            price=parse_price(block.get("price")),
            description=block.get("description"),
        ))
    return products


def load_products(path: str, limit: Optional[int] = None) -> List[ProductItem]:
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "jsonl":
        return load_jsonl(path, limit)
    if ext in ("csv", "txt", "xls", "xlsx"):
        return load_dataframe(path, limit)
    if ext in ("pdf", "doc", "docx", "ppt", "pptx"):
        return load_doc(path, limit)
    return []


async def init_rag(dir: str) -> LightRAG:
    rag = LightRAG(
        working_dir=dir,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()            # setup vector/graph/KV stores :contentReference[oaicite:2]{index=2}
    await initialize_pipeline_status()         # prepare pipeline metadata :contentReference[oaicite:3]{index=3}
    return rag


async def main():
    # decide files or patterns here; example uses a sample CSV
    items = load_products("./sample_data/sample_products.csv", limit=5)
    rag = await init_rag(WORKING_DIR)
    
    # batch-insert using ainsert() to avoid nested loops :contentReference[oaicite:4]{index=4}
    docs = [item.to_document() for item in items]
    batch = 10
    for i in tqdm(range(0, len(docs), batch), desc="indexing"):
        chunk = docs[i : i + batch]
        await rag.ainsert(                         # use the async insert method
            [d["content"] for d in chunk],
            ids=[d["id"] for d in chunk]
        )                                         # async bulk insert :contentReference[oaicite:5]{index=5}

    # test hybrid query 
    res = await rag.query(
        "Show me product recommendations for kitchen",
        param=QueryParam(mode="hybrid")
    )
    print(f"found {len(res.chunks)} chunks")

    # finalize storage 
    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
