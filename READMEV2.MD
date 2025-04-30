# Retail Recommender System (v2)

A multi-agent retail product recommendation system with sentiment analysis trained on Amazon reviews. This Streamlit-based web app lets users query products, retrieves relevant items via a hybrid search, analyzes sentiment, and generates structured feedback.

## Features

- Intent Classification: Rasa NLU server running locally with custom domain/config.
- Product Retrieval: LightRAG hybrid search over CSV product data (simple_indexer.py).
- Sentiment Analysis: DistilBERT model fine-tuned on Amazon reviews (`sohan-ai/sentiment-analysis-model-amazon-reviews`).
- Orchestration: PydanticAI agents with LangGraph handling pipeline steps.
- Streamlit App: Interactive UI displaying products, color-coded sentiment badges, confidence scores, and overall feedback.

## Architecture

```
User → Streamlit App (app.py)
       ↳ invokes Orchestrator →
            • IntentAgent (Rasa API)
            • RetrievalAgent (LightRAG via simple_indexer.py)
            • SentimentAgent (Transformers model)
            • FeedbackAgent (structured insights)
       ↳ returns JSON → Streamlit renders products & feedback
``` 

## Installation & Setup

1. Clone repo and create a virtual environment:
   ```bash
   git clone <repo_url>
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare data (optional):
   - Place Amazon review JSONL and metadata in `/Downloads` and run indexing scripts.
   - Sample data available in `sample_data/sample_products.csv`.

## Running Rasa NLU

```bash
# from project root, with venv active
./rasa_venv/bin/rasa run --enable-api --cors "*"
```

Ensure Rasa starts without errors.

## Running the Streamlit App

```bash
# from project root, with venv active
streamlit run app.py
```

Open browser at `http://localhost:8501`.

## Usage

- Enter a product query or choose an example.
- View retrieved products with:
  - Title, price, category, description.
  - Amazon sentiment badge (✅/❌/⚪) with progress bar and confidence.
- See overall feedback: summary, key points, actionable insights, sentiment score.

## Directory Structure

```
/src
  /agents           # PydanticAI orchestrator and agent code
  /data_indexing    # simple_indexer.py for CSV indexing
/sample_data        # sample_products.csv, sample_products.xlsx
/app.py             # Streamlit front-end
/READMEV2.MD        # This file
/requirements.txt   # Python deps
``` 

## Results & Feedback

- Sentiment aligns with Amazon review trends.
- Hybrid search returns top-k relevant products.
- Feedback agent provides actionable recommendations.
- UI improvements: color coding, badges, progress bars.

---
Built with PydanticAI, Transformers, Rasa, and Streamlit. Feel free to explore and modify agents or data workflows for your use case!
