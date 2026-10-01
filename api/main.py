import os
import time
from typing import List, Optional

import torch
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI(title="KalkiAtlas API", version="1.0.0")

device = "cuda" if torch.cuda.is_available() else "cpu"

embed_model = SentenceTransformer(
    "intfloat/multilingual-e5-base",
    device=device,
)

rerank_tokenizer = AutoTokenizer.from_pretrained(
    "BAAI/bge-reranker-base"
)
rerank_model = AutoModelForSequenceClassification.from_pretrained(
    "BAAI/bge-reranker-base"
).to(device).eval()

api_key_header = APIKeyHeader(
    name="X-Kalki-Key",
    auto_error=False,
)

async def verify_key(
    key: str = Security(api_key_header),
):
    expected = os.getenv("KALKI_API_KEY", "change-this-key")

    if not key or key != expected:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return key


class EmbedRequest(BaseModel):
    texts: List[str]
    input_type: Optional[str] = "search_document"


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_n: Optional[int] = 3


@app.get("/")
@app.get("/v1/health")
def health():
    return {
        "status": "ONLINE",
        "engine": "KalkiAtlas",
        "device": device,
    }


@app.post("/v1/embed")
def embed(
    request: EmbedRequest,
    _: str = Depends(verify_key),
):
    prefix = (
        "query: "
        if request.input_type == "search_query"
        else "passage: "
    )

    texts = [prefix + text for text in request.texts]

    vectors = embed_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    return {
        "embeddings": vectors,
        "count": len(vectors),
    }


@app.post("/v1/rerank")
def rerank(
    request: RerankRequest,
    _: str = Depends(verify_key),
):
    started = time.time()
    pairs = [[request.query, document]
             for document in request.documents]

    with torch.no_grad():
        inputs = rerank_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        scores = rerank_model(**inputs).logits.view(-1)
        scores = scores.float().cpu().tolist()

    results = [
        {
            "index": index,
            "document": document,
            "relevance_score": float(score),
        }
        for index, (document, score)
        in enumerate(zip(request.documents, scores))
    ]

    results.sort(
        key=lambda item: item["relevance_score"],
        reverse=True,
    )

    return {
        "results": results[:request.top_n],
        "latency_ms": round(
            (time.time() - started) * 1000,
            2,
        ),
    }
