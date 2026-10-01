from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

import numpy as np
import onnxruntime as ort

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


app = FastAPI(
    title="KalkiAtlas API",
    description="Embedding and Reranking API",
    version="1.0.0"
)


MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = None
session = None


# =========================================================
# LOAD MODEL
# =========================================================

def get_engine():

    global tokenizer
    global session

    if tokenizer is None or session is None:

        tokenizer_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="tokenizer.json"
        )

        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="onnx/model.onnx"
        )

        tokenizer = Tokenizer.from_file(
            tokenizer_path
        )

        tokenizer.enable_truncation(
            max_length=256
        )

        tokenizer.enable_padding(
            length=256,
            pad_id=0,
            pad_token="[PAD]"
        )

        options = ort.SessionOptions()

        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1

        session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"]
        )

    return tokenizer, session


# =========================================================
# EMBEDDING ENGINE
# =========================================================

def encode_texts(texts):

    tokenizer, session = get_engine()

    encoded = tokenizer.encode_batch(texts)

    input_ids = np.array(
        [item.ids for item in encoded],
        dtype=np.int64
    )

    attention_mask = np.array(
        [item.attention_mask for item in encoded],
        dtype=np.int64
    )

    token_type_ids = np.zeros_like(
        input_ids,
        dtype=np.int64
    )

    required_inputs = {
        item.name
        for item in session.get_inputs()
    }

    inputs = {}

    if "input_ids" in required_inputs:
        inputs["input_ids"] = input_ids

    if "attention_mask" in required_inputs:
        inputs["attention_mask"] = attention_mask

    if "token_type_ids" in required_inputs:
        inputs["token_type_ids"] = token_type_ids

    outputs = session.run(
        None,
        inputs
    )

    token_embeddings = outputs[0]

    mask = attention_mask[..., None].astype(
        np.float32
    )

    summed = np.sum(
        token_embeddings * mask,
        axis=1
    )

    counts = np.clip(
        mask.sum(axis=1),
        1e-9,
        None
    )

    embeddings = summed / counts

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    embeddings = embeddings / np.clip(
        norms,
        1e-12,
        None
    )

    return embeddings


# =========================================================
# REQUEST MODELS
# =========================================================

class EmbedRequest(BaseModel):
    texts: List[str]


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_n: int = 5


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "name": "KalkiAtlas API",
        "version": "1.0.0",
        "status": "online"
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# =========================================================
# EMBEDDING API
# =========================================================

@app.post("/v1/embed")
def embed(request: EmbedRequest):

    if not request.texts:

        raise HTTPException(
            status_code=400,
            detail="texts cannot be empty"
        )

    try:

        vectors = encode_texts(
            request.texts
        )

        return {
            "model": "kalki-embed-v1",
            "embeddings": vectors.tolist()
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =========================================================
# RERANK API
# =========================================================

@app.post("/v1/rerank")
def rerank(request: RerankRequest):

    if not request.query.strip():

        raise HTTPException(
            status_code=400,
            detail="query cannot be empty"
        )

    if not request.documents:

        raise HTTPException(
            status_code=400,
            detail="documents cannot be empty"
        )

    try:

        texts = [
            request.query
        ] + request.documents

        embeddings = encode_texts(
            texts
        )

        query_vector = embeddings[0]

        document_vectors = embeddings[1:]

        scores = np.dot(
            document_vectors,
            query_vector
        )

        results = []

        for index, (
            document,
            score
        ) in enumerate(
            zip(
                request.documents,
                scores
            )
        ):

            results.append(
                {
                    "index": index,
                    "document": document,
                    "relevance_score": round(
                        float(score),
                        4
                    )
                }
            )

        results.sort(
            key=lambda item:
            item["relevance_score"],
            reverse=True
        )

        top_n = max(
            1,
            min(
                request.top_n,
                len(results)
            )
        )

        return {
            "model": "kalki-rerank-v1",
            "results": results[:top_n]
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
