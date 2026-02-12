import os
import numpy as np
from openai import OpenAI
from typing import List

EMBEDDING_MODEL = "text-embedding-3-small"

def get_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_texts(texts: List[str], batch_size: int = 50) -> np.ndarray:
    client = get_client()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch = [t.replace("\n", " ") for t in batch]
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL, input=batch
            )
            batch_embs = [d.embedding for d in response.data]
            all_embeddings.extend(batch_embs)
        except Exception as e:
            print(f"Error embedding batch: {e}")
            all_embeddings.extend([np.zeros(1536) for _ in batch])

    return np.array(all_embeddings, dtype="float32")

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return embeddings / norms