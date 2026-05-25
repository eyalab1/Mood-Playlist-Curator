"""Voyage AI embedding wrapper.

Two public functions:
  embed_documents(texts) - for items stored in the corpus (song lyrics)
  embed_query(text)      - for the live query the user is searching with

Voyage's input_type flag tells the model whether to optimize the vector
for storage or for searching. Using it correctly noticeably improves
retrieval quality.
"""

import logging
import time

import numpy as np
import voyageai
import voyageai.error

from config import EMBEDDING_MODEL, VOYAGE_API_KEY

logger = logging.getLogger(__name__)

# Voyage hard limit: 128 documents per request.
_MAX_DOCS_PER_BATCH = 128
# With a payment method on file the free-tier 10K tokens/min throttle is
# lifted, so we use large batches and no inter-batch pause. The retry/backoff
# below stays as a safety net in case the new limits take a few minutes to
# activate. (To run card-free again, set the pause back to ~62 and the token
# budget to ~8000.)
_MAX_TOKENS_PER_BATCH = 100000
_SECONDS_BETWEEN_BATCHES = 0
_RATE_LIMIT_BACKOFF = 65
_MAX_RETRIES = 4


_client: voyageai.Client | None = None


def _get_client() -> voyageai.Client:
    global _client
    if _client is None:
        if not VOYAGE_API_KEY:
            raise RuntimeError("Missing VOYAGE_API_KEY in .env")
        _client = voyageai.Client(api_key=VOYAGE_API_KEY)
    return _client


def _estimate_tokens(text: str) -> int:
    """Rough heuristic: about 4 characters per token."""
    return max(1, len(text) // 4)


def _make_batches(texts: list[str]) -> list[list[str]]:
    """Group texts into batches that stay under the per-batch token budget
    and the 128-document limit."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for t in texts:
        tok = _estimate_tokens(t)
        too_many_docs = len(current) >= _MAX_DOCS_PER_BATCH
        too_many_tokens = current_tokens + tok > _MAX_TOKENS_PER_BATCH
        if current and (too_many_docs or too_many_tokens):
            batches.append(current)
            current, current_tokens = [], 0
        current.append(t)
        current_tokens += tok
    if current:
        batches.append(current)
    return batches


def _embed_batch(client, batch: list[str], input_type: str) -> list[list[float]]:
    """Embed one batch, retrying with a long pause if rate-limited."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result = client.embed(
                texts=batch, model=EMBEDDING_MODEL, input_type=input_type
            )
            return result.embeddings
        except voyageai.error.RateLimitError:
            if attempt == _MAX_RETRIES:
                raise
            logger.info(
                "Rate limited; waiting %ds before retry %d/%d...",
                _RATE_LIMIT_BACKOFF,
                attempt,
                _MAX_RETRIES,
            )
            time.sleep(_RATE_LIMIT_BACKOFF)
    return []  # unreachable


def iter_embed_documents(texts: list[str]):
    """Yield (start_index, batch_embeddings) as each throttled batch finishes.

    Lets callers persist results incrementally so a crash mid-run does not
    lose progress. Small jobs are a single batch with no waiting; large jobs
    are token-budgeted batches with a ~60s pause between them to stay under
    the free-tier 10K-tokens/min limit.
    """
    if not texts:
        return
    client = _get_client()
    batches = _make_batches(texts)
    pos = 0
    for idx, batch in enumerate(batches, start=1):
        embeddings = _embed_batch(client, batch, "document")
        yield pos, embeddings
        pos += len(batch)
        if len(batches) > 1:
            logger.info("Embedded batch %d/%d (%d tracks)", idx, len(batches), len(batch))
        if idx < len(batches):
            time.sleep(_SECONDS_BETWEEN_BATCHES)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed corpus documents (song lyrics), throttled for the free tier."""
    out: list[list[float]] = []
    for _, batch_embeddings in iter_embed_documents(texts):
        out.extend(batch_embeddings)
    return out


def embed_query(text: str) -> list[float]:
    """Embed a single search query (e.g. a user mood prompt)."""
    client = _get_client()
    result = client.embed(
        texts=[text],
        model=EMBEDDING_MODEL,
        input_type="query",
    )
    return result.embeddings[0]


def to_blob(embedding: list[float]) -> bytes:
    """Serialize an embedding to bytes for SQLite storage (float32)."""
    return np.asarray(embedding, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    """Deserialize a stored embedding back into a numpy float32 array."""
    return np.frombuffer(blob, dtype=np.float32)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = "rainy sunday afternoon, melancholy but hopeful"
    vec = embed_query(sample)
    print(f"Query: {sample!r}")
    print(f"Vector length: {len(vec)}")
    print(f"First 5 dims: {vec[:5]}")

    docs = ["lonely highway driving at night", "joyful summer beach party"]
    vecs = embed_documents(docs)
    print(f"\nDoc embeddings: {len(vecs)} vectors of length {len(vecs[0])}")
