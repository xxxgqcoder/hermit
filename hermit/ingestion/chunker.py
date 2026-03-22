from tokenizers import Tokenizer

from hermit.config import (
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DENSE_MODEL,
    MODEL_ROOT,
)

_tokenizer: Tokenizer | None = None


def _get_tokenizer() -> Tokenizer:
    global _tokenizer
    if _tokenizer is None:
        repo_dir = MODEL_ROOT / f"models--{DENSE_MODEL.replace('/', '--')}"
        tok_path = next(repo_dir.rglob("tokenizer.json"))
        _tokenizer = Tokenizer.from_file(str(tok_path))
    return _tokenizer


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks by token count.

    Uses the embedding model's tokenizer to count tokens, ensuring
    consistent chunk sizes across Chinese and English text.

    Empty input returns an empty list. Short texts that fit in one chunk
    are returned as a single-element list.
    """
    if not text.strip():
        return []

    tokenizer = _get_tokenizer()
    encoding = tokenizer.encode(text)
    offsets = encoding.offsets

    if len(offsets) <= DEFAULT_CHUNK_TOKENS:
        return [text]

    chunks = []
    start = 0
    while start < len(offsets):
        end = min(start + DEFAULT_CHUNK_TOKENS, len(offsets))
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunk = text[char_start:char_end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(offsets):
            break
        start = end - DEFAULT_CHUNK_OVERLAP_TOKENS
    return chunks
