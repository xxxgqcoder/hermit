from hermit.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count.

    Empty input returns an empty list. Short texts that fit in one chunk
    are returned as a single-element list.
    """
    if not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks
