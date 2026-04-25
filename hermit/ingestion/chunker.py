import re

from tokenizers import Tokenizer

from hermit.config import (
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DENSE_MODEL,
    MODEL_ROOT,
)

# ── Markdown semantic-block patterns ────────────────────────────

_ATX_RE = re.compile(r'^#{1,6}(\s|$)')
# Standard list markers + common Unicode bullets (•·–—▪▸◦)
_LIST_RE = re.compile(r'^\s*([-*+•·\u2013\u2014\u25aa\u25b8\u25e6]|\d+[.)]) ')
_FENCE_RE = re.compile(r'^(`{3,}|~{3,})')
# Standard Markdown image ![alt](url)  OR  Obsidian wiki-link ![[path]]
_IMG_RE = re.compile(r'^!\[{1,2}[^\]]*\]{1,2}(\([^)]*\))?$')
# Thematic break: 3+ identical chars from [-*_] with optional spaces between
_HR_RE = re.compile(r'^[-*_](\s*[-*_]){2,}\s*$')
# Setext underline: one or more = or - (nothing else on the line)
_SETEXT_RE = re.compile(r'^(=+|-+)\s*$')

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


# ── Markdown semantic-block parser ──────────────────────────────


def _fence_char(stripped: str) -> str | None:
    """Return the fence character ('`' or '~') if this line opens a fenced block."""
    m = _FENCE_RE.match(stripped)
    return m.group(1)[0] if m else None


def _fence_open_len(stripped: str, fc: str) -> int:
    """Count how many fence characters open the block (e.g. 3 for ```)."""
    return len(stripped) - len(stripped.lstrip(fc))


def _is_special_start(lines: list[str], idx: int) -> bool:
    """Return True if the line at *idx* starts a new special block.

    Used to terminate paragraph collection.
    """
    s = lines[idx].strip()
    if not s:
        return True
    if _ATX_RE.match(s):
        return True
    if _FENCE_RE.match(s):
        return True
    if s == '$$':
        return True
    if s.startswith('|'):
        return True
    if s.startswith('>'):
        return True
    if _HR_RE.match(s):
        return True
    if _LIST_RE.match(s):
        return True
    if _IMG_RE.match(s):
        return True
    return False


def parse_md_blocks(text: str) -> list[str]:
    """Parse Markdown text into a list of semantic block strings.

    Each element is one logical unit:
    - YAML frontmatter  (``---`` … ``---`` at file start)
    - Fenced blocks     (``` or ~~~, including code / mermaid / etc.)
    - Math blocks       (``$$`` … ``$$``)
    - ATX headings      (``#`` through ``######``)
    - Setext headings   (text line + ``===`` / ``---`` underline)
    - Tables            (consecutive ``|``-prefixed lines)
    - Blockquotes       (consecutive ``>``-prefixed lines)
    - Horizontal rules  (``---``, ``***``, ``___``, …)
    - Lists             (entire list, including sub-items, as one block)
    - Standalone images (a line containing only ``![…](…)``)
    - Paragraphs        (consecutive non-blank, non-special lines)

    Blank lines are treated as separators and do not produce blocks.
    """
    if not text.strip():
        return []

    lines = text.splitlines()
    n = len(lines)
    blocks: list[str] = []
    i = 0

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # ── blank line: separator, skip ─────────────────────────
        if not stripped:
            i += 1
            continue

        # ── 1. YAML frontmatter (only at file position 0) ───────
        if i == 0 and stripped == '---':
            j = i + 1
            while j < n and lines[j].strip() != '---':
                j += 1
            end = j + 1 if j < n else j
            blocks.append('\n'.join(lines[i:end]))
            i = end
            continue

        # ── 2. Fenced block (``` / ~~~) ──────────────────────────
        fc = _fence_char(stripped)
        if fc is not None:
            open_len = _fence_open_len(stripped, fc)
            j = i + 1
            while j < n:
                s = lines[j].strip()
                # Closing fence: same char, at least as many chars, nothing else
                if s.startswith(fc * open_len) and not s.lstrip(fc).strip():
                    j += 1
                    break
                j += 1
            blocks.append('\n'.join(lines[i:j]))
            i = j
            continue

        # ── 3. Math block ($$ … $$) ──────────────────────────────
        if stripped == '$$':
            j = i + 1
            while j < n and lines[j].strip() != '$$':
                j += 1
            end = j + 1 if j < n else j
            blocks.append('\n'.join(lines[i:end]))
            i = end
            continue

        # ── 4. ATX heading (# through ######) ───────────────────
        if _ATX_RE.match(stripped):
            blocks.append(line)
            i += 1
            continue

        # ── 5. Table (consecutive | lines) ──────────────────────
        if stripped.startswith('|'):
            j = i + 1
            while j < n and lines[j].strip().startswith('|'):
                j += 1
            blocks.append('\n'.join(lines[i:j]))
            i = j
            continue

        # ── 6. Blockquote (consecutive > lines) ─────────────────
        if stripped.startswith('>'):
            j = i + 1
            while j < n and lines[j].strip().startswith('>'):
                j += 1
            blocks.append('\n'.join(lines[i:j]))
            i = j
            continue

        # ── 7. Horizontal rule (check before list to win over * * *) ──
        if _HR_RE.match(stripped):
            blocks.append(line)
            i += 1
            continue

        # ── 8. List (entire list as one block) ──────────────────
        if _LIST_RE.match(stripped):
            j = i + 1
            while j < n:
                ln = lines[j]
                s = ln.strip()
                if not s:
                    # Blank line: look ahead for list continuation
                    k = j + 1
                    while k < n and not lines[k].strip():
                        k += 1
                    if k < n:
                        ns = lines[k].strip()
                        nl = lines[k]
                        if _LIST_RE.match(ns) or (nl and nl[0] in (' ', '\t')):
                            j = k  # skip blanks, resume at continuation
                            continue
                    break
                if _LIST_RE.match(s) or (ln and ln[0] in (' ', '\t')):
                    j += 1
                else:
                    break
            blocks.append('\n'.join(lines[i:j]))
            i = j
            continue

        # ── 9. Standalone image ──────────────────────────────────
        if _IMG_RE.match(stripped):
            blocks.append(line)
            i += 1
            continue

        # ── 10. Setext heading (text line + === / --- underline) ─
        if i + 1 < n:
            next_s = lines[i + 1].strip()
            if next_s and _SETEXT_RE.match(next_s):
                blocks.append(line + '\n' + lines[i + 1])
                i += 2
                continue

        # ── 11. Paragraph ────────────────────────────────────────
        j = i + 1
        while j < n:
            if not lines[j].strip():
                break
            if _is_special_start(lines, j):
                break
            j += 1
        blocks.append('\n'.join(lines[i:j]))
        i = j

    return blocks


def chunk_markdown(
    text: str,
    blocks_per_chunk: int = 4,
    overlap: int = 1,
) -> list[str]:
    """Chunk Markdown by grouping consecutive semantic blocks.

    Splits *text* into semantic blocks via :func:`parse_md_blocks`, then
    produces overlapping chunks using a sliding window.

    Args:
        text: Raw Markdown text.
        blocks_per_chunk: Number of semantic blocks per chunk (default 4).
        overlap: Number of blocks shared between consecutive chunks (default 1).

    Returns:
        List of chunk strings joined with ``\\n\\n``.  Empty input returns ``[]``.
    """
    blocks = parse_md_blocks(text)
    if not blocks:
        return []

    if len(blocks) <= blocks_per_chunk:
        return ['\n\n'.join(blocks)]

    stride = blocks_per_chunk - overlap
    chunks: list[str] = []
    start = 0
    while start < len(blocks):
        chunk_blocks = blocks[start:start + blocks_per_chunk]
        chunks.append('\n\n'.join(chunk_blocks))
        if start + blocks_per_chunk >= len(blocks):
            break
        start += stride
    return chunks
