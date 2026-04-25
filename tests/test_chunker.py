"""Tests for Markdown semantic-block parsing and chunking.

Covers:
  - parse_md_blocks(): one test per semantic block type
  - parse_md_blocks(): mixed-content integration
  - chunk_markdown(): sliding-window behaviour and edge cases
"""

from pathlib import Path

import pytest

from hermit.ingestion.chunker import parse_md_blocks, chunk_markdown

FIXTURES = Path(__file__).parent / "fixtures" / "md"


# ── helpers ─────────────────────────────────────────────────────


def _load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# ── ATX headings ────────────────────────────────────────────────


def test_atx_headings_count():
    blocks = parse_md_blocks(_load("atx_headings.md"))
    headings = [b for b in blocks if b.strip().startswith("#")]
    assert len(headings) == 6


def test_atx_headings_each_is_own_block():
    blocks = parse_md_blocks(_load("atx_headings.md"))
    # Each ATX heading must occupy its own block (no multi-heading blocks)
    for b in blocks:
        lines = [ln for ln in b.splitlines() if ln.strip().startswith("#")]
        assert len(lines) <= 1, f"Multiple headings in one block: {b!r}"


def test_atx_headings_paragraph_separated():
    blocks = parse_md_blocks(_load("atx_headings.md"))
    paragraphs = [b for b in blocks if not b.strip().startswith("#")]
    assert len(paragraphs) == 1
    assert "Some paragraph text" in paragraphs[0]


# ── Setext headings ─────────────────────────────────────────────


def test_setext_headings_count():
    blocks = parse_md_blocks(_load("setext_heading.md"))
    # Both setext headings + paragraph = 3 blocks
    assert len(blocks) == 3


def test_setext_heading_h1_contains_underline():
    blocks = parse_md_blocks(_load("setext_heading.md"))
    h1 = blocks[0]
    assert "First Level Heading" in h1
    assert "=" in h1


def test_setext_heading_h2_contains_underline():
    blocks = parse_md_blocks(_load("setext_heading.md"))
    h2 = blocks[1]
    assert "Second Level Heading" in h2
    assert "-" in h2


# ── Fenced blocks ────────────────────────────────────────────────


def test_fenced_blocks_count():
    blocks = parse_md_blocks(_load("fenced_blocks.md"))
    fenced = [b for b in blocks if b.strip().startswith("```")]
    assert len(fenced) == 2


def test_fenced_block_python_content():
    blocks = parse_md_blocks(_load("fenced_blocks.md"))
    py_block = next(b for b in blocks if "```python" in b)
    assert "def hello" in py_block
    assert 'return "world"' in py_block


def test_fenced_block_mermaid_content():
    blocks = parse_md_blocks(_load("fenced_blocks.md"))
    mm_block = next(b for b in blocks if "```mermaid" in b)
    assert "graph TD" in mm_block
    assert "A --> B" in mm_block


def test_fenced_block_is_single_block():
    blocks = parse_md_blocks(_load("fenced_blocks.md"))
    for b in blocks:
        if b.strip().startswith("```"):
            # Opening and closing fence must both be in the same block
            fence_lines = [ln for ln in b.splitlines() if ln.strip().startswith("```")]
            assert len(fence_lines) == 2, f"Unmatched fences in: {b!r}"


# ── Math blocks ─────────────────────────────────────────────────


def test_math_blocks_count():
    blocks = parse_md_blocks(_load("math_blocks.md"))
    math = [b for b in blocks if b.strip().startswith("$$")]
    assert len(math) == 2


def test_math_block_e_equals_mc2():
    blocks = parse_md_blocks(_load("math_blocks.md"))
    b = next(b for b in blocks if "mc^2" in b)
    assert b.strip().startswith("$$")
    assert b.strip().endswith("$$")


def test_math_block_euler():
    blocks = parse_md_blocks(_load("math_blocks.md"))
    b = next(b for b in blocks if "e^{i" in b)
    assert b.strip().startswith("$$")
    assert b.strip().endswith("$$")


def test_inline_math_stays_in_paragraph():
    blocks = parse_md_blocks(_load("math_blocks.md"))
    para = next(b for b in blocks if "$x = y$" in b)
    # Inline math is part of a paragraph block, not a standalone $$ block
    assert not para.strip().startswith("$$")


# ── Table ────────────────────────────────────────────────────────


def test_table_is_single_block():
    blocks = parse_md_blocks(_load("table_block.md"))
    table_blocks = [b for b in blocks if b.strip().startswith("|")]
    assert len(table_blocks) == 1


def test_table_contains_all_rows():
    blocks = parse_md_blocks(_load("table_block.md"))
    table = next(b for b in blocks if b.strip().startswith("|"))
    assert "Name" in table      # header
    assert "|---" in table      # separator row
    assert "foo" in table
    assert "bar" in table
    assert "baz" in table


def test_table_surrounding_paragraphs():
    blocks = parse_md_blocks(_load("table_block.md"))
    non_table = [b for b in blocks if not b.strip().startswith("|")]
    assert any("Here is a table" in b for b in non_table)
    assert any("End of table" in b for b in non_table)


# ── Blockquote ───────────────────────────────────────────────────


def test_blockquote_count():
    blocks = parse_md_blocks(_load("blockquote.md"))
    quotes = [b for b in blocks if b.strip().startswith(">")]
    assert len(quotes) == 2


def test_blockquote_first_is_single_block():
    blocks = parse_md_blocks(_load("blockquote.md"))
    first_quote = next(b for b in blocks if b.strip().startswith(">"))
    # All three "> " lines must be in the same block
    assert first_quote.count(">") >= 3


# ── List (entire list as one block) ─────────────────────────────


def test_unordered_list_is_one_block():
    blocks = parse_md_blocks(_load("list_block.md"))
    ul = next(b for b in blocks if "item 1" in b)
    # All top-level items and sub-items must be in the same block
    assert "item 2" in ul
    assert "sub-item A" in ul
    assert "sub-item B" in ul
    assert "item 3" in ul


def test_ordered_list_is_one_block():
    blocks = parse_md_blocks(_load("list_block.md"))
    ol = next(b for b in blocks if "first ordered" in b)
    assert "second ordered" in ol
    assert "third ordered" in ol


def test_two_lists_are_two_blocks():
    blocks = parse_md_blocks(_load("list_block.md"))
    list_blocks = [b for b in blocks if "item" in b or "ordered" in b]
    assert len(list_blocks) == 2


def test_paragraph_between_lists_is_own_block():
    blocks = parse_md_blocks(_load("list_block.md"))
    assert any("Paragraph between lists" in b for b in blocks)


# ── Standalone image ─────────────────────────────────────────────


def test_standalone_image_is_own_block():
    blocks = parse_md_blocks(_load("standalone_image.md"))
    img_blocks = [b for b in blocks if b.strip().startswith("!")]
    assert len(img_blocks) == 1
    assert "Architecture Diagram" in img_blocks[0]


def test_standalone_image_not_merged_with_paragraph():
    blocks = parse_md_blocks(_load("standalone_image.md"))
    # "Before image." and "After image." must each be separate from the image block
    assert any("Before image" in b for b in blocks)
    assert any("After image" in b for b in blocks)
    for b in blocks:
        if "Before image" in b or "After image" in b:
            assert "!" not in b, f"Paragraph merged with image: {b!r}"


# ── Horizontal rule ──────────────────────────────────────────────


def test_horizontal_rule_count():
    blocks = parse_md_blocks(_load("horizontal_rule.md"))
    hrs = [b for b in blocks if b.strip() in ("---", "***", "___")]
    assert len(hrs) == 2


def test_horizontal_rule_separates_sections():
    blocks = parse_md_blocks(_load("horizontal_rule.md"))
    assert any("Section one" in b for b in blocks)
    assert any("Section two" in b for b in blocks)
    assert any("Section three" in b for b in blocks)


# ── YAML frontmatter ─────────────────────────────────────────────


def test_yaml_frontmatter_is_first_block():
    blocks = parse_md_blocks(_load("yaml_frontmatter.md"))
    assert blocks[0].startswith("---")
    assert "title:" in blocks[0]
    assert "author:" in blocks[0]


def test_yaml_frontmatter_followed_by_heading():
    blocks = parse_md_blocks(_load("yaml_frontmatter.md"))
    assert len(blocks) == 3
    assert blocks[1].strip().startswith("#")


# ── Mixed document (integration) ────────────────────────────────


def test_mixed_doc_block_count():
    """mixed_doc.md has exactly 10 semantic blocks."""
    blocks = parse_md_blocks(_load("mixed_doc.md"))
    assert len(blocks) == 10


def test_mixed_doc_block_types():
    blocks = parse_md_blocks(_load("mixed_doc.md"))
    headings = [b for b in blocks if b.strip().startswith("#")]
    tables = [b for b in blocks if b.strip().startswith("|")]
    lists = [b for b in blocks if b.strip().startswith("-")]
    fenced = [b for b in blocks if b.strip().startswith("```")]
    assert len(headings) == 4   # #, ##, ###, ##
    assert len(tables) == 1
    assert len(lists) == 1
    assert len(fenced) == 1


# ── chunk_markdown: edge cases ───────────────────────────────────


def test_chunk_markdown_empty():
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n  ") == []


def test_chunk_markdown_single_block():
    result = chunk_markdown("# Hello\n")
    assert result == ["# Hello"]


def test_chunk_markdown_exactly_four_blocks():
    text = "# H1\n\n# H2\n\n# H3\n\n# H4\n"
    result = chunk_markdown(text)
    assert len(result) == 1
    assert "# H1" in result[0]
    assert "# H4" in result[0]


def test_chunk_markdown_five_blocks_gives_two_chunks():
    text = "# H1\n\n# H2\n\n# H3\n\n# H4\n\n# H5\n"
    result = chunk_markdown(text)
    assert len(result) == 2
    # chunk 0: H1..H4, chunk 1: H4..H5 (overlap = H4)
    assert "# H1" in result[0]
    assert "# H4" in result[0]
    assert "# H4" in result[1]
    assert "# H5" in result[1]


def test_chunk_markdown_overlap():
    """With stride=3, the last block of chunk N == first block of chunk N+1."""
    blocks = parse_md_blocks(_load("mixed_doc.md"))
    chunks = chunk_markdown(_load("mixed_doc.md"))
    # mixed_doc has 10 blocks → 3 chunks: [0:4], [3:7], [6:10]
    assert len(chunks) == 3
    # Block at index 3 (overlap between chunk 0 and chunk 1)
    assert blocks[3] in chunks[0]
    assert blocks[3] in chunks[1]
    # Block at index 6 (overlap between chunk 1 and chunk 2)
    assert blocks[6] in chunks[1]
    assert blocks[6] in chunks[2]


def test_chunk_markdown_all_blocks_covered():
    """Every semantic block must appear in at least one chunk."""
    blocks = parse_md_blocks(_load("mixed_doc.md"))
    chunks = chunk_markdown(_load("mixed_doc.md"))
    all_chunk_text = "\n\n".join(chunks)
    for b in blocks:
        assert b in all_chunk_text, f"Block not found in any chunk: {b!r}"


def test_chunk_markdown_custom_window():
    text = "\n\n".join(f"# H{i}" for i in range(9))  # 9 blocks
    # blocks_per_chunk=3, overlap=1 → stride=2
    # starts: 0, 2, 4, 6 → 4 chunks ([0:3],[2:5],[4:7],[6:9])
    result = chunk_markdown(text, blocks_per_chunk=3, overlap=1)
    assert len(result) == 4
    for chunk in result:
        assert chunk.strip()


# ── chunk_markdown: non-.md path not affected ───────────────────
# (scanner integration verified separately in test_scan_and_queue)
