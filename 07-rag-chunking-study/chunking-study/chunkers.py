"""The four chunking strategies under study. Each function takes the loaded
handbook files ([(filename, text), ...]) and returns a list of chunk dicts:
{"text": ..., "source": <filename>}. Only the semantic chunker calls Nova, so
only it takes a CallTracker.
"""
import re

from common import CallTracker, call_nova

# --- 1. fixed-size ---------------------------------------------------------


def chunk_fixed_size(files: list[tuple[str, str]], chunk_words: int, overlap_words: int) -> list[dict]:
    """Slide a fixed-size window (chunk_words) over each document's words,
    stepping by (chunk_words - overlap_words) so consecutive chunks share
    overlap_words of context. Simplest and cheapest strategy: no LLM calls,
    no structure awareness — just a word count."""
    chunks = []
    for source, text in files:
        words = text.split()
        start = 0
        while start < len(words):
            chunk_text = " ".join(words[start : start + chunk_words])
            chunks.append({"text": chunk_text, "source": source})
            if start + chunk_words >= len(words):
                break
            start += chunk_words - overlap_words
    return chunks


# --- 2. separator-based (markdown headings / paragraphs) -------------------

_HEADING_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)


def chunk_separator(files: list[tuple[str, str]]) -> list[dict]:
    """Split on markdown headings first; within a heading section with no
    further headings, split on blank-line-separated paragraphs so no chunk
    spans more than one heading section."""
    chunks = []
    for source, text in files:
        sections = _split_on_headings(text)
        for section in sections:
            for paragraph in _split_paragraphs(section):
                if paragraph.strip():
                    chunks.append({"text": paragraph.strip(), "source": source})
    return chunks


def _split_on_headings(text: str) -> list[str]:
    """Cut text at every markdown heading line, each returned section running
    from one heading up to (not including) the next. Text before the first
    heading (e.g. YAML frontmatter) is kept as its own leading section rather
    than silently dropped."""
    positions = [m.start() for m in _HEADING_RE.finditer(text)]
    if not positions:
        return [text]
    if positions[0] != 0:
        positions.insert(0, 0)  # keep any preamble (e.g. frontmatter) as its own section
    positions.append(len(text))
    return [text[positions[i] : positions[i + 1]] for i in range(len(positions) - 1)]


def _split_paragraphs(section: str) -> list[str]:
    """Split on blank lines (one or more), the common markdown paragraph
    separator, dropping any all-whitespace fragments."""
    return [p for p in re.split(r"\n\s*\n", section) if p.strip()]


# --- 3. sentence-based -------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z#\-*\d])")


def chunk_sentences(files: list[tuple[str, str]], sentences_per_chunk: int) -> list[dict]:
    """Flatten each document to one line (so markdown line-wrapping doesn't
    look like sentence breaks), split into sentences on '.', '!', '?' followed
    by a capital/heading/list marker, then group sentences_per_chunk at a
    time into a chunk."""
    chunks = []
    for source, text in files:
        flat = " ".join(text.split())  # collapse markdown line breaks before splitting
        sentences = [s for s in _SENTENCE_RE.split(flat) if s.strip()]
        for start in range(0, len(sentences), sentences_per_chunk):
            group = sentences[start : start + sentences_per_chunk]
            chunks.append({"text": " ".join(group), "source": source})
    return chunks


# --- 4. semantic / LLM-based -------------------------------------------------

_BOUNDARY_SYSTEM = (
    "You split a document into topically coherent chunks. Given numbered "
    "paragraphs, return ONLY a comma-separated list of paragraph numbers that "
    "should START a new chunk (always including 1). Do not explain, just the "
    "numbers, e.g. '1,4,9'."
)


def chunk_semantic(files: list[tuple[str, str]], tracker: CallTracker, max_paragraphs_per_call: int = 40) -> list[dict]:
    """Ask Nova 2 Lite where topics change within each document's paragraphs,
    then split there. Long documents are processed in batches of
    max_paragraphs_per_call (one Nova call each) so no paragraph is silently
    dropped. Falls back to one chunk per batch if Nova's reply can't be
    parsed, so a bad response degrades gracefully instead of crashing.
    """
    chunks = []
    for source, text in files:
        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            continue
        boundaries = _semantic_boundaries(paragraphs, tracker, max_paragraphs_per_call)
        for start, end in zip(boundaries, boundaries[1:] + [len(paragraphs)]):
            chunk_text = "\n\n".join(paragraphs[start:end]).strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "source": source})
    return chunks


def _semantic_boundaries(paragraphs: list[str], tracker: CallTracker, max_paragraphs_per_call: int) -> list[int]:
    """Global paragraph-index boundaries (always includes 0), gathered one
    Nova call per max_paragraphs_per_call-sized batch."""
    boundaries: list[int] = []
    for batch_start in range(0, len(paragraphs), max_paragraphs_per_call):
        batch = paragraphs[batch_start : batch_start + max_paragraphs_per_call]
        numbered = "\n".join(f"{i + 1}. {p.strip()[:300]}" for i, p in enumerate(batch))
        try:
            reply = call_nova(_BOUNDARY_SYSTEM, numbered, tracker)
            local = sorted({int(n) - 1 for n in re.findall(r"\d+", reply) if 0 < int(n) <= len(batch)})
            if not local or local[0] != 0:
                local = [0] + local
        except (KeyError, ValueError, RuntimeError):
            # A malformed reply or a transient Bedrock error shouldn't abort the
            # whole build — fall back to treating the batch as one chunk.
            local = [0]
        boundaries.extend(batch_start + i for i in local)
    return boundaries
