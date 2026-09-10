from pathlib import Path
import pypdf
import pandas as pd


def extract_text(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"The file at {file_path} does not exist.")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix == ".csv":
        df = pd.read_csv(path)
        return df.to_string(index=False)
    elif suffix == ".txt":
        return path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(
    text: str,
    strategy: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    if strategy == "recursive":
        return _recursive_chunk(text, chunk_size, chunk_overlap)
    elif strategy == "fixed":
        return _fixed_chunk(text, chunk_size, chunk_overlap)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _recursive_chunk(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Splits on the largest available separator first (paragraphs), and only
    falls back to smaller separators (sentences, then words) for pieces
    that are still too big. This mirrors what LangChain's splitter does
    internally, without importing it.
    """
    separators = ["\n\n", "\n", ". ", " "]
    return _split_recursive(text, separators, chunk_size, chunk_overlap)


def _split_recursive(text: str, separators: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # No separators left — hard-split by character count
        return _fixed_chunk(text, chunk_size, chunk_overlap)

    sep = separators[0]
    parts = text.split(sep)

    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = (current + sep + part) if current else part

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                # This single part is still too big — recurse with the next separator
                chunks.extend(_split_recursive(part, separators[1:], chunk_size, chunk_overlap))
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    # Apply overlap by prepending the tail of the previous chunk
    if chunk_overlap > 0:
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                prev_tail = chunks[i - 1][-chunk_overlap:]
                overlapped.append(prev_tail + chunk)
        return overlapped

    return chunks


def _fixed_chunk(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Simple fixed-size character chunking — your second strategy."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    chunks: list[str] = []
    step = chunk_size - chunk_overlap

    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(text):
            break

    return chunks