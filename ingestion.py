import io
import os
import re
from pypdf import PdfReader
import docx
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
from db import get_conn

_model = None

_tesseract_cmd = os.environ.get("TESSERACT_CMD")
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def extract_pdf_pages(file_path: str) -> list[tuple[int, str]]:
    """Returns [(page_number, page_text), ...], 1-indexed. Falls back to OCR
    per page if the PDF has no real text layer (scanned pages)."""
    reader = PdfReader(file_path)
    page_texts = [page.extract_text() or "" for page in reader.pages]
    total_chars = sum(len(t.strip()) for t in page_texts)
    avg_chars_per_page = total_chars / max(len(page_texts), 1)

    if avg_chars_per_page > 20:
        return [(i + 1, t) for i, t in enumerate(page_texts)]

    pages = []
    doc = fitz.open(file_path)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        pages.append((i + 1, pytesseract.image_to_string(img)))
    doc.close()
    return pages


def extract_text(file_path: str, filetype: str) -> str:
    """For non-paginated formats (no real page concept)."""
    filetype = filetype.lower()
    if filetype == "docx":
        d = docx.Document(file_path)
        text = "\n".join(p.text for p in d.paragraphs)
    elif filetype in ("txt", "md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file type: {filetype}. Supported: pdf, docx, txt, md")
    return text.replace("\x00", "")


def chunk_text(text: str, target_words: int = 250, overlap_words: int = 40) -> list[str]:
    """Chunk at paragraph/sentence boundaries instead of a raw word count, so
    a chunk never starts or ends mid-sentence -- this measurably improves
    embedding quality, since a half-sentence weakens the meaning captured
    for both chunks it got split across."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]

    chunks = []
    current_words: list[str] = []

    def flush():
        if current_words:
            chunks.append(" ".join(current_words))

    for para in paragraphs:
        para_words = para.split()

        if len(para_words) > target_words * 1.5:
            # A single paragraph too large on its own -- split at sentence
            # boundaries instead of mid-sentence.
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent_words = sent.split()
                if not sent_words:
                    continue
                if current_words and len(current_words) + len(sent_words) > target_words:
                    flush()
                    current_words = current_words[-overlap_words:] if overlap_words else []
                current_words.extend(sent_words)
            continue

        if current_words and len(current_words) + len(para_words) > target_words:
            flush()
            current_words = current_words[-overlap_words:] if overlap_words else []
        current_words.extend(para_words)

    flush()
    return [c for c in chunks if c.strip()]


def ingest_resource(resource_id: int, file_path: str, filetype: str) -> int:
    """Extract, chunk, embed, and store a resource's content.
    PDFs are chunked per-page so page numbers can be cited honestly later.
    Other formats have no real page concept."""
    filetype = filetype.lower()

    if filetype == "pdf":
        pages = extract_pdf_pages(file_path)
        pages = [(n, t.replace("\x00", "")) for n, t in pages]
        full_text = "\n".join(t for _, t in pages)
        return _ingest_pages(resource_id, full_text, pages)

    text = extract_text(file_path, filetype)
    return ingest_text(resource_id, text)


def _ingest_pages(resource_id: int, full_text: str, pages: list[tuple[int, str]]) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE resources SET raw_text = %s WHERE id = %s", (full_text, resource_id))
    conn.commit()

    model = get_model()
    chunk_index = 0
    total_chunks = 0
    for page_number, page_text in pages:
        if not page_text.strip():
            continue
        chunks = chunk_text(page_text)
        if not chunks:
            continue
        embeddings = model.encode(chunks).tolist()
        for chunk, emb in zip(chunks, embeddings):
            cur.execute(
                """INSERT INTO resource_chunks (resource_id, chunk_text, chunk_index, embedding, page_number)
                   VALUES (%s, %s, %s, %s, %s)""",
                (resource_id, chunk, chunk_index, emb, page_number),
            )
            chunk_index += 1
            total_chunks += 1

    conn.commit()
    cur.close()
    conn.close()
    return total_chunks


def ingest_text(resource_id: int, text: str) -> int:
    """Chunk, embed, and store text with no page concept (docx/txt/md, or
    already-generated documents like discussion exports)."""
    chunks = chunk_text(text)
    if not chunks:
        return 0

    model = get_model()
    embeddings = model.encode(chunks).tolist()

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE resources SET raw_text = %s WHERE id = %s", (text, resource_id))
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        cur.execute(
            """INSERT INTO resource_chunks (resource_id, chunk_text, chunk_index, embedding)
               VALUES (%s, %s, %s, %s)""",
            (resource_id, chunk, i, emb),
        )
    conn.commit()
    cur.close()
    conn.close()
    return len(chunks)
