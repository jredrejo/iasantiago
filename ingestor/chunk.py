from pypdf import PdfReader


def pdf_to_chunks(path: str, chunk_size=900, overlap=120):
    reader = PdfReader(path)
    chunks = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        # troceado simple
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            seg = text[start:end]
            chunks.append({"page": i, "text": seg})
            start = end - overlap if end - overlap > start else end
    return chunks
