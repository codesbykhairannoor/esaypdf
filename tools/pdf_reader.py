import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text content from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"
