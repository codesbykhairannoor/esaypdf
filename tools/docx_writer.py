from docx import Document
import os

def create_docx(text_content: str, output_path: str) -> str:
    """Generates a DOCX file from the provided text content."""
    try:
        doc = Document()
        
        # Simple parsing for markdown-like structure (just headers and paragraphs for now)
        lines = text_content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('# '):
                doc.add_heading(line[2:], level=1)
            elif line.startswith('## '):
                doc.add_heading(line[3:], level=2)
            elif line.startswith('### '):
                doc.add_heading(line[4:], level=3)
            else:
                doc.add_paragraph(line)
                
        doc.save(output_path)
        return f"Successfully saved to {output_path}"
    except Exception as e:
        return f"Error creating DOCX: {e}"
