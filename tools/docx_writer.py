from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import re

def parse_markdown_to_paragraph(paragraph, text):
    """Simple parser for bold and italic markdown within a paragraph."""
    # Split text by bold markers
    parts = re.split(r'(\*\*.*?\*\*)', text)
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            # It's bold
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        else:
            # Handle italics within non-bold parts
            sub_parts = re.split(r'(\*.*?\*)', part)
            for sub_part in sub_parts:
                if sub_part.startswith('*') and sub_part.endswith('*'):
                    run = paragraph.add_run(sub_part[1:-1])
                    run.italic = True
                else:
                    paragraph.add_run(sub_part)

def create_docx(text_content: str, output_path: str) -> str:
    """Generates an academically formatted DOCX file."""
    try:
        doc = Document()
        
        # Setup Academic Styles
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Times New Roman'
        font.size = Pt(12)
        
        paragraph_format = style.paragraph_format
        paragraph_format.line_spacing = 1.5
        paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

        lines = text_content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('# '):
                h = doc.add_heading(level=1)
                run = h.add_run(line[2:])
                run.font.name = 'Times New Roman'
                run.font.color.rgb = None # Reset color to default black
            elif line.startswith('## '):
                h = doc.add_heading(level=2)
                run = h.add_run(line[3:])
                run.font.name = 'Times New Roman'
                run.font.color.rgb = None
            elif line.startswith('### '):
                h = doc.add_heading(level=3)
                run = h.add_run(line[4:])
                run.font.name = 'Times New Roman'
                run.font.color.rgb = None
            elif line.startswith('- ') or line.startswith('* '):
                # Bullet point
                p = doc.add_paragraph(style='List Bullet')
                parse_markdown_to_paragraph(p, line[2:])
            elif re.match(r'^\d+\.\s', line):
                # Numbered list
                p = doc.add_paragraph(style='List Number')
                parse_markdown_to_paragraph(p, line[line.find(' ')+1:])
            else:
                p = doc.add_paragraph()
                parse_markdown_to_paragraph(p, line)
                
        doc.save(output_path)
        return f"Successfully saved to {output_path}"
    except Exception as e:
        return f"Error creating DOCX: {e}"
