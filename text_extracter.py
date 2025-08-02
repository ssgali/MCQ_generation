import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    if pdf_path == "":
        return ""
    try: 
        text = ""
        doc = fitz.open(pdf_path)  # ← direct path input
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return ""
