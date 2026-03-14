import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_file) -> str:
    """
    Accepts either:
    - A Streamlit UploadedFile object (BytesIO-like)
    - A file path string
    """
    if pdf_file is None or pdf_file == "":
        return ""
    try:
        # Streamlit returns a BytesIO-like object, not a path
        if hasattr(pdf_file, "read"):
            data = pdf_file.read()
            doc = fitz.open(stream=data, filetype="pdf")
        else:
            doc = fitz.open(pdf_file)

        text = ""
        for page in doc:
            text += page.get_text()                 # type: ignore
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"[text_extracter] Failed to extract PDF: {e}")
        return ""
