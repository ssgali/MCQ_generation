from pypdf import PdfReader
import re
import sys

def text_getter(name):
    text = """"""
    try:
        reader = PdfReader(name)
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            text = text + page.extract_text()
            text = re.sub(r'\d+$', '', text)
        return text
    
    except FileNotFoundError:
        print("Please Provide a valid FileName")
        sys.exit()

     