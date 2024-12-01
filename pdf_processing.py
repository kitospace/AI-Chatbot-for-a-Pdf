import pdfplumber
import re
from nltk.tokenize import sent_tokenize
import nltk

# Download necessary resources
nltk.download('punkt')


def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""  # Handle pages with no text
        if not text.strip():
            raise ValueError("The uploaded PDF has no extractable text.")
        return text
    except Exception as e:
        raise RuntimeError(f"Error reading PDF file: {e}")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text.strip()


def preprocess_text(text, chunk_size=1000):
    sentences = sent_tokenize(text)
    if not sentences:
        raise ValueError("No valid sentences found in the text.")
    chunks, chunk = [], []
    chunk_word_count = 0

    for sentence in sentences:
        chunk_word_count += len(sentence.split())
        chunk.append(sentence)
        if chunk_word_count >= chunk_size:
            chunks.append(" ".join(chunk))
            chunk, chunk_word_count = [], 0

    if chunk:  # Add the last chunk
        chunks.append(" ".join(chunk))
    if not chunks:
        raise ValueError("No valid chunks could be created from the text.")
    return chunks


# http://192.168.209.84:5000
