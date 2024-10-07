import os
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import pytesseract
from vm_ai_helpers import img_processing

def read_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file.
    
    Parameters:
    file_path (str): The path to the PDF file.
    
    Returns:
    str: Extracted text from the PDF.
    """
    reader = PdfReader(file_path)
    text = []
    
    # Iterate through all the pages and extract text
    for page in reader.pages:
        text.append(page.extract_text())
    
    return "\n".join(text)

def read_word(file_path: str) -> str:
    """
    Extracts text from a Word document.
    
    Parameters:
    file_path (str): The path to the Word file.
    
    Returns:
    str: Extracted text from the Word document.
    """
    doc = Document(file_path)
    text = []
    
    # Iterate through all the paragraphs in the document
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    
    return "\n".join(text)

def read_images_in_folder(folder_path: str) -> str:
    """
    Performs OCR on a series of images in a folder.
    
    Parameters:
    folder_path (str): The path to the folder containing images.
    
    Returns:
    str: The concatenated OCR text from all the images.
    """
    text = []
    # List all image files in the folder (sequentially named)
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))])
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        # Perform OCR on the image
        ocr_text = img_processing.extract_text_from_image(image_path)
        text.append(ocr_text)
    
    return "\n".join(text)

def read_document(file_path: str) -> str:
    """
    Determines the type of document (PDF, Word, Images) and extracts text accordingly.
    
    Parameters:
    file_path (str): The path to the file or folder containing images.
    
    Returns:
    str: Extracted text from the document.
    """
    if file_path.endswith('.pdf'):
        # Handle PDF files
        return read_pdf(file_path)
    elif file_path.endswith('.docx'):
        # Handle Word files
        return read_word(file_path)
    elif os.path.isdir(file_path):
        # Handle a folder of images (treat as a single document)
        return read_images_in_folder(file_path)
    else:
        raise ValueError("Unsupported file format or structure")

def main():
    # Example usage: Replace with your actual file path or folder
    file_path = '/data/document_name'
    
    try:
        document_text = read_document(file_path)
        print(document_text)  # This is where you would pass the text to the summarization function
    except Exception as e:
        print(f"Error processing document: {e}")

if __name__ == "__main__":
    main()
