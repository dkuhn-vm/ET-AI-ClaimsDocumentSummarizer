import os
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
#from vm_ai_helpers import img_processing, text_processing, summarizers
import img_processing, text_processing, summarizers
import tempfile

def read_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file, handling both text-based and image-based PDFs.
    
    Parameters:
    file_path (str): The path to the PDF file.
    
    Returns:
    str: Extracted text from the PDF, either directly or via OCR on images.
    """
    reader = PdfReader(file_path)  # Initialize PDF reader
    text = []  # List to store extracted text
    
    # Iterate through all the pages in the PDF
    for page_num, page in enumerate(reader.pages):
        # Attempt to extract text from the page
        page_text = page.extract_text()
        
        if page_text:
            # If text is found, append it
            text.append(page_text)
        else:
            # If no text is found, assume the page might be an image
            print(f"Page {page_num} is likely an image. Extracting text using OCR.")
            # Convert the PDF page to an image using pdf2image
            images = convert_from_path(file_path, first_page=page_num + 1, last_page=page_num + 1)
            
            for image in images:
                # Save the image to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_image_file:
                    # Save the PIL image to the temp file
                    image.save(temp_image_file.name)
                    
                    # Perform OCR on the temporary image file using the path
                    ocr_text = text_processing.clean_text(img_processing.extract_text_from_image(temp_image_file.name))
                    text.append(ocr_text)
    
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
        text.append(text_processing.clean_text(paragraph.text))
    
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
        ocr_text = text_processing.clean_text(img_processing.extract_text_from_image(image_path))
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

def test_distilbart(file_path) -> None:
    try:
        document_text = read_document(file_path)
        summary = summarizers.summarize_distilbart(document_text)
        summary = summarizers.summarize_distilbart(summary)
        summary = summarizers.summarize_distilbart(summary)
        summary = summarizers.summarize_distilbart(summary)
        print(summary)  # This is where you would pass the text to the summarization function
    except Exception as e:
        print(f"Error processing document: {e}")

def test_ollama(file_path, model_name="gemma") -> None:
    try:
        document_text = read_document(file_path)
        summary = summarizers.summarize_with_ollama(document_text, model_name)
        print(model_name, summary)  # This is where you would pass the text to the summarization function
    except Exception as e:
        print(f"Error processing document: {e}")

def main() -> None:
    # Example usage: Replace with your actual file path or folder
    #file_path = '../data/Sample Expert Report.pdf'
    file_path = '../data/Sample Police Report.pdf'
    
    #test_distilbart(file_path)
    test_ollama(file_path)
    #test_ollama(file_path, "llama3.2")

if __name__ == "__main__":
    main()
