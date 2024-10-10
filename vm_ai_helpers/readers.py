import os
from PyPDF2 import PdfReader
from docx import Document
from pdf2image import convert_from_path
import tempfile
from tqdm import tqdm
if __name__ == "__main__":
    import img_processing, text_processing, summarizers
else:
    from vm_ai_helpers import img_processing, text_processing, summarizers

import pandas as pd
import csv  # Python's CSV module to handle quoting

def clean_csv(file_path: str, output_path: str) -> None:
    """
    Cleans a CSV file by removing line breaks within quoted fields, preserving line breaks between records.
    
    :param file_path: Path to the input CSV file.
    :param output_path: Path to save the cleaned CSV file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            for row in reader:
                # Join multiline fields within the row (fields with newlines inside them)
                clean_row = [field.replace('\n', ' ').replace('\r', ' ') for field in row]
                writer.writerow(clean_row)
        print(f"CSV cleaned successfully. Saved to {output_path}")
    
    except UnicodeDecodeError:
        print("Error reading CSV with 'utf-8' encoding, falling back to 'ISO-8859-1'")
        try:
            with open(file_path, 'r', encoding='ISO-8859-1') as infile, open(output_path, 'w', encoding='utf-8', newline='') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile)

                for row in reader:
                    clean_row = [field.replace('\n', ' ').replace('\r', ' ') for field in row]
                    writer.writerow(clean_row)
            print(f"CSV cleaned successfully with ISO-8859-1 encoding. Saved to {output_path}")
        except Exception as e:
            print(f"Error reading CSV with 'ISO-8859-1' encoding: {e}")
            return

    except Exception as e:
        print(f"Error cleaning CSV: {e}")

def read_csv_with_progress(file_path: str, encoding='utf-8', chunk_size=500) -> pd.DataFrame:
    """
    Cleans the CSV to remove line breaks within fields, then reads it in chunks, processes each chunk individually, 
    and returns a concatenated DataFrame.
    
    :param file_path: Path to the original CSV file.
    :param encoding: Encoding to use for reading the CSV.
    :param chunk_size: Number of rows per chunk.
    :return: A pandas DataFrame containing all the data.
    """
    # Generate a temporary cleaned file path
    cleaned_csv_path = file_path.replace(".csv", "_cleaned.csv")
    
    # Step 1: Clean the CSV to remove line breaks within fields
    clean_csv(file_path, cleaned_csv_path)
    
    all_chunks = []  # To hold all processed chunks
    
    try:
        # Calculate total rows for progress tracking
        total_rows = sum(1 for _ in open(cleaned_csv_path, encoding=encoding)) - 1 # Subtract 1 for header
        print(f"Total rows: {total_rows}")
        
        # Step 2: Read the cleaned CSV file in chunks
        chunks = pd.read_csv(cleaned_csv_path, encoding=encoding, chunksize=chunk_size, 
                             low_memory=False, quoting=csv.QUOTE_MINIMAL)
        
        chunk_count = 0  # Counter to track chunks processed
        
        # Iterate through chunks and process each chunk
        for chunk in tqdm(chunks, total=total_rows // chunk_size, desc="Reading and processing CSV"):
            print(f"Processing chunk {chunk_count}")
            processed_chunk = process_chunk(chunk)  # Process the chunk and return it
            all_chunks.append(processed_chunk)  # Append processed chunk to list
            chunk_count += 1
        
        # Concatenate all processed chunks into a single DataFrame
        final_df = pd.concat(all_chunks, ignore_index=True)
        print(f"Finished processing {chunk_count} chunks")
        return final_df
    
    except UnicodeDecodeError:
        print(f"Failed to read CSV with {encoding} encoding, falling back to ISO-8859-1")
        return read_csv_with_progress(file_path, encoding='ISO-8859-1', chunk_size=chunk_size)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of failure
    
def detect_text_columns(chunk: pd.DataFrame) -> list:
    """
    Detects columns that contain mostly text data.
    
    :param chunk: A pandas DataFrame chunk.
    :return: A list of column names that likely contain textual data.
    """
    text_columns = []
    for col in chunk.columns:
        # Check if the majority of the data in the column is string (text)
        if chunk[col].apply(lambda x: isinstance(x, str)).mean() > 0.5:  # More than 50% of the column is text
            text_columns.append(col)
    return text_columns


def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a chunk of the CSV file by summarizing incidents in the chunk, detecting text columns dynamically.
    
    :param chunk: A pandas DataFrame chunk containing part of the CSV data.
    :return: A pandas DataFrame chunk with additional processed information.
    """
    try:
        # Detect text-based columns
        text_columns = detect_text_columns(chunk)
        print(f"Detected text columns: {text_columns}")
        
        # If no text columns are found, raise an error or skip the chunk
        if not text_columns:
            raise ValueError("No text-based columns found for summarization.")
        
        # Create the 'Processed Summary' column if it doesn't exist
        if 'Processed Summary' not in chunk.columns:
            print("Adding 'Processed Summary' column to the chunk")
            chunk['Processed Summary'] = ""  # Initialize an empty column
        
        # Iterate through each row of the chunk and process the incident data
        for index, row in chunk.iterrows():
            # Combine text from relevant columns for summarization
            incident_text = " ".join([str(row[col]) for col in text_columns if isinstance(row[col], str)])
            
            # Actual summarization using Ollama model
            summary = summarizers.summarize_incident(incident_text, model_name="gemma")
            #print(f"Processed summary for row {index}: {summary}")  # Debug print for each row

            chunk.loc[index, 'Processed Summary'] = summary  # Store the summary in the 'Processed Summary' column
    except Exception as e:
        print(f"Error processing chunk: {e}")
    
    return chunk  # Return the processed chunk

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
        summary = summarizers.summarize_claims_with_ollama(document_text, model_name)
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
