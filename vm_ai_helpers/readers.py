import os
from PyPDF2 import PdfReader
from docx import Document
from pdf2image import convert_from_path
import tempfile
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
if __name__ == "__main__":
    import img_processing, text_processing, summarizers
else:
    from vm_ai_helpers import img_processing, text_processing, summarizers

import pandas as pd
import csv
import threading
import signal

# Create a lock object for file writing
write_lock = threading.Lock()

# Create a global flag to stop the processing when Ctrl+C is pressed
should_exit = False

def signal_handler(sig, frame):
    global should_exit
    print("Gracefully shutting down...")
    should_exit = True  # Set the exit flag to True

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

def clean_csv(file_path: str, output_path: str) -> None:
    try:
        with open(file_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            for row in reader:
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

def process_chunk_parallel(chunk: pd.DataFrame, chunk_id: int) -> pd.DataFrame:
    """
    Process each chunk in parallel without writing to file immediately.
    
    :param chunk: A pandas DataFrame chunk containing part of the CSV data.
    :param chunk_id: An identifier for the chunk.
    :return: A pandas DataFrame chunk with additional processed information.
    """
    global should_exit
    if should_exit:
        print(f"Stopping chunk processing for chunk {chunk_id} due to exit signal.")
        return pd.DataFrame()  # Return empty dataframe if the process is interrupted

    try:
        # Ensure 'Processed Summary' column exists in the chunk
        if 'Processed Summary' not in chunk.columns:
            print(f"Adding 'Processed Summary' column to chunk {chunk_id}")
            chunk['Processed Summary'] = ""  # Initialize the column with empty strings

        # Detect text-based columns and process the chunk
        text_columns = detect_text_columns(chunk)
        if not text_columns:
            raise ValueError(f"No text-based columns found for summarization in chunk {chunk_id}.")

        for index, row in chunk.iterrows():
            if should_exit:  # Check if we should exit in the middle of processing
                print(f"Stopping row processing for chunk {chunk_id} due to exit signal.")
                return pd.DataFrame()  # Return empty dataframe if the process is interrupted
            
            print(f"Processing row {index}: {row['Number']} {row['Summary'][:100]}")

            # Combine all the relevant text from text columns for summarization
            incident_text = "\n".join([f"{col}: {str(row[col])}" for col in text_columns if isinstance(row[col], str)])  # Ensure proper newlines for better formatting
            summary = summarizers.summarize_incident(incident_text, model_name="gemma")
            chunk.loc[index, 'Processed Summary'] = summary  # Store the summary for each incident

    except Exception as e:
        print(f"Error processing chunk {chunk_id}: {e}")

    return chunk

def read_csv_with_parallel_processing(file_path: str, encoding='utf-8', chunk_size=500, max_workers=8, debug=False, sample_size=100) -> pd.DataFrame:
    """
    Reads a CSV file in chunks, processes each chunk in parallel, and returns the final processed DataFrame.
    Handles Ctrl+C gracefully to stop all processes. If utf-8 encoding fails, it falls back to ISO-8859-1.
    
    :param file_path: Path to the CSV file.
    :param encoding: Encoding of the CSV file (default is 'utf-8').
    :param chunk_size: Number of rows to process in each chunk (default is 500).
    :param max_workers: Maximum number of workers for parallel processing (default is 8).
    :param debug: Boolean flag to indicate whether debugging is enabled.
    :param sample_size: Number of random records to sample if debugging is enabled.
    :return: Processed pandas DataFrame.
    """
    global should_exit
    should_exit = False  # Reset exit flag

    cleaned_csv_path = file_path.replace(".csv", "_cleaned.csv")
    clean_csv(file_path, cleaned_csv_path)  # Ensure the CSV is cleaned of newlines

    all_chunks = []

    try:
        # Try reading with the provided encoding
        total_rows = sum(1 for _ in open(cleaned_csv_path, encoding=encoding)) - 1
        print(f"Total rows: {total_rows}")
        
        # Read CSV in chunks
        chunks = pd.read_csv(cleaned_csv_path, encoding=encoding, chunksize=chunk_size, low_memory=False, quoting=csv.QUOTE_MINIMAL)

        # If debugging, select a random sample of rows to process
        if debug:
            print(f"Debug mode enabled. Sampling {sample_size} random records for processing.")
            full_df = pd.concat(chunks)  # Load the entire CSV into memory
            full_df_sample = full_df.sample(n=sample_size, random_state=42)  # Take a random sample
            chunks = [full_df_sample]  # Replace chunks with the sampled data
            total_rows = sample_size

        futures = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for chunk_id, chunk in enumerate(tqdm(chunks, total=total_rows // chunk_size, desc="Reading and processing CSV")):
                if should_exit:
                    print("Exiting process due to Ctrl+C")
                    break  # Stop submitting new chunks if we should exit
                futures.append(executor.submit(process_chunk_parallel, chunk, chunk_id))
            
            for future in as_completed(futures):
                if should_exit:
                    break  # Stop waiting for other futures if exit was requested
                processed_chunk = future.result()
                all_chunks.append(processed_chunk)

        # Concatenate all processed chunks into a single DataFrame
        if not should_exit:
            final_df = pd.concat(all_chunks, ignore_index=True)
            print(f"Finished processing {len(all_chunks)} chunks")
            return final_df
        else:
            print("Process terminated by user.")
            return pd.DataFrame()

    except UnicodeDecodeError as e:
        # If utf-8 fails, attempt to fall back to ISO-8859-1
        print(f"Encoding error with 'utf-8', falling back to 'ISO-8859-1': {e}")
        try:
            chunks = pd.read_csv(cleaned_csv_path, encoding='ISO-8859-1', chunksize=chunk_size, low_memory=False, quoting=csv.QUOTE_MINIMAL)
            
            if debug:
                print(f"Debug mode enabled. Sampling {sample_size} random records for processing.")
                full_df = pd.concat(chunks)  # Load the entire CSV into memory
                full_df_sample = full_df.sample(n=sample_size, random_state=42)  # Take a random sample
                chunks = [full_df_sample]  # Replace chunks with the sampled data
                total_rows = sample_size

            futures = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for chunk_id, chunk in enumerate(tqdm(chunks, total=total_rows // chunk_size, desc="Reading and processing CSV (fallback)")):
                    if should_exit:
                        print("Exiting process due to Ctrl+C")
                        break
                    futures.append(executor.submit(process_chunk_parallel, chunk, chunk_id))
                
                for future in as_completed(futures):
                    if should_exit:
                        break
                    processed_chunk = future.result()
                    all_chunks.append(processed_chunk)

            if not should_exit:
                final_df = pd.concat(all_chunks, ignore_index=True)
                print(f"Finished processing {len(all_chunks)} chunks with fallback encoding")
                return final_df
            else:
                print("Process terminated by user.")
                return pd.DataFrame()

        except Exception as e:
            print(f"Error reading CSV with fallback encoding: {e}")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return pd.DataFrame()

def detect_text_columns(chunk: pd.DataFrame) -> list:
    text_columns = []
    for col in chunk.columns:
        if chunk[col].apply(lambda x: isinstance(x, str)).mean() > 0.5:
            text_columns.append(col)
    return text_columns

def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    try:
        text_columns = detect_text_columns(chunk)
        print(f"Detected text columns: {text_columns}")
        
        if not text_columns:
            raise ValueError("No text-based columns found for summarization.")
        
        if 'Processed Summary' not in chunk.columns:
            print("Adding 'Processed Summary' column to the chunk")
            chunk['Processed Summary'] = ""
        
        for index, row in chunk.iterrows():
            incident_text = " ".join([str(row[col]) for col in text_columns if isinstance(row[col], str)])
            summary = summarizers.summarize_incident(incident_text, model_name="gemma")
            chunk.loc[index, 'Processed Summary'] = summary
    except Exception as e:
        print(f"Error processing chunk: {e}")
    
    return chunk

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
