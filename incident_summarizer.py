import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import signal
from tqdm import tqdm
import csv
from typing import Tuple
from vm_ai_helpers import summarizers

# Get number of CPU cores
cpu_count = multiprocessing.cpu_count()

# Global flag to stop processing when Ctrl+C is pressed
should_exit = False

def signal_handler(sig, frame):
    global should_exit
    print("\nGracefully shutting down...")
    should_exit = True

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

def clean_csv(file_path: str, output_path: str) -> None:
    """
    Cleans the input CSV by removing newlines and saving it with the specified encoding.
    Falls back to 'ISO-8859-1' if 'utf-8' encoding fails.
    """
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
            
            # Print the incident number and summary for monitoring
            print(f"Processing row {index}: {row['Number']}")

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

    # Clean the CSV file before processing
    cleaned_csv_path = file_path.replace(".csv", "_cleaned.csv")
    clean_csv(file_path, cleaned_csv_path)

    # Load CSV data (or sample, if in debug mode)
    df, total_rows = load_csv_data(cleaned_csv_path, encoding, debug, sample_size)

    if df is None or total_rows == 0:
        print("No data found in CSV.")
        return pd.DataFrame()

    # **Correctly limit the sample size in debug mode**: Sample the data here
    if debug:
        print(f"Running in debug mode, sampling {sample_size} records out of {total_rows}.")
        df = df.sample(n=min(sample_size, total_rows), random_state=42)
        total_rows = len(df)  # Update total_rows after sampling
        chunk_size = sample_size  # Make the chunk size equal to the sample size to ensure the chunking is minimal

    # Process data in chunks
    return process_csv_chunks(df, total_rows, chunk_size, max_workers)

def load_csv_data(file_path: str, encoding: str, debug: bool, sample_size: int) -> Tuple[pd.DataFrame, int]:
    """
    Loads the CSV data, with optional sampling for debug mode. Handles encoding fallbacks.
    
    :param file_path: Path to the cleaned CSV file.
    :param encoding: Encoding to use when reading the file.
    :param debug: Whether to enable sampling (debug mode).
    :param sample_size: Number of rows to sample if debugging.
    :return: Tuple containing the DataFrame and total number of rows.
    """
    try:
        # Try loading the CSV file into a DataFrame
        df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
        total_rows = len(df)
        return df, total_rows

    except UnicodeDecodeError as e:
        print(f"Encoding error with 'utf-8', falling back to 'ISO-8859-1': {e}")
        try:
            df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
            total_rows = len(df)
            return df, total_rows
        except Exception as e:
            print(f"Error loading CSV with fallback encoding: {e}")
            return None, 0
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None, 0

def process_csv_chunks(df: pd.DataFrame, total_rows: int, chunk_size: int, max_workers: int) -> pd.DataFrame:
    """
    Processes the CSV data in chunks using parallel processing.
    
    :param df: DataFrame to process.
    :param total_rows: Total number of rows in the DataFrame.
    :param chunk_size: Number of rows to process in each chunk.
    :param max_workers: Maximum number of workers for parallel processing.
    :return: Processed DataFrame.
    """
    global should_exit
    all_chunks = []
    futures = []
    
    # Split the DataFrame into chunks
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, total_rows, chunk_size)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for chunk_id, chunk in enumerate(tqdm(chunks, total=len(chunks), desc="Processing CSV")):
            if should_exit:
                print("Exiting process due to Ctrl+C")
                break
            futures.append(executor.submit(process_chunk_parallel, chunk, chunk_id))

        # Collect results
        for future in as_completed(futures):
            if should_exit:
                print("Stopping due to Ctrl+C.")
                break
            processed_chunk = future.result()
            if not processed_chunk.empty:
                all_chunks.append(processed_chunk)

    # Concatenate all processed chunks
    if not should_exit:
        final_df = pd.concat(all_chunks, ignore_index=True)
        print(f"Finished processing {len(all_chunks)} chunks")
        return final_df
    else:
        print("Process terminated by user.")
        return pd.DataFrame()

def detect_text_columns(chunk: pd.DataFrame) -> list:
    text_columns = []
    for col in chunk.columns:
        if chunk[col].apply(lambda x: isinstance(x, str)).mean() > 0.5:
            text_columns.append(col)
    return text_columns

def process_incident(row, columns, model_name="gemma"):
    """
    Function to process a single incident for parallel processing.
    
    :param row: A single row from the DataFrame.
    :param columns: List of columns from the DataFrame.
    :param model_name: The model used for summarization.
    :return: The processed summary.
    """
    if should_exit:
        return None  # Exit early if flag is set
    
    # Construct the structured incident text with headers and values
    incident_data = "\n".join([f"{col}: {row[col]}" for col in columns if pd.notna(row[col])])
    
    # Pass structured incident data to the LLM for summarization
    summary = summarizers.summarize_incident(incident_data, model_name)
    
    return summary

def summarize_incidents_from_csv(file_path, model_name="gemma2", debug=False, sample_size=100, output_file="output.md") -> None:
    """
    Summarizes incidents from a CSV file using Ollama models and writes only the final trend summary to an output markdown file.
    
    :param file_path: Path to the CSV file.
    :param model_name: Name of the Ollama model used for summarization (default is 'gemma').
    :param debug: Boolean to indicate debug mode.
    :param sample_size: Number of records to sample if debugging.
    :param output_file: Path to the output markdown file for storing the final trend summary.
    """
    global should_exit  # Reference the global flag for exit
    executor = None  # Initialize executor for later shutdown handling

    try:
        # Read and process the CSV file with progress tracking
        if cpu_count > 8: 
            max_workers = 8  # Limit to 8 workers for large CPU counts
        else:
            max_workers = cpu_count  # Use all available CPUs for smaller counts
        if debug:
            print("Debug Mode On")
            incidents_df = read_csv_with_parallel_processing(file_path=file_path, max_workers=max_workers, debug=debug, sample_size=sample_size)
        else:
            incidents_df = read_csv_with_parallel_processing(file_path=file_path, max_workers=max_workers)
        
        all_summaries = []

        # List of columns to pass for constructing incident text
        columns = incidents_df.columns

        # Use ProcessPoolExecutor to parallelize incident summarization
        with ProcessPoolExecutor(max_workers=cpu_count) as executor:
            futures = [executor.submit(process_incident, row, columns, model_name) for _, row in incidents_df.iterrows()]
            
            for index, future in enumerate(as_completed(futures)):
                if should_exit:
                    print("Early exit triggered. Shutting down executor...")
                    executor.shutdown(wait=False)
                    break  # Stop processing if Ctrl+C was pressed

                summary = future.result()
                if summary is not None:
                    all_summaries.append(summary)
                    incidents_df.loc[index, 'Processed Summary'] = summary

        if should_exit:
            print("Exiting before combining summaries due to Ctrl+C.")
            return  # Exit early if the process was interrupted

        # Combine all summaries for final trend analysis
        combined_summaries = " ".join(all_summaries)
        print(f"Combined Summaries for Final Trend Summary:\n{combined_summaries[:1000]}...")  # Optional: Print part of combined summaries for debugging
        
        # Generate the final trend summary
        final_summary = summarizers.summarize_trend(combined_summaries, model_name)
        
        # Write only the final trend summary to the output file
        with open(output_file, 'w') as f:
            f.write("# Final Trend Summary\n\n")
            f.write(final_summary)
        
        print("Final Trend Summary:", final_summary)
    
    except Exception as e:
        print(f"Error processing incidents: {e}")
    finally:
        # Ensure executor shutdown if it's still running
        if executor:
            executor.shutdown(wait=True)

def main() -> None:
    # Example usage: Replace with your actual file path
    file_path = 'incident_data/Incidents.csv'
    #summarize_incidents_from_csv(file_path=file_path)
    summarize_incidents_from_csv(file_path=file_path,model_name="gemma2", debug=True, sample_size=100)
    #summarize_incidents_from_csv(file_path=file_path,model_name="llama3.1", debug=True, sample_size=600)

if __name__ == "__main__":
    main()
