import warnings
from transformers import pipeline
from datasets import Dataset
import torch
from typing import List

# Suppress the FutureWarning related to clean_up_tokenization_spaces
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# Determine if GPU is available, else use CPU
device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_id: int = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU in Hugging Face pipeline

print("Device:", device)

def summarize_distilbart(text: str) -> str:
    """
    Summarizes a given text by splitting it into manageable chunks, summarizing each chunk,
    and then combining the results into a single summary.

    :param text: The full text to be summarized.
    :return: A single string containing the combined summaries of each chunk of the original text.
    """
    
    # Define the size of each chunk of text. The model has a maximum token length it can handle,
    # so we split the input text into smaller pieces to stay within those limits.
    chunk_size: int = 1024
    
    # Split the input text into chunks of size `chunk_size`.
    # This ensures that large documents are broken into smaller sections that the model can process.
    chunks: List[str] = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Create a Hugging Face Dataset from the chunks.
    # This dataset structure makes it easier to batch process the chunks using the summarization pipeline.
    dataset = Dataset.from_dict({"text": chunks})

    # Initialize the Hugging Face summarization pipeline with the chosen model and device.
    # The `device=device_id` argument allows the pipeline to use GPU (0) or CPU (-1) depending on availability.
    summarization = pipeline(
        "summarization", 
        model="sshleifer/distilbart-cnn-12-6", 
        device=device_id, 
        clean_up_tokenization_spaces=True  # Explicitly set this to True or False
    )
    # Determine the batch size for processing.
    # Use a larger batch size for GPU to take advantage of parallel processing, while CPU uses smaller batches.
    batch_size: int = 8 if device_id == 0 else 1

    # Use the summarization pipeline to summarize each chunk of text in batches.
    # This improves efficiency when using a GPU, as it processes multiple chunks in parallel.
    summaries = summarization(
        dataset["text"],  # Pass the text chunks from the dataset
        batch_size=batch_size,  # Use the appropriate batch size for the available hardware
        max_length=100,  # Maximum length of the generated summary
        min_length=20,  # Minimum length of the generated summary
        truncation=True,  # Ensure overly long text is truncated to fit within model limits
        do_sample=False  # Disable sampling for deterministic summarization results
    )

    # Extract the actual summary text from each result in the list of summaries.
    # Each summary is returned as a dictionary, so we need to pull out the 'summary_text' field.
    summary_text: List[str] = [summary['summary_text'] for summary in summaries]

    # Combine the individual summaries into one cohesive summary by joining them with spaces.
    return " ".join(summary_text)

def main() -> None:
    """
    Entry point for testing the summarize_distilbart function.
    This is a placeholder function and currently raises a NotImplementedError.
    """
    raise NotImplementedError("Main testing not implemented")

if __name__ == "__main__":
    main()
