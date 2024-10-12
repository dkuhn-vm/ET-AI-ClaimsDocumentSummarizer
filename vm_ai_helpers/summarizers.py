import warnings
from transformers import pipeline
from datasets import Dataset
import torch
from typing import List
if __name__ == "__main__":
    import ollama_funcs
else:
    from vm_ai_helpers import ollama_funcs

# Suppress the FutureWarning related to clean_up_tokenization_spaces
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# Determine if GPU is available, else use CPU
device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_id: int = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU in Hugging Face pipeline

print("Device:", device)

def summarize_text(file_name):
    with open(file_name, "r") as file:
        # Read the contents of the file into a string
        text = file.read()
    return summarize_distilbart(text)

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


def summarize_claims_with_ollama(text: str, model_name: str = "gemma") -> str:
    """
    Summarizes a given text by querying an Ollama model and handling streaming responses.
    
    :param text: The full text to be summarized.
    :return: A single string containing the summarized text from the Ollama model.
    """
    # Define the system prompt to guide the models behavior and tone
    system_prompt = """
        You are an AI assistant designed to help P&C insurance claims adjusters. 
        Your goal is to generate clear and concise summaries of lengthy documents, such as loss reports, policy documents, medical reports, and repair estimates. 
        Summarize the key details like dates, amounts, parties involved, liability, and any critical findings, while avoiding unnecessary detail.
        """
    
    # Define the prompt for summarization
    user_prompt = f"Summarize the following document:\n\n{text}\n\nProvide a concise and relevant summary for a claims adjuster."
    
    return ollama_funcs.call_ollama(text, system_prompt, user_prompt, model_name)

def summarize_incident(incident_text: str, model_name: str = "gemma") -> str:
    """
    Summarizes a single incident using an Ollama model, including domain and product information.
    
    :param incident_text: The full text of the incident to be summarized.
    :param model_name: The name of the Ollama model to use.
    :return: A summarized string of the incident along with domain and product information.
    """
    print(f"Summarizing incident: {incident_text[:100]}...")

    # Define the updated system prompt for summarizing individual incidents
    system_prompt = """
        You are an AI assistant that helps IT teams at a Property and Casualty (P&C) insurance carrier analyze incident reports.
        Your goal is to summarize each incident and identify key information such as the technical issue, its cause, impact, affected systems, 
        and resolution. Additionally, for each incident, clearly identify the relevant insurance domain (e.g., Underwriting, Sales Marketing, 
        Documents, Distribution, Billing, Claims, Corporate functions) and insurance product line (Personal, Commercial, Life) impacted by the incident.
    """
    
    user_prompt = f"Summarize the following incident:\n\n{incident_text}\n\nPlease provide a concise summary including the domain and product."
    
    # Get the summarized incident text using the new prompt
    return ollama_funcs.call_ollama(incident_text, system_prompt, user_prompt, model_name)

def summarize_trend(combined_text: str, model_name: str = "gemma") -> str:
    """
    Summarizes a combined set of incident summaries to identify trends and provide an overarching analysis,
    including domain and product details.
    
    :param combined_text: The full text of combined incident summaries.
    :param model_name: The name of the Ollama model to use.
    :return: A summarized string that captures trends and patterns across incidents, with domain and product information.
    """
    # Define an updated system prompt for summarizing trends across incidents
    system_prompt = """
        You are an AI assistant tasked with helping IT teams at a Property and Casualty (P&C) insurance carrier analyze patterns 
        and trends from multiple incident reports. Your goal is to provide a high-level summary that highlights common technical issues, 
        recurring problems, and overall trends across many incidents. Additionally, make sure to identify any trends in the impacted insurance domains 
        (e.g., Claims, Underwriting, Billing, etc.) and insurance product categories (Personal, Commercial, Life) affected by the incidents.
    """
    
    user_prompt = f"Summarize the following combined incident summaries:\n\n{combined_text}\n\nPlease provide a high-level trend summary, including domain and product trends."
    
    # Get the summarized trend text using the new prompt
    return ollama_funcs.call_ollama(combined_text, system_prompt, user_prompt, model_name)
def main() -> None:
    """
    Entry point for testing the summarize_distilbart function.
    This is a placeholder function and currently raises a NotImplementedError.
    """
    #print("Summary-Long:", summarize_text("../data/long_text.txt"))
    #print("Summary-Medium:", summarize_text("../data/mid_text.txt"))
    #print("Summary-Short:", summarize_text("../data/short_text.txt"))

    with open("../data/long_text.txt", "r") as file:
        # Read the contents of the file into a string
        text = file.read()
    print("Summary-Ollama-gemma", summarize_claims_with_ollama(text, "gemma"))
    print("Summary-Ollama-llama", summarize_claims_with_ollama(text, "llama3.2"))



if __name__ == "__main__":
    main()
