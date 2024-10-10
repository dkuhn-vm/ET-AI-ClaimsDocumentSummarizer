from vm_ai_helpers import summarizers, readers


def test_ollama(file_path, model_name="gemma") -> None:
    try:
        document_text = readers.read_document(file_path)
        summary = summarizers.summarize_claims_with_ollama(document_text, model_name)
        print(model_name, summary)  # This is where you would pass the text to the summarization function
    except Exception as e:
        print(f"Error processing document: {e}")

def main() -> None:
    # Example usage: Replace with your actual file path or folder
    #file_path = '../data/Sample Expert Report.pdf'
    file_path = 'data/Sample Police Report.pdf'
    
    #test_distilbart(file_path)
    test_ollama(file_path)
    #test_ollama(file_path, "llama3.2")

if __name__ == "__main__":
    main()