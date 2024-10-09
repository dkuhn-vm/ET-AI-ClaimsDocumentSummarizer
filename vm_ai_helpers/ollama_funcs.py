import subprocess
import time
import requests
import ollama

ollama_site: str = "http://localhost:11434"

def is_ollama_running() -> bool:
    """
    Checks if Ollama server is running by sending a request to the API endpoint.
    """
    try:
        response = requests.get(f"{ollama_site}/api/models")
        if response.status_code == 200:
            print("Ollama is already running.")
            return True
        return False
    except requests.ConnectionError as e:
        print(f"Connection Error: {e}")
        return False

def start_ollama_server() -> None:
    """
    Starts the Ollama server by running 'ollama serve' if it is not already running.
    """
    if is_ollama_running():
        print("Ollama server is already running. No need to start it again.")
        return  # Do not start the server if it's already running
    
    try:
        print("Starting Ollama server...")
        subprocess.Popen(["ollama", "serve"])
        # Give some time for the server to start
        time.sleep(5)  # Wait 5 seconds to give Ollama time to start
    except Exception as e:
        print(f"Error starting Ollama server: {str(e)}")

def is_model_loaded(model_name: str) -> bool:
    """
    Checks if the specified model is loaded in Ollama.
    """
    try:
        response = requests.get(f"{ollama_site}/api/models")
        if response.status_code == 200:
            models = response.json()
            return model_name in models
        return False
    except requests.ConnectionError:
        return False

def load_model(model_name: str) -> None:
    """
    Loads the specified model into Ollama.
    """
    try:
        response = requests.post(
            f"{ollama_site}/api/models/load",
            json={"model": model_name}
        )
        if response.status_code == 200:
            print(f"Model {model_name} is being loaded...")
        else:
            print(f"Failed to load model {model_name}: {response.status_code}")
    except requests.ConnectionError as e:
        print(f"Error connecting to Ollama API: {str(e)}")

def call_ollama(text: str, system_prompt: str, user_prompt: str, model_name: str = "gemma") -> str:
    try:
        # Combine system and user prompts
        complete_prompt = system_prompt + "\n" + user_prompt

        generate = ollama.generate(model=model_name, prompt=complete_prompt)

        # Return the generated text from the response
        return generate['response']

    except Exception as e:
        return f"Error: Could not connect to the Ollama API. Exception: {str(e)}"
    except Exception as e:
        return f"Error: Could not connect to the Ollama API. Exception: {str(e)}"

# Example usage of the above functions to call Ollama summarization
if __name__ == "__main__":
    text = "Sample long text to summarize..."
    system_prompt = "You are an assistant helping summarize text."
    user_prompt = "Summarize the following text:"
    print(call_ollama(text, system_prompt, user_prompt, "gemma"))
