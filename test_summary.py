from transformers import pipeline
from datasets import Dataset
import torch
from vm_ai_helpers import summarizers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def summarize_text(file_name):
    with open(file_name, "r") as file:
        # Read the contents of the file into a string
        text = file.read()
    return summarizers.summarize_distilbart(text)

def main():
    print("Summary-Long:", summarize_text("data/long_text.txt"))
    print("Summary-Medium:", summarize_text("data/mid_text.txt"))
    print("Summary-Short:", summarize_text("data/short_text.txt"))

    # open pdf
    

if __name__ == "__main__":
   main()