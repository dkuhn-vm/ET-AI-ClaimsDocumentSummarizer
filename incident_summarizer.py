from vm_ai_helpers import readers, summarizers
import multiprocessing

# Get number of CPU cores
cpu_count = multiprocessing.cpu_count()

def summarize_incidents_from_csv(file_path, model_name="gemma", debug=False, sample_size=100) -> None:
    """
    Summarizes incidents from a CSV file using Ollama models and returns a final trend summary.
    
    :param file_path: Path to the CSV file.
    :param model_name: Name of the Ollama model used for summarization (default is 'gemma').
    """
    try:
        # Read and process the CSV file with progress tracking
        incidents_df = readers.read_csv_with_parallel_processing(file_path=file_path, max_workers=cpu_count)
        
        all_summaries = []
        
        for index, row in incidents_df.iterrows():
            # Construct the structured incident text with headers and values
            incident_data = "\n".join([f"{col}: {row[col]}" for col in incidents_df.columns if pd.notna(row[col])])

            # Pass structured incident data to the LLM for summarization
            summary = summarizers.summarize_incident(incident_data, model_name)
            
            all_summaries.append(summary)
            incidents_df.loc[index, 'Processed Summary'] = summary

        # Combine all summaries for final trend analysis
        combined_summaries = " ".join(all_summaries)
        print(f"Combined Summaries for Final Trend Summary:\n{combined_summaries[:1000]}...")  # Optional: Print part of combined summaries for debugging
        
        # Generate the final trend summary
        final_summary = summarizers.summarize_trend(combined_summaries, model_name)
        
        print("Final Trend Summary:", final_summary)
    
    except Exception as e:
        print(f"Error processing incidents: {e}")

def main() -> None:
    # Example usage: Replace with your actual file path
    file_path = 'incident_data/Incidents.csv'
    summarize_incidents_from_csv(file_path=file_path, debug=True, sample_size=500)

if __name__ == "__main__":
    main()
