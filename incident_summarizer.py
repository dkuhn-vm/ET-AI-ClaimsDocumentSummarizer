from vm_ai_helpers import readers, summarizers

def summarize_incidents_from_csv(file_path, model_name="gemma") -> None:
    """
    Summarizes incidents from a CSV file using Ollama models and returns a final trend summary.
    
    :param file_path: Path to the CSV file.
    :param model_name: Name of the Ollama model used for summarization (default is 'gemma').
    """
    try:
        # Read and process the CSV file with progress tracking
        incidents_df = readers.read_csv_with_progress(file_path)
        
        # Now you can access the 'Processed Summary' column or do further processing
        all_summaries = []
        
        for index, row in incidents_df.iterrows():
            # Summarize each incident individually
            incident_text = f"Summary: {row['Summary']}; Incident Area: {row['Incident Area']}; " \
                            f"Group: {row['Group']}; Environment: {row['Environment']}; Class: {row['Class']}; " \
                            f"Related CI Name: {row['Related - CI Name']}; Related CI Family: {row['Related - CI Family']}"
            summary = summarizers.summarize_incident(incident_text, model_name)
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
    summarize_incidents_from_csv(file_path)

if __name__ == "__main__":
    main()
