from flask import Flask, request, jsonify
from flask_cors import CORS
from vm_ai_helpers import summarizers, readers
import incident_summarizer
import os

app = Flask(__name__)
CORS(app)  # Enable CORS

def check_file_uploaded(request) -> str:
    try:
        # Check if the request contains a file
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        # Get the uploaded file
        uploaded_file = request.files['file']

        # Save the file temporarily for processing
        file_path = os.path.join('/tmp', uploaded_file.filename)
        uploaded_file.save(file_path)

        return file_path

    except Exception as e:
        app.logger.error(f"Error during summarization: {str(e)}")
        return jsonify({'error': 'Internal server error. Please try again later.'}), 500


@app.route('/summarize', methods=['POST'])
def summarize_document():
    try:
        file_path = check_file_uploaded(request)

        # Extract text from the document
        document_text = readers.read_document(file_path)

        if not document_text:
            return jsonify({'error': 'No text could be extracted from the document'}), 400

        # Summarize the extracted text
        summary = summarizers.summarize_claims_with_ollama(document_text)

        # Return the summarized text as a response
        return jsonify({'summary': summary}), 200
    
    except Exception as e:
        app.logger.error(f"Error during summarization: {str(e)}")
        return jsonify({'error': 'Internal server error. Please try again later.'}), 500
    
@app.route('/incidents', methods=['POST'])
def summarize_incidents():
    try:
        file_path = check_file_uploaded(request)

        # Extract text from the document
        document_text = readers.read_document(file_path)

        if not document_text:
            return jsonify({'error': 'No text could be extracted from the document'}), 400

        # Summarize the extracted text
        summary = incident_summarizer.summarize_incidents_from_csv(document_text)

        # Return the summarized text as a response
        return jsonify({'summary': summary}), 200
    
    except Exception as e:
        app.logger.error(f"Error during summarization: {str(e)}")
        return jsonify({'error': 'Internal server error. Please try again later.'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    # Start the Flask service
    app.run(host='0.0.0.0', port=5000, debug=True)
