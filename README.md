to run the service
```console
$ curl -X POST http://localhost:5000/summarize -H "Content-Type: application/json" -d '{"file_path": "/path/to/document_or_folder"}'
```

example
```json
{
  "summary": "This is the summarized version of the document..."
}
```