import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

def load_text_file():
    # Create a temporary text file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"This is a sample text file for testing.")
        temp_file_path = temp_file.name

    # Load the text file using TextLoader
    loader = TextLoader(temp_file_path)
    documents = loader.load()

    # Print the loaded documents
    for doc in documents:
        print(doc.page_content)

    # Clean up the temporary file
    os.remove(temp_file_path)

if __name__ == "__main__":
    load_text_file()
