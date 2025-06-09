import os
import zipfile
import shutil
import fitz  # PyMuPDF
from docx import Document
from datetime import datetime
from typing import List
import lancedb
import pyarrow as pa
import numpy as np
import pytz
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Set Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAUhKKil9ZSreVmNzF-_aOb2K-e03AzoY0"

# Folder Setup
BASE_FOLDER = "UserStories"
UPLOAD_FOLDER = os.path.join(BASE_FOLDER, "uploaded_docs")
SUCCESS_FOLDER = os.path.join(BASE_FOLDER, "success")
FAILURE_FOLDER = os.path.join(BASE_FOLDER, "failure")
LANCE_DB_PATH = os.path.join(BASE_FOLDER, "my_lance_db")

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUCCESS_FOLDER, exist_ok=True)
os.makedirs(FAILURE_FOLDER, exist_ok=True)
os.makedirs(LANCE_DB_PATH, exist_ok=True)

# ----- Text Extraction Functions -----
def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    return "".join(page.get_text() for page in doc)

def extract_text_from_docx(file_path: str) -> str:
    if not zipfile.is_zipfile(file_path):
        raise ValueError(f"'{file_path}' is not a valid DOCX file.")
    doc = Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs)

def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def embed_text(text: str, model: GoogleGenerativeAIEmbeddings) -> List[float]:
    return model.embed_query(text)

def get_file_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    return {"pdf": "pdf", "docx": "docx", "txt": "txt"}.get(ext[1:], "unknown")

def generate_story_id(file_name: str) -> str:
    return f"{os.path.splitext(file_name)[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

def get_current_ist_timestamp() -> str:
    return datetime.now(pytz.timezone("Asia/Kolkata")).isoformat()

# ----- Main Processing Function -----
def main():
    # Load Gemini embedding model
    model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Connect to LanceDB
    db = lancedb.connect(LANCE_DB_PATH)
    schema = pa.schema([
    ("story_id", pa.string()),
    ("story desc vector", pa.list_(pa.float32())),
    ("file_name", pa.string()),
    ("processed_flags", pa.string()),
    ("timestamp", pa.string()),
    ("test_cases", pa.list_(pa.float32())),  # ✅ Vector for LLM-generated test cases
    ])

    # Open or create table
    
    if "user_stories" in db.table_names():
        table = db.open_table("user_stories")
    else:
        table = db.create_table("user_stories", schema=schema)

    all_rows = []

    # Process each file
    for file_name in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, file_name)
        if not os.path.isfile(file_path):
            continue

        file_type = get_file_type(file_path)
        if file_type == "unknown":
            print(f"⏭ Skipping unsupported file: {file_name}")
            shutil.move(file_path, os.path.join(FAILURE_FOLDER, file_name))
            continue

        try:
            # Step 1: Extract text
            if file_type == "pdf":
                text = extract_text_from_pdf(file_path)
            elif file_type == "docx":
                text = extract_text_from_docx(file_path)
            elif file_type == "txt":
                text = extract_text_from_txt(file_path)
            else:
                raise ValueError("Unsupported file type")

            if not text.strip():
                raise ValueError("Empty text extracted")

            # Step 2: Embed
            vector = embed_text(text, model)
            story_id = generate_story_id(file_name)
            timestamp = get_current_ist_timestamp()

            # Step 3: Show content
            print(f"\n📄 File: {file_name}")
            print(f"🆔 Story ID: {story_id}")
            print(f"📌 Text Preview:\n{text[:300]}...")

            # Step 4: Store
            all_rows.append({
                all_rows.append({
    "story_id": story_id,
    "story_desc_vector": vector,           
    "file_name": file_name,
    "processed_flags": "NO",             
    "timestamp": timestamp,
    "test_cases": []                     
})

            })


            # Step 5: Move to success
            shutil.move(file_path, os.path.join(SUCCESS_FOLDER, file_name))

        except Exception as e:
            print(f"❌ Failed to process {file_name}: {e}")
            shutil.move(file_path, os.path.join(FAILURE_FOLDER, file_name))

    # Insert rows
    if all_rows:
        table.add(all_rows)
        print(f"\n✅ {len(all_rows)} files inserted into LanceDB.")
    else:
        print("⚠ No new files inserted.")

if __name__== "__main__":
    main()