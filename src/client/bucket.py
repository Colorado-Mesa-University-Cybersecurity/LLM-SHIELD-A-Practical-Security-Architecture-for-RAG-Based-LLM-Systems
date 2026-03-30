import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]  # service_role or a key that can read objects

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

bucket = "Company files"
filenames = [
    "EMAILS.csv",
    "Notice of Intention to Foreclose.docx",
    "OFFICE MANUAL, POLICIES & PROCEDURES.docx",
    "PERSONAL ASSISTANT AUTHORIZATION FORM.docx",
    "RENTER LEASE AGREEMENT.docx",
]

os.makedirs("downloads", exist_ok=True)

for name in filenames:
    path_in_bucket = name  # change if the file lives in a folder like "some/folder/name.docx"
    local_path = os.path.join("downloads", name)

    try:
        file_bytes = supabase.storage.from_(bucket).download(path_in_bucket)
        # file_bytes is raw bytes; write to disk
        with open(local_path, "wb") as f:
            f.write(file_bytes)
        print(f"Downloaded: {name} -> {local_path}")
    except Exception as e:
        print(f"Failed to download {name}: {e}")