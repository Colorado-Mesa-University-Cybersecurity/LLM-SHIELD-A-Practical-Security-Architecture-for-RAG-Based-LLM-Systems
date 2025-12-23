#!/usr/bin/env python3
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
BUCKET = os.environ.get("SUPABASE_BUCKET", "Company files")
EMAIL = os.environ.get("SUPABASE_EMAIL", "company@elmivyproperties.com")
PASSWORD = os.environ.get("SUPABASE_PASSWORD", "123")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# sign in; AuthResponse has .user and .session in your client version
auth_response = supabase.auth.sign_in_with_password({"email": EMAIL, "password": PASSWORD})

# extract user id and access token (adjust if your client returns a different shape)
UID = auth_response.user.id
ACCESS_TOKEN = auth_response.session.access_token

# create a client that uses the user's access token for storage calls
user_client = create_client(SUPABASE_URL, ACCESS_TOKEN)

# list objects under the user's prefix using the SDK list method
# many supabase-py versions accept (prefix, options_dict)
list_opts = {
    "limit": 1000,
    "offset": 0,
    "sortBy": {"column": "name", "order": "desc"},
}
raw_list = supabase.storage.from_(BUCKET).list(UID, list_opts)

# The SDK can return a plain list, or (data, error) in some builds.
# Normalize to a list of entries.
if isinstance(raw_list, tuple) and len(raw_list) >= 1:
    entries = raw_list[0] or []
elif isinstance(raw_list, dict) and raw_list.get("data") is not None:
    entries = raw_list.get("data") or []
else:
    entries = raw_list or []

print("Storage list response (entries count):", len(entries))
for e in entries:
    # entry may be dict-like or object-like
    if isinstance(e, dict):
        name = e.get("name")
    else:
        name = getattr(e, "name", None)
    print("-", name)

# Create downloads directory
os.makedirs("downloads", exist_ok=True)

def make_object_path(uid: str, name: str) -> str:
    if not name:
        return None
    uid_clean = uid.rstrip("/")
    if name.startswith(uid_clean + "/") or name.startswith("/" + uid_clean + "/"):
        return name.lstrip("/")
    return f"{uid_clean}/{name.lstrip('/')}"

# Download each file
for entry in entries:
    if isinstance(entry, dict):
        name = entry.get("name")
    else:
        name = getattr(entry, "name", None)
    if not name:
        print("[skip] entry with no name:", entry)
        continue
    object_path = make_object_path(UID, name)
    local_path = os.path.join("downloads", os.path.basename(name))
    try:
        file_bytes = user_client.storage.from_(BUCKET).download(object_path)
        # supabase-py sometimes returns a dict on error, or bytes on success
        if isinstance(file_bytes, dict) and file_bytes.get("statusCode"):
            raise RuntimeError(f"storage error: {file_bytes}")
        with open(local_path, "wb") as f:
            f.write(file_bytes)
        print(f"[ok] Downloaded: {object_path} -> {local_path}")
    except Exception as e:
        print(f"[fail] {object_path}: {e}")

print("Done. Check ./downloads for downloaded files.")