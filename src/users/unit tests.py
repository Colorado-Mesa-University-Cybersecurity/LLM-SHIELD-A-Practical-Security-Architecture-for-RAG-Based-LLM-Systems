import os
import tempfile
import shutil
from pathlib import Path
import io
import pytest
from rag import (
    mask_digits_keep_last,
    redact_text,
    sanitize_metadata_key,
    load_csv,
    SupabaseSession,
)


def test_mask_digits_keep_last_basic():
    assert mask_digits_keep_last("4111-1111-1111-1111", keep=4).endswith("1111")
    assert mask_digits_keep_last("1234", keep=4) == "****"
    assert mask_digits_keep_last("", keep=4) == ""
    assert mask_digits_keep_last("no digits here", keep=4) == "no digits here"


def test_redact_text_masks_cc_ssn_and_email():
    text = "Charge card 4111111111111111 and SSN 123-45-6789. Contact alice@example.com."
    redacted = redact_text(text)
    # Credit card sequence should be masked leaving last 4 digits
    assert "1111" in redacted
    assert "4111111111111111" not in redacted
    # SSN masked
    assert "***-**-****" in redacted
    # Email local part partially masked
    assert "a****@example.com" in redacted or "a***@example.com" in redacted


def test_sanitize_metadata_key_replaces_bad_chars_and_truncates():
    key = "some bad/key:with*chars and spaces"
    sk = sanitize_metadata_key(key, max_len=20)
    assert " " not in sk
    assert "/" not in sk
    assert ":" not in sk
    assert len(sk) <= 20


def make_temp_csv(tmpdir, rows):
    path = Path(tmpdir) / "test.csv"
    header = "col1,col2\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for r in rows:
            f.write(f"{r[0]},{r[1]}\n")
    return path


def test_load_csv_single_row_docs(tmp_path):
    rows = [("a1", "b1"), ("a2", "b2"), ("a3", "b3")]
    path = make_temp_csv(tmp_path, rows)
    docs = load_csv(path, rows_per_doc=1)
    assert len(docs) == 3
    # Each doc should contain both columns
    assert "col1: a1" in docs[0].page_content
    assert docs[0].metadata["row_index"] == 0


def test_load_csv_multi_row_docs(tmp_path):
    rows = [("r1c1", "r1c2"), ("r2c1", "r2c2"), ("r3c1", "r3c2"), ("r4c1", "r4c2")]
    path = make_temp_csv(tmp_path, rows)
    docs = load_csv(path, rows_per_doc=2)
    # With 4 rows and rows_per_doc=2 we expect 2 docs
    assert len(docs) == 2
    # The metadata for group should contain 'row_start' and 'row_end'
    assert "row_start" in docs[0].metadata
    assert docs[0].metadata["row_end"] == 1


# --- SupabaseSession SDK-shape normalization tests (mocked client) ---


class FakeBucket:
    def __init__(self, list_return, download_map):
        self._list_return = list_return
        self._download_map = download_map

    def list(self, prefix, opts):
        return self._list_return

    def download(self, object_path):
        # Simulate success returning raw bytes or an error dict
        if object_path in self._download_map:
            return self._download_map[object_path]
        return {"statusCode": 404, "message": "not found"}


class FakeStorage:
    def __init__(self, list_return, download_map):
        self._list_return = list_return
        self._download_map = download_map

    def from_(self, bucket):
        return FakeBucket(self._list_return, self._download_map)


class FakeClient:
    def __init__(self, list_return, download_map):
        self.storage = FakeStorage(list_return, download_map)


@pytest.mark.parametrize(
    "raw_list,expected_names",
    [
        # tuple-style response: (data, error)
        (([{"name": "a.txt"}, {"name": "b.txt"}], None), ["a.txt", "b.txt"]),
        # dict with data key
        ({"data": [{"name": "x.pdf"}, {"name": "y.pdf"}]}, ["x.pdf", "y.pdf"]),
        # plain list
        ([{"name": "one.csv"}, {"name": "two.csv"}], ["one.csv", "two.csv"]),
    ],
)
def test_list_objects_handles_various_sdk_shapes(tmp_path, raw_list, expected_names):
    sb = SupabaseSession.from_config({"SUPABASE_URL": "https://example", "SUPABASE_KEY": "key"})
    # Mark as authenticated for list_objects path
    sb.user_id = "uid123"
    sb.access_token = "fake-token"  # ensure is_authenticated() returns True
    # Prepare a fake client that returns the given raw_list
    sb.user_client = FakeClient(list_return=raw_list, download_map={})
    names = sb.list_objects("mybucket", prefix="")  # should normalize shapes
    assert set(names) == set(expected_names)


def test_download_object_and_download_objects_to_dir(tmp_path):
    sb = SupabaseSession.from_config({"SUPABASE_URL": "https://example", "SUPABASE_KEY": "key"})
    sb.user_id = "uid123"
    sb.access_token = "fake-token"  # ensure authenticated
    # Simulate one file path and its bytes
    obj_path = "uid123/notes.txt"
    content_bytes = b"hello world"
    sb.user_client = FakeClient(list_return=[{"name": "notes.txt"}], download_map={obj_path: content_bytes})

    # Single-download
    got = sb.download_object("mybucket", obj_path)
    assert isinstance(got, (bytes, bytearray))
    assert got == content_bytes

    # Bulk download into a directory
    dest = tmp_path / "out"
    dest.mkdir()
    results = sb.download_objects_to_dir("mybucket", [obj_path], str(dest))
    assert len(results) == 1
    path, ok, err = results[0]
    assert ok is True
    assert err is None
    # Check file written
    out_file = dest / "notes.txt"
    assert out_file.exists()
    with open(out_file, "rb") as f:
        assert f.read() == content_bytes