import hashlib

import pytest

from nids import data
from nids.config import RAW_COLUMNS


def test_load_split_adds_labels(frame):
    assert list(frame.columns) == RAW_COLUMNS + ["category", "is_attack"]
    assert (frame.loc[frame["label"] == "normal", "is_attack"] == 0).all()
    assert (frame.loc[frame["label"] == "neptune", "category"] == "dos").all()


def test_load_split_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="nids download"):
        data.load_split("train", tmp_path)


def test_sha256(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"intrusion")
    assert data.sha256(p) == hashlib.sha256(b"intrusion").hexdigest()


def _serve(tmp_path, monkeypatch, payload: bytes, correct_hash: bool):
    """Point the downloader at a local file:// mirror holding ``payload``."""
    src = tmp_path / "mirror"
    src.mkdir()
    (src / "only.txt").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest() if correct_hash else "0" * 64
    monkeypatch.setattr(data, "SPLITS", {"train": ("only.txt", digest)})
    return src.as_uri() + "/"


def test_download_verifies_and_skips_existing(tmp_path, monkeypatch):
    base = _serve(tmp_path, monkeypatch, b"a,b\n", correct_hash=True)
    dest = tmp_path / "dest"
    data.download(dest, base_url=base)
    assert (dest / "only.txt").read_bytes() == b"a,b\n"
    data.download(dest, base_url="file:///nonexistent/")  # verified copy: no fetch


def test_download_rejects_bad_checksum(tmp_path, monkeypatch):
    base = _serve(tmp_path, monkeypatch, b"tampered", correct_hash=False)
    dest = tmp_path / "dest"
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        data.download(dest, base_url=base)
    assert not any(dest.iterdir())
