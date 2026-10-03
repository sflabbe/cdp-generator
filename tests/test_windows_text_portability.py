"""Frozen byte checks remain strict while CRLF checkout damage is repairable."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "frozen_normalizer", ROOT / "scripts/normalize_frozen_text.py"
)
assert SPEC and SPEC.loader
NORMALIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NORMALIZER)


def test_all_original_frozen_hashes_still_hold():
    expected = NORMALIZER.frozen_hashes(ROOT)
    assert NORMALIZER.normalize_frozen_text(ROOT, expected) == []


def test_crlf_checkout_is_repaired_to_original_bytes(tmp_path):
    data = "Qualification — exact UTF-8 bytes\nSecond line\n".encode()
    target = tmp_path / "frozen.md"
    target.write_bytes(data.replace(b"\n", b"\r\n"))
    expected = {"frozen.md": hashlib.sha256(data).hexdigest()}
    assert NORMALIZER.normalize_frozen_text(tmp_path, expected) == ["frozen.md"]
    assert target.read_bytes() == data
    assert NORMALIZER.normalize_frozen_text(tmp_path, expected) == []


def test_content_change_fails_before_any_repairs(tmp_path):
    data = b"Original\n"
    good = tmp_path / "good.txt"
    bad = tmp_path / "bad.txt"
    good.write_bytes(data.replace(b"\n", b"\r\n"))
    bad.write_bytes(b"Edited\r\n")
    expected = dict.fromkeys(("good.txt", "bad.txt"), hashlib.sha256(data).hexdigest())
    with pytest.raises(NORMALIZER.FrozenContentError, match="no files were changed"):
        NORMALIZER.normalize_frozen_text(tmp_path, expected)
    assert good.read_bytes() == b"Original\r\n"
    assert bad.read_bytes() == b"Edited\r\n"
