"""Repair only CRLF checkout conversion of frozen artifacts; never change hashes.

Run from any directory with: python -X utf8 scripts/normalize_frozen_text.py
Unknown content changes fail before any file is written.
"""

import ast
import hashlib
import json
from pathlib import Path


class FrozenContentError(ValueError):
    """Frozen content differs by more than Git newline conversion."""


def frozen_hashes(root: Path) -> dict[str, str]:
    manifest = root / "qualification/standards/g1_integration_manifest.json"
    expected: dict[str, str] = json.loads(manifest.read_text(encoding="utf-8"))["frozen_artifacts"]
    for relative, constant in (
        ("tests/test_cdpm2_authority_contract.py", "FROZEN_G1"),
        ("tests/test_cdpm2_schema.py", "FROZEN_G2A"),
    ):
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == constant for target in node.targets
            ):
                for name, digest in ast.literal_eval(node.value).items():
                    if name in expected and expected[name] != digest:
                        raise FrozenContentError(f"Conflicting expected hashes for {name}")
                    expected[name] = digest
                break
        else:
            raise FrozenContentError(f"Missing {constant} in {relative}")
    return expected


def normalize_frozen_text(root: Path, expected: dict[str, str]) -> list[str]:
    repairs: list[tuple[Path, bytes, str]] = []
    # Validate the entire set before making any changes.
    for relative, digest in expected.items():
        path = root / relative
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() == digest:
            continue
        canonical = data.replace(b"\r\n", b"\n")
        if hashlib.sha256(canonical).hexdigest() != digest:
            raise FrozenContentError(
                f"{relative}: content differs from its frozen hash; no files were changed."
            )
        repairs.append((path, canonical, relative))
    for path, canonical, _ in repairs:
        path.write_bytes(canonical)
    return [relative for _, _, relative in repairs]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    repaired = normalize_frozen_text(root, frozen_hashes(root))
    for relative in repaired:
        print(f"Restored frozen LF bytes: {relative}")
    print(f"Frozen artifacts verified; {len(repaired)} CRLF checkout conversions repaired.")


if __name__ == "__main__":
    main()
