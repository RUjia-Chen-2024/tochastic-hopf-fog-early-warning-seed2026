"""Regenerate SHA256SUMS.txt for every package file except the manifest itself."""

from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "SHA256SUMS.txt"

entries = []
for path in sorted(p for p in ROOT.rglob("*") if p.is_file() and p != MANIFEST):
    digest = sha256(path.read_bytes()).hexdigest()
    entries.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")

MANIFEST.write_text("\n".join(entries) + "\n", encoding="utf-8")
print(f"Wrote {len(entries)} SHA256 entries to {MANIFEST.name}")
