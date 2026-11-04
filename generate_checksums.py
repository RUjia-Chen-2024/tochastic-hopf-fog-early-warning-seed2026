"""Regenerate checksums for published files, excluding local Git and input files."""

from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "SHA256SUMS.txt"

IGNORED_DIRECTORIES = {
    ".git", "__pycache__", ".pytest_cache", ".venv", "venv",
    "raw_data", "source_data", "checkpoints",
}


def package_files():
    """Yield release files in both Git checkouts and downloaded source archives."""
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or path == MANIFEST:
            continue
        if any(part in IGNORED_DIRECTORIES for part in relative.parts):
            continue
        if path.name == ".env" or path.suffix in {".pyc", ".pyo", ".key", ".pem"}:
            continue
        if path.name.endswith("_force_paths.npy"):
            continue
        if relative.as_posix() == "paper/manuscript_seed2026.pdf":
            continue
        yield path


if __name__ == "__main__":
    entries = []
    for path in package_files():
        digest = sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    with MANIFEST.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(entries) + "\n")
    print(f"Wrote {len(entries)} SHA256 entries to {MANIFEST.name}")
