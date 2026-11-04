"""Verify every file listed in SHA256SUMS.txt and reject missing/unlisted files."""

from hashlib import sha256
from pathlib import Path

from generate_checksums import package_files


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "SHA256SUMS.txt"

expected = {}
for line in MANIFEST.read_text(encoding="utf-8").splitlines():
    digest, relative_path = line.split("  ", 1)
    expected[relative_path] = digest

actual_paths = {
    path.relative_to(ROOT).as_posix()
    for path in package_files()
}
missing = sorted(actual_paths - expected.keys())
unexpected = sorted(expected.keys() - actual_paths)
changed = sorted(
    relative_path
    for relative_path, digest in expected.items()
    if relative_path in actual_paths
    and sha256((ROOT / relative_path).read_bytes()).hexdigest() != digest
)

if missing or unexpected or changed:
    raise SystemExit(
        f"Checksum validation failed. Missing entries={missing}; "
        f"unexpected entries={unexpected}; changed files={changed}"
    )

print(f"PASS: {len(expected)} package files match SHA256SUMS.txt")
