"""Fetch pinned upstream prompt bytes, verify Git blob IDs, record SHA-256 + diff.

No LLM/API credentials are loaded. Existing different files are never overwritten.
Run from the project root: python scripts/snapshot_autoontik.py
"""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

COMMIT = "9368d6149050f3085a9450e03e4d4c1052722525"
FILES = {
    "prompt_1_with_types_and_qualifiers.txt": "9a415457ff1c16aa1a45d3449adefe6876f0ed71",
    "triple_match.txt": "9fd7230a789940451e1f69323ba75250379b5d3c",
    "gold_triple_groundedness.txt": "5907b6ad70514d816da55af5ca33441b5c3c028c",
    "cluster_entity_names.txt": "7fe2f358b72a92d295f3e19cb9e4a0895415240f",
    "cluster_relations.txt": "fce77000150ba49c8c9ddcc273043066676cc236",
}


def save_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite different file: {path}")
        return
    with path.open("xb") as f:
        f.write(content)


def main() -> None:
    manifest = {"repository": "https://github.com/screemix/AutoOntik", "commit": COMMIT,
                "source_branch": "dev", "files": []}
    for name, expected in FILES.items():
        url = f"https://raw.githubusercontent.com/screemix/AutoOntik/{COMMIT}/src/ontodisco/utils/prompts/{name}"
        with urlopen(url, timeout=60) as response:
            content = response.read()
        blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if blob != expected:
            raise ValueError(f"Upstream blob mismatch: {name}")
        path = Path("prompts/autoontik") / name
        save_once(path, content)
        manifest["files"].append({"path": str(path), "url": url, "git_blob_sha1": blob,
                                  "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)})
    save_once(Path("external/autoontik_source_manifest.json"),
              (json.dumps(manifest, indent=2) + "\n").encode())
    baseline = Path("prompts/baseline/propmt_1_types_qualifiers.txt")
    current = Path("prompts/autoontik/prompt_1_with_types_and_qualifiers.txt")
    diff = "".join(difflib.unified_diff(baseline.read_text().splitlines(True),
                                       current.read_text().splitlines(True),
                                       fromfile=str(baseline), tofile=str(current)))
    save_once(Path("reports/autoontik_vs_baseline.diff"), diff.encode())
    print("Verified 5 pinned AutoOntik prompts; manifest and baseline diff saved.")


if __name__ == "__main__":
    main()
