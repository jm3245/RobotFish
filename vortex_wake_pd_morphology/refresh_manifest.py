"""Refresh SHA-256 records for all curated bundle files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "source_manifest.json"


def main() -> None:
    previous = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    by_path = {entry["path"]: entry for entry in previous.get("files", [])}
    entries = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or ".git" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel == MANIFEST.name or rel.startswith("flow_040_navigation/smoke_") or rel.startswith("presentation/research-update/"):
            continue
        entry = dict(by_path.get(rel, {}))
        entry.update(path=rel, size_bytes=path.stat().st_size,
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        entry.setdefault("source", "Created in this curated bundle")
        if rel == "legacy_stack/01_Baseline_Controller/source_snapshot/fish_N4_entity.xml":
            entry["source"] = r"D:\Fish_Baseline_PD_and_SAC_20261028\01_Baseline_Controller\source_snapshot\fish_N4_entity.xml"
        entries.append(entry)
    MANIFEST.write_text(json.dumps({"name": ROOT.name, "file_count": len(entries),
                                    "files": entries}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"Recorded {len(entries)} files in {MANIFEST}")


if __name__ == "__main__":
    main()
