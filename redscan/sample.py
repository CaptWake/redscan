import hashlib
from pathlib import Path

import magic

from redscan.ember import AnalysisError, FileType, detect_file_type


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_metadata(path: Path, file_type: FileType | None = None) -> dict[str, object]:
    resolved_path = path.expanduser().resolve()
    try:
        stat = resolved_path.stat()
        mime_type = magic.from_file(str(resolved_path), mime=True)
        sha256 = sha256_file(resolved_path)
    except Exception as exc:
        raise AnalysisError(f"Could not inspect {resolved_path}: {exc}") from exc

    return {
        "path": str(resolved_path),
        "name": resolved_path.name,
        "size": stat.st_size,
        "sha256": sha256,
        "mime_type": mime_type,
        "file_type": (file_type or detect_file_type(resolved_path)).name.lower(),
    }
