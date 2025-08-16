import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def file_info(path: PathLike) -> Dict[str, Any]:
    p = Path(path)
    try:
        stat = p.stat()
        return {"exists": True, "size": stat.st_size, "mtime": stat.st_mtime}
    except FileNotFoundError:
        return {"exists": False}


def hash_json(data: Any) -> str:
    s = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_json(path: PathLike) -> Optional[Dict[str, Any]]:
    try:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_json(path: PathLike, data: Dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
