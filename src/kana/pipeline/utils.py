import hashlib
import json
import os
from typing import Any, Dict, Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def file_info(path: str) -> Dict[str, Any]:
    try:
        stat = os.stat(path)
        return {"exists": True, "size": stat.st_size, "mtime": stat.st_mtime}
    except FileNotFoundError:
        return {"exists": False}


def hash_json(data: Any) -> str:
    s = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_json(path: str, data: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
