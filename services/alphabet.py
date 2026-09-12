"""Alphabet service — in-memory queries over alphabet.json (no database)."""
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from utils.logging_config import get_logger

logger = get_logger(__name__)

# Module-level cache
_entries: List[Dict[str, Any]] = []
_loaded = False


def _load_entries() -> List[Dict[str, Any]]:
    global _entries, _loaded
    if _loaded:
        return _entries

    alphabet_path = Path(__file__).parent.parent / "alphabet.json"
    if not alphabet_path.exists():
        logger.error(f"alphabet.json not found at {alphabet_path}")
        _loaded = True
        return _entries

    with open(alphabet_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entries = []
    for i, letter_data in enumerate(data.get("letters", []), start=1):
        letter = letter_data.get("letter", "").strip()
        if not letter:
            continue

        examples_raw = letter_data.get("examples", {})
        parsed_examples = {}
        for pos in ["initial", "middle", "final"]:
            parsed_examples[pos] = []
            for ex in examples_raw.get(pos, []):
                bulos = str(ex.get("BULOS", "")).strip()
                filipino = str(ex.get("FILIPINO", "")).strip()
                english = str(ex.get("ENGLISH", "")).strip() if ex.get("ENGLISH") else None
                if bulos and filipino:
                    parsed_examples[pos].append({
                        "bulos": bulos,
                        "filipino": filipino,
                        "english": english or "",
                    })

        entries.append({
            "id": str(i),
            "letter": letter,
            "position": i,
            "initial_examples": parsed_examples["initial"],
            "middle_examples": parsed_examples["middle"],
            "final_examples": parsed_examples["final"],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        })

    _entries = entries
    _loaded = True
    logger.info(f"AlphabetService: loaded {len(_entries)} letters from alphabet.json")
    return _entries


class AlphabetService:
    """Queries alphabet.json in memory — no database required."""

    def __init__(self):
        _load_entries()
        logger.info("AlphabetService initialized")

    async def get_by_letter(self, letter: str) -> Dict[str, Any]:
        entries = _load_entries()
        for e in entries:
            if e["letter"].lower() == letter.lower():
                return e
        return None

    async def get_all_letters(self, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        entries = _load_entries()
        return entries[skip: skip + limit]

    async def count_letters(self) -> int:
        return len(_load_entries())

    async def import_from_json(self, json_path: str, force_reimport: bool = False, dry_run: bool = False) -> Dict[str, Any]:
        """Re-parse alphabet.json and reload the in-memory cache."""
        global _entries, _loaded
        start = time.time()
        _loaded = False
        entries = _load_entries()
        duration = time.time() - start
        logger.info(f"Alphabet reloaded: {len(entries)} letters in {duration:.2f}s")
        return {
            "total_entries": len(entries),
            "inserted": len(entries),
            "skipped": 0,
            "errors": 0,
            "duration": duration,
            "error_details": []
        }
