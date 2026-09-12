"""Dictionary service — in-memory queries over dictionary.json (no database)."""
import json
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from utils.logging_config import get_logger

logger = get_logger(__name__)

# Module-level cache — loaded once at first use
_entries: List[Dict[str, Any]] = []
_loaded = False


def _normalize(text: str) -> str:
    """Accent-strip + lowercase for matching."""
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn').lower().strip()


def _normalize_category_key(category: str) -> str:
    match = re.search(r'\(([^)]+)\)', category)
    english_name = match.group(1) if match else category
    key = english_name.lower()
    key = re.sub(r'[^a-z0-9\s-]', '', key)
    key = re.sub(r'\s+', '-', key)
    key = re.sub(r'-+', '-', key)
    return key.strip('-')


def _load_entries() -> List[Dict[str, Any]]:
    global _entries, _loaded
    if _loaded:
        return _entries

    dictionary_path = Path(__file__).parent.parent / "dictionary.json"
    if not dictionary_path.exists():
        logger.error(f"dictionary.json not found at {dictionary_path}")
        _loaded = True
        return _entries

    with open(dictionary_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entries = []
    for cat in data.get("categories", []):
        category = cat.get("category", "Unknown")
        category_key = _normalize_category_key(category)
        for e in cat.get("entries", []):
            bulos = str(e.get("BULOS", "")).strip()
            filipino = str(e.get("FILIPINO", "")).strip()
            english = str(e.get("ENGLISH", "")).strip()
            if bulos and filipino and english and bulos != "—" and filipino != "—":
                entries.append({
                    "bulos": bulos,
                    "filipino": filipino,
                    "english": english,
                    "category": category,
                    "category_key": category_key,
                    "metadata": {"source": "dictionary.json", "import_version": "1.0"},
                })

    _entries = entries
    _loaded = True
    logger.info(f"DictionaryService: loaded {len(_entries)} entries from dictionary.json")
    return _entries


class DictionaryService:
    """Queries dictionary.json in memory — no database required."""

    def __init__(self):
        _load_entries()
        logger.info("DictionaryService initialized")

    async def lookup_word(self, word: str, source_language: str) -> List[Dict[str, Any]]:
        entries = _load_entries()
        field_map = {"bul": "bulos", "tl": "filipino", "en": "english"}
        field = field_map.get(source_language, "bulos")
        norm_word = _normalize(word)

        results = [
            e for e in entries
            if _normalize(e.get(field, "")).startswith(norm_word)
        ]
        logger.info(f"lookup_word '{word}' ({source_language}): {len(results)} results")
        return results[:50]

    async def search_words(self, query: str, language: str = "all", limit: int = 50) -> List[Dict[str, Any]]:
        entries = _load_entries()
        norm_query = _normalize(query)
        field_map = {"bul": "bulos", "tl": "filipino", "en": "english"}

        if language == "all":
            results = [
                e for e in entries
                if any(norm_query in _normalize(e.get(f, "")) for f in ["bulos", "filipino", "english"])
            ]
        else:
            field = field_map.get(language, "bulos")
            results = [e for e in entries if norm_query in _normalize(e.get(field, ""))]

        logger.info(f"search_words '{query}' ({language}): {len(results)} results")
        return results[:limit]

    async def get_by_category(self, category_key: str, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        entries = _load_entries()
        results = [e for e in entries if e.get("category_key") == category_key]
        return results[skip: skip + limit]

    async def get_all_categories(self) -> List[Dict[str, Any]]:
        entries = _load_entries()
        counts: Dict[str, Dict[str, Any]] = {}
        for e in entries:
            key = e["category_key"]
            if key not in counts:
                counts[key] = {"category": e["category"], "category_key": key, "entry_count": 0}
            counts[key]["entry_count"] += 1
        return sorted(counts.values(), key=lambda c: c["entry_count"], reverse=True)

    async def count_entries(self) -> int:
        return len(_load_entries())

    async def get_version_metadata(self) -> Dict[str, Any]:
        return {
            "version": "1.0",
            "last_updated": datetime.utcnow(),
            "total_entries": await self.count_entries(),
            "changelog": "Loaded from dictionary.json"
        }

    async def update_version_metadata(self, changelog: str = None) -> None:
        # No-op — metadata is derived from the JSON file at runtime
        logger.info(f"update_version_metadata called: {changelog}")

    async def import_from_json(self, json_path: str, force_reimport: bool = False, dry_run: bool = False) -> Dict[str, Any]:
        """Re-parse dictionary.json and reload the in-memory cache."""
        import time
        global _entries, _loaded

        start = time.time()
        _loaded = False  # Force reload
        entries = _load_entries()

        duration = time.time() - start
        logger.info(f"Dictionary reloaded: {len(entries)} entries in {duration:.2f}s")
        return {
            "total_entries": len(entries),
            "inserted": len(entries),
            "skipped": 0,
            "errors": 0,
            "duration": duration,
            "error_details": []
        }
