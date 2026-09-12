"""
Dictionary Reload Script
------------------------
Reloads dictionary.json into the in-memory DictionaryService cache by calling
the admin API endpoint. The server must be running.

Usage:
    python scripts/import_dictionary.py [--url http://localhost:8000] [--admin-key YOUR_KEY]
"""
import argparse
import asyncio
import httpx
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def reload_dictionary(base_url: str, admin_key: str, force: bool = False) -> None:
    url = f"{base_url}/api/v1/dictionary/import?force_reimport={str(force).lower()}"

    async with httpx.AsyncClient(timeout=30) as client:
        logger.info(f"Calling {url}")
        response = await client.post(url, headers={"X-Admin-Key": admin_key})

    if response.status_code == 200:
        data = response.json()
        print("\n✓ Dictionary reloaded successfully")
        print(f"  Total entries : {data.get('total_entries', 0)}")
        print(f"  Inserted      : {data.get('inserted', 0)}")
        print(f"  Duration      : {data.get('duration', 0):.2f}s")
    else:
        print(f"\n✗ Failed: HTTP {response.status_code}")
        print(f"  {response.text}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Reload dictionary.json into the API cache")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--admin-key",
                        default=os.environ.get("ADMIN_API_KEY", "dev-admin-key-change-in-production"))
    parser.add_argument("--force", action="store_true",
                        help="Force full reload (clears existing cache)")
    args = parser.parse_args()

    asyncio.run(reload_dictionary(args.url, args.admin_key, args.force))


if __name__ == "__main__":
    main()
