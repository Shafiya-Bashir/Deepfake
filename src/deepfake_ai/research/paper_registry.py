"""Queryable registry of all 31 curated research papers."""

import json
from pathlib import Path
from typing import List, Dict, Any


class PaperRegistry:
    def __init__(self, metadata_path: str = 'papers/metadata/papers_metadata.json'):
        self.path = Path(metadata_path)
        self._papers = []
        if self.path.exists():
            with open(self.path, 'r') as f:
                self._papers = json.load(f).get('papers', [])

    def list_all(self) -> List[Dict[str, Any]]:
        return self._papers

    def filter_by_category(self, category: str) -> List[Dict[str, Any]]:
        return [p for p in self._papers if category.lower() in p.get('category', '').lower()]

    def search_by_keyword(self, query: str) -> List[Dict[str, Any]]:
        q = query.lower()
        return [p for p in self._papers if q in p.get('title', '').lower() or q in p.get('key_contribution', '').lower()]
