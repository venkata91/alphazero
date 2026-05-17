#!/usr/bin/env python3
"""Build a client-side search index for the GitHub Pages site.

Crawls all HTML pages in the repo, extracts title + section headings +
section body text, and writes search-index.json that search.js consumes.

Run from repo root:
    python3 tools/build_search_index.py

Output: search-index.json (~50KB, gzipped to ~15KB by GH Pages).
"""
from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path


SKIP_TAGS = {"style", "script", "svg", "noscript"}


class PageIndexer(HTMLParser):
    """Extract {title, sections: [{id, heading, level, text}]} from one HTML page.

    Headings (h1-h4) define section boundaries. Text between two headings is
    accumulated into the earlier heading's section body.
    """

    def __init__(self):
        super().__init__()
        self.title: str = ""
        self.sections: list[dict] = []
        self._in_title = False
        self._in_heading = False
        self._heading_text_buf: list[str] = []
        self._heading_id = ""
        self._heading_level = 0
        self._body_text_buf: list[str] = []
        self._current_section: dict | None = None
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        attrs_dict = dict(attrs)
        if tag == "title":
            self._in_title = True
        elif tag in {"h1", "h2", "h3", "h4"}:
            self._flush_current_section()
            self._in_heading = True
            self._heading_level = int(tag[1])
            self._heading_id = attrs_dict.get("id", "")
            self._heading_text_buf = []

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return
        if tag == "title":
            self._in_title = False
        elif tag in {"h1", "h2", "h3", "h4"}:
            self._in_heading = False
            heading_text = " ".join("".join(self._heading_text_buf).split()).strip()
            self._current_section = {
                "id": self._heading_id,
                "heading": heading_text,
                "level": self._heading_level,
                "_text": [],
            }

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        if self._in_title:
            self.title += data
        elif self._in_heading:
            self._heading_text_buf.append(data)
        elif self._current_section is not None:
            self._current_section["_text"].append(data)

    def _flush_current_section(self):
        if self._current_section is None:
            return
        text = " ".join("".join(self._current_section.pop("_text")).split())
        if self._current_section["heading"] or text:
            self._current_section["text"] = text
            self.sections.append(self._current_section)
        self._current_section = None

    def finalize(self):
        self._flush_current_section()


def crawl(repo_root: Path) -> list[dict]:
    """Walk the site's HTML files, build the index."""
    candidates = sorted(
        {*repo_root.glob("*.html"), *repo_root.glob("docs/**/*.html")}
    )
    pages = []
    for path in candidates:
        rel = str(path.relative_to(repo_root))
        if any(part.startswith(".") for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        indexer = PageIndexer()
        try:
            indexer.feed(text)
        except Exception as e:
            print(f"  skipped {rel}: {type(e).__name__}: {e}")
            continue
        indexer.finalize()
        pages.append(
            {
                "url": rel,
                "title": indexer.title.strip() or rel,
                "sections": indexer.sections,
            }
        )
    return pages


def trim_for_size(pages: list[dict], max_body_chars: int = 1200) -> None:
    """Cap each section's body text length so the JSON stays small."""
    for page in pages:
        for section in page["sections"]:
            text = section.get("text", "")
            if len(text) > max_body_chars:
                section["text"] = text[:max_body_chars]


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    pages = crawl(repo_root)
    trim_for_size(pages)
    out = repo_root / "search-index.json"
    out.write_text(json.dumps(pages, separators=(",", ":"), ensure_ascii=False))
    n_pages = len(pages)
    n_sections = sum(len(p["sections"]) for p in pages)
    size_kb = out.stat().st_size / 1024
    print(
        f"Indexed {n_pages} pages, {n_sections} sections "
        f"→ {out.name} ({size_kb:.1f} KB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
