#!/usr/bin/env python3
"""Render a markdown spec document as a styled standalone HTML page.

Wraps the rendered markdown in the comic-notebook template that matches
concepts.html. Inlines all CSS so the output is fully self-contained
(only Google Fonts is fetched from a CDN).

Usage:
    python tools/render_spec.py path/to/spec.md
    python tools/render_spec.py path/to/spec.md --out path/to/spec.html

Markdown features supported (via Python-Markdown):
    - fenced code blocks (```python ... ```)
    - tables
    - automatic table of contents
    - headers, lists, blockquotes, inline code, links
    - YAML-like frontmatter (--- ... --- at top), used for title + metadata
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import markdown
except ImportError:
    sys.stderr.write(
        "Missing dependency 'markdown'. Install with:\n"
        "    pip install --user markdown\n"
    )
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Frontmatter parsing                                                          #
# --------------------------------------------------------------------------- #

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Extract a simple YAML-ish frontmatter block.

    Supports only top-level `key: value` lines; no nested structures.
    Returns (metadata, body_without_frontmatter)."""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    fm_block, body = match.group(1), match.group(2)
    meta: dict[str, str] = {}
    for line in fm_block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta, body


# --------------------------------------------------------------------------- #
# CSS (matches concepts.html palette and feel)                                 #
# --------------------------------------------------------------------------- #

STYLES = """
:root {
  color-scheme: light;
  --bg: #fbf3e2;
  --bg-soft: #fff8ec;
  --paper: #ffffff;
  --paper-tint: #fffaf0;
  --ink: #1f2547;
  --ink-soft: #4a5680;
  --ink-faint: #6b7290;
  --blue: #3a55c4;
  --blue-deep: #2a3f9a;
  --blue-soft: rgba(58, 85, 196, 0.18);
  --blue-bg: rgba(58, 85, 196, 0.08);
  --orange: #ee8a30;
  --orange-deep: #d6741e;
  --orange-soft: rgba(238, 138, 48, 0.22);
  --orange-bg: rgba(238, 138, 48, 0.08);
  --green: #3aa860;
  --green-bg: rgba(58, 168, 96, 0.10);
  --line: #2a3a8a;
  --line-faint: rgba(42, 58, 138, 0.20);
  --shadow: 4px 4px 0 rgba(31, 37, 71, 0.07);
  --hand: 'Patrick Hand', 'Kalam', 'Comic Neue', system-ui, sans-serif;
  --cursive: 'Caveat', 'Patrick Hand', cursive;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  font-family: var(--hand);
  font-size: 18px;
  line-height: 1.6;
  color: var(--ink);
  background:
    radial-gradient(circle at 8% 8%, rgba(238, 138, 48, 0.10), transparent 30%),
    radial-gradient(circle at 92% 6%, rgba(58, 85, 196, 0.10), transparent 30%),
    radial-gradient(circle at 50% 100%, rgba(238, 138, 48, 0.05), transparent 40%),
    var(--bg);
  background-attachment: fixed;
}
body::before {
  content: "";
  position: fixed;
  inset: 0;
  background-image: radial-gradient(rgba(31, 37, 71, 0.05) 1px, transparent 1px);
  background-size: 22px 22px;
  pointer-events: none;
  z-index: -1;
  opacity: 0.65;
}

a {
  color: var(--blue);
  text-decoration: none;
  border-bottom: 2px dotted var(--line-faint);
  padding-bottom: 1px;
}
a:hover { color: var(--orange-deep); border-bottom-color: var(--orange); }

.shell {
  width: min(960px, calc(100% - 32px));
  margin: 0 auto;
  padding: 40px 0 80px;
}

.hero, .spec {
  background: var(--paper);
  border: 2.5px solid var(--line);
  border-radius: 26px;
  box-shadow: var(--shadow);
  padding: 30px 38px;
}
.hero { margin-bottom: 20px; position: relative; }
.hero::before, .hero::after {
  content: "";
  position: absolute;
  width: 26px; height: 26px;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23ee8a30' stroke-width='2' stroke-linecap='round'><path d='M12 2 L12 8 M12 16 L12 22 M2 12 L8 12 M16 12 L22 12 M5 5 L9 9 M15 15 L19 19 M5 19 L9 15 M15 9 L19 5'/></svg>");
  background-repeat: no-repeat;
  background-size: contain;
  opacity: 0.8;
}
.hero::before { top: 18px; right: 28px; transform: rotate(12deg); }
.hero::after  { bottom: 18px; left: 28px; width: 20px; height: 20px; transform: rotate(-18deg); opacity: 0.55; }

.kicker {
  margin: 0 0 6px;
  font-family: var(--cursive);
  font-weight: 700;
  color: var(--orange);
  font-size: 1.3rem;
}
.hero h1 {
  margin: 0;
  color: var(--blue);
  font-family: var(--cursive);
  font-weight: 700;
  font-size: clamp(2.2rem, 6vw, 3.6rem);
  line-height: 1.1;
  letter-spacing: 1px;
}
.hero-meta {
  margin-top: 14px;
  padding-top: 12px;
  border-top: 2px dashed var(--line-faint);
  display: flex;
  flex-wrap: wrap;
  gap: 18px;
  color: var(--ink-soft);
  font-size: 0.96rem;
}
.hero-meta span strong {
  color: var(--blue-deep);
  font-family: var(--cursive);
  font-weight: 700;
  margin-right: 6px;
}

/* Rendered markdown body */
.spec h1, .spec h2, .spec h3, .spec h4 {
  font-family: var(--cursive);
  letter-spacing: 0.5px;
  font-weight: 700;
  line-height: 1.15;
  margin: 28px 0 10px;
}
.spec h1 { font-size: 2.4rem; color: var(--blue);      border-bottom: 2.5px solid var(--blue-soft); padding-bottom: 6px; margin-top: 8px; }
.spec h2 { font-size: 1.85rem; color: var(--blue);     margin-top: 32px; }
.spec h3 { font-size: 1.45rem; color: var(--blue-deep); }
.spec h4 { font-size: 1.2rem;  color: var(--orange-deep); }

.spec p { margin: 10px 0; color: var(--ink-soft); }
.spec strong { color: var(--blue-deep); }
.spec em { color: var(--ink); }

.spec ul, .spec ol {
  padding-left: 26px;
  color: var(--ink-soft);
}
.spec li { margin: 4px 0; }
.spec li > p { margin: 4px 0; }

.spec blockquote {
  margin: 14px 0;
  padding: 12px 18px;
  border-left: 4px solid var(--orange);
  background: var(--orange-bg);
  border-radius: 10px;
  color: var(--ink);
}
.spec blockquote p { margin: 4px 0; color: var(--ink); }

.spec code {
  font-family: var(--mono);
  font-size: 0.92em;
  padding: 1.5px 6px;
  border-radius: 6px;
  background: var(--blue-bg);
  border: 1.5px dashed var(--blue-soft);
  color: var(--blue-deep);
}
.spec pre {
  margin: 14px 0;
  padding: 14px 18px;
  border-radius: 14px;
  background: var(--paper-tint);
  border: 2px solid var(--line-faint);
  overflow-x: auto;
  line-height: 1.45;
}
.spec pre code {
  display: block;
  padding: 0;
  background: transparent;
  border: none;
  color: var(--ink);
  font-size: 0.92rem;
  white-space: pre;
}

.spec table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  margin: 16px 0;
  background: var(--paper-tint);
  border: 2px solid var(--line-faint);
  border-radius: 14px;
  overflow: hidden;
}
.spec table th, .spec table td {
  padding: 9px 14px;
  text-align: left;
  border-bottom: 1.5px solid var(--line-faint);
  vertical-align: top;
  font-size: 0.96rem;
}
.spec table th {
  background: var(--blue-bg);
  color: var(--blue-deep);
  font-family: var(--cursive);
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: 0.3px;
}
.spec table tr:last-child td { border-bottom: none; }
.spec table tr:hover td { background: var(--orange-bg); }

.spec hr {
  border: none;
  border-top: 2.5px dashed var(--line-faint);
  margin: 28px 0;
}

.toc {
  margin: 18px 0 6px;
  padding: 14px 18px;
  background: var(--paper-tint);
  border: 2px dashed var(--line-faint);
  border-radius: 14px;
  font-size: 0.96rem;
}
.toc > ul { padding-left: 22px; margin: 4px 0; }
.toc a { border-bottom: none; }
.toc a:hover { text-decoration: underline; }

.page-nav {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 10px 20px;
  margin-bottom: 18px;
  background: var(--paper);
  border: 2px solid var(--line-faint);
  border-radius: 14px;
  font-size: 0.94rem;
  font-family: var(--hand);
}
.page-nav a {
  color: var(--blue);
  border-bottom: none;
  font-family: var(--cursive);
  font-weight: 700;
  font-size: 1rem;
}
.page-nav a:hover { color: var(--orange-deep); }
.page-nav .crumb { color: var(--ink-faint); }

@media (max-width: 720px) {
  .shell { padding: 24px 0 64px; }
  .hero, .spec { padding: 22px 22px; }
  body { font-size: 17px; }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
}
"""

# --------------------------------------------------------------------------- #
# HTML template                                                                #
# --------------------------------------------------------------------------- #

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Caveat:wght@600;700&amp;family=Patrick+Hand&amp;display=swap" />
  <style>{styles}</style>
</head>
<body>
  <main class="shell">
    <nav class="page-nav">
      <a href="{index_path}">&#8592; All docs</a>
      <span class="crumb">AlphaZero &middot; {crumb}</span>
    </nav>
    <header class="hero">
      <p class="kicker">{kicker}</p>
      <h1>{heading}</h1>
{meta_block}
    </header>
    <article class="spec">{body}</article>
  </main>
</body>
</html>
"""


def build_meta_block(meta: dict[str, str]) -> str:
    """Render frontmatter as a meta strip inside the hero."""
    if not meta:
        return ""
    items = []
    # Render a fixed set of common fields first, then any extras.
    preferred_order = ["date", "status", "sub_project", "target"]
    seen = set()
    for key in preferred_order:
        if key in meta:
            items.append(
                f"      <span><strong>{key.replace('_', ' ')}:</strong>{meta[key]}</span>"
            )
            seen.add(key)
    for key, value in meta.items():
        if key in seen or key in ("title",):
            continue
        items.append(
            f"      <span><strong>{key.replace('_', ' ')}:</strong>{value}</span>"
        )
    if not items:
        return ""
    return '      <div class="hero-meta">\n' + "\n".join(items) + "\n      </div>"


def extract_kicker_and_heading(meta: dict[str, str], body: str) -> tuple[str, str]:
    """Derive page kicker + h1 heading.

    Title precedence:
        1. `title:` in frontmatter
        2. First `# Heading` line in the body (stripped from body downstream)
        3. Falls back to "Design"
    Kicker precedence:
        1. `kicker:` in frontmatter
        2. "Spec" if a date is in frontmatter
        3. "Notes"
    """
    title = meta.get("title")
    if not title:
        m = re.search(r"^#\s+(.+?)\s*$", body, flags=re.MULTILINE)
        title = m.group(1) if m else "Design"
    kicker = meta.get("kicker") or ("Spec" if "date" in meta else "Notes")
    return kicker, title


def strip_leading_h1(body: str, heading: str) -> str:
    """Remove the first `# Heading` from body if it duplicates the page title."""
    return re.sub(
        rf"^#\s+{re.escape(heading)}\s*\n+",
        "",
        body,
        count=1,
        flags=re.MULTILINE,
    )


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def compute_index_path(out_path: Path) -> tuple[str, str]:
    """Return (relative_path_to_index, crumb_label) for the nav bar.

    Counts how many directory levels deep out_path is relative to the repo
    root (assumed to be where index.html lives) and builds the appropriate
    relative path back up.  The crumb is derived from the parent directories
    and the file stem.
    """
    parts = out_path.parts
    # Number of parent directories above the file (not counting the file itself)
    depth = len(parts) - 1  # will be resolved relative to cwd at render time
    # Build relative path: depth levels of "../" then "index.html"
    index_path = "../" * depth + "index.html" if depth > 0 else "index.html"

    # Derive crumb from parent dir name + stem
    parent = out_path.parent.name  # e.g. "specs" or "plans"
    stem = out_path.stem           # e.g. "2026-05-15-connect4-design"
    # Strip leading date from stem
    crumb_name = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", stem).replace("-", " ").title()
    crumb = f"{parent.title()} · {crumb_name}"
    return index_path, crumb


def render(md_path: Path, out_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)
    kicker, heading = extract_kicker_and_heading(meta, body)
    body = strip_leading_h1(body, heading)

    md = markdown.Markdown(
        extensions=[
            "fenced_code",
            "tables",
            "sane_lists",
            "attr_list",
            "toc",
            "md_in_html",
        ],
        extension_configs={
            "toc": {"title": "Contents", "permalink": False},
        },
        output_format="html5",
    )
    rendered_body = md.convert(body)

    index_path, crumb = compute_index_path(out_path)

    html = HTML_TEMPLATE.format(
        title=heading,
        styles=STYLES,
        kicker=kicker,
        heading=heading,
        index_path=index_path,
        crumb=crumb,
        meta_block=build_meta_block(meta),
        body=rendered_body,
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"rendered: {md_path} → {out_path}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a markdown spec doc as a styled standalone HTML page."
    )
    parser.add_argument("input", type=Path, help="Path to the .md spec file")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .html path (default: same path with .html extension)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        sys.stderr.write(f"not a file: {args.input}\n")
        return 1

    out_path = args.out or args.input.with_suffix(".html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render(args.input, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
