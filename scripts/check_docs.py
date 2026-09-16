"""Check repository-local inline Markdown links and ATX heading anchors.

No network access or third-party packages. Fenced code is ignored. This is a
small check for this repository's documentation, not a full Markdown parser.
"""
from pathlib import Path
import os
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "venv", "build", "__pycache__", "mnist_data", ".claude"}
LINK = re.compile(r"!?\[[^\]\n]*\]\((<[^>]+>|[^\s)]+)(?:\s+\"[^\"]*\")?\)")


def prose_lines(text):
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is None:
            yield number, line


def anchors(text):
    result = set()
    counts = {}
    for _, line in prose_lines(text):
        heading = re.match(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not heading:
            continue
        title = re.sub(r"!?\[([^]]+)\]\([^)]*\)", r"\1", heading.group(1))
        slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        result.add(slug if count == 0 else f"{slug}-{count}")
    return result


def main():
    files = []
    for directory, children, names in os.walk(ROOT):
        children[:] = sorted(name for name in children if name not in SKIP)
        files.extend(Path(directory) / name for name in sorted(names) if name.endswith(".md"))
    errors = []
    checked = 0
    anchor_cache = {}
    for source in files:
        for number, line in prose_lines(source.read_text(encoding="utf-8")):
            for match in LINK.finditer(line):
                destination = match.group(1).strip("<>")
                url = urlsplit(destination)
                if url.scheme or url.netloc:
                    continue
                checked += 1
                target = (source.parent / unquote(url.path)).resolve() if url.path else source
                error = None
                if not target.exists():
                    error = "missing path"
                elif url.fragment and target.suffix == ".md":
                    if target not in anchor_cache:
                        anchor_cache[target] = anchors(target.read_text(encoding="utf-8"))
                    if unquote(url.fragment) not in anchor_cache[target]:
                        error = "missing heading"
                if error:
                    errors.append(f"{source.relative_to(ROOT)}:{number}: {error}: {destination}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Documentation OK: {len(files)} Markdown files, {checked} local links checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
