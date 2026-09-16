# Repository maintenance scripts

[check_docs.py](check_docs.py) checks local inline Markdown links, image paths,
and heading anchors without network access or extra dependencies.

Run from the repository root:

```bash
make docs-check
```

The check skips fenced code, generated build output, environments, and downloaded
data. It supports the inline links and ATX headings used in this repository;
it is not a complete Markdown parser and does not check external website links.

See the [documentation index](../docs/README.md) for the reading path.
