from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^]]*\]\(([^)]+)\)")


def broken_links() -> list[str]:
    failures: list[str] = []
    for document in sorted(ROOT.rglob("*.md")):
        if any(part in {".git", ".venv"} for part in document.relative_to(ROOT).parts):
            continue
        text = document.read_text(encoding="utf-8")
        for target in LINK.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_text = unquote(target.split("#", 1)[0]).strip("<>")
            if path_text and not (document.parent / path_text).exists():
                failures.append(f"{document.relative_to(ROOT)} -> {target}")
    return failures


def main() -> None:
    failures = broken_links()
    if failures:
        raise SystemExit("Broken Markdown links:\n" + "\n".join(failures))
    print("Markdown links: ok")


if __name__ == "__main__":
    main()
