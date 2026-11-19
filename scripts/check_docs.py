"""Check that the docs carry no unaccounted assumptions.

Rules, over docs/*.md:
- no line may contain TBD;
- every line containing ASSUMED must cite an open-measurement id like (OM-3),
  and that id must be defined in docs/ARCHITECTURE.md under
  "## Open measurements" as a list item starting with **OM-3**.

Run: python scripts/check_docs.py   (exit 1 with a list of problems)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted((ROOT / "docs").glob("*.md"))
CITE = re.compile(r"\(OM-(\d+)\)")
DEFINE = re.compile(r"^\s*[-*]\s+\*\*OM-(\d+)\*\*")


def defined_ids(architecture: str) -> set[str]:
    section = architecture.split("## Open measurements", 1)
    if len(section) < 2:
        return set()
    body = section[1].split("\n## ", 1)[0]
    return {m.group(1) for line in body.splitlines() if (m := DEFINE.match(line))}


def problems() -> list[str]:
    arch = (ROOT / "docs/ARCHITECTURE.md").read_text()
    ids = defined_ids(arch)
    out = []
    for doc in DOCS:
        for n, line in enumerate(doc.read_text().splitlines(), 1):
            where = f"{doc.relative_to(ROOT)}:{n}"
            if "TBD" in line:
                out.append(f"{where}: TBD left in docs")
            if "ASSUMED" in line and not DEFINE.match(line):
                cited = CITE.findall(line)
                if not cited:
                    out.append(f"{where}: ASSUMED without an (OM-n) reference")
                out += [f"{where}: (OM-{c}) is not defined" for c in cited if c not in ids]
    return out


if __name__ == "__main__":
    found = problems()
    for p in found:
        print(p)
    print("docs check:", "FAIL" if found else "ok")
    sys.exit(1 if found else 0)
