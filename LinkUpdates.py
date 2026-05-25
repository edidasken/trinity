#!/usr/bin/env python3
"""
LinkUpdates.py

Deep-planning notes for .sixth
===============================

Purpose
-------
This script normalizes broken internal links in repos that use a parent base
href such as::

    <base href="../">

That layout makes browser-facing URLs like ``../app.feed/feed.html`` resolve
above the repo root. The fix is usually to strip the leading ``../`` so the
link becomes ``app.feed/feed.html``.

What it changes
---------------
By default it rewrites a small, exact set of safe prefixes:

- ``../app.`` -> ``app.``
- ``../Styles/`` -> ``Styles/``
- ``../Images/`` -> ``Images/``
- ``../Scripts/`` -> ``Scripts/``
- ``../Data/`` -> ``Data/``

Why the scope is conservative
-----------------------------
This repo's broken-link pattern shows up in two kinds of places:

1. HTML files that declare ``<base href="../">``.
2. Shared browser scripts that still contain browser URLs as string literals.

The script does **not** blindly rewrite every ``../app.`` link in the tree.
That matters because some pages legitimately use ``../app.grow/...`` when they
do **not** have the parent base href.

How the script decides what to scan
-----------------------------------
1. It finds HTML files that contain the base marker.
2. It treats the parent directories of those HTML files as "rooted app dirs".
3. It scans those rooted app dirs recursively.
4. It also scans shared browser-code dirs such as ``Scripts/`` and ``Views/``.
5. It rewrites only the exact prefixes listed in ``DEFAULT_RULES``.

How to use it
-------------
Dry-run first:

    python3 LinkUpdates.py

Apply changes:

    python3 LinkUpdates.py --apply

Show unified diffs while applying:

    python3 LinkUpdates.py --apply --show-diff

If a repo keeps browser JS outside the detected app dirs, add it explicitly:

    python3 LinkUpdates.py --include-dir . --apply

If a repo uses a different base href, change ``BASE_MARKER`` or pass
``--base-marker`` on the command line.

How to adapt it in another repo
-------------------------------
When copying this script into a new repo, the only things that usually need
changing are:

- ``BASE_MARKER`` if the page base path differs.
- ``DEFAULT_RULES`` if your repo uses different top-level resource folders.
- ``DEFAULT_SHARED_DIRS`` if browser scripts live somewhere other than
  ``Scripts/`` or ``Views/``.
- ``--include-dir`` if you keep browser code in the repo root.

Operational rule for .sixth
---------------------------
Always follow the same flow:

1. Run the script in dry-run mode.
2. Inspect the file list and the proposed replacements.
3. Apply the patch.
4. Run the site and click the updated links.
5. If a repo has nonstandard folder layout, update the rules before applying.

This script is intentionally small and editable so .sixth can copy it into any
repo and tune the constants without having to re-derive the workflow.
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Set, Tuple

BASE_MARKER = '<base href="../">'

DEFAULT_RULES: List[Tuple[str, str]] = [
    ("../app.", "app."),
    ("../Styles/", "Styles/"),
    ("../Images/", "Images/"),
    ("../Scripts/", "Scripts/"),
    ("../Data/", "Data/"),
]

DEFAULT_SHARED_DIRS = ("Scripts", "Views")
DEFAULT_SKIP_DIRS = {".git", "node_modules", "dist", "build", "coverage", ".next", ".vite", "out"}
TEXT_EXTENSIONS = {
    ".html", ".htm", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".css", ".json", ".md", ".svg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize broken internal links in base-rooted HTML/JS files."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repo root to scan (default: current directory).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the changes back to disk. Without this flag the script is dry-run only.",
    )
    parser.add_argument(
        "--show-diff",
        action="store_true",
        help="Print a unified diff for each changed file.",
    )
    parser.add_argument(
        "--base-marker",
        default=BASE_MARKER,
        help="HTML marker that identifies a base-rooted app (default: <base href=\"../\">).",
    )
    parser.add_argument(
        "--include-dir",
        action="append",
        default=[],
        help="Extra directory to scan. Can be repeated (for example: --include-dir .).",
    )
    parser.add_argument(
        "--skip-dir",
        action="append",
        default=[],
        help="Directory name to skip during recursion. Can be repeated.",
    )
    parser.add_argument(
        "--rule",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Add a custom replacement rule. Can be repeated.",
    )
    return parser.parse_args()


def parse_rule(raw: str) -> Tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Invalid rule '{raw}'. Expected OLD=NEW.")
    old, new = raw.split("=", 1)
    if not old:
        raise ValueError(f"Invalid rule '{raw}'. OLD cannot be empty.")
    return old, new


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS


def should_skip(path: Path, skip_dirs: Set[str]) -> bool:
    return any(part in skip_dirs for part in path.parts)


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def discover_rooted_dirs(root: Path, base_marker: str, skip_dirs: Set[str]) -> Set[Path]:
    rooted_dirs: Set[Path] = set()

    for html in root.rglob("*.html"):
        if should_skip(html, skip_dirs):
            continue
        text = read_text(html)
        if text is None:
            continue
        if base_marker in text:
            rooted_dirs.add(html.parent.resolve())

    return rooted_dirs


def normalize_include_dirs(root: Path, include_dirs: Sequence[str]) -> List[Path]:
    dirs: List[Path] = []
    for raw in include_dirs:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate.exists():
            dirs.append(candidate)
    return dirs


def build_scan_dirs(
    root: Path,
    rooted_dirs: Set[Path],
    include_dirs: Sequence[str],
    shared_dirs: Sequence[str],
) -> List[Path]:
    ordered: List[Path] = []
    seen: Set[Path] = set()

    def add_dir(candidate: Path) -> None:
        resolved = candidate.resolve()
        if resolved.exists() and resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)

    for name in shared_dirs:
        add_dir((root / name))

    for candidate in normalize_include_dirs(root, include_dirs):
        add_dir(candidate)

    for candidate in sorted(rooted_dirs):
        add_dir(candidate)

    return ordered


def iter_files(base: Path) -> Iterator[Path]:
    if base.is_file():
        yield base
        return

    for path in base.rglob("*"):
        if path.is_file():
            yield path


def apply_rules(text: str, rules: Sequence[Tuple[str, str]]) -> Tuple[str, int]:
    total = 0
    updated = text
    for old, new in rules:
        if old not in updated:
            continue
        count = updated.count(old)
        updated = updated.replace(old, new)
        total += count
    return updated, total


def render_diff(path: Path, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(True),
            after.splitlines(True),
            fromfile=str(path),
            tofile=str(path),
        )
    )


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    skip_dirs = set(DEFAULT_SKIP_DIRS) | set(args.skip_dir)

    try:
        extra_rules = [parse_rule(raw) for raw in args.rule]
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    rules = list(DEFAULT_RULES) + extra_rules

    rooted_dirs = discover_rooted_dirs(root, args.base_marker, skip_dirs)
    scan_dirs = build_scan_dirs(root, rooted_dirs, args.include_dir, DEFAULT_SHARED_DIRS)

    if not scan_dirs:
        print("No scan directories found. Nothing to do.")
        return 0

    print(f"Root: {root}")
    if rooted_dirs:
        print("Rooted app dirs:")
        for d in sorted(rooted_dirs):
            print(f"  - {d.relative_to(root) if d.is_relative_to(root) else d}")
    else:
        print("Rooted app dirs: none found")

    print("Scan dirs:")
    for d in scan_dirs:
        print(f"  - {d.relative_to(root) if d.is_relative_to(root) else d}")

    scanned = 0
    changed_files = 0
    total_replacements = 0

    for base in scan_dirs:
        for path in iter_files(base):
            if should_skip(path, skip_dirs):
                continue
            if not is_text_file(path):
                continue

            before = read_text(path)
            if before is None:
                continue

            after, replacements = apply_rules(before, rules)
            if replacements == 0:
                continue

            scanned += 1
            total_replacements += replacements

            rel = path.relative_to(root) if path.is_relative_to(root) else path
            if args.show_diff:
                print(render_diff(path, before, after))
            else:
                print(f"[{'APPLY' if args.apply else 'DRY-RUN'}] {rel} ({replacements} replacement{'s' if replacements != 1 else ''})")

            if args.apply:
                path.write_text(after, encoding="utf-8")
                changed_files += 1

    print(
        f"Summary: {scanned} file{'s' if scanned != 1 else ''} matched, "
        f"{changed_files} file{'s' if changed_files != 1 else ''} written, "
        f"{total_replacements} replacement{'s' if total_replacements != 1 else ''} total."
    )

    if not args.apply:
        print("Dry-run only. Re-run with --apply to write the changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
