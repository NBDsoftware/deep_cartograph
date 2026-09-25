#!/usr/bin/env python3
"""Set the release version everywhere it appears, and roll the CHANGELOG.

Single source of truth for a release: run this with the new version and every tracked
version string (setup.py, setup.cfg, CITATION.cff, docs/conf.py) plus the CHANGELOG
`[Unreleased]` section are updated consistently. Driven by
`.github/workflows/release-prepare.yml` (bump) and `release-publish.yml` (--notes-only);
safe to run locally with --dry-run to preview.

Usage:
    python scripts/bump_version.py 0.2.0 [--notes-out FILE] [--dry-run]
"""

import argparse
import difflib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/NBDsoftware/deep_cartograph"

# Bare semver, optionally with a pre-release suffix (existing tags have no leading "v").
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-.]\w+)*$")

# Each pattern below must match exactly once — a silent no-op is the main failure mode of a
# search-and-replace release, so we treat it as fatal.
#
# The package's own pin in the conda env files, should one ever pin it from git.
PIN_RE = re.compile(
    r"(deep_cartograph @ git\+https://github\.com/NBDsoftware/deep_cartograph@)\S+"
)

LITERAL_PATTERNS = {
    "setup.py": re.compile(r"(version=')[^']+(')"),
    "setup.cfg": re.compile(r"(?m)^(version = )\S+()"),
    "CITATION.cff": re.compile(r'(?m)^(version: ")[^"]*(")'),
    "docs/conf.py": re.compile(r"(?m)^(release = ')[^']*(')"),
}

# No env file pins the package from git yet; list them here if that changes.
PIN_FILES: list[str] = []

CHANGELOG = "CHANGELOG.md"
# Deliberately does not consume the trailing newline, so the blank line separating the
# heading from its body survives the substitution.
UNRELEASED_RE = re.compile(r"(?m)^## \[Unreleased\][ \t]*$")


class BumpError(Exception):
    """A target file did not look the way we expect — abort before writing anything."""


def substitute_once(path: Path, pattern: re.Pattern, replacement: str) -> str:
    """Apply `pattern` to `path`, requiring exactly one match."""
    text = path.read_text()
    new_text, count = pattern.subn(replacement, text)
    if count != 1:
        raise BumpError(
            f"{path.relative_to(REPO)}: expected 1 match for {pattern.pattern!r}, found {count}"
        )
    return new_text


def roll_changelog(text: str, version: str, today: str) -> str:
    """Rename [Unreleased] to the new version and open a fresh empty [Unreleased]."""
    if re.search(rf"(?m)^## \[{re.escape(version)}\]", text):
        raise BumpError(f"{CHANGELOG}: a section for {version} already exists")
    if not UNRELEASED_RE.search(text):
        raise BumpError(f"{CHANGELOG}: no '## [Unreleased]' heading found")

    text = UNRELEASED_RE.sub(
        f"## [Unreleased]\n\n## [{version}] - {today}", text, count=1
    )

    # Add the link reference next to the existing ones at the bottom of the file.
    link = f"[{version}]: {REPO_URL}/releases/tag/{version}"
    link_refs = list(re.finditer(r"(?m)^\[\d[^\]]*\]: \S+$", text))
    if link_refs:
        first = link_refs[0]
        text = text[: first.start()] + link + "\n" + text[first.start() :]
    else:
        text = text.rstrip("\n") + f"\n\n{link}\n"
    return text


def release_notes(changelog: str, version: str) -> str:
    """Extract the body of the new version's section, for the GitHub release notes."""
    match = re.search(
        rf"(?m)^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)",
        changelog,
        re.DOTALL,
    )
    body = match.group(1).strip() if match else ""
    return body if body else f"Release {version}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="new version, e.g. 0.2.0 (no leading 'v')")
    parser.add_argument(
        "--notes-out",
        metavar="FILE",
        help="write the new CHANGELOG section body here (release notes)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print a diff instead of writing"
    )
    parser.add_argument(
        "--notes-only",
        action="store_true",
        help="only extract the version's existing CHANGELOG section; change no files "
        "(used by the publish workflow, after the bump has already been merged)",
    )
    args = parser.parse_args()

    version = args.version.strip()
    if not VERSION_RE.match(version):
        print(
            f"error: {version!r} is not a bare version like 0.2.0 (no leading 'v')",
            file=sys.stderr,
        )
        return 2

    if args.notes_only:
        notes = release_notes((REPO / CHANGELOG).read_text(), version)
        if args.notes_out:
            Path(args.notes_out).write_text(notes + "\n")
        else:
            print(notes)
        return 0

    today = datetime.now(timezone.utc).date().isoformat()
    updates: dict[Path, str] = {}

    try:
        for rel, pattern in LITERAL_PATTERNS.items():
            path = REPO / rel
            updates[path] = substitute_once(path, pattern, rf"\g<1>{version}\g<2>")

        for rel in PIN_FILES:
            path = REPO / rel
            updates[path] = substitute_once(path, PIN_RE, rf"\g<1>{version}")

        changelog_path = REPO / CHANGELOG
        changelog = roll_changelog(changelog_path.read_text(), version, today)
        updates[changelog_path] = changelog
    except BumpError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    notes = release_notes(changelog, version)

    if args.dry_run:
        for path, new_text in updates.items():
            diff = difflib.unified_diff(
                path.read_text().splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{path.relative_to(REPO)}",
                tofile=f"b/{path.relative_to(REPO)}",
            )
            sys.stdout.writelines(diff)
        print(f"\n--- release notes for {version} ---\n{notes}")
        return 0

    for path, new_text in updates.items():
        path.write_text(new_text)
        print(f"updated {path.relative_to(REPO)}")

    if args.notes_out:
        Path(args.notes_out).write_text(notes + "\n")
        print(f"wrote release notes to {args.notes_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
