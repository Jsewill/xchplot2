#!/usr/bin/env python3
"""Check local links in tracked Markdown without contacting external sites."""
from collections import Counter
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit


def markdown(text):
    # ponytail: inline links and ATX headings only; use a Markdown parser if
    # the guides adopt reference links, setext headings, or richer markup.
    headings, links, counts = set(), [], Counter()
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            delimiter = marker[1]
            if fence is None:
                fence = delimiter
            elif delimiter[0] == fence[0] and len(delimiter) >= len(fence):
                fence = None
            continue
        if fence:
            continue
        heading = re.match(r"^ {0,3}#{1,6}\s+(.+?)(?:\s+#+)?$", line)
        if heading:
            slug = re.sub(r"[^\w -]", "", heading[1].lower()).replace(" ", "-")
            suffix = f"-{counts[slug]}" if counts[slug] else ""
            counts[slug] += 1
            headings.add(slug + suffix)
        line = re.sub(r"(`+).*?\1", "", line)
        links.extend((number, match[1]) for match in re.finditer(r"\]\(([^)]+)\)", line))
    return headings, links


def check(root, files, tracked):
    errors = []
    for file in files:
        _, links = markdown((root / file).read_text())
        for number, destination in links:
            url = urlsplit(destination)
            if url.scheme or url.netloc:
                continue
            target = (root / file).parent / unquote(url.path) if url.path else root / file
            try:
                relative = target.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                relative = ""
            reason = None
            if relative not in tracked or not target.exists():
                reason = "target is missing or not tracked"
            elif target.suffix == ".md" and url.fragment:
                headings, _ = markdown(target.read_text())
                if unquote(url.fragment) not in headings:
                    reason = "section does not exist"
            if reason:
                errors.append(f"{file}:{number}: {destination}: {reason}")
    return errors


def self_test():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "README.md").write_text(
            "# Guide\n[API](Reference.md#cpu-workers)\n"
            "[Repeat](Reference.md#cpu-workers-1)\n[Web](https://example.com)\n"
            "```md\n[Example](missing.md)\n```\n`[Example](missing.md)`\n")
        (root / "Reference.md").write_text("# CPU workers\n## CPU workers\n")
        tracked = {"README.md", "Reference.md"}
        assert not check(root, ["README.md"], tracked)
        with (root / "README.md").open("a") as output:
            output.write("[Broken](Reference.md#absent)\n[Missing](missing.md)\n"
                         "[Outside](../private.md)\n[Local](#absent)\n")
        assert len(check(root, ["README.md"], tracked)) == 4
    print("Documentation link check self-test passed.")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    else:
        root = Path(__file__).resolve().parents[2]
        tracked = set(subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z"], text=True).split("\0"))
        files = sys.argv[1:] or sorted(path for path in tracked if path.endswith(".md"))
        errors = check(root, files, tracked | set(files))
        print("\n".join(errors) if errors else f"Checked local links in {len(files)} Markdown files.")
        sys.exit(bool(errors))
