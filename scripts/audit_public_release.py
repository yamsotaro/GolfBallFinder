#!/usr/bin/env python3
"""Audit a Git worktree and reachable history before making it public.

Findings deliberately contain only a rule name, path, and optional commit ID.
Matched values are never printed or written to disk.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


MAX_TEXT_BYTES = 5 * 1024 * 1024
IGNORED_PRIVATE_KEY_SUFFIXES = {".key", ".pem"}
BLOCKED_SUFFIXES = {
    ".p8": "apple_private_key_file",
    ".p12": "certificate_private_key_file",
    ".cer": "certificate_file",
    ".mobileprovision": "provisioning_profile_file",
    ".env": "environment_secret_file",
}
BLOCKED_BASENAMES = {
    "credentials.json": "credential_file",
    "secrets.json": "secret_file",
    "secrets.yaml": "secret_file",
    "secrets.yml": "secret_file",
}
TEXT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key_material",
        re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    ),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{20,}\b")),
    (
        "authorization_header",
        re.compile(r"(?im)^\s*Authorization\s*:\s*(?:Bearer|Basic)\s+\S{8,}"),
    ),
    (
        "credential_in_url",
        re.compile(r"https?://[^\s/:@]+:[^\s/@]+@", re.IGNORECASE),
    ),
    (
        "credential_query_parameter",
        re.compile(
            r"https?://\S+[?&](?:token|access_token|api_key|key|signature|x-amz-signature)=[^\s&]{8,}",
            re.IGNORECASE,
        ),
    ),
    (
        "literal_secret_assignment",
        re.compile(
            r"(?im)^\s*(?:export\s+)?[A-Za-z0-9_.-]*"
            r"(?:password|passwd|api[_-]?key|private[_-]?key|client[_-]?secret|"
            r"access[_-]?token|bearer[_-]?token|issuer[_-]?id|key[_-]?id)"
            r"[A-Za-z0-9_.-]*\s*[:=]\s*[\"']?"
            r"(?!\$|%|\{|<|example|replace|your_|none|null|true|false)"
            r"[A-Za-z0-9/+_.:@?&=-]{8,}",
            re.IGNORECASE,
        ),
    ),
    (
        "personal_windows_path",
        re.compile(r"\b[A-Za-z]:\\Users\\(?!\.\.\.|<|%|\$)[^\\\s]+\\", re.IGNORECASE),
    ),
    (
        "personal_email",
        re.compile(
            r"\b[A-Z0-9._%+-]+@(?!example\.(?:com|org|net)\b|users\.noreply\.github\.com\b)"
            r"[A-Z0-9.-]+\.[A-Z]{2,}\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True, order=True)
class Finding:
    scope: str
    rule: str
    path: str
    commit: str | None = None

    def sanitized(self) -> str:
        commit = f" commit={self.commit[:12]}" if self.commit else ""
        return f"scope={self.scope} rule={self.rule} path={self.path}{commit}"


def run_git(root: Path, *args: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=text,
        encoding="utf-8" if text else None,
        errors="replace" if text else None,
    )
    return result.stdout


def path_finding(path: str, scope: str, commit: str | None) -> Finding | None:
    normalized = PurePosixPath(path.replace("\\", "/"))
    lower_name = normalized.name.lower()
    suffix = normalized.suffix.lower()
    rule = BLOCKED_BASENAMES.get(lower_name) or BLOCKED_SUFFIXES.get(suffix)
    if rule is None and lower_name.startswith(".env."):
        rule = "environment_secret_file"
    return Finding(scope, rule, str(normalized), commit) if rule else None


def scan_text(text: str, path: str, scope: str, commit: str | None) -> set[Finding]:
    findings: set[Finding] = set()
    for rule, pattern in TEXT_RULES:
        if pattern.search(text):
            findings.add(Finding(scope, rule, path, commit))
    return findings


def scan_bytes(data: bytes, path: str, scope: str, commit: str | None) -> set[Finding]:
    if len(data) > MAX_TEXT_BYTES or b"\x00" in data:
        return set()
    return scan_text(data.decode("utf-8", errors="replace"), path, scope, commit)


def candidate_worktree_paths(root: Path) -> list[str]:
    output = run_git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    return sorted(path for path in str(output).split("\0") if path)


def candidate_ignored_worktree_paths(root: Path) -> list[str]:
    output = run_git(
        root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "-z",
    )
    return sorted(path for path in str(output).split("\0") if path)


def scan_worktree(root: Path) -> set[Finding]:
    findings: set[Finding] = set()
    for relative in candidate_worktree_paths(root):
        blocked = path_finding(relative, "worktree", None)
        if blocked:
            findings.add(blocked)
        path = root / PurePosixPath(relative)
        if path.is_file():
            findings.update(scan_bytes(path.read_bytes(), relative, "worktree", None))
    return findings


def scan_ignored_worktree(root: Path) -> set[Finding]:
    """Check ignored names and possible private-key containers without exposing data.

    Ignored dependency/data/build trees can contain tens of thousands of large files. Their
    content is not a Git publication candidate, but secret-bearing filenames must still be
    surfaced and ignored PEM/KEY files must not contain private-key material.
    """
    findings: set[Finding] = set()
    private_key_rule = dict(TEXT_RULES)["private_key_material"]
    for relative in candidate_ignored_worktree_paths(root):
        blocked = path_finding(relative, "ignored-worktree", None)
        if blocked:
            findings.add(blocked)
        path = root / PurePosixPath(relative)
        if (
            path.suffix.lower() in IGNORED_PRIVATE_KEY_SUFFIXES
            and path.is_file()
            and path.stat().st_size <= MAX_TEXT_BYTES
        ):
            text = path.read_text(encoding="utf-8", errors="replace")
            if private_key_rule.search(text):
                findings.add(
                    Finding("ignored-worktree", "private_key_material", relative, None)
                )
    return findings


def reachable_commits(root: Path) -> list[str]:
    return [commit for commit in str(run_git(root, "rev-list", "--all")).splitlines() if commit]


def tree_paths(root: Path, commit: str) -> list[str]:
    output = run_git(root, "ls-tree", "-r", "--name-only", "-z", commit)
    return [path for path in str(output).split("\0") if path]


def blob(root: Path, commit: str, path: str) -> bytes:
    return bytes(run_git(root, "show", f"{commit}:{path}", text=False))


def scan_history(root: Path) -> set[Finding]:
    findings: set[Finding] = set()
    for commit in reachable_commits(root):
        for path in tree_paths(root, commit):
            blocked = path_finding(path, "history", commit)
            if blocked:
                findings.add(blocked)
            findings.update(scan_bytes(blob(root, commit, path), path, "history", commit))
    return findings


def scan_history_metadata(root: Path) -> set[Finding]:
    findings: set[Finding] = set()
    email_rule = dict(TEXT_RULES)["personal_email"]
    output = str(run_git(root, "log", "--all", "--format=%H%x00%ae%x00%ce%x00"))
    fields = output.split("\0")
    for index in range(0, len(fields) - 2, 3):
        commit = fields[index].strip()
        if not commit:
            continue
        if email_rule.search(fields[index + 1]) or email_rule.search(fields[index + 2]):
            findings.add(
                Finding("history-metadata", "personal_email", "<commit-metadata>", commit)
            )
    return findings


def audit(root: Path, include_history: bool = True) -> list[Finding]:
    findings = scan_worktree(root)
    findings.update(scan_ignored_worktree(root))
    if include_history:
        findings.update(scan_history(root))
        findings.update(scan_history_metadata(root))
    return sorted(findings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--worktree-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    findings = audit(root, include_history=not args.worktree_only)
    if findings:
        print(f"Public-release audit failed with {len(findings)} sanitized finding(s):")
        for finding in findings:
            print(finding.sanitized())
        raise SystemExit(1)
    history = "disabled" if args.worktree_only else "all reachable commits"
    print(
        "Public-release audit passed: tracked/untracked/ignored worktree + "
        f"{history}; secret values were never emitted"
    )


if __name__ == "__main__":
    main()
