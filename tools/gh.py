#!/usr/bin/env python3
"""Minimal GitHub REST helper used by the packaging automation.

Deliberately dependency-light: only `requests` is required, so CI can run this
without installing the full dev toolchain.

Credentials are read from the environment first, then from the repository's
secret file. The secret file name is configurable through
``CUDA_AUR_SECRETS_FILE`` so a project can move away from a plain ``.env``
without touching code.

Usage:
    gh.py get   /repos/{owner}/{repo}
    gh.py post  /repos/{owner}/{repo}/issues --data '{"title": "..."}'
    gh.py ci    [--limit 5]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover - developer guidance only
    sys.exit("requests is required; install it with `uv pip install requests`")

API = "https://api.github.com"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SECRET_FILE = ".env"


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a minimal KEY=VALUE file, ignoring comments and blank lines."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def load_config() -> dict[str, str]:
    """Resolve credentials from the environment, then the secret file.

    The environment always wins so CI can inject a token without writing a
    secret file to disk.
    """
    secret_name = os.environ.get("CUDA_AUR_SECRETS_FILE", DEFAULT_SECRET_FILE)
    file_values = _parse_env_file(REPO_ROOT / secret_name)

    config: dict[str, str] = {}
    for key in ("GITHUB_TOKEN", "GITHUB_OWNER", "GITHUB_REPO"):
        config[key] = os.environ.get(key) or file_values.get(key, "")

    return config


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def request(method: str, path: str, payload: dict | None = None) -> tuple[int, object]:
    """Perform an API call and return (status_code, decoded_body)."""
    config = load_config()
    token = config["GITHUB_TOKEN"]
    if not token:
        raise SystemExit(
            "no GITHUB_TOKEN found; set it in the environment or in "
            f"{REPO_ROOT / DEFAULT_SECRET_FILE}"
        )

    url = path if path.startswith("http") else f"{API}{path}"
    response = requests.request(
        method,
        url,
        headers=_headers(token),
        json=payload,
        timeout=30,
    )
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return response.status_code, body


def cmd_ci(args: argparse.Namespace) -> int:
    """Print the recent workflow runs, or say plainly that there are none."""
    config = load_config()
    repo = args.repo or config["GITHUB_REPO"]
    if not repo:
        raise SystemExit("no repository configured")

    status, body = request("GET", f"/repos/{repo}/actions/runs?per_page={args.limit}")
    if status != 200:
        print(f"error {status}: {body}", file=sys.stderr)
        return 1

    runs = body.get("workflow_runs", [])
    if not runs:
        print(f"{repo}: no workflow runs (total_count={body.get('total_count', 0)})")
        return 0

    for run in runs:
        print(
            f"{run['name']}: {run['status']}"
            f"/{run.get('conclusion') or '-'} [{run['head_branch']}] {run['html_url']}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for method in ("get", "post", "patch"):
        p = sub.add_parser(method, help=f"HTTP {method.upper()}")
        p.add_argument("path")
        p.add_argument("--data", help="JSON request body")

    ci = sub.add_parser("ci", help="show recent workflow runs")
    ci.add_argument("--repo", help="owner/repo override")
    ci.add_argument("--limit", type=int, default=5)

    args = parser.parse_args(argv)

    if args.command == "ci":
        return cmd_ci(args)

    payload = json.loads(args.data) if getattr(args, "data", None) else None
    status, body = request(args.command.upper(), args.path, payload)
    print(json.dumps(body, indent=2) if not isinstance(body, str) else body)
    return 0 if status < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())
