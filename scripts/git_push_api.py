#!/usr/bin/env python3
"""Idempotent push of the committed tree to GitHub via the git-data API.

Blob SHAs are the deterministic git object hashes, so already-uploaded blobs
are detected with a GET and skipped. Every request is retried on network
errors and 5xx; trees/commits retry on 403 (secondary rate limits).
"""
import base64
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

TOKEN = sys.argv[1]
COMMIT_MSG = sys.argv[2] if len(sys.argv) > 2 else (
    "Continual learning framework on openpi: INFO (prompt-bank) with "
    "BASE/ER/EWC/LWF baselines\n\n"
    "- Rename GEN method to INFO (agents/info.py, InfoAgent, exp_name INFO)\n"
    "- Keep BASE, ER, EWC, LWF continual learning agents\n"
    "- Archive legacy code (old SNN variants, MAR, Stellar, viz tooling, "
    "logs, assets/docs/third_party) out of the repo\n"
    "- Add README with install, data prep, training, and evaluation guide"
)
OWNER, REPO = "Zhaolarry", "Info-VLA"
API = f"https://api.github.com/repos/{OWNER}/{REPO}"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "info-vla-push",
}


def api(method, path, payload=None, ok403=False):
    data = json.dumps(payload).encode() if payload is not None else None
    last = None
    for attempt in range(6):
        req = urllib.request.Request(API + path, data=data, headers=HEADERS, method=method)
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            if e.code == 403 and ok403 and attempt < 5:
                wait = 20 * (attempt + 1)
                print(f"  403 on {method} {path}, wait {wait}s")
                time.sleep(wait)
                last = e
                continue
            if e.code == 404:
                return None
            print(f"HTTP {e.code} on {method} {path}: {body}")
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e
        wait = min(3 * (2 ** attempt), 40)
        print(f"  net error ({type(last).__name__}), retry in {wait}s")
        time.sleep(wait)
    raise last


# 1. list tree entries of local HEAD
out = subprocess.run(
    ["git", "ls-tree", "-r", "HEAD"], capture_output=True, text=True, check=True
).stdout
entries = []
for line in out.splitlines():
    meta, path = line.split("\t", 1)
    mode, typ, sha = meta.split()
    entries.append((mode, typ, sha, path))
print(f"files: {len(entries)}")

# 2. blobs: skip if already on the remote (git blob sha is deterministic)
tree_entries = []
uploaded = skipped = 0
for i, (mode, typ, git_sha, path) in enumerate(entries):
    assert typ == "blob", f"unexpected non-blob: {path}"
    if api("GET", f"/git/blobs/{git_sha}") is not None:
        skipped += 1
    else:
        with open(path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode()
        api("POST", "/git/blobs", {"content": b64, "encoding": "base64"})
        uploaded += 1
    tree_entries.append({"path": path, "mode": mode, "type": "blob", "sha": git_sha})
    if (i + 1) % 25 == 0:
        print(f"  blobs {i + 1}/{len(entries)} (new {uploaded}, skipped {skipped})")
    time.sleep(0.1)
print(f"blobs done: {uploaded} new, {skipped} already present")

# 3. tree
tree = api("POST", "/git/trees", {"tree": tree_entries}, ok403=True)
print("tree:", tree["sha"])

# 4. remote HEAD as parent
ref = api("GET", "/git/refs/heads/main")
parent = ref["object"]["sha"]
print("parent:", parent)

# 5. commit
commit = api(
    "POST",
    "/git/commits",
    {
        "message": COMMIT_MSG,
        "tree": tree["sha"],
        "parents": [parent],
    },
    ok403=True,
)
print("commit:", commit["sha"])

# 6. update ref (fast-forward)
api("PATCH", "/git/refs/heads/main", {"sha": commit["sha"]}, ok403=True)
print("OK: main updated")
