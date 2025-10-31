"""
Minimal VPT contractor downloader (no sudo, stdlib only).

- Fetches OpenAI VPT contractor manifests (6xx/7xx/8xx/9xx).
- Estimates sizes via HEAD, keeps total under --target-gb (default 45 GB).
- Downloads MP4 + JSONL pairs with basic resume.
- Writes summary CSV and checks sizes.

Usage:
  python grab_vpt_mvs.py --out ./vpt_mvs --target-gb 45 --min-pairs 12 --max-pairs 200

Tip:
  Run it once; if you want more data, raise --target-gb and re-run. Already-downloaded files are skipped/resumed.
"""
import argparse
import concurrent.futures as cf
import csv
import json
import os
import sys
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

MANIFEST_URLS = [
    # Public VPT contractor snapshot manifests hosted on Azure Blob
    "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_6xx_Jun_29.json",
    "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_7xx_Apr_6.json",
    "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_8xx_Jun_29.json",
    "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_9xx_Jun_29.json",
]

DEFAULT_HEADERS = {
    "User-Agent": "VPT-MVS/1.0 (+https://github.com/) Python-urllib",
    # Range & If-Range are set dynamically for resume
}

def human(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"

def http_head(url, timeout=20):
    req = Request(url, method="HEAD", headers=DEFAULT_HEADERS)
    with urlopen(req, timeout=timeout) as resp:
        headers = dict(resp.headers)
        cl = headers.get("Content-Length")
        length = int(cl) if cl and cl.isdigit() else None
        accept_ranges = headers.get("Accept-Ranges", "").lower() == "bytes"
        return length, accept_ranges, headers

def http_get(url, out_path, expected_size=None, accept_ranges=True, timeout=60, chunk=2*1024*1024):
    """
    Streaming GET with basic resume support if accept_ranges is True and a partial file exists.
    Returns True if the resulting file size matches expected_size (if provided), else False.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mode = "wb"
    start = 0
    if os.path.exists(out_path) and accept_ranges:
        start = os.path.getsize(out_path)
        if expected_size and start >= expected_size:
            # Already complete
            return True
        mode = "ab"

    headers = DEFAULT_HEADERS.copy()
    if accept_ranges and start > 0:
        headers["Range"] = f"bytes={start}-"

    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout) as resp, open(out_path, mode) as f:
        while True:
            data = resp.read(chunk)
            if not data:
                break
            f.write(data)

    if expected_size is not None:
        final = os.path.getsize(out_path)
        return final == expected_size
    return True

def load_manifests(urls):
    records = []
    for u in urls:
        try:
            with urlopen(Request(u, headers=DEFAULT_HEADERS), timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, list):
                    for item in data:
                        records.append(item)
                elif isinstance(data, dict):
                    records.extend(list(data.values()))
        except Exception as e:
            print(f"[WARN] Could not load manifest {u}: {e}", file=sys.stderr)
    return records

# Try multiple possible key names for URLs, since manifests vary slightly
VIDEO_KEYS = ["video_url", "mp4_url", "video", "video_mp4", "url"]
ACTIONS_KEYS = ["actions_jsonl_url", "actions_url", "jsonl_url", "actions", "label"]

def extract_urls(rec):
    v, a = None, None
    for k in VIDEO_KEYS:
        if k in rec and isinstance(rec[k], str) and rec[k].startswith("http"):
            v = rec[k]; break
    for k in ACTIONS_KEYS:
        if k in rec and isinstance(rec[k], str) and rec[k].startswith("http"):
            a = rec[k]; break
    return v, a

def pick_pairs(records, target_bytes, min_pairs=12, max_pairs=200):
    """
    Build a list of (video_url, jsonl_url, sizes) until target_bytes is reached.
    We stop when we hit max_pairs or would exceed target_bytes by > 5%.
    """
    chosen = []
    total = 0
    for rec in records:
        v, a = extract_urls(rec)
        if not v or not a:
            continue
        try:
            v_size, v_ranges, _ = http_head(v)
            a_size, a_ranges, _ = http_head(a)
        except Exception as e:
            # Skip entries with HEAD failures
            continue
        if v_size is None or a_size is None:
            # Skip if size is unknown
            continue
        pair_size = v_size + a_size
        # If we've got at least min_pairs and adding this would push us > target +5%, stop
        if len(chosen) >= min_pairs and (total + pair_size) > int(target_bytes * 1.05):
            break
        chosen.append({
            "video_url": v, "video_size": v_size, "video_ranges": v_ranges,
            "jsonl_url": a, "jsonl_size": a_size, "jsonl_ranges": a_ranges
        })
        total += pair_size
        if len(chosen) >= max_pairs or total >= target_bytes:
            break
    return chosen, total

def safe_name_from_url(u):
    p = urlparse(u)
    base = os.path.basename(p.path)
    return base or (p.netloc.replace(":","_")+"_"+str(int(time.time())))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./vpt_mvs", help="Output directory for data files")
    ap.add_argument("--target-gb", type=float, default=45.0, help="Target total download budget (GB)")
    ap.add_argument("--min-pairs", type=int, default=12, help="Minimum MP4+JSONL pairs")
    ap.add_argument("--max-pairs", type=int, default=200, help="Maximum MP4+JSONL pairs")
    ap.add_argument("--parallel", type=int, default=4, help="Parallel downloads")
    args = ap.parse_args()

    target_bytes = int(args.target_gb * (1024**3))
    os.makedirs(args.out, exist_ok=True)
    print(f"[INFO] Loading manifests…")
    records = load_manifests(MANIFEST_URLS)
    print(f"[INFO] Manifests entries: {len(records)}")

    print(f"[INFO] Selecting pairs up to ~{human(target_bytes)} (min={args.min_pairs}, max={args.max_pairs}) …")
    pairs, estimated = pick_pairs(records, target_bytes, args.min_pairs, args.max_pairs)
    if len(pairs) < args.min_pairs:
        print(f"[ERROR] Could not find enough pairs with known sizes. Found: {len(pairs)}", file=sys.stderr)
        sys.exit(2)
    print(f"[INFO] Selected {len(pairs)} pairs, estimated total {human(estimated)}")

    # Prepare download plan
    dl_jobs = []
    rows = []
    for i, p in enumerate(pairs):
        v_url = p["video_url"]; a_url = p["jsonl_url"]
        v_out = os.path.join(args.out, safe_name_from_url(v_url))
        a_out = os.path.join(args.out, safe_name_from_url(a_url))
        rows.append({
            "index": i,
            "video_url": v_url, "video_path": v_out, "video_size": p["video_size"],
            "jsonl_url": a_url, "jsonl_path": a_out, "jsonl_size": p["jsonl_size"]
        })
        dl_jobs.append(("video", v_url, v_out, p["video_size"], p["video_ranges"]))
        dl_jobs.append(("jsonl", a_url, a_out, p["jsonl_size"], p["jsonl_ranges"]))

    # Download in parallel with small pool
    ok = True
    def _task(job):
        kind, url, outp, sz, rng = job
        try:
            res = http_get(url, outp, expected_size=sz, accept_ranges=rng)
            return (kind, outp, res, sz)
        except Exception as e:
            return (kind, outp, False, sz)

    print(f"[INFO] Downloading ({len(dl_jobs)} files) with {args.parallel} workers…")
    with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
        for kind, outp, success, sz in ex.map(_task, dl_jobs):
            if success:
                print(f"[OK] {kind:5s} {os.path.basename(outp)} ({human(os.path.getsize(outp))})")
            else:
                print(f"[FAIL] {kind:5s} {os.path.basename(outp)}", file=sys.stderr)
                ok = False

    # Verify sizes and write CSV
    total_bytes = 0
    for r in rows:
        vs = os.path.getsize(r["video_path"]) if os.path.exists(r["video_path"]) else 0
        js = os.path.getsize(r["jsonl_path"]) if os.path.exists(r["jsonl_path"]) else 0
        r["video_downloaded"] = vs
        r["jsonl_downloaded"] = js
        total_bytes += vs + js

    csv_path = os.path.join(args.out, "download_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[INFO] Wrote summary: {csv_path}")
    print(f"[INFO] Downloaded total: {human(total_bytes)}")
    if not ok:
        print("[WARN] Some files failed; rerun to resume missing ones.", file=sys.stderr)
    # Basic pair-check
    missing_json = [r for r in rows if not os.path.exists(r["jsonl_path"])]
    missing_mp4  = [r for r in rows if not os.path.exists(r["video_path"])]
    if missing_json or missing_mp4:
        print(f"[WARN] Missing JSONL pairs: {len(missing_json)}; missing MP4s: {len(missing_mp4)}", file=sys.stderr)

if __name__ == "__main__":
    main()
