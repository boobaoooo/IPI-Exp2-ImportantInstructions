"""Read-only runtime monitor for the Full Experiment (run it in its OWN window, next to the batch).

It never loads a model, never sends a chat request, never stops or changes anything (also not the
owner's Ollama). Each tick it records, appending one JSON line to the log and one short line on screen:
  * our instance: server version, the 3 model digests (/api/tags) vs the manifest, loaded models with
    context_length and GPU residency (/api/ps)
  * the owner's instance (default 127.0.0.1:11434): version and loaded models (must stay empty)
  * GPU: utilisation, memory, temperature, and the compute processes (nvidia-smi)
  * disk free on the output drive, and batch progress (records in raw_runs.jsonl, time of the last record,
    lock file present) so a stalled batch is visible
and raises ALERT lines when something that matters for the experiment changes:
  version drift | digest drift | model missing | served context != 16384 | model not 100% on GPU |
  other instance holds a model | foreign GPU compute process | GPU too hot | disk low | batch stalled.
Alerts do not stop the batch by themselves (the driver has its own guards and stops itself); they tell the
operator to look and are kept as evidence.

Usage (from Z:\\p2, venv active, same env vars as the batch):
  python monitor_runtime.py --handoff P1_to_P2_full_execution_v1.0 --expected-endpoint http://127.0.0.1:11435 --out-dir output_full
  python monitor_runtime.py ... --once        # one check, exit 0 = OK, 1 = alert(s), 2 = cannot read
Stop it with Ctrl+C. Log: monitor_log.jsonl (next to --out-dir unless --log is given).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_CONTEXT = 16384
PROC_ROW = re.compile(r"^\|\s+(\d+)\s+\S+\s+\S+\s+(\d+)\s+(C\+G|C|G)\s+(.+?)\s+(\S+)\s+\|\s*$")


def http_json(url: str, timeout: float = 10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def native_root(url: str) -> str:
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def parse_gpu_processes(table: str) -> list[dict]:
    """Rows of the process table in plain `nvidia-smi` output (works on WDDM, where memory is N/A)."""
    out = []
    for line in table.splitlines():
        m = PROC_ROW.match(line)
        if m:
            out.append({"gpu": int(m[1]), "pid": int(m[2]), "type": m[3], "name": m[4].strip()})
    return out


def gpu_info() -> dict | None:
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        if q.returncode != 0 or not q.stdout.strip():
            return None
        name, util, used, total, temp = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
        table = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
        return {"gpu": name, "util_pct": int(util), "mem_used_mib": int(used), "mem_total_mib": int(total),
                "temp_c": int(temp), "processes": parse_gpu_processes(table)}
    except Exception:  # noqa: BLE001
        return None


def residency(m: dict) -> str:
    size, vram = m.get("size", 0), m.get("size_vram", 0)
    return "100% GPU" if vram and vram >= size else ("partial CPU" if vram else "CPU only")


def evaluate(snap: dict, expected: dict) -> list[str]:
    """Pure function: alerts for one snapshot (so it can be unit-tested without any server)."""
    A: list[str] = []
    ours = snap.get("ours") or {}
    if ours.get("error"):
        A.append(f"cannot read our instance: {ours['error']}")
        return A
    if ours.get("version") != expected["ollama"]:
        A.append(f"VERSION DRIFT: our server reports {ours.get('version')!r}, expected {expected['ollama']!r}")
    for model, want in expected["digests"].items():
        got = ours.get("digests", {}).get(model)
        if got is None:
            A.append(f"MODEL MISSING: {model} not listed by /api/tags")
        elif got.split("sha256:")[-1] != want:
            A.append(f"DIGEST DRIFT: {model} is {got}, expected {want}")
    for m in ours.get("loaded", []):
        if m.get("context_length") != EXPECTED_CONTEXT:
            A.append(f"CONTEXT: {m['name']} served with {m.get('context_length')}, expected {EXPECTED_CONTEXT}")
        if m.get("residency") != "100% GPU":
            A.append(f"RESIDENCY: {m['name']} is '{m.get('residency')}' (VRAM interference?)")
    other = snap.get("other")
    if other and other.get("loaded"):
        A.append(f"OTHER INSTANCE has model(s) loaded: {[m['name'] for m in other['loaded']]} (GPU interference risk)")
    g = snap.get("gpu")
    if g:
        procs = g.get("processes", [])
        compute = [p for p in procs if p["type"] == "C"]
        inf = [p for p in compute if re.search(r"llama|ollama", p["name"], re.I)]
        foreign = [p for p in compute if p not in inf]
        if foreign:
            A.append(f"FOREIGN GPU COMPUTE PROCESS: {[p['name'] for p in foreign]}")
        n_loaded = len(ours.get("loaded", []))
        if len(inf) > max(n_loaded, 1):
            A.append(f"{len(inf)} llama/ollama compute processes on the GPU but only {n_loaded} model(s) loaded "
                     f"in our instance (someone else may be running inference)")
        if g["temp_c"] >= expected["max_temp_c"]:
            A.append(f"GPU TEMPERATURE {g['temp_c']} C >= {expected['max_temp_c']} C")
    disk = snap.get("disk_free_gb")
    if disk is not None and disk < expected["min_free_gb"]:
        A.append(f"DISK LOW: {disk:.1f} GB free (< {expected['min_free_gb']} GB)")
    b = snap.get("batch")
    if b and b.get("lock"):
        # "activity" = the newest of: last record written, lock file created. After a resume the raw file
        # can be hours old while the lock is fresh; counting only the raw file gave a false stall alert.
        ages = [a for a in (b.get("minutes_since_last_record"), b.get("minutes_since_lock")) if a is not None]
        if ages and min(ages) > expected["stall_minutes"]:
            A.append(f"BATCH STALLED?: lock file present but no new record and no new batch start for "
                     f"{min(ages):.0f} min")
    return A


def take_snapshot(args, expected) -> dict:
    snap: dict = {"utc": datetime.now(timezone.utc).isoformat()}
    root = native_root(args.endpoint)
    try:
        ver = http_json(root + "/api/version").get("version")
        tags = http_json(root + "/api/tags").get("models", [])
        ps = http_json(root + "/api/ps").get("models", [])
        snap["ours"] = {
            "endpoint": root, "version": ver,
            "digests": {m["name"]: m.get("digest") for m in tags},
            "loaded": [{"name": m["name"], "context_length": m.get("context_length"), "residency": residency(m),
                        "size": m.get("size"), "size_vram": m.get("size_vram")} for m in ps]}
    except Exception as e:  # noqa: BLE001
        snap["ours"] = {"error": f"{type(e).__name__}: {e}"}
    if args.other_ollama:
        try:
            r2 = native_root(args.other_ollama)
            snap["other"] = {"endpoint": r2, "version": http_json(r2 + "/api/version").get("version"),
                             "loaded": [{"name": m["name"], "size_vram": m.get("size_vram")}
                                        for m in http_json(r2 + "/api/ps").get("models", [])]}
        except Exception as e:  # noqa: BLE001
            snap["other"] = {"endpoint": args.other_ollama, "not_reachable": f"{type(e).__name__}", "loaded": []}
    snap["gpu"] = gpu_info()
    if args.out_dir:
        p = args.out_dir
        while not p.exists() and p != p.parent:
            p = p.parent
        snap["disk_free_gb"] = round(shutil.disk_usage(p).free / 2**30, 1)
        raw = args.out_dir / "raw_runs.jsonl"
        lock = args.out_dir / ".run_full.lock"
        snap["batch"] = {
            "lock": (args.out_dir / ".run_full.lock").exists(),
            "raw_records": (sum(1 for _ in raw.open(encoding="utf-8")) if raw.exists() else 0),
            "minutes_since_last_record": ((time.time() - raw.stat().st_mtime) / 60 if raw.exists() else None),
            "minutes_since_lock": ((time.time() - lock.stat().st_mtime) / 60 if lock.exists() else None)}
    snap["alerts"] = evaluate(snap, expected)
    return snap


def line(snap: dict) -> str:
    o = snap.get("ours", {})
    g = snap.get("gpu") or {}
    b = snap.get("batch") or {}
    loaded = ",".join(f"{m['name']}[{m['residency']}]" for m in o.get("loaded", [])) or "-"
    other = snap.get("other") or {}
    return (f"{snap['utc'][11:19]} v={o.get('version')} loaded={loaded} "
            f"other={other.get('version', 'n/a')}/{len(other.get('loaded', []))} "
            f"gpu={g.get('util_pct', '?')}%/{g.get('mem_used_mib', '?')}MiB/{g.get('temp_c', '?')}C "
            f"records={b.get('raw_records', '-')} disk={snap.get('disk_free_gb', '-')}GB "
            f"{'ALERTS=' + str(len(snap['alerts'])) if snap['alerts'] else 'ok'}")


def main() -> int:
    import run_full as rf
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--handoff", type=Path, default=Path("P1_to_P2_full_execution_v1.0"))
    ap.add_argument("--manifest-rel", default=rf.DEFAULT_MANIFEST_REL)
    ap.add_argument("--expected-ollama", default=rf.DEFAULT_OLLAMA)
    ap.add_argument("--endpoint", default=os.getenv("OLLAMA_BASE_URL", rf.DEFAULT_BASE_URL))
    ap.add_argument("--expected-endpoint", default=None, help="refuse to monitor any other endpoint")
    ap.add_argument("--other-ollama", default=rf.DEFAULT_OTHER_OLLAMA)
    ap.add_argument("--no-other-check", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None, help="batch output folder (progress, disk, lock)")
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--interval", type=float, default=60.0, help="seconds between checks")
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    ap.add_argument("--stall-minutes", type=float, default=30.0)
    ap.add_argument("--max-temp-c", type=int, default=85)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    if args.no_other_check:
        args.other_ollama = None
    if args.expected_endpoint and rf._hostport(args.expected_endpoint) != rf._hostport(args.endpoint):
        print(f"STOP: monitoring {rf._hostport(args.endpoint)} but expected endpoint is "
              f"{rf._hostport(args.expected_endpoint)}", file=sys.stderr)
        return 2
    try:
        rows = rf.read_rows(args.handoff / args.manifest_rel)
    except FileNotFoundError as e:
        print(f"STOP: manifest not found: {e.filename}", file=sys.stderr)
        return 2
    expected = {"ollama": args.expected_ollama, "min_free_gb": args.min_free_gb, "stall_minutes": args.stall_minutes,
                "max_temp_c": args.max_temp_c,
                "digests": {r["expected_resolved_model_name"]: r["expected_model_digest"] for r in rows}}
    log = args.log or ((args.out_dir.parent if args.out_dir else Path(".")) / "monitor_log.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"monitoring {args.endpoint} every {args.interval:.0f}s -> {log}  (read-only; Ctrl+C to stop)")
    try:
        while True:
            snap = take_snapshot(args, expected)
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(snap, ensure_ascii=False) + "\n")
            print(line(snap))
            for a in snap["alerts"]:
                print("   ALERT:", a)
            if args.once:
                if snap.get("ours", {}).get("error"):
                    return 2
                return 1 if snap["alerts"] else 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nmonitor stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
