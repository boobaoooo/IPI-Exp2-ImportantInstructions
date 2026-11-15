"""Chunk controller for the Full Experiment remaining batch (P2).

It does NOT change any scientific logic. It only calls the frozen run_full.py again and again, in this order:
    LLM-A block -> LLM-B block -> LLM-C block        (manifest order is kept inside every block)
one chunk (default 50 runs) per call, and between chunks it validates the output folder. Resume is the normal
resume of run_full.py (accepted runs are skipped, nothing is re-run or overwritten).

Per chunk:   run_full.py --slots <LLM> --chunk-size N   ->  exit code check  ->  validator  ->  progress check
It STOPS (and says why) when ANY of these happens:
  * run_full.py exit code != 0      (3 runtime drift/GPU residency, 4 disk/lock, 5 failures/attempts exhausted,
                                     6 data-contract break, 2 gate/args, 130 interrupted)
  * validator reports an ERROR, or any quarantined attempt exists
  * a chunk made NO progress twice in a row (accepted count unchanged)  -> persistent technical failure
  * the batch is confirmed stalled: no new record for --stall-minutes while run_full.py is running
    (the controller stops that child process and exits 7; evidence already on disk is kept)
  * Ctrl+C: run_full.py gets the same Ctrl+C and stops itself (exit 130); the controller waits up to 2 min for that
    and then ends the child. Evidence of finished runs stays; an unfinished run is simply redone on resume.
It does NOT stop for retryable technical failures that run_full.py still has attempts for: the next chunk
call simply retries them (max 4 attempts per run_id is enforced by run_full.py / the validator).

Exit codes of this script: 0 all requested blocks complete and validated | 8 stopped by a stop condition
(details printed, in controller log) | 7 stall | 2 bad arguments / cannot start.

Windows usage (own window, venv active, same env vars as the earlier chunks):
  python batch_controller.py --handoff P1_to_P2_full_execution_v1.1 ^
      --manifest-rel core\\full_experiment_execution_manifest_v1.1.csv ^
      --expected-manifest-sha 94760ff7985160b903b4d037940f2a3c1363d11e662eb77f0be66b68b30214ac ^
      --expected-endpoint http://127.0.0.1:11435 --out output_full --log-dir logs_full
Same command again = resume (safe at any time).
Files written: <log-dir>/chunk_<LLM>_<n>_<UTC>.log (full output of every run_full.py call),
               <log-dir>/controller_log.jsonl (one line per chunk: exit code, accepted before/after, seconds,
               validator summary), <log-dir>/final_validation.json (when everything is complete).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import run_full as rf
import validate_outputs as vo
from p2runner import RUNNER_VERSION

EXIT_OK, EXIT_ARGS, EXIT_STALL, EXIT_STOP = 0, 2, 7, 8
BLOCKS = ["LLM-A", "LLM-B", "LLM-C"]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_line(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def validate(out: Path, rows, args, *, slots=None, expect_complete=False) -> dict:
    return vo.validate_directory(out, rows, expected_ollama=args.expected_ollama, runner_version=RUNNER_VERSION,
                                 expect_complete=expect_complete, coverage_slots=set(slots) if slots else None)


def short(rep: dict) -> dict:
    return {k: rep[k] for k in ("raw_records", "accepted_runs", "quarantined", "missing_runs")} | \
           {"errors": len(rep["errors"]), "warnings": len(rep["warnings"])}


def run_chunk(cmd: list[str], logfile: Path, raw_path: Path, stall_min: float, poll_s: float) -> tuple[int | None, bool]:
    """Run one run_full.py call, tee its output to the screen and to logfile. Returns (exit code, stalled)."""
    stalled = False
    with logfile.open("w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        last = {"t": time.monotonic()}

        def pump() -> None:
            for ln in proc.stdout:  # type: ignore[union-attr]
                last["t"] = time.monotonic()
                sys.stdout.write(ln)
                sys.stdout.flush()
                lf.write(ln)
                lf.flush()
        th = threading.Thread(target=pump, daemon=True)
        th.start()
        try:
            while proc.poll() is None:
                time.sleep(poll_s)
                # activity = any line printed by run_full.py (one per finished run) OR a raw record newer than that
                quiet = (time.monotonic() - last["t"]) / 60
                if raw_path.exists():
                    quiet = min(quiet, (time.time() - raw_path.stat().st_mtime) / 60)
                if quiet > stall_min:
                    stalled = True
                    print(f"\nSTALL: no output and no new record for {quiet:.0f} min (> {stall_min:.0f}). "
                          f"Stopping this run_full.py call; evidence on disk is kept.", file=sys.stderr)
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
        except KeyboardInterrupt:
            print("\nCtrl+C: waiting for run_full.py to stop by itself (max 2 min)...", file=sys.stderr)
            try:
                proc.wait(timeout=120)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
            th.join(timeout=5)
            return 130, False
        th.join(timeout=10)
        return proc.returncode, stalled


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--handoff", type=Path, required=True)
    ap.add_argument("--manifest-rel", required=True)
    ap.add_argument("--expected-manifest-sha", required=True)
    ap.add_argument("--expected-endpoint", required=True)
    ap.add_argument("--expected-ollama", default=rf.DEFAULT_OLLAMA)
    ap.add_argument("--out", type=Path, default=Path("output_full"))
    ap.add_argument("--log-dir", type=Path, default=Path("logs_full"))
    ap.add_argument("--blocks", nargs="+", choices=BLOCKS, default=BLOCKS, help="default: LLM-A LLM-B LLM-C in that order")
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--max-chunks", type=int, default=0, help="stop cleanly after this many chunks (0 = no limit); "
                                                              "for a first supervised pass")
    ap.add_argument("--stall-minutes", type=float, default=30.0)
    ap.add_argument("--poll-seconds", type=float, default=10.0)
    ap.add_argument("--run-full", type=Path, default=Path(__file__).with_name("run_full.py"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="everything after --extra is passed to run_full.py unchanged (only for rehearsals)")
    args = ap.parse_args()
    if args.chunk_size < 1 or args.max_chunks < 0:
        print("STOP: --chunk-size must be >= 1 and --max-chunks >= 0", file=sys.stderr)
        return EXIT_ARGS
    try:
        rows = rf.read_rows(args.handoff / args.manifest_rel)
    except FileNotFoundError as e:
        print(f"STOP: manifest not found: {e.filename}", file=sys.stderr)
        return EXIT_ARGS
    ctl = args.log_dir / "controller_log.jsonl"
    raw_path = args.out / "raw_runs.jsonl"
    print(f"batch_controller: blocks={args.blocks} chunk={args.chunk_size} out={args.out} logs={args.log_dir}")
    log_line(ctl, {"utc": utc(), "event": "start", "blocks": args.blocks, "chunk_size": args.chunk_size,
                   "manifest_sha_expected": args.expected_manifest_sha, "runner_version": RUNNER_VERSION})
    chunks_done = 0
    for block in args.blocks:
        no_progress = 0
        while True:
            block_total = sum(1 for r in rows if r["source_llm_id"] == block)
            if args.out.exists():
                rep0 = validate(args.out, rows, args, slots=[block], expect_complete=True)
                before = block_total - rep0["missing_runs"]
                real_errors = [e for e in rep0["errors"] if "no accepted attempt" not in e]
                if real_errors or rep0["quarantined"]:
                    print(f"STOP before {block}: validator found {len(real_errors)} error(s) / "
                          f"{rep0['quarantined']} quarantined attempt(s):", *real_errors[:8], sep="\n  ", file=sys.stderr)
                    log_line(ctl, {"utc": utc(), "event": "stop_before_chunk", "block": block, "report": short(rep0),
                                   "errors": real_errors[:20]})
                    return EXIT_STOP
                if rep0["missing_runs"] == 0:
                    print(f"\n=== {block}: complete (all {rep0['coverage_rows']} runs accepted) ===")
                    log_line(ctl, {"utc": utc(), "event": "block_complete", "block": block, "report": short(rep0)})
                    break
            else:
                before = 0
            if args.max_chunks and chunks_done >= args.max_chunks:
                print(f"\nStopping cleanly after {chunks_done} chunk(s) (--max-chunks).")
                log_line(ctl, {"utc": utc(), "event": "max_chunks_reached", "chunks": chunks_done})
                return EXIT_OK
            chunks_done += 1
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            logfile = args.log_dir / f"chunk_{block}_{chunks_done:03d}_{stamp}.log"
            args.log_dir.mkdir(parents=True, exist_ok=True)
            cmd = [args.python, "-u", str(args.run_full), "--handoff", str(args.handoff), "--manifest-rel", args.manifest_rel,
                   "--expected-manifest-sha", args.expected_manifest_sha, "--expected-endpoint", args.expected_endpoint,
                   "--expected-ollama", args.expected_ollama, "--out", str(args.out), "--slots", block,
                   "--chunk-size", str(args.chunk_size), "--checkpoint-every", str(args.checkpoint_every), *args.extra]
            print(f"\n=== {block} chunk #{chunks_done}: accepted so far in this block {before}/{block_total} ===")
            t0 = time.monotonic()
            code, stalled = run_chunk(cmd, logfile, raw_path, args.stall_minutes, args.poll_seconds)
            secs = round(time.monotonic() - t0, 1)
            rep = validate(args.out, rows, args, slots=[block], expect_complete=True) if args.out.exists() else None
            after = (block_total - rep["missing_runs"]) if rep else 0
            real_errors = [e for e in (rep["errors"] if rep else []) if "no accepted attempt" not in e]
            log_line(ctl, {"utc": utc(), "event": "chunk", "block": block, "chunk": chunks_done, "exit_code": code,
                           "stalled": stalled, "seconds": secs, "accepted_before": before, "accepted_after": after,
                           "report": short(rep) if rep else None, "errors": real_errors[:20], "logfile": str(logfile)})
            print(f"--- chunk #{chunks_done} done: exit={code} accepted {before} -> {after} in {secs:.0f}s; "
                  f"validator errors={len(real_errors)} quarantined={rep['quarantined'] if rep else 0}")
            if stalled:
                return EXIT_STALL
            if code != 0:
                print(f"STOP: run_full.py exit code {code} (see {logfile}). Report to P1; do not continue.", file=sys.stderr)
                return EXIT_STOP
            if real_errors or (rep and rep["quarantined"]):
                print(f"STOP: validator found {len(real_errors)} error(s) / {rep['quarantined']} quarantined attempt(s) "
                      f"after the chunk:", *real_errors[:8], sep="\n  ", file=sys.stderr)
                return EXIT_STOP
            if after == before:
                no_progress += 1
                print(f"WARNING: chunk made no progress ({no_progress}/2).")
                if no_progress >= 2:
                    print("STOP: two chunks in a row without any new accepted run (persistent technical failure).",
                          file=sys.stderr)
                    log_line(ctl, {"utc": utc(), "event": "stop_no_progress", "block": block})
                    return EXIT_STOP
            else:
                no_progress = 0
    final = validate(args.out, rows, args, slots=args.blocks, expect_complete=True)
    (args.log_dir / "final_validation.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    log_line(ctl, {"utc": utc(), "event": "finished", "report": short(final)})
    full = len(args.blocks) == len(BLOCKS)
    print(f"\nFINAL: accepted {final['accepted_runs']}/{final['manifest_rows']}, errors={len(final['errors'])}, "
          f"quarantined={final['quarantined']}")
    if final["errors"] or final["quarantined"]:
        print("STOP: final validation not clean:", *final["errors"][:8], sep="\n  ", file=sys.stderr)
        return EXIT_STOP
    print("ALL REQUESTED BLOCKS COMPLETE AND VALIDATED." + ("" if full else " (not all three blocks were requested)"))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
