"""Output schema / data-contract validator for the Full Experiment outputs (read-only).

Two ways to use it:
  1. After the batch (or at any time), on a whole output folder:
       python validate_outputs.py --handoff <dir> --out output_full [--expect-complete] [--json report.json]
  2. run_full.py imports validate_attempt() and applies it to EVERY attempt right after the run,
     before the record is accepted (errors -> quarantine + STOP, exit 6).

Severity: ERROR = the record or folder breaks the contract with the frozen manifest or the data contract
(identity, manifest fields, generation settings, versions, payload/vectors, raw/eval separation, evidence files).
WARNING = worth a look but not a contract break (trajectory shape, timing, hidden-reasoning markers, evaluator
errors). Nothing here changes any data; it only reads.

Checks per attempt (raw + eval record vs. its manifest row):
  schema keys and types | canonical run_id + all manifest fields | attempt_id 1..4 | frozen generation settings
  | versions (runner, AgentDojo, reference commit, Ollama) | model identity (requested/resolved/digest)
  | payload SHA256 and injection vectors | benign rows carry no payload | no evaluation fields inside raw
  | eval record matches raw, evaluation_only, snapshots exist and are readable gzip | events/tool calls consistent.
Checks over the whole folder: JSON lines parse (partial last line reported) | (run_id, attempt) unique | every
run_id is in the manifest | attempt ledger consistent (explicit accepted attempt, at most one per run)
| code and prompt fingerprints identical across the batch | same logical condition -> same initial environment
hash across LLMs | resource_log entry per attempt | coverage (with --expect-complete: every row accepted).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import sys
from datetime import datetime
from pathlib import Path

RAW_KEYS = {
    "schema_version", "run_id", "scenario_id", "scenario_group_id", "suite", "user_task_id", "injection_task_id",
    "condition", "repetition_id", "attempt_id", "split", "source_llm", "model_id", "provider", "started_at",
    "completed_at", "reproducibility", "injection", "events", "llm_calls", "usage_total", "execution_status",
    "termination_reason", "final_output_present", "steps_used", "runner_traceback"}
REPRO_KEYS = {
    "benchmark_version", "agentdojo_version", "agentdojo_git_commit", "agentdojo_reference_commit",
    "runner_version", "runner_source_sha256", "sdk", "api_route", "system_prompt_name", "system_prompt_sha256",
    "agent_workflow", "available_tools", "tool_permissions", "tool_choice", "generation_settings",
    "model_identity", "max_agent_steps", "timeout_s", "retry_policy", "initial_environment_sha256",
    "attack_template_id", "attack_payload_sha256", "selected_injection_vector", "manifest_source"}
EVAL_ONLY_KEYS = {"evaluation_only", "user_task_succeeded", "benchmark_security", "final_output_text",
                  "snapshots", "evaluator_error"}
STATUSES = {"completed", "step_limit", "timeout", "model_api_error", "runner_error", "interrupted"}
RERUN = {"model_api_error", "timeout", "runner_error", "interrupted"}
EVENT_TYPES = {"tool_call", "tool_result", "agent_message", "agent_final", "injected_content_delivered",
               "execution_error"}
SCHEMA_VERSION = "raw_run/0.1"
MAX_ATTEMPTS = 4
FROZEN_GEN = {"temperature": 0.0, "seed": 42, "parallel_tool_calls": False, "context_length_env": 16384,
              "thinking_override": "omitted", "top_p": "inherited_default", "top_k": "inherited_default",
              "num_predict": "inherited_default", "max_tokens": "inherited_default"}


def _vec(s: str) -> list[str]:
    return sorted(v.strip() for v in re.split(r"[;,|]", s or "") if v.strip())


def _ts(s):
    try:
        return datetime.fromisoformat(s)
    except Exception:  # noqa: BLE001
        return None


def validate_attempt(raw: dict, ev: dict | None, row: dict, out_dir: Path, *,
                     expected_ollama: str, runner_version: str,
                     agentdojo_version: str = "0.1.35",
                     reference_commit: str = "a75aba7631d3ca5fb7ab938965c97ead2f9ff84b",
                     check_files: bool = True) -> tuple[list[str], list[str]]:
    """(errors, warnings) for one attempt. Pure function of the record, its manifest row and the files."""
    E: list[str] = []
    W: list[str] = []
    rid = row["run_id"]

    def e(msg):
        E.append(msg)

    def w(msg):
        W.append(msg)

    miss = RAW_KEYS - set(raw)
    if miss:
        e(f"raw: missing keys {sorted(miss)}")
        return E, W                      # nothing else can be trusted
    if raw["schema_version"] != SCHEMA_VERSION:
        e(f"raw: schema_version {raw['schema_version']!r} != {SCHEMA_VERSION!r}")
    leaked = EVAL_ONLY_KEYS & set(raw)
    if leaked:
        e(f"raw: evaluation-only field(s) inside the raw record (data-contract leak): {sorted(leaked)}")

    # --- identity vs manifest
    cond = (row["condition"] or "").strip()
    pairs = {
        "run_id": rid, "suite": row["suite"], "user_task_id": row["user_task_id"],
        "scenario_group_id": row["scenario_group_id"], "condition": cond, "split": row["split"],
        "source_llm": row["source_llm_id"], "model_id": row["requested_model_name"], "provider": "ollama",
        "scenario_id": f"{row['scenario_group_id']}__{cond}",
        "injection_task_id": (row.get("injection_task_id") or "").strip() or None,
    }
    for k, want in pairs.items():
        if raw[k] != want:
            e(f"raw.{k} = {raw[k]!r}, manifest says {want!r}")
    rep = int((row.get("repetition_id") or "rep1").replace("rep", "") or 1)
    if raw["repetition_id"] != rep:
        e(f"raw.repetition_id = {raw['repetition_id']!r}, manifest says {rep}")
    a = raw["attempt_id"]
    if not isinstance(a, int) or isinstance(a, bool) or not 1 <= a <= MAX_ATTEMPTS:
        e(f"raw.attempt_id = {a!r} (must be an integer 1..{MAX_ATTEMPTS})")
    if raw["execution_status"] not in STATUSES:
        e(f"raw.execution_status = {raw['execution_status']!r} is not a known status")

    # --- reproducibility block
    rp = raw["reproducibility"]
    miss = REPRO_KEYS - set(rp)
    if miss:
        e(f"raw.reproducibility: missing keys {sorted(miss)}")
        return E, W
    if rp["runner_version"] != runner_version or rp["runner_version"] != (row.get("runner_version") or "").strip():
        e(f"runner_version {rp['runner_version']!r} (code {runner_version!r}, manifest {row.get('runner_version')!r})")
    if rp["agentdojo_version"] != agentdojo_version:
        e(f"agentdojo_version {rp['agentdojo_version']!r} != {agentdojo_version!r}")
    if rp["agentdojo_reference_commit"] != reference_commit:
        e(f"agentdojo_reference_commit {rp['agentdojo_reference_commit']!r} != {reference_commit!r}")
    if rp["agentdojo_git_commit"] is not None:
        e(f"agentdojo_git_commit must stay null, got {rp['agentdojo_git_commit']!r}")
    if rp["benchmark_version"] != "v1":
        e(f"benchmark_version {rp['benchmark_version']!r} != 'v1'")
    gs = rp["generation_settings"]
    for k, want in FROZEN_GEN.items():
        if gs.get(k) != want or type(gs.get(k)) is not type(want):
            e(f"generation_settings.{k} = {gs.get(k)!r}, frozen value {want!r}")
    extra = set(gs) - set(FROZEN_GEN) - {"base_url"}
    if extra:
        e(f"generation_settings has unexpected keys {sorted(extra)} (possible changed request settings)")
    mi = rp["model_identity"]
    for k, want in (("requested_model", row["requested_model_name"]),
                    ("resolved_model", row["expected_resolved_model_name"]),
                    ("ollama_version", expected_ollama)):
        if mi.get(k) != want:
            e(f"model_identity.{k} = {mi.get(k)!r}, expected {want!r}")
    if (mi.get("model_digest") or "").split("sha256:")[-1] != row["expected_model_digest"]:
        e(f"model_identity.model_digest = {mi.get('model_digest')!r}, expected {row['expected_model_digest']!r}")
    if mi.get("identity_error"):
        e(f"model_identity.identity_error = {mi['identity_error']!r}")
    for k, want in (("max_agent_steps", 15), ("timeout_s", 180), ("tool_choice", "auto"),
                    ("api_route", "chat.completions")):
        if rp[k] != want:
            e(f"reproducibility.{k} = {rp[k]!r}, frozen value {want!r}")
    if not rp["initial_environment_sha256"]:
        w("initial_environment_sha256 is empty (environment not built?)")

    # --- payload / vectors
    inj = raw["injection"]
    if cond == "ipi":
        if rp["attack_payload_sha256"] != row["payload_sha256"]:
            e(f"attack_payload_sha256 {rp['attack_payload_sha256']!r} != manifest {row['payload_sha256']!r}")
        if sorted(rp["selected_injection_vector"] or []) != _vec(row["injection_vector_ids"]):
            e(f"selected_injection_vector {rp['selected_injection_vector']!r} != manifest {_vec(row['injection_vector_ids'])!r}")
        if rp["attack_template_id"] != ((row.get("attack_template") or "").strip() or None):
            e(f"attack_template_id {rp['attack_template_id']!r} != manifest {row.get('attack_template')!r}")
        if inj.get("ipi_present") is not True:
            e("injection.ipi_present must be true for an IPI row")
        if inj.get("injection_inserted") is not True and raw["execution_status"] != "runner_error":
            e(f"injection.injection_inserted = {inj.get('injection_inserted')!r} for an IPI row (payload not in environment)")
    else:
        if rp["attack_payload_sha256"] is not None or rp["selected_injection_vector"]:
            e("benign row carries a payload hash or injection vectors")
        if inj.get("ipi_present") is not False:
            e("injection.ipi_present must be false for a benign row")

    # --- trajectory consistency (warnings: shape, not contract)
    events = raw["events"]
    if [x.get("event_index") for x in events] != list(range(len(events))):
        w("events: event_index is not 0..n-1 in order")
    bad_types = {x.get("event_type") for x in events} - EVENT_TYPES
    if bad_types:
        w(f"events: unknown event types {sorted(bad_types)}")
    calls = [x.get("tool_call_id") for x in events if x.get("event_type") == "tool_call"]
    results = [x.get("tool_call_id") for x in events if x.get("event_type") == "tool_result"]
    if sorted(calls) != sorted(results) and raw["execution_status"] in ("completed", "step_limit"):
        w(f"events: tool_call ids {len(calls)} vs tool_result ids {len(results)} do not pair up")
    if raw["usage_total"].get("llm_calls") != len(raw["llm_calls"]):
        w("usage_total.llm_calls != len(llm_calls)")
    if raw["execution_status"] == "completed" and not any(x.get("event_type") == "agent_final" for x in events):
        w("status completed but no agent_final event")
    n_ev = len(events)
    for item in inj.get("exposure_evidence") or []:
        if not isinstance(item.get("event_index"), int) or not 0 <= item["event_index"] < n_ev:
            w("exposure_evidence points to an event_index that does not exist")
    if '<think>' in json.dumps(events, ensure_ascii=False).lower():
        w("a <think> marker appears in the trajectory (hidden reasoning must not enter the raw log)")
    t0, t1 = _ts(raw["started_at"]), _ts(raw["completed_at"])
    if not t0 or not t1 or t1 < t0:
        w("started_at/completed_at missing, unparsable or out of order")
    elif (t1 - t0).total_seconds() > rp["timeout_s"] + 90:
        w(f"run took {(t1 - t0).total_seconds():.0f}s (> timeout {rp['timeout_s']}s + 90s slack)")

    # --- evaluation record
    if ev is None:
        e("no evaluation record for this attempt")
    else:
        if ev.get("run_id") != rid or ev.get("attempt_id") != a:
            e(f"eval record is for ({ev.get('run_id')!r}, {ev.get('attempt_id')!r}), raw is ({rid!r}, {a!r})")
        if ev.get("evaluation_only") is not True:
            e("eval record is not marked evaluation_only")
        if ev.get("evaluator_error"):
            w(f"evaluator_error: {ev['evaluator_error']}")
        if "user_task_succeeded" in ev:
            if not isinstance(ev["user_task_succeeded"], bool):
                e("eval.user_task_succeeded is not a boolean")
            if pairs["injection_task_id"] and not isinstance(ev.get("benchmark_security"), bool):
                e("eval.benchmark_security missing/not boolean for a row with an injection task")
            sn = ev.get("snapshots") or {}
            if set(sn) != {"pre", "post"}:
                e("eval record has evaluation results but no pre/post snapshot entries")
            elif check_files:
                want_stem = f"{rid}__att{a}"
                for tag, rel in sn.items():
                    if rel != f"snapshots/{want_stem}.{tag}.json.gz":
                        e(f"snapshot path {rel!r} does not follow the run_id + __att{a} naming")
                        continue
                    f = out_dir / rel
                    try:
                        with gzip.open(f, "rt") as fh:
                            json.load(fh)
                    except FileNotFoundError:
                        e(f"snapshot file missing: {rel}")
                    except Exception as exc:  # noqa: BLE001
                        e(f"snapshot unreadable ({type(exc).__name__}): {rel}")
        elif raw["execution_status"] in ("completed", "step_limit"):
            w("completed run without evaluation results (see evaluator_error)")
    return E, W


def _read_jsonl(path: Path, errors: list[str]) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    lines = path.read_text(encoding="utf-8").split("\n")
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            where = "LAST line (partial write after a crash?)" if i >= len([x for x in lines if x.strip()]) else f"line {i}"
            errors.append(f"{path.name}: unparsable JSON at {where}: {exc.msg}")
    return out


def validate_directory(out: Path, rows: list[dict], *, expected_ollama: str, runner_version: str,
                       expect_complete: bool = False, coverage_slots: set[str] | None = None) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    by_id = {r["run_id"]: r for r in rows}
    raw = _read_jsonl(out / "raw_runs.jsonl", errors)
    evs = _read_jsonl(out / "eval_evidence.jsonl", errors)
    ledger = _read_jsonl(out / "attempt_ledger.jsonl", errors)
    q_raw = _read_jsonl(out / "quarantine_raw.jsonl", errors)
    res_log = _read_jsonl(out / "resource_log.jsonl", errors)
    meta = _read_jsonl(out / "run_metadata_full.jsonl", errors)

    key = lambda r: (r.get("run_id"), r.get("attempt_id"))  # noqa: E731
    for name, recs in (("raw_runs", raw), ("eval_evidence", evs), ("attempt_ledger", ledger)):
        c = collections.Counter(key(r) for r in recs if name != "attempt_ledger" or r.get("outcome") != "exhausted_no_accepted_attempt")
        for k, n in c.items():
            if n > 1:
                errors.append(f"{name}.jsonl: (run_id, attempt) {k} appears {n} times (duplicate/overwritten evidence)")
    ev_by = {key(r): r for r in evs}
    led_by = {key(r): r for r in ledger if r.get("outcome") != "exhausted_no_accepted_attempt"}
    res_keys = {key(r) for r in res_log}

    n_err_rec = 0
    for r in raw:
        row = by_id.get(r.get("run_id"))
        if row is None:
            errors.append(f"raw_runs.jsonl: run_id not in the manifest: {r.get('run_id')!r}")
            continue
        E, W = validate_attempt(r, ev_by.get(key(r)), row, out, expected_ollama=expected_ollama,
                                runner_version=runner_version)
        n_err_rec += bool(E)
        errors += [f"{r['run_id']} att{r.get('attempt_id')}: {m}" for m in E]
        warnings += [f"{r['run_id']} att{r.get('attempt_id')}: {m}" for m in W]
        L = led_by.get(key(r))
        if L is None:
            errors.append(f"{r['run_id']} att{r.get('attempt_id')}: no attempt_ledger entry")
        else:
            want = "failed_retryable" if r.get("execution_status") in RERUN else "accepted"
            if L.get("outcome") != want:
                errors.append(f"{r['run_id']} att{r.get('attempt_id')}: ledger outcome {L.get('outcome')!r}, "
                              f"status {r.get('execution_status')!r} implies {want!r}")
        if key(r) not in res_keys:
            warnings.append(f"{r['run_id']} att{r.get('attempt_id')}: no resource_log entry")
    for k in ev_by:
        if k not in {key(r) for r in raw} and k not in {key(r) for r in q_raw}:
            errors.append(f"eval_evidence.jsonl: record {k} has no raw record")

    # accepted attempts per run
    acc: dict[str, list[int]] = collections.defaultdict(list)
    for r in raw:
        if r.get("execution_status") not in RERUN:
            acc[r["run_id"]].append(r["attempt_id"])
    for rid, atts in acc.items():
        if len(atts) > 1:
            errors.append(f"{rid}: more than one accepted attempt {sorted(atts)} (rerun although accepted)")
    attempts_per = collections.Counter(r["run_id"] for r in raw)
    for rid, n in attempts_per.items():
        if n > MAX_ATTEMPTS:
            errors.append(f"{rid}: {n} attempts recorded (limit {MAX_ATTEMPTS})")

    # fingerprints constant across the batch
    for fld in ("runner_source_sha256", "system_prompt_sha256"):
        vals = {r["reproducibility"].get(fld) for r in raw if "reproducibility" in r}
        if len(vals) > 1:
            errors.append(f"{fld} differs inside one batch ({len(vals)} variants): code/prompt changed mid-batch")
    # same logical condition -> same initial environment across LLMs / attempts
    env = collections.defaultdict(set)
    for r in raw:
        row = by_id.get(r.get("run_id"))
        if row and "reproducibility" in r:
            env[row["logical_condition_id"]].add(r["reproducibility"].get("initial_environment_sha256"))
    for k, s in env.items():
        if len(s) > 1:
            errors.append(f"logical condition {k}: {len(s)} different initial environment hashes (fresh-env contract)")
    if len({m.get("manifest_sha256") for m in meta}) > 1:
        errors.append("run_metadata_full.jsonl: more than one manifest SHA256 used in this folder")

    # coverage
    accepted_ids = set(acc)
    cover_rows = [r for r in rows if not coverage_slots or r["source_llm_id"] in coverage_slots]
    missing = [r["run_id"] for r in cover_rows if r["run_id"] not in accepted_ids]
    if expect_complete and missing:
        errors.append(f"{len(missing)} manifest row(s) have no accepted attempt, e.g. {missing[:3]}")
    if q_raw:
        warnings.append(f"{len(q_raw)} quarantined attempt(s) exist (not in raw_runs.jsonl); P1 decides")
    referenced = {p for e in evs for p in (e.get("snapshots") or {}).values()}
    snap_dir = out / "snapshots"
    if snap_dir.exists():
        orphan = [f.name for f in snap_dir.iterdir() if f"snapshots/{f.name}" not in referenced]
        if orphan:
            warnings.append(f"{len(orphan)} snapshot file(s) not referenced by any eval record (quarantined/orphan), "
                            f"e.g. {orphan[:2]}")
    per = collections.Counter()
    for rid in accepted_ids:
        if rid in by_id:
            r = by_id[rid]
            per[(r["source_llm_id"], r["condition"], r["split"])] += 1
    return {"raw_records": len(raw), "eval_records": len(evs), "ledger_entries": len(ledger),
            "quarantined": len(q_raw), "accepted_runs": len(accepted_ids), "manifest_rows": len(rows),
            "missing_runs": len(missing), "coverage_rows": len(cover_rows), "records_with_errors": n_err_rec,
            "errors": errors, "warnings": warnings,
            "accepted_by_llm_condition_split": {"|".join(k): v for k, v in sorted(per.items())}}


def main() -> int:
    import run_full as rf
    from p2runner import RUNNER_VERSION
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--handoff", type=Path, default=Path("P1_to_P2_full_execution_v1.0"))
    ap.add_argument("--manifest-rel", default=rf.DEFAULT_MANIFEST_REL)
    ap.add_argument("--out", type=Path, default=Path("output_full"))
    ap.add_argument("--expected-ollama", default=rf.DEFAULT_OLLAMA)
    ap.add_argument("--runner-version", default=RUNNER_VERSION)
    ap.add_argument("--expect-complete", action="store_true", help="every manifest row must have an accepted attempt")
    ap.add_argument("--slots", nargs="+", choices=["LLM-A", "LLM-B", "LLM-C"],
                    help="with --expect-complete: only require these source LLMs (e.g. after finishing one block)")
    ap.add_argument("--json", type=Path, help="also write the full report here")
    ap.add_argument("--max-print", type=int, default=40)
    args = ap.parse_args()
    try:
        rows = rf.read_rows(args.handoff / args.manifest_rel)
    except FileNotFoundError as exc:
        print(f"STOP: file not found: {exc.filename}", file=sys.stderr)
        return 2
    if not args.out.exists():
        print(f"STOP: output folder not found: {args.out}", file=sys.stderr)
        return 2
    rep = validate_directory(args.out, rows, expected_ollama=args.expected_ollama,
                             runner_version=args.runner_version, expect_complete=args.expect_complete,
                             coverage_slots=set(args.slots) if args.slots else None)
    print(f"raw={rep['raw_records']} eval={rep['eval_records']} ledger={rep['ledger_entries']} "
          f"quarantined={rep['quarantined']} | accepted runs {rep['accepted_runs']}/{rep['manifest_rows']} "
          f"(missing {rep['missing_runs']} of {rep['coverage_rows']} required) | attempts with errors: {rep['records_with_errors']}")
    for title, items in (("ERRORS", rep["errors"]), ("WARNINGS", rep["warnings"])):
        print(f"\n{title}: {len(items)}")
        for m in items[:args.max_print]:
            print("  -", m)
        if len(items) > args.max_print:
            print(f"  ... and {len(items) - args.max_print} more (use --json for the full list)")
    if args.json:
        args.json.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nVALIDATION " + ("FAILED" if rep["errors"] else "PASSED"))
    return 1 if rep["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
