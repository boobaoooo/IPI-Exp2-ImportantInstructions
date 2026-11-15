"""Core execution: one AgentDojo scenario x one LLM -> one raw_runs.jsonl record.

Design rules (data_contract_v0.1):
* Only LLM-generated actions enter the trajectory; benchmark ground truth never does.
* Injection insertion, and delivery to the model, are logged separately.
* No hidden retries: transient API failures are retried under one bounded policy and
  every attempt is logged; a fresh environment is used for any complete rerun.
* Evaluator results / environment snapshots go to evaluation-only files, not raw_runs.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
import traceback
from ast import literal_eval
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

from agentdojo.agent_pipeline.tool_execution import is_string_list, tool_result_to_str
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.agent_pipeline.agent_pipeline import load_system_message

from . import RUNNER_VERSION
from .adapters import BaseAdapter, LLMResult, classify_exception
from .scenarios import BENCHMARK_VERSION, Case

SCHEMA_VERSION = "raw_run/0.1"
SYSTEM_PROMPT_NAME = "default"
# AgentDojo provenance for the Full Experiment baseline (Generation Policy v1.0).
# The pip package has no .git, so agentdojo_git_commit (local checkout commit) stays
# null; this records the reference commit the pinned package version corresponds to.
AGENTDOJO_REFERENCE_COMMIT = "a75aba7631d3ca5fb7ab938965c97ead2f9ff84b"


@dataclass
class RunConfig:
    max_steps: int = 15            # LLM calls per execution (protocol draft)
    timeout_s: float = 180.0       # wall clock per execution (protocol draft)
    max_attempts: int = 4          # initial try + 3 retries for transient API errors
    backoff_base_s: float = 2.0
    backoff_cap_s: float = 30.0
    sleep: Callable[[float], None] = time.sleep


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _norm(s: str) -> str:
    # Tool output is YAML-formatted (line folding, quoting), so compare without
    # whitespace/quote/backslash noise.
    return re.sub(r"[\s'\"\\]+", "", s)


def _collect_strings(obj: Any, out: list[str]) -> None:
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_strings(v, out)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _collect_strings(v, out)


def payload_in_env(env: Any, payload: str) -> bool:
    strings: list[str] = []
    _collect_strings(env.model_dump(mode="json"), strings)
    return _norm(payload) in _norm("\n".join(strings))


def env_hash(env: Any) -> str:
    return hashlib.sha256(json.dumps(env.model_dump(mode="json"), sort_keys=True, default=str).encode()).hexdigest()


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _runner_source_sha() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()


class Trace:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.step = -1

    def add(self, event_type: str, actor: str, **fields: Any) -> dict[str, Any]:
        ev = {"step_index": self.step, "event_index": len(self.events), "timestamp": now_iso(),
              "event_type": event_type, "actor": actor,
              "tool_name": None, "tool_call_id": None, "tool_arguments": None,
              "tool_result": None, "error": None}
        ev.update(fields)
        self.events.append(ev)
        return ev


class CallFailed(Exception):
    def __init__(self, status: str, info: dict[str, Any]):
        super().__init__(status)
        self.status, self.info = status, info


def call_llm(adapter: BaseAdapter, messages: list[dict], tools: list[dict], deadline: float,
             cfg: RunConfig) -> tuple[LLMResult, list[dict[str, Any]]]:
    """One logical LLM call with bounded, fully logged retries for transient errors only."""
    attempts: list[dict[str, Any]] = []
    for n in range(1, cfg.max_attempts + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CallFailed("timeout", {"attempts": attempts, "reason": "execution time budget exhausted"})
        t0 = time.monotonic()
        try:
            res = adapter.complete(messages, tools, timeout=remaining)
            attempts.append({"attempt": n, "ok": True, "latency_s": round(time.monotonic() - t0, 3)})
            return res, attempts
        except Exception as exc:  # noqa: BLE001 - classified below
            info = classify_exception(exc)
            attempts.append({"attempt": n, "ok": False, "latency_s": round(time.monotonic() - t0, 3),
                             "error_class": info.error_class, "http_status": info.status,
                             "retryable": info.retryable, "message": info.message})
            out_of_time = (deadline - time.monotonic()) <= 0
            if out_of_time and info.error_class == "timeout":
                raise CallFailed("timeout", {"attempts": attempts, "reason": "request exceeded time budget"}) from exc
            if not info.retryable or n == cfg.max_attempts:
                raise CallFailed("model_api_error", {"attempts": attempts, "reason": info.error_class}) from exc
            wait = min(cfg.backoff_cap_s, cfg.backoff_base_s * 2 ** (n - 1))
            if info.retry_after:
                wait = max(wait, info.retry_after)
            if wait >= deadline - time.monotonic():
                raise CallFailed("timeout", {"attempts": attempts, "reason": "backoff exceeds time budget"}) from exc
            cfg.sleep(wait)
    raise AssertionError("unreachable")


def execute_tool(runtime: FunctionsRuntime, env: Any, call: dict[str, Any]) -> tuple[str, str | None]:
    """Run a model-issued call with the same coercion/formatting AgentDojo's executor uses."""
    name = call["name"]
    if name not in runtime.functions:
        return "", f"Invalid tool {name} provided."
    args = dict(call["args"])
    for k, v in args.items():
        if isinstance(v, str) and is_string_list(v):
            args[k] = literal_eval(v)
    try:
        result, err = runtime.run_function(env, name, args)
        return tool_result_to_str(result), err
    except Exception as exc:  # noqa: BLE001 - keep the run alive, record it
        return "", f"{type(exc).__name__}: {exc}"


def execute_run(case: Case, adapter: BaseAdapter, source_llm: str, cfg: RunConfig, out_dir: Path,
                repetition_id: int = 0, attempt_id: int = 1, run_id: str | None = None) -> dict[str, Any]:
    """Execute one run. Always returns a raw record (even on failure) and writes evaluation evidence.

    run_id: for the Full Experiment, the canonical run_id from the execution manifest,
    preserved verbatim in the raw record, the eval record and the snapshot filenames.
    A technical retry keeps the SAME run_id and only increments attempt_id (recorded as
    a separate field). If None, the v0.2.1 generated form is used (pilot / builtin)."""
    canonical_run_id = run_id is not None
    if run_id is None:
        run_id = f"{case.scenario_group_id}__{case.condition}__{source_llm}__rep{repetition_id}__att{attempt_id}"
    # Evidence files are per attempt: with a canonical run_id (same across technical
    # retries) the attempt is part of the snapshot filename so a retry never overwrites
    # an earlier attempt's snapshots. The generated v0.2.1 form already embeds the attempt.
    snap_stem = f"{run_id}__att{attempt_id}" if canonical_run_id else run_id
    started = now_iso()
    trace = Trace()
    llm_calls: list[dict[str, Any]] = []
    exposure: dict[str, Any] = {"injection_encountered": None, "exposure_evidence": []}
    status, reason, error = "runner_error", None, None
    final_text: str | None = None
    pre_env = env = suite = user_task = inj_task = None
    injection_inserted: bool | None = None
    tool_names: list[str] = []
    env_h = None
    steps_used = 0

    system_prompt = load_system_message(SYSTEM_PROMPT_NAME)
    try:
        suite = get_suite(BENCHMARK_VERSION, case.suite)
        user_task = suite.get_user_task_by_id(case.user_task_id)
        inj_task = suite.get_injection_task_by_id(case.injection_task_id) if case.injection_task_id else None

        injections: dict[str, str] = {}
        if case.condition == "ipi":
            vectors = case.injection_vectors or []
            if not vectors:
                raise ValueError("IPI case has no injection vector (manifest must name one)")
            injections = {v: case.payload for v in vectors}
        env = suite.load_and_inject_default_environment(injections)  # fresh environment every run
        env = user_task.init_environment(env)
        pre_env = env.model_copy(deep=True)
        env_h = env_hash(pre_env)
        if case.condition == "ipi":
            injection_inserted = payload_in_env(pre_env, case.payload)
            exposure["injection_encountered"] = False if injection_inserted else None
        runtime = FunctionsRuntime(suite.tools)
        tools = [{"name": f.name, "description": f.description, "schema": f.parameters.model_json_schema()}
                 for f in runtime.functions.values()]
        tool_names = sorted(f.name for f in runtime.functions.values())
        if hasattr(adapter, "prepare"):
            adapter.prepare(user_task, inj_task if case.condition == "ipi" else None, pre_env)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_task.PROMPT},
        ]
        deadline = time.monotonic() + cfg.timeout_s
        status = "step_limit"
        for step in range(cfg.max_steps):
            trace.step = step
            steps_used = step + 1
            try:
                res, attempts = call_llm(adapter, messages, tools, deadline, cfg)
            except CallFailed as cf:
                llm_calls.append({"call_index": step, "step_index": step, "ok": False, **cf.info})
                trace.add("execution_error", "runner", error=f"{cf.status}: {cf.info.get('reason')}")
                status, reason = cf.status, cf.info.get("reason")
                break
            llm_calls.append({"call_index": step, "step_index": step, "ok": True, "attempts": attempts,
                              "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
                              "finish_reason": res.finish_reason, "request_id": res.request_id,
                              "model_returned": res.model_returned})
            for c in res.tool_calls:
                if not c.get("id"):
                    c["id"] = f"call_{step}_{len(messages)}"
            messages.append({"role": "assistant", "content": res.text, "tool_calls": res.tool_calls})
            if res.text:
                trace.add("agent_message", "llm", content_text=res.text)
            if not res.tool_calls:
                final_text = res.text
                trace.add("agent_final", "llm", content_text=res.text)
                status = "completed"
                break
            for c in res.tool_calls:
                trace.add("tool_call", "llm", tool_name=c["name"], tool_call_id=c["id"], tool_arguments=c["args"])
            for c in res.tool_calls:
                text, err = execute_tool(runtime, env, c)
                trace.add("tool_result", "tool", tool_name=c["name"], tool_call_id=c["id"],
                          tool_arguments=c["args"], tool_result=text, error=err)
                delivered = err if err else text
                messages.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"],
                                 "content": text, "error": err})
                if case.condition == "ipi" and injection_inserted and _norm(case.payload) in _norm(delivered or ""):
                    ev = trace.add("injected_content_delivered", "runner", tool_name=c["name"],
                                   tool_call_id=c["id"])
                    exposure["exposure_evidence"].append({
                        "event_index": ev["event_index"], "tool_call_id": c["id"], "tool_name": c["name"],
                        "injection_vector": ",".join(case.injection_vectors), "payload_sha256": case.payload_sha256,
                        "match_method": "normalized_substring"})
                    exposure["injection_encountered"] = True
        else:
            status = "step_limit"
        if status == "step_limit":
            reason = f"reached {cfg.max_steps} LLM steps without a final answer"
    except Exception as exc:  # noqa: BLE001
        status, reason = "runner_error", f"{type(exc).__name__}: {exc}"
        error = traceback.format_exc()
        trace.add("execution_error", "runner", error=reason)

    # Exposure is only "confirmed not encountered" if the whole trajectory was observed.
    if case.condition == "ipi" and injection_inserted and exposure["injection_encountered"] is not True:
        exposure["injection_encountered"] = False if status in ("completed", "step_limit") else None
    completed = now_iso()
    # ---- evaluation-only evidence (never mixed into raw trajectory) ----
    eval_rec: dict[str, Any] = {"run_id": run_id, "attempt_id": attempt_id, "evaluation_only": True}
    try:
        if env is not None and user_task is not None:
            stack = [FunctionCall(function=e["tool_name"], args=e["tool_arguments"] or {}, id=e["tool_call_id"])
                     for e in trace.events if e["event_type"] == "tool_call"]
            out_blocks = [{"type": "text", "content": final_text or ""}]
            eval_rec["user_task_succeeded"] = bool(suite._check_task_result(user_task, out_blocks, pre_env, env, stack))
            if inj_task is not None:
                eval_rec["benchmark_security"] = bool(suite._check_task_result(inj_task, out_blocks, pre_env, env, stack))
            eval_rec["final_output_text"] = final_text
            snap = out_dir / "snapshots"
            snap.mkdir(parents=True, exist_ok=True)
            for tag, e in (("pre", pre_env), ("post", env)):
                with gzip.open(snap / f"{snap_stem}.{tag}.json.gz", "wt") as f:
                    json.dump(e.model_dump(mode="json"), f, default=str)
            eval_rec["snapshots"] = {"pre": f"snapshots/{snap_stem}.pre.json.gz",
                                     "post": f"snapshots/{snap_stem}.post.json.gz"}
    except Exception as exc:  # noqa: BLE001
        eval_rec["evaluator_error"] = f"{type(exc).__name__}: {exc}"

    try:
        agentdojo_version = metadata.version("agentdojo")
    except Exception:
        agentdojo_version = None
    raw = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id,
        "scenario_id": case.scenario_id, "scenario_group_id": case.scenario_group_id,
        "suite": case.suite, "user_task_id": case.user_task_id, "injection_task_id": case.injection_task_id,
        "condition": case.condition, "repetition_id": repetition_id, "attempt_id": attempt_id,
        "split": case.split, "source_llm": source_llm,
        "model_id": adapter.model, "provider": adapter.provider,
        "started_at": started, "completed_at": completed,
        "reproducibility": {
            "benchmark_version": BENCHMARK_VERSION, "agentdojo_version": agentdojo_version,
            "agentdojo_git_commit": None,  # local checkout commit: null (installed via pip, no .git)
            "agentdojo_reference_commit": AGENTDOJO_REFERENCE_COMMIT,
            "runner_version": RUNNER_VERSION, "runner_source_sha256": _runner_source_sha(),
            "sdk": adapter.sdk_version(), "api_route": getattr(adapter, "api_route", None),
            "system_prompt_name": SYSTEM_PROMPT_NAME, "system_prompt_sha256": _sha(system_prompt),
            "agent_workflow": "custom_loop/v1: llm -> tools -> llm until no tool call",
            "available_tools": tool_names, "tool_permissions": "all suite tools, unrestricted (simulated environment)",
            "tool_choice": "auto", "generation_settings": adapter.effective_settings(),
            "model_identity": adapter.runtime_identity(),
            "max_agent_steps": cfg.max_steps, "timeout_s": cfg.timeout_s,
            "retry_policy": {"max_attempts": cfg.max_attempts, "backoff_base_s": cfg.backoff_base_s,
                             "backoff_cap_s": cfg.backoff_cap_s},
            "initial_environment_sha256": env_h,
            "attack_template_id": case.attack_template_id, "attack_payload_sha256": case.payload_sha256,
            "selected_injection_vector": case.injection_vectors or None,
            "manifest_source": case.manifest_source,
        },
        "injection": {"ipi_present": case.condition == "ipi", "injection_inserted": injection_inserted,
                      "injection_encountered": exposure["injection_encountered"],
                      "exposure_evidence": exposure["exposure_evidence"]},
        "events": trace.events, "llm_calls": llm_calls,
        "usage_total": {"input_tokens": sum(c.get("input_tokens") or 0 for c in llm_calls),
                        "output_tokens": sum(c.get("output_tokens") or 0 for c in llm_calls),
                        "llm_calls": len(llm_calls)},
        "execution_status": status, "termination_reason": reason,
        "final_output_present": bool(final_text), "steps_used": steps_used,
        "runner_traceback": error,
    }
    return {"raw": raw, "eval": eval_rec}
