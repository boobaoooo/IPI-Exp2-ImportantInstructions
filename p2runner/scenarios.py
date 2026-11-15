"""Scenario loading: Person 1's pilot_manifest / pilot_payloads / checksums.

The real files were not available while writing this module, so the loaders accept
several column / key spellings and fail loudly (listing what they found) rather than
guessing. A `builtin` dev fallback rebuilds the 9 pilot groups from the AgentDojo
InjecAgent template so the pipeline can be tested offline. Never use the fallback
for the real pilot: P1's payload file is the source of truth.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BENCHMARK_VERSION = "v1"
ATTACK_TEMPLATE_ID = "injecagent-fixed-v1"

# The 9 pilot groups from P1's plan (suite, user_task, injection_task).
BUILTIN_PILOT = [
    ("banking", "user_task_0", "injection_task_0"),
    ("banking", "user_task_1", "injection_task_4"),
    ("slack", "user_task_0", "injection_task_1"),
    ("slack", "user_task_5", "injection_task_2"),
    ("travel", "user_task_0", "injection_task_0"),
    ("travel", "user_task_7", "injection_task_1"),
    ("workspace", "user_task_0", "injection_task_0"),
    ("workspace", "user_task_22", "injection_task_1"),
    ("workspace", "user_task_28", "injection_task_0"),
]
# Vector stated in P1's notes for the smoke-test scenario.
KNOWN_VECTORS = {("banking", "user_task_1", "injection_task_4"): ["injection_incoming_transaction"]}


@dataclass
class Case:
    scenario_id: str
    scenario_group_id: str
    suite: str
    user_task_id: str
    injection_task_id: str | None
    condition: str  # "benign" | "ipi"
    injection_vectors: list[str] = field(default_factory=list)
    payload: str | None = None
    payload_sha256: str | None = None
    attack_template_id: str | None = None
    split: str | None = None
    manifest_source: str = "file"
    extra: dict[str, Any] = field(default_factory=dict)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_checksums(checksum_file: Path, base_dir: Path | None = None) -> dict[str, bool]:
    """Verify a `sha256sum`-style file. Returns {filename: ok}. Raises on mismatch."""
    base = base_dir or checksum_file.parent
    results: dict[str, bool] = {}
    for line in checksum_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition(" ")
        name = name.strip().lstrip("*").strip()
        target = base / Path(name).name
        ok = target.exists() and sha256_file(target) == digest.lower()
        results[name] = ok
    bad = [n for n, ok in results.items() if not ok]
    if bad:
        raise ValueError(f"Checksum mismatch or missing file: {bad}")
    return results


def _pick(row: dict[str, str], *names: str) -> str | None:
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n].strip()
    return None


def _norm_condition(v: str) -> str:
    v = v.lower()
    if v in ("benign", "control", "clean"):
        return "benign"
    if v in ("ipi", "attack", "injected", "malicious"):
        return "ipi"
    raise ValueError(f"Unknown condition {v!r}")


def _payload_entries(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("payloads"), (list, dict)):
        data = data["payloads"]
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)]
    if isinstance(data, dict):
        out = []
        for k, v in data.items():
            e = dict(v) if isinstance(v, dict) else {"payload": v}
            e.setdefault("_key", k)
            out.append(e)
        return out
    raise ValueError("Unrecognised payload file structure")


_TEXT_KEYS = ("payload", "payload_text", "text", "injection", "attack_text", "content")


def _entry_text(e: dict[str, Any]) -> str | None:
    for k in _TEXT_KEYS:
        if isinstance(e.get(k), str) and e[k]:
            return e[k]
    return None


def find_payload(entries: list[dict[str, Any]], case: Case) -> dict[str, Any]:
    ids = {case.scenario_id, case.scenario_group_id}
    triple = (case.suite, case.user_task_id, case.injection_task_id)
    for e in entries:
        keys = {str(e.get(k)) for k in ("scenario_id", "scenario_group_id", "group_id", "_key") if e.get(k)}
        if ids & keys:
            return e
    for e in entries:
        if (e.get("suite"), e.get("user_task_id") or e.get("user_task"),
                e.get("injection_task_id") or e.get("injection_task")) == triple:
            return e
    raise KeyError(f"No payload entry for {case.scenario_group_id}; entry keys seen: "
                   f"{sorted({k for e in entries for k in e})}")


def load_manifest(manifest: Path, payloads: Path, checksums: Path | None = None) -> list[Case]:
    if checksums:
        verify_checksums(checksums, manifest.parent)
    entries = _payload_entries(json.loads(payloads.read_text()))
    with manifest.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Manifest is empty")
    cases: list[Case] = []
    for row in rows:
        suite = _pick(row, "suite")
        ut = _pick(row, "user_task_id", "user_task")
        it = _pick(row, "injection_task_id", "injection_task")
        if not (suite and ut):
            raise ValueError(f"Manifest row lacks suite/user_task; columns: {list(row)}")
        group = _pick(row, "scenario_group_id", "group_id") or f"{suite}__{ut}__{it}"
        vec = _pick(row, "injection_vectors", "injection_vector_id", "injection_vector", "vector")
        vectors = [v.strip() for v in vec.replace(";", ",").replace("|", ",").split(",") if v.strip()] if vec else []
        cond_raw = _pick(row, "condition")
        conds = [_norm_condition(cond_raw)] if cond_raw else ["benign", "ipi"]
        for cond in conds:
            sid = _pick(row, "scenario_id") if cond_raw else None
            sid = sid or f"{group}__{cond}"
            case = Case(sid, group, suite, ut, it, cond, vectors, split=_pick(row, "split"),
                        extra={k: v for k, v in row.items() if v})
            if cond == "ipi":
                e = find_payload(entries, case)
                text = _entry_text(e)
                if text is None:
                    raise KeyError(f"Payload entry has no text field; keys: {list(e)}")
                declared = e.get("sha256") or e.get("payload_sha256")
                actual = sha256_text(text)
                if declared and declared != actual:
                    raise ValueError(f"Payload SHA-256 mismatch for {group}")
                case.payload, case.payload_sha256 = text, actual
                case.attack_template_id = e.get("attack_template_id") or e.get("attack_template") or _pick(case.extra, "attack_template_id") or ATTACK_TEMPLATE_ID
                if not case.injection_vectors and (e.get("injection_vector_id") or e.get("injection_vector")):
                    case.injection_vectors = [e.get("injection_vector_id") or e.get("injection_vector")]
            cases.append(case)
    return cases


def build_builtin_pilot() -> list[Case]:
    """DEV ONLY fallback: rebuild pilot cases from AgentDojo's InjecAgent template."""
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.attacks.baseline_attacks import InjecAgentAttack
    from agentdojo.task_suite.load_suites import get_suite

    class _Dummy(BasePipelineElement):
        name = "dummy"

        def query(self, *a, **k):  # pragma: no cover
            raise RuntimeError

    cases: list[Case] = []
    for suite_name, ut, it in BUILTIN_PILOT:
        suite = get_suite(BENCHMARK_VERSION, suite_name)
        atk = InjecAgentAttack(suite, _Dummy())
        inj = atk.attack(suite.get_user_task_by_id(ut), suite.get_injection_task_by_id(it))
        vectors = KNOWN_VECTORS.get((suite_name, ut, it)) or sorted(inj)
        payload = next(iter(inj.values()))
        group = f"{suite_name}__{ut}__{it}"
        for cond in ("benign", "ipi"):
            c = Case(f"{group}__{cond}", group, suite_name, ut, it, cond, vectors,
                     manifest_source="builtin_dev_fallback")
            if cond == "ipi":
                c.payload, c.payload_sha256, c.attack_template_id = payload, sha256_text(payload), ATTACK_TEMPLATE_ID
            cases.append(c)
    return cases
