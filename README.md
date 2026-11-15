# Experiment 2: IPI with AgentDojo "Important Instructions" Payload

This repository contains the execution package and raw agent trajectories for **Experiment 2** of the project *Beyond Payload Presence: Evaluating Whether Behavioral Detection of Indirect Prompt Injection Transfers Across LLMs* (SIIT, Thammasat University, 2026). The runs were executed by the P2 (execution) role. **The IPI trajectories used in the paper come from this experiment.**

Experiment 1 (https://github.com/boobaoooo/IPI-Exp1-Baseline-InjecAgent) used a fixed InjecAgent-style template, which produced too few attacker-induced actions for supervised learning. Experiment 2 keeps everything else identical and changes only the payload template to AgentDojo's `important_instructions` attack **without names** (`agentdojo_important_instructions_no_names_fixed_v0.1`). This template wraps the attacker goal in an `<INFORMATION>` block that is presented as a message from the user.

Unchanged from Experiment 1: tasks, attacker goals, injection vectors, models and digests, runner and driver, generation policy, and splits. The same payload is used for every LLM, and its SHA-256 is logged per run. **Benign controls are not re-run here.** The paper uses the benign runs from Experiment 1.

## Staged design

| Stage | What | LLM | Runs | Accepted |
|---|---|---|---:|---:|
| 1. Paired pilot | 40 conditions (10 per suite, selected without looking at outcomes), each run under both templates in counterbalanced AB/BA order | Qwen (LLM-B) | 40 baseline rerun + 40 important | 40 + 39 |
| 2. Cross-model pilot | The same 40 conditions | Gemma (LLM-A), LFM (LLM-C) | 80 baseline rerun + 80 important | 80 + 77 (Gemma 37, LFM 40) |
| 3. Main run | The remaining 260 IPI conditions per LLM, important template only | All three | 780 | 774 |

**Totals under the important-instructions template:** 900 planned and 890 accepted (Gemma 296, Qwen 294, LFM 300). Ten runs on nine conditions timed out on all 4 attempts. A condition enters the paper's matched set only if all three LLMs accepted it, which gives 291 matched IPI conditions (873 trajectories).

The 120 **baseline reruns** from stages 1–2 repeat Experiment 1 conditions. They were used to check run-to-run stability: 45 of 120 gave identical call sequences, and the mean number of tool calls and state-changing calls showed no systematic shift.

Stage 3 was launched under a documented protocol amendment, made before the behavioral gate was formally signed off. Baseline-versus-important comparisons outside the 40 paired conditions are therefore historical, not contemporaneous.

## Settings (identical to Experiment 1)

AgentDojo 0.1.35 (commit `a75aba76`); Ollama 0.35.0; `gemma4:12b` (6114515d…), `qwen3.5:9b` (6488c96f…), `lfm2.5:latest` (9cf75615…); temperature 0, seed 42, context 16,384, `parallel_tool_calls=false`, at most 15 steps, 180 s timeout, at most 4 attempts and only after a technical failure; `p2runner` v0.2.2 (`runner_source_sha256` 32395aea…); driver `run_full.py` (fingerprint `36e2dae4…`); single workstation with an NVIDIA RTX 3060 (12 GB). The GPU driver was updated between the experiments (591.86 → 610.88).

## Repository layout

```
config/
  full_experiment_fixed_ipi_payloads_IMPORTANT_NO_NAMES_v1.0.json   Frozen payloads (important template)
  full_experiment_execution_manifest_PAYLOAD_EVAL_900_v1.0.csv      900-run manifest
  full_experiment_execution_manifest_REMAINING780_v1.0.csv          Stage-3 manifest
  mapping_baseline_to_ii_900.csv        Exp-1 run_id ↔ Exp-2 run_id, old/new payload hashes
  pilot_*                               Stage-1/2 condition list and schedules
  PILOT_120_RUN_IDS_EXCLUDED_from_780.csv
  FREEZE_PROVENANCE_*.json, MODEL_SLOTS_AND_COUNTERS.json
p2runner/                               Runner package v0.2.2 (4 files); source hash `32395aea…`, identical to Experiment 1
batch_controller.py, validate_outputs.py, monitor_runtime.py   Helper scripts used during the runs
code/
  run_full.py                           Frozen batch driver (same as Exp 1)
  p2_payload_launcher_*.py              Launchers that point the frozen driver at the new manifest/payloads
  run_paired.py, verify_paired.py       Stage-1 paired pilot
  RUN_FULL_CHANGE_NOTE_TH.md            Driver change note (Thai)
data/
  stage1_pilot_qwen_80/{P0_baseline_rerun,P1_important_instructions}/
  stage2_pilot_gemma_lfm_160/{GEMMA,LFM}_{P0_baseline_rerun,P1_important_instructions}/
  stage3_main_780/
    raw_runs.jsonl, attempt_ledger.jsonl, checkpoint_log.jsonl,
    run_metadata_full.jsonl (host name and OS user redacted), resource_log.jsonl, schema_warnings.jsonl
logs/       Console logs of each stage (converted to UTF-8) and the stage-3 summary
```

The record format is the same as in Experiment 1 (`raw_run/0.1`). Model-reported multiple tool calls per step occurred despite `parallel_tool_calls=false` (172 of the 890 accepted IPI runs: Qwen 77, LFM 74, Gemma 21). Every such call was executed and logged.

## Running the code

The runner package and helper scripts are included, with SHA-256 values in `SHA256SUMS.txt` (newline-normalised; `monitor_runtime.py` is `244fa6747b297827f19a73d78de434ce23d73ebd45fa1d0ceccb513c522140e2`). On the execution machine all of these files sat in one working folder next to `run_full.py`; to reproduce that layout, copy `code/run_full.py` next to `p2runner/`. Running also needs AgentDojo 0.1.35, the three Ollama models at the digests above, and the frozen manifests in `config/`. The package was not re-run for this release.

## What is not included

- **Ground-truth labels** (`ipi_affected`), adjudication evidence, the exposure overlay, **AgentDojo evaluator outcomes** (`eval_evidence.jsonl`), and **environment snapshots**. These are owned by the adjudication role (P1).

## Notes

- The runner's `injection_encountered` flag misses multi-line payloads (it strips backslashes before matching). The paper uses a separate exposure overlay, which finds the full payload in 809 of the 890 accepted runs.
- Payload strings and e-mail addresses in the data are fictional AgentDojo benchmark content.

## Related

- Experiment 1 (baseline template, benign controls): https://github.com/boobaoooo/IPI-Exp1-Baseline-InjecAgent
- Detector pipeline and results: https://github.com/a3troooooo/IPI-Cross-LLM-Research-Archive
