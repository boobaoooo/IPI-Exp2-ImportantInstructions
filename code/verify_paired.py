#!/usr/bin/env python3
from pathlib import Path
import csv,hashlib,json,sys
BASE=Path(__file__).resolve().parent
with (BASE/'MANIFEST_SHA256.json').open(encoding='utf8') as f: M=json.load(f)
for path,digest in M['sha256'].items():
 p=BASE/path
 assert p.is_file(), f'missing {p}'
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest, f'bad SHA {path}'

def load(x):
 with (BASE/x).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
a=load('handoff/core/P0_QWEN_40_RERUN_manifest.csv')
b=load('handoff/core/P1_QWEN_40_IMPORTANT_manifest.csv')
p=load('pilot_40_pairs_qwen_NON_OUTCOME_SELECTED.csv')
s=load('matched_run_schedule_80.csv')
assert len(a)==len(b)==len(p)==40 and len(s)==80
assert set(x['pair_id'] for x in a)==set(x['pair_id'] for x in b)==set(x['pair_id'] for x in p)
A={x['pair_id']:x for x in a};B={x['pair_id']:x for x in b}
ALLOWED={'run_id','payload_sha256','payload_source_file','attack_template','injection_policy_version','fixed_ipi_payloads_sha256'}
for k in A:
 o,n=A[k],B[k]
 assert o['source_llm_id']==n['source_llm_id']=='LLM-B'
 assert all(o[t]==n[t] for t in o if t not in ALLOWED),(k,'unintended diff')
 assert o['run_id'].startswith('p0_rerun_v1__full_v1__llm_b__')
 assert n['run_id'].startswith('payload_v0_1__full_v1__llm_b__')
for arm,stem in [('P0','full_experiment_fixed_ipi_payloads_v1.0.json'),('P1','full_experiment_fixed_ipi_payloads_IMPORTANT_NO_NAMES_v1.0.json')]:
 payload={e['pair_id']:e for e in json.loads((BASE/'handoff'/'core'/stem).read_text(encoding='utf8'))}
 for k,row in (A if arm=='P0' else B).items():
  assert hashlib.sha256(payload[k]['payload'].encode('utf8')).hexdigest()==row['payload_sha256']
  assert set(payload[k]['injections'])==set(row['injection_vector_ids'].replace(';',',').replace('|',',').split(','))
assert len(set(x['run_id'] for x in s))==80
assert set(x['run_id'] for x in s)==set(x['run_id'] for x in a+b)
print('PASS: all archive files SHA256; 40 P0 and 40 P1 frozen rows matched; identical condition, vectors, runner/generation settings; 80-run AB/BA schedule')
