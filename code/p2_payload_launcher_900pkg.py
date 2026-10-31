#!/usr/bin/env python3
"""P2 executable launcher: offline verify, dry-run, gate, smoke, pilot, full, status.
Requires original installed P2 runtime folder with p2runner/, validate_outputs.py and Ollama instance.
"""
from pathlib import Path
import argparse,csv,hashlib,json,os,subprocess,sys,collections
HERE=Path(__file__).resolve().parent
HANDOFF=HERE/'handoff'; CORE=HANDOFF/'core'
MAN='core/full_experiment_execution_manifest_PAYLOAD_EVAL_v1.0.csv'
PAY='core/full_experiment_fixed_ipi_payloads_IMPORTANT_NO_NAMES_v1.0.json'
OLDMAN=HANDOFF/'reference/ORIGINAL_FROZEN_manifest_v1.1_READ_ONLY.csv'
OLDPAY=HANDOFF/'reference/ORIGINAL_FROZEN_payloads_v1.0_READ_ONLY.json'
REFERENCE=HERE/'plans/MAPPING_900_BLIND.csv'
PILOT=HERE/'plans/pilot_40_pairs_qwen_NON_OUTCOME_SELECTED.csv'
FREEZE=HERE/'FREEZE_PROVENANCE.json'
FP={'runner.py':'38991e2687a5805ddd78ca1572874846fbe43ade5bc9bd821f0afa96a9d9c03e','adapters.py':'2e98b1276832c0cb5696c774c743c7abccb531c8d6a059f19f959c7910c8479e','scenarios.py':'d5b7ff8900dc5354a86227186bb7d58f7a94049e4ad8131cdd2afe8660e6597d','__init__.py':'83c19fe9c044c5a33602c9ace2800a5bafb243bca3950bd7b8f39b6471e4246b'}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def normsha(path):return hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def read_csv(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def verify():
 errs=[];meta=json.loads(FREEZE.read_text(encoding='utf8'))
 for x,h in meta['file_sha256'].items():
  path=HERE/x
  if not path.is_file() or sha(path)!=h:errs.append('file SHA mismatch: '+x)
 if errs:raise RuntimeError('\n'.join(errs))
 old=read_csv(OLDMAN);orig=[r for r in old if r['condition']=='ipi'];new=read_csv(CORE/MAN.split('/')[-1]);payload=json.loads((CORE/PAY.split('/')[-1]).read_text(encoding='utf8'));op=json.loads(OLDPAY.read_text(encoding='utf8'))
 idx={x['pair_id']:x for x in payload};opidx={x['pair_id']:x for x in op};mp=read_csv(REFERENCE)
 assert len(old)==1164 and len(new)==len(orig)==len(mp)==900 and len(idx)==len(opidx)==300
 assert len({x['run_id'] for x in new})==900
 origidx={x['run_id']:x for x in orig}; mpidx={r['candidate_run_id']:r for r in mp}
 allowed={'run_id','payload_sha256','payload_source_file','attack_template','injection_policy_version','fixed_ipi_payloads_sha256'}
 counts=collections.Counter()
 for n in new:
  counts[n['source_llm_id']]+=1
  m=mpidx[n['run_id']];o=origidx[m['reference_run_id']];p=idx[n['pair_id']]
  assert n['run_id']=='payload_v0_1__'+o['run_id']
  assert all(n[k]==o[k] for k in n if k not in allowed),n['run_id']
  assert n['payload_sha256']==p['payload_sha256']==hashlib.sha256(p['payload'].encode('utf8')).hexdigest()
  assert o['payload_sha256']==opidx[n['pair_id']]['payload_sha256']
  assert n['fixed_ipi_payloads_sha256']==meta['payload_sha256']
  assert n['payload_lookup_key']==p['pair_id']
  assert len(p['injection_vector_ids'])==int(n['injection_vector_count'])==len(p['injections'])
  assert set(p['injection_vector_ids'])==set(n['injection_vector_ids'].replace(';',',').replace('|',',').split(','))
  assert all(t==p['payload'] for t in p['injections'].values())
 assert counts=={'LLM-A':300,'LLM-B':300,'LLM-C':300},counts
 assert meta['manifest_sha256']==sha(CORE/MAN.split('/')[-1])
 assert meta['payload_sha256']==sha(CORE/PAY.split('/')[-1])
 print('PASS / offline archive integrity — 900 new IPI rows, 300 original pairs, all 3 model identities, all vector mappings, payload hashes, no leaked old outcomes.')
def check_p2(root):
 problems=[]
 for x,h in FP.items():
  p=root/'p2runner'/x
  if not p.is_file(): problems.append('Missing P2 source: '+str(p))
  elif normsha(p)!=h:problems.append(f'P2 source changed: {x} sha256(normalized)={normsha(p)} expected={h}')
 if not (root/'validate_outputs.py').is_file():problems.append('Missing original validate_outputs.py at '+str(root))
 if problems:raise RuntimeError('\n'.join(problems)+'\nDO NOT reimplement/bypass a failed gate; have P2 share exact original source with P1.')
 print('PASS / original P2 source fingerprints and validate_outputs.py present')
def accepted_summary(out):
 mf=read_csv(CORE/MAN.split('/')[-1]);planned=set(x['run_id'] for x in mf)
 if not (out/'attempt_ledger.jsonl').exists():print('No ledger yet: 0/900 accepted');return
 le=[json.loads(s) for s in (out/'attempt_ledger.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
 status=collections.Counter();acc={};exhausted=set()
 for a in le:
  if a.get('outcome')=='accepted':acc[a['run_id']]=max(int(a['attempt_id']),acc.get(a['run_id'],0))
  status[a.get('outcome','none')]+=1
  if a.get('outcome')=='exhausted_no_accepted_attempt':exhausted.add(a['run_id'])
 by=collections.Counter({m:sum(x['source_llm_id']==m and x['run_id'] in acc for x in mf) for m in ['LLM-A','LLM-B','LLM-C']})
 print(json.dumps({'accepted':len(acc),'planned':len(planned),'by_model':dict(by),'exhausted':len(exhausted),'attempt_outcomes':dict(status),'unexpected_run_ids':len(set(acc)-planned),'note':'strict attack success not computed; P1-only post-state evaluator required'},indent=2))
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('command',choices=['verify','check-runtime-files','dry-run','gate','smoke','pilot','all','status'])
 ap.add_argument('--p2-root',type=Path,default=Path.cwd(),help='original P2 folder containing p2runner and validate_outputs.py; default cwd')
 ap.add_argument('--out',type=Path,help='dedicated sidecar output; default <p2-root>/output_payload_eval_important_v1_0')
 ap.add_argument('--endpoint',default='http://127.0.0.1:11435')
 ap.add_argument('--slots',nargs='+',choices=['LLM-A','LLM-B','LLM-C'],default=None)
 ap.add_argument('--chunk-size',type=int,default=25)
 ap.add_argument('--other-ollama',default='http://127.0.0.1:11434')
 a=ap.parse_args();verify()
 root=a.p2_root.resolve();out=(a.out or (root/'output_payload_eval_important_v1_0')).resolve()
 if a.command=='verify':return 0
 if a.command=='status':accepted_summary(out);return 0
 check_p2(root)
 if a.command=='check-runtime-files':return 0
 env=os.environ.copy();env['PYTHONPATH']=str(root)+os.pathsep+env.get('PYTHONPATH','')
 # ORIGINAL runner reads variables from operator environment, no implicit modification; fail gate if absent/wrong
 cmd=[sys.executable,str(HERE/'runner/run_full.py'),'--handoff',str(HANDOFF),'--manifest-rel',MAN,'--payload-rel',PAY,
      '--expected-manifest-sha',json.loads(FREEZE.read_text(encoding='utf8'))['manifest_sha256'],
      '--expected-payload-sha',json.loads(FREEZE.read_text(encoding='utf8'))['payload_sha256'],
      '--expected-endpoint',a.endpoint,'--other-ollama',a.other_ollama,'--out',str(out)]
 if a.slots:cmd.extend(['--slots',*a.slots])
 if a.command=='dry-run':cmd.append('--dry-run')
 elif a.command=='gate':cmd.append('--gate-only')
 elif a.command=='smoke':
  # exactly one single-vector and one multi-vector for Qwen; checked against package manifest
  mf=[x for x in read_csv(CORE/MAN.split('/')[-1]) if x['source_llm_id']=='LLM-B']
  one=next(x['run_id'] for x in mf if int(x['injection_vector_count'])==1)
  multi=next(x['run_id'] for x in mf if int(x['injection_vector_count'])>1)
  cmd.extend(['--only',one,multi]);print('smoke ids:',one,multi)
 elif a.command=='pilot':
  pairs={x['pair_id'] for x in read_csv(PILOT)}
  mf=read_csv(CORE/MAN.split('/')[-1]);rr=[x['run_id'] for x in mf if x['source_llm_id']=='LLM-B' and x['pair_id'] in pairs]
  assert len(rr)==40,'pilot ids incomplete';cmd.extend(['--only',*rr,'--chunk-size',str(a.chunk_size)])
 elif a.command=='all':cmd.extend(['--chunk-size',str(a.chunk_size)])
 print('Using original 0.2.2 runner code and P2 source; separate output:',out,flush=True)
 print('Command:',subprocess.list2cmdline(cmd),flush=True)
 return subprocess.call(cmd,cwd=str(root),env=env)
if __name__=='__main__':
 try:raise SystemExit(main())
 except (RuntimeError,AssertionError,FileNotFoundError,ValueError,KeyError) as e:
  print('STOP:',str(e),file=sys.stderr);raise SystemExit(2)
