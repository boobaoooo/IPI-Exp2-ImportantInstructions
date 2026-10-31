#!/usr/bin/env python3
"""This wrapper invokes the bundled frozen old runner, not a replacement LLM agent."""
from pathlib import Path
import argparse,os,subprocess,sys,csv,hashlib,collections
HERE=Path(__file__).resolve().parent
P2FP={'runner.py':'38991e2687a5805ddd78ca1572874846fbe43ade5bc9bd821f0afa96a9d9c03e','adapters.py':'2e98b1276832c0cb5696c774c743c7abccb531c8d6a059f19f959c7910c8479e','scenarios.py':'d5b7ff8900dc5354a86227186bb7d58f7a94049e4ad8131cdd2afe8660e6597d','__init__.py':'83c19fe9c044c5a33602c9ace2800a5bafb243bca3950bd7b8f39b6471e4246b'}
NAME={'P0':('P0_QWEN_40_RERUN_manifest.csv','full_experiment_fixed_ipi_payloads_v1.0.json','output_pilot_P0_rerun_v1'),'P1':('P1_QWEN_40_IMPORTANT_manifest.csv','full_experiment_fixed_ipi_payloads_IMPORTANT_NO_NAMES_v1.0.json','output_pilot_P1_important_v1')}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def call(arm,root,out,only=None,mode='run'):
 man,pay,folder=NAME[arm];m=HERE/'handoff'/'core'/man;p=HERE/'handoff'/'core'/pay
 cmd=[sys.executable,str(HERE/'runner'/'run_full.py'),'--handoff',str(HERE/'handoff'),'--manifest-rel','core/'+man,'--payload-rel','core/'+pay,'--expected-manifest-sha',sha(m),'--expected-payload-sha',sha(p),'--expected-endpoint','http://127.0.0.1:11435','--other-ollama','http://127.0.0.1:11434','--out',str(root/folder)]
 if only:cmd+=['--only',*only]
 if mode=='dry':cmd+=['--dry-run']
 elif mode=='gate':cmd+=['--gate-only']
 print('COMMAND:',subprocess.list2cmdline(cmd),flush=True)
 env=os.environ.copy();env['PYTHONPATH']=str(root)+os.pathsep+env.get('PYTHONPATH','')
 return subprocess.call(cmd,cwd=str(root),env=env)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['preflight','gate','smoke','pilot']);ap.add_argument('--p2-root',required=True,type=Path);a=ap.parse_args()
 import verify_paired
 root=a.p2_root.resolve()
 for f,h in P2FP.items():
  p=root/'p2runner'/f
  if not p.is_file() or hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=h:raise RuntimeError('P2 SOURCE FINGERPRINT FAIL: '+str(p))
 if not (root/'validate_outputs.py').is_file():raise RuntimeError('missing old validator')
 if a.command=='preflight':
  for arm in NAME:
   e=call(arm,root,root/NAME[arm][2],mode='dry')
   if e:return e
  return 0
 if a.command=='gate':
  for arm in NAME:
   e=call(arm,root,root/NAME[arm][2],mode='gate')
   if e:return e
  return 0
 with (HERE/'matched_run_schedule_80.csv').open(encoding='utf8') as f:r=list(csv.DictReader(f))
 if a.command=='smoke':
  data={}
  for arm in NAME:
   cand=[x for x in r if x['payload_arm']==arm]
   with (HERE/'handoff'/'core'/NAME[arm][0]).open(encoding='utf8') as f: mf={x['run_id']:x for x in csv.DictReader(f)}
   selected=[next(x['run_id'] for x in cand if int(mf[x['run_id']]['injection_vector_count'])==n) for n in [1,2] if any(int(mf[x['run_id']]['injection_vector_count'])==n for x in cand)]
   if len(selected)<2: selected=[next(x['run_id'] for x in cand if int(mf[x['run_id']]['injection_vector_count'])==1),next(x['run_id'] for x in cand if int(mf[x['run_id']]['injection_vector_count'])>1)]
   data[arm]=selected
  for arm,items in data.items():
   ec=call(arm,root,root/NAME[arm][2],only=items)
   if ec:return ec
  return 0
 # Paired fixed blocks: 4 blocks, each arm 10 pairs; same arm output directory used across blocks and smoke
 blocks=collections.OrderedDict()
 for x in r:blocks.setdefault((x['block'],x['order_pos'],x['payload_arm']),[]).append(x['run_id'])
 for (block,order,arm),ids in blocks.items():
  print(f'BEGIN {block} #{order} arm={arm} count={len(ids)}',flush=True)
  ec=call(arm,root,root/NAME[arm][2],only=ids)
  if ec:return ec
 print('DONE. DO NOT EXPAND TO FULL 900 UNTIL P1 SIGN-OFF. Strict ASR remains P1-only.')
 return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except (ValueError,AssertionError,RuntimeError,FileNotFoundError) as e:
  print('STOP:',str(e),file=sys.stderr);raise SystemExit(2)
