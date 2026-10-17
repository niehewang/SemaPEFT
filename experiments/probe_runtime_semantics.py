#!/usr/bin/env python
import argparse,json
from pathlib import Path
from semapeft.runtime import load_vllm,load_sglang

ap=argparse.ArgumentParser(description='Execute a pinned runtime LoRA loader in an isolated namespace.')
ap.add_argument('--runtime',choices=['vllm','sglang'],required=True)
ap.add_argument('--version',required=True)
ap.add_argument('--runtime-source',required=True,help='Local source snapshot root for the pinned runtime version.')
ap.add_argument('--adapter',required=True,help='Local PEFT adapter directory.')
ap.add_argument('--out',default='runtime_semantics.json')
a=ap.parse_args()
res=load_vllm(a.adapter,a.runtime_source,a.version) if a.runtime=='vllm' else load_sglang(a.adapter,a.runtime_source,a.version)
def compact(x):
    y={k:v for k,v in x.items() if k not in ('rows','state')}
    if 'rows' in x:y['rows']=[{k:v for k,v in r.items() if k not in ('A','B')} for r in x['rows']]
    if 'state' in x:y['state']=[{k:v for k,v in r.items() if k!='tensor'} for r in x['state']]
    return y
Path(a.out).write_text(json.dumps(compact(res),indent=2),encoding='utf-8')
print(json.dumps(compact(res),indent=2))
