#!/usr/bin/env python
"""Aggregate systems timings using the corrected E5B v1.1 process-level protocol."""
import argparse,csv,json
from collections import defaultdict
import numpy as np
from semapeft.stats import process_summary
ap=argparse.ArgumentParser();ap.add_argument('raw_csv');ap.add_argument('--out',default='process_summary.csv');ap.add_argument('--resamples',type=int,default=10000);a=ap.parse_args()
rows=list(csv.DictReader(open(a.raw_csv,encoding='utf-8')));groups=defaultdict(lambda:defaultdict(list))
for r in rows:
    if r.get('process_status')!='COMPLETE': continue
    groups[(r['cell_id'],r['route'])][int(r['process_seed'])].append(float(r['wall_seconds']))
out=[]
for (cell,route),byseed in sorted(groups.items()):
    pmed=[float(np.median(v)) for _,v in sorted(byseed.items())]
    s=process_summary(pmed,resamples=a.resamples)
    out.append({'cell_id':cell,'route':route,'process_count':len(pmed),'median_wall_seconds':s['median'],'wall_p25_seconds':s['p25'],'wall_p75_seconds':s['p75'],'ci95_low':s['ci95'][0],'ci95_high':s['ci95'][1],'process_medians_json':json.dumps(pmed)})
with open(a.out,'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print(json.dumps({'groups':len(out),'out':a.out},indent=2))
