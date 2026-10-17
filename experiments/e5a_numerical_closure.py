#!/usr/bin/env python
"""Public reproduction of the E5A high-precision numerical-closure grid.

The default run covers the synthetic d/r grid and module-count grid. A real anchor NPZ can
be supplied with --anchor-npz; it must contain A_ref, B_ref, A_target, B_target, scale_ref,
scale_target as in the paper experiment artifact.
"""
import argparse,csv,json,math,hashlib
from pathlib import Path
import numpy as np, torch
from semapeft.certificate import lowrank_certificate,dense_certificate,compare_certificates
from semapeft.numerics import relerr

def make(seed,d,r,M):
    g=torch.Generator(device='cpu').manual_seed(seed)
    Ar=torch.randn(M,r,d,generator=g,dtype=torch.float32)/math.sqrt(d);Br=torch.randn(M,d,r,generator=g,dtype=torch.float32)/math.sqrt(r)
    At=Ar+0.05*torch.randn(M,r,d,generator=g,dtype=torch.float32)/math.sqrt(d);Bt=Br+0.05*torch.randn(M,d,r,generator=g,dtype=torch.float32)/math.sqrt(r)
    return Ar,Br,At,Bt,1.0,1.0

def ihash(x):
    h=hashlib.sha256();
    for t in x[:4]:h.update(t.contiguous().numpy().tobytes(order='C'))
    h.update(np.asarray(x[4:],dtype=np.float64).tobytes());return h.hexdigest()

def eval_one(cell,seed,tensors,device,tol):
    Ar,Br,At,Bt,sr,st=tensors; xs=[x.to(device) for x in (Ar,Br,At,Bt)]
    r1=lowrank_certificate(*xs,sr,st,compute_dtype=torch.float32); r2=dense_certificate(*xs,sr,st,compute_dtype=torch.float32)
    r3=lowrank_certificate(*xs,sr,st,compute_dtype=torch.float64); r4=dense_certificate(*xs,sr,st,compute_dtype=torch.float64)
    hp=compare_certificates(r3,r4)
    return {**cell,'seed':seed,'input_sha256':ihash(tensors),
      'lr32_dense32_frob_disagreement':relerr(r1.relative_frobenius,r2.relative_frobenius),
      'lr32_dense32_spec_disagreement':relerr(r1.relative_spectral,r2.relative_spectral),
      'lr64_dense64_frob_disagreement':hp['frobenius_relative_disagreement'],
      'lr64_dense64_spec_disagreement':hp['spectral_relative_disagreement'],
      'lr32_frob_error_to_dense64':relerr(r1.relative_frobenius,r4.relative_frobenius),
      'dense32_frob_error_to_dense64':relerr(r2.relative_frobenius,r4.relative_frobenius),
      'lr32_spec_error_to_dense64':relerr(r1.relative_spectral,r4.relative_spectral),
      'dense32_spec_error_to_dense64':relerr(r2.relative_spectral,r4.relative_spectral),
      'high_precision_pass':hp['frobenius_relative_disagreement']<=tol and hp['spectral_relative_disagreement']<=tol}

ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');ap.add_argument('--out',default='e5a_public.csv');ap.add_argument('--anchor-npz');ap.add_argument('--tolerance',type=float,default=1e-10)
a=ap.parse_args();rows=[];seeds=[11,23,47,71,89]
for d in [256,512,1024,2048,4096]:
  for r in [4,8,16,32]:
    for s in seeds: rows.append(eval_one({'family':'dimension_rank','d':d,'r':r,'M':1},s,make(s,d,r,1),a.device,a.tolerance))
for M in [1,8,32,128]:
  for s in seeds: rows.append(eval_one({'family':'module_count','d':1024,'r':8,'M':M},s,make(s,1024,8,M),a.device,a.tolerance))
if a.anchor_npz:
  z=np.load(a.anchor_npz);ts=(torch.from_numpy(z['A_ref']).unsqueeze(0).float(),torch.from_numpy(z['B_ref']).unsqueeze(0).float(),torch.from_numpy(z['A_target']).unsqueeze(0).float(),torch.from_numpy(z['B_target']).unsqueeze(0).float(),float(z['scale_ref'][0]),float(z['scale_target'][0]))
  for s in seeds: rows.append(eval_one({'family':'real_anchor','d':2048,'r':16,'M':1},s,ts,a.device,a.tolerance))
fields=list(rows[0]);
with open(a.out,'w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print(json.dumps({'rows':len(rows),'all_high_precision_pass':all(r['high_precision_pass'] for r in rows),'max_frob':max(r['lr64_dense64_frob_disagreement'] for r in rows),'max_spec':max(r['lr64_dense64_spec_disagreement'] for r in rows)},indent=2))
