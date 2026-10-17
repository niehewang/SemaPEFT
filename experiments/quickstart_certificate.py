#!/usr/bin/env python
import json, math, torch
from semapeft.certificate import p64_compare

g=torch.Generator().manual_seed(11); d=512; r=8
Ar=torch.randn(r,d,generator=g,dtype=torch.float32)/math.sqrt(d)
Br=torch.randn(d,r,generator=g,dtype=torch.float32)/math.sqrt(r)
At=Ar + 0.05*torch.randn(r,d,generator=g,dtype=torch.float32)/math.sqrt(d)
Bt=Br + 0.05*torch.randn(d,r,generator=g,dtype=torch.float32)/math.sqrt(r)
print(json.dumps(p64_compare(Ar,Br,At,Bt,tolerance=1e-10),indent=2))
