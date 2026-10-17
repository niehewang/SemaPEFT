from __future__ import annotations
import numpy as np

def bootstrap_median_ci(process_medians, *, resamples=10000, seed=20261014, alpha=0.05):
    vals=np.asarray(process_medians,dtype=float)
    if vals.size < 2: return (float(vals[0]),float(vals[0])) if vals.size else (None,None)
    rng=np.random.default_rng(seed)
    boots=np.median(vals[rng.integers(0,len(vals),(resamples,len(vals)))],axis=1)
    return tuple(float(x) for x in np.quantile(boots,[alpha/2,1-alpha/2]))

def process_summary(process_medians, *, resamples=10000, seed=20261014):
    vals=np.asarray(process_medians,dtype=float)
    if not vals.size:return {"median":None,"p25":None,"p75":None,"ci95":[None,None]}
    lo,hi=bootstrap_median_ci(vals,resamples=resamples,seed=seed)
    return {"median":float(np.median(vals)),"p25":float(np.quantile(vals,.25)),"p75":float(np.quantile(vals,.75)),"ci95":[lo,hi]}
