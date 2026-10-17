from __future__ import annotations
import numpy as np

def relerr(a,b,eps=1e-300):
    return abs(float(a)-float(b))/max(abs(float(b)),eps)

def gamma_n(n,u):
    x=float(n)*float(u)
    return x/(1.0-x) if x < 1.0 else float("inf")

def float32_diagnostics(n_values=(8,16,32,256,512,1024,2048,4096)):
    eps=float(np.finfo(np.float32).eps); u=eps/2.0
    return {"epsilon":eps,"unit_roundoff":u,"gamma_n":{int(n):gamma_n(n,u) for n in n_values}}
