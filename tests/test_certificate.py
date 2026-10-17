import math, torch
from semapeft.certificate import lowrank_certificate,dense_certificate,p64_compare

def factors(seed=11,d=48,r=4,M=3):
    g=torch.Generator().manual_seed(seed)
    Ar=torch.randn(M,r,d,generator=g,dtype=torch.float32)/math.sqrt(d);Br=torch.randn(M,d,r,generator=g,dtype=torch.float32)/math.sqrt(r)
    At=Ar+0.02*torch.randn(M,r,d,generator=g,dtype=torch.float32)/math.sqrt(d);Bt=Br+0.02*torch.randn(M,d,r,generator=g,dtype=torch.float32)/math.sqrt(r)
    return Ar,Br,At,Bt

def test_p64_lowrank_dense_agree():
    x=factors(); out=p64_compare(*x,tolerance=1e-10)
    assert out['pass']; assert out['frobenius_relative_disagreement'] <= 1e-10; assert out['spectral_relative_disagreement'] <= 1e-10

def test_identical_operator_zero_gap():
    Ar,Br,_,_=factors(M=1); lr=lowrank_certificate(Ar,Br,Ar,Br,2.0,2.0); de=dense_certificate(Ar,Br,Ar,Br,2.0,2.0)
    assert lr.relative_frobenius < 1e-12; assert de.relative_frobenius == 0.0
