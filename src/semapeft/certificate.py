from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import math
import torch

@dataclass(frozen=True)
class CertificateResult:
    relative_frobenius: float
    relative_spectral: float
    diff_frobenius_norm: float
    reference_frobenius_norm: float
    diff_spectral_norm: float
    reference_spectral_norm: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

def _batched(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.ndim == 2 else x

def _validate(A_ref, B_ref, A_target, B_target):
    xs=[_batched(x) for x in (A_ref,B_ref,A_target,B_target)]
    Ar,Br,At,Bt=xs
    if Ar.ndim != 3 or Br.ndim != 3:
        raise ValueError("Expected A:[M,r,in], B:[M,out,r] (or unbatched 2-D tensors).")
    if Ar.shape[0] != Br.shape[0] or At.shape[0] != Bt.shape[0] or Ar.shape[0] != At.shape[0]:
        raise ValueError("Reference and target module counts differ.")
    if Ar.shape[2] != At.shape[2] or Br.shape[1] != Bt.shape[1]:
        raise ValueError("Reference and target operator dimensions differ.")
    return Ar,Br,At,Bt

def lowrank_certificate(
    A_ref: torch.Tensor, B_ref: torch.Tensor,
    A_target: torch.Tensor, B_target: torch.Tensor,
    scale_ref: float = 1.0, scale_target: float = 1.0,
    *, compute_dtype: torch.dtype = torch.float64, chunk_modules: int = 32,
) -> CertificateResult:
    """Exact low-rank QR/core certificate used by SemaPEFT.

    Package aggregation matches the paper experiments: module Frobenius squares sum,
    while the package spectral norm is the maximum per-module spectral norm.
    """
    Ar,Br,At,Bt=_validate(A_ref,B_ref,A_target,B_target)
    device=Ar.device
    Ar,Br,At,Bt=[x.to(device=device,dtype=compute_dtype) for x in (Ar,Br,At,Bt)]
    M=Ar.shape[0]
    df2=torch.zeros((),device=device,dtype=compute_dtype)
    rf2=torch.zeros((),device=device,dtype=compute_dtype)
    dsmax=torch.zeros((),device=device,dtype=compute_dtype)
    rsmax=torch.zeros((),device=device,dtype=compute_dtype)
    tiny=torch.finfo(compute_dtype).tiny
    for s in range(0,M,chunk_modules):
        ar,br,at,bt=Ar[s:s+chunk_modules],Br[s:s+chunk_modules],At[s:s+chunk_modules],Bt[s:s+chunk_modules]
        U=torch.cat([scale_target*bt,-scale_ref*br],dim=2)
        V=torch.cat([at,ar],dim=1)
        gU=U.transpose(1,2)@U; gV=V@V.transpose(1,2)
        df2+=(gU*gV.transpose(1,2)).sum()
        Ur=scale_ref*br; Vr=ar
        gUr=Ur.transpose(1,2)@Ur; gVr=Vr@Vr.transpose(1,2)
        rf2+=(gUr*gVr.transpose(1,2)).sum()
        _,Ru=torch.linalg.qr(U,mode='reduced'); _,Rv=torch.linalg.qr(V.transpose(1,2),mode='reduced')
        ds=torch.linalg.svdvals(Ru@Rv.transpose(1,2))[:,0]
        _,Rur=torch.linalg.qr(Ur,mode='reduced'); _,Rvr=torch.linalg.qr(Vr.transpose(1,2),mode='reduced')
        rs=torch.linalg.svdvals(Rur@Rvr.transpose(1,2))[:,0]
        dsmax=torch.maximum(dsmax,ds.max()); rsmax=torch.maximum(rsmax,rs.max())
    dfn=torch.sqrt(torch.clamp_min(df2,0)); rfn=torch.sqrt(torch.clamp_min(rf2,0))
    return CertificateResult(
        relative_frobenius=float((dfn/(rfn+tiny)).detach().cpu()),
        relative_spectral=float((dsmax/(rsmax+tiny)).detach().cpu()),
        diff_frobenius_norm=float(dfn.detach().cpu()), reference_frobenius_norm=float(rfn.detach().cpu()),
        diff_spectral_norm=float(dsmax.detach().cpu()), reference_spectral_norm=float(rsmax.detach().cpu()),
    )

def dense_certificate(
    A_ref: torch.Tensor, B_ref: torch.Tensor,
    A_target: torch.Tensor, B_target: torch.Tensor,
    scale_ref: float = 1.0, scale_target: float = 1.0,
    *, compute_dtype: torch.dtype = torch.float64, chunk_modules: int = 8,
) -> CertificateResult:
    """Dense materialization oracle/reference for the same certificate quantities."""
    Ar,Br,At,Bt=_validate(A_ref,B_ref,A_target,B_target)
    device=Ar.device
    Ar,Br,At,Bt=[x.to(device=device,dtype=compute_dtype) for x in (Ar,Br,At,Bt)]
    M=Ar.shape[0]
    df2=torch.zeros((),device=device,dtype=compute_dtype)
    rf2=torch.zeros((),device=device,dtype=compute_dtype)
    dsmax=torch.zeros((),device=device,dtype=compute_dtype)
    rsmax=torch.zeros((),device=device,dtype=compute_dtype)
    tiny=torch.finfo(compute_dtype).tiny
    for s in range(0,M,chunk_modules):
        ar,br,at,bt=Ar[s:s+chunk_modules],Br[s:s+chunk_modules],At[s:s+chunk_modules],Bt[s:s+chunk_modules]
        R=scale_ref*(br@ar); G=scale_target*(bt@at); D=G-R
        df2+=(D*D).sum(); rf2+=(R*R).sum()
        ds=torch.linalg.matrix_norm(D,ord=2,dim=(-2,-1)); rs=torch.linalg.matrix_norm(R,ord=2,dim=(-2,-1))
        dsmax=torch.maximum(dsmax,ds.max()); rsmax=torch.maximum(rsmax,rs.max())
    dfn=torch.sqrt(torch.clamp_min(df2,0)); rfn=torch.sqrt(torch.clamp_min(rf2,0))
    return CertificateResult(
        relative_frobenius=float((dfn/(rfn+tiny)).detach().cpu()),
        relative_spectral=float((dsmax/(rsmax+tiny)).detach().cpu()),
        diff_frobenius_norm=float(dfn.detach().cpu()), reference_frobenius_norm=float(rfn.detach().cpu()),
        diff_spectral_norm=float(dsmax.detach().cpu()), reference_spectral_norm=float(rsmax.detach().cpu()),
    )

def _rel(a: float,b: float,eps: float=1e-300) -> float:
    return abs(float(a)-float(b))/max(abs(float(b)),eps)

def compare_certificates(lowrank: CertificateResult, dense: CertificateResult) -> dict[str, Any]:
    return {
        "frobenius_relative_disagreement": _rel(lowrank.relative_frobenius,dense.relative_frobenius),
        "spectral_relative_disagreement": _rel(lowrank.relative_spectral,dense.relative_spectral),
        "lowrank": lowrank.as_dict(), "dense": dense.as_dict(),
    }

def p64_compare(A_ref,B_ref,A_target,B_target,scale_ref=1.0,scale_target=1.0,tolerance: float=1e-10):
    """Stable P64 profile: float32 storage values upcast to float64 at certificate boundary."""
    xs=[x.to(torch.float32) for x in (A_ref,B_ref,A_target,B_target)]
    lr=lowrank_certificate(*xs,scale_ref,scale_target,compute_dtype=torch.float64)
    de=dense_certificate(*xs,scale_ref,scale_target,compute_dtype=torch.float64)
    out=compare_certificates(lr,de)
    out["tolerance"]=float(tolerance)
    out["pass"]=(out["frobenius_relative_disagreement"]<=tolerance and out["spectral_relative_disagreement"]<=tolerance)
    return out
