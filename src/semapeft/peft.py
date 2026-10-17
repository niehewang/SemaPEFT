from __future__ import annotations
from pathlib import Path
from typing import Any
import json, math, shutil
import torch
from safetensors.torch import load_file, save_file

def canonical_adapter_base(base: str) -> str:
    x=str(base)
    for prefix in ("base_model.model.","base_model."):
        if x.startswith(prefix): x=x[len(prefix):]
    return x

def canonical_leaf(name: str) -> str:
    s=str(name)
    for leaf in ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"):
        if s.endswith(leaf) or f".{leaf}" in s: return leaf
    return s.split(".")[-1]

def lora_scale(config: dict[str,Any], module: str, rank: int) -> float:
    alpha=float(config["lora_alpha"])
    for key,val in (config.get("alpha_pattern") or {}).items():
        if module.endswith(key) or module == key: alpha=float(val)
    return alpha/(math.sqrt(rank) if config.get("use_rslora") else rank)

def load_lora_pairs(adapter_dir: str|Path):
    p=Path(adapter_dir); cfg=json.loads((p/'adapter_config.json').read_text(encoding='utf-8'))
    tensors=load_file(str(p/'adapter_model.safetensors'),device='cpu')
    pairs={}
    for k,A in tensors.items():
        if not k.endswith('.lora_A.weight'): continue
        base=k[:-len('.lora_A.weight')]; bk=base+'.lora_B.weight'
        if bk not in tensors: continue
        mod=base.split('.')[-1]
        pairs[canonical_adapter_base(base)]={"module":mod,"A":A.float(),"B":tensors[bk].float(),"rank":int(A.shape[0]),"Akey":k,"Bkey":bk}
    return pairs,cfg,tensors

def operator_map_reference(adapter_dir: str|Path, strength: float=1.0):
    pairs,cfg,_=load_lora_pairs(adapter_dir)
    return {n:{"A":x['A'],"B":x['B'],"scale":lora_scale(cfg,x['module'],x['rank'])*float(strength)} for n,x in pairs.items()}

def operator_gap(reference: dict, realized: dict):
    num=den=0.0; per=[]
    for name,r in reference.items():
        if name not in realized:
            return {"relative_frobenius":float('inf'),"max_module_relative_frobenius":float('inf'),"per_module":per+[{'module':name,'status':'MISSING'}]}
        g=realized[name]
        A1=r['A'].double(); B1=r['B'].double()*float(r['scale']); A2=g['A'].double(); B2=g['B'].double()*float(g['scale'])
        if A1.shape==A2.shape and torch.equal(A1,A2):
            dB=B1-B2; n2=float(torch.sum((dB.T@dB)*(A1@A1.T))); d2=float(torch.sum((B1.T@B1)*(A1@A1.T)))
        else:
            D=B1@A1-B2@A2; n2=float(torch.sum(D*D)); R=B1@A1; d2=float(torch.sum(R*R))
        rel=math.sqrt(max(n2,0.0))/(math.sqrt(max(d2,0.0))+1e-30)
        num+=n2; den+=d2; per.append({'module':name,'relative_frobenius':rel})
    return {"relative_frobenius":math.sqrt(max(num,0.0))/(math.sqrt(max(den,0.0))+1e-30),"max_module_relative_frobenius":max([x.get('relative_frobenius',0) for x in per] or [0]),"per_module":per}

def lower_adapter_to_scale(adapter_dir: str|Path, output_dir: str|Path, target_scale: float, *, clear_patterns: bool=True):
    """Proof-carrying exact scale lowering by absorbing reference/target scale into B.

    The serialized float32 factor values are preserved except for the deterministic B scaling.
    """
    src=Path(adapter_dir); dst=Path(output_dir); dst.mkdir(parents=True,exist_ok=True)
    pairs,cfg,tensors=load_lora_pairs(src)
    out={k:v.clone() for k,v in tensors.items()}
    manifest=[]
    for _,x in pairs.items():
        ref=lora_scale(cfg,x['module'],x['rank']); ratio=ref/float(target_scale)
        out[x['Bkey']]=out[x['Bkey']]*ratio
        manifest.append({'module':x['module'],'rank':x['rank'],'reference_scale':ref,'target_scale':float(target_scale),'B_multiplier':ratio})
    new_cfg=dict(cfg)
    if clear_patterns:
        new_cfg['rank_pattern']={}; new_cfg['alpha_pattern']={}; new_cfg['use_rslora']=False
    # encode a global alpha/r corresponding to the requested runtime scale
    r=int(new_cfg.get('r') or next(iter(pairs.values()))['rank'])
    new_cfg['lora_alpha']=float(target_scale)*r
    save_file(out,str(dst/'adapter_model.safetensors'))
    (dst/'adapter_config.json').write_text(json.dumps(new_cfg,indent=2),encoding='utf-8')
    return manifest

def drop_undeclared_tensors(adapter_dir: str|Path, output_dir: str|Path):
    """Remove serialized LoRA factors whose leaf module is not declared in target_modules."""
    src=Path(adapter_dir); dst=Path(output_dir); dst.mkdir(parents=True,exist_ok=True)
    _,cfg,tensors=load_lora_pairs(src); targets=set(cfg.get('target_modules') or [])
    keep={k:v for k,v in tensors.items() if canonical_leaf(k) in targets}
    save_file(keep,str(dst/'adapter_model.safetensors'))
    (dst/'adapter_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    return sorted(keep)
