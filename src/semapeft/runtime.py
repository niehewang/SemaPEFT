from __future__ import annotations
from pathlib import Path
from dataclasses import MISSING,dataclass,field,fields
from collections.abc import Sequence as GenericSequence
from types import SimpleNamespace
from typing import Dict,List,Optional
import hashlib, json, logging, math, os, re, sys, types
import numpy as np
import torch, safetensors
from torch import nn
from safetensors import safe_open
from .source_extract import extract_class, extract_func, find_py_with

def sha256_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def _load_tensor_dict(path):
    path=Path(path); st=path/'adapter_model.safetensors'
    if st.exists():
        with safe_open(st,framework='pt') as f:return {k:f.get_tensor(k) for k in f.keys()}
    bp=path/'adapter_model.bin'
    if bp.exists(): return torch.load(bp,map_location='cpu')
    raise FileNotFoundError(path)

def _expected_modules(tensors):
    out=set()
    for k in tensors:
        m=re.match(r'.*\.([^.]+)\.lora_[AB]\.weight$',k)
        if m: out.add(m.group(1))
    return out or {'q_proj','v_proj','k_proj','o_proj','gate_proj','up_proj','down_proj'}

def load_vllm(adapter, runtime_root, version='0.31.0'):
    """Execute the exact pinned vLLM LoRA loader source in an isolated namespace.

    `runtime_root` must point to a local checkout/source snapshot of the requested version.
    This intentionally avoids importing unrelated top-level serving/CUDA initialization.
    """
    adapter=Path(adapter); root=Path(runtime_root)
    peft_path=find_py_with(root,'PEFTHelper','vllm/lora/peft_helper.py')
    weights_path=find_py_with(root,'LoRALayerWeights','vllm/lora/lora_weights.py')
    model_path=find_py_with(root,'LoRAModel','vllm/lora/lora_model.py')
    utils_path=next(iter(root.rglob('vllm/lora/utils.py')))
    class DummyLoRAConfig:max_lora_rank=4096;lora_dtype=torch.float32
    class DummyTensorizerConfig:
        def __init__(self,**kw):self.tensorizer_dir=None
    def init_logger(name):return logging.getLogger(name)
    pg={'json':json,'math':math,'os':os,'MISSING':MISSING,'dataclass':dataclass,'field':field,'fields':fields,
        'Literal':__import__('typing').Literal,'LoRAConfig':DummyLoRAConfig,'TensorizerConfig':DummyTensorizerConfig,'init_logger':init_logger}
    exec('from __future__ import annotations\n'+extract_class(peft_path,'PEFTHelper'),pg)
    PEFTHelper=pg['PEFTHelper']; helper=PEFTHelper.from_local_dir(str(adapter),None)
    ug={'re':re,'os':os}
    for fn in ['parse_fine_tuned_lora_name','is_base_embedding_weights','get_lora_id']:
        try:exec('from __future__ import annotations\n'+extract_func(utils_path,fn),ug)
        except Exception:pass
    parse=ug['parse_fine_tuned_lora_name']; isbase=ug.get('is_base_embedding_weights',lambda n:False); gid=ug.get('get_lora_id',lambda:1)
    wg={'torch':torch,'GenericSequence':GenericSequence,'PEFTHelper':PEFTHelper,'PIN_MEMORY':False}
    exec('from __future__ import annotations\n'+extract_class(weights_path,'LoRALayerWeights'),wg); LW=wg['LoRALayerWeights']
    class DummyWeightsMapper:pass
    class DummyFull:
        def __init__(self,**kw):self.__dict__.update(kw)
    class DummyMoE:pass
    mg={'os':os,'torch':torch,'safetensors':safetensors,'safe_open':safe_open,'PEFTHelper':PEFTHelper,
        'LoRALayerWeights':LW,'LoRAFullModuleWeights':DummyFull,'WeightsMapper':DummyWeightsMapper,
        'TensorizerConfig':DummyTensorizerConfig,'PIN_MEMORY':False,'parse_fine_tuned_lora_name':parse,
        'is_base_embedding_weights':isbase,'get_lora_id':gid,'MoEEPLoadSpec':DummyMoE,'_is_remote_expert_key':lambda *a,**k:False}
    exec('from __future__ import annotations\n'+extract_class(model_path,'LoRAModel'),mg); LM=mg['LoRAModel']
    tensors=_load_tensor_dict(adapter); exp=_expected_modules(tensors)
    lm=LM.from_local_checkpoint(str(adapter),exp,helper,lora_model_id=1,device='cpu',dtype=torch.float32,
        model_vocab_size=128,weights_mapper=None,tensorizer_config_dict=None,skip_prefixes=None,moe_ep_spec=None)
    rows=[]
    for name,lw in lm.loras.items():
        A=lw.lora_a.detach().float().cpu();B=lw.lora_b.detach().float().cpu()
        rows.append({'module':name,'rank':int(lw.rank),'alpha':float(lw.lora_alpha),'scale':float(lw.scaling),
                     'A_shape':list(A.shape),'B_shape':list(B.shape),'A':A.numpy(),'B':B.numpy()})
    return {'runtime':f'vllm-{version}','helper_scale':float(helper.vllm_lora_scaling_factor),'rows':rows,
            'source_hashes':{'peft':sha256_file(peft_path),'weights':sha256_file(weights_path),'model':sha256_file(model_path)}}

def _sglang_classes(root):
    root=Path(root); cfg_path=next(iter(root.rglob('sglang/srt/lora/lora_config.py'))); lora_path=next(iter(root.rglob('sglang/srt/lora/lora.py'))); utils_path=next(iter(root.rglob('sglang/srt/lora/utils.py'))); layer_utils=find_py_with(root,'get_layer_id')
    def snapshot_download(*a,**k):raise RuntimeError('remote disabled in isolated semantic extraction')
    cg={'json':json,'logging':logging,'os':os,'Dict':Dict,'Optional':Optional,'snapshot_download':snapshot_download}
    exec('from __future__ import annotations\n'+extract_class(cfg_path,'LoRAConfig'),cg);Cfg=cg['LoRAConfig']
    ug={'re':re,'torch':torch,'Iterable':__import__('typing').Iterable,'List':List,'Optional':Optional,'Set':__import__('typing').Set,'Tuple':__import__('typing').Tuple,'Union':__import__('typing').Union,'logging':logging}
    exec('from __future__ import annotations\n'+extract_func(utils_path,'get_normalized_target_modules'),ug); norm=ug['get_normalized_target_modules']
    lg={'re':re};exec('from __future__ import annotations\n'+extract_func(layer_utils,'get_layer_id'),lg); layerid=lg['get_layer_id']
    for name in ['sglang','sglang.srt','sglang.srt.lora']:
        m=types.ModuleType(name);m.__path__=[];sys.modules[name]=m
    um=types.ModuleType('sglang.srt.lora.utils');um.get_normalized_target_modules=norm;sys.modules['sglang.srt.lora.utils']=um
    class LoadConfig:pass
    class Backend:pass
    class Loader:pass
    class AutoConfig:pass
    gg={'logging':logging,'re':re,'Dict':Dict,'List':List,'Optional':Optional,'torch':torch,'nn':nn,'LoadConfig':LoadConfig,
        'get_layer_id':layerid,'BaseLoRABackend':Backend,'LoRAConfig':Cfg,'DefaultModelLoader':Loader,'AutoConfig':AutoConfig,
        '_ROUTED_EXPERT_PATTERN':re.compile(r'\.experts\.'),'logger':logging.getLogger('sglang-isolated')}
    exec('from __future__ import annotations\n'+extract_class(lora_path,'LoRALayer'),gg);Layer=gg['LoRALayer'];gg['LoRALayer']=Layer
    exec('from __future__ import annotations\n'+extract_class(lora_path,'LoRAAdapter'),gg);Adapter=gg['LoRAAdapter']
    return Cfg,Adapter,{'lora':sha256_file(lora_path),'config':sha256_file(cfg_path),'utils':sha256_file(utils_path)}

def load_sglang(adapter, runtime_root, version='0.5.20'):
    adapter=Path(adapter); Cfg,Adapter,src=_sglang_classes(runtime_root); tensors=_load_tensor_dict(adapter)
    ids=[]
    for k in tensors:
        m=re.search(r'\.layers\.(\d+)\.',k)
        if m:ids.append(int(m.group(1)))
    n=max(ids)+1 if ids else 1; cfg=Cfg(path=str(adapter))
    ad=Adapter(uid='semapeft',config=cfg,base_hf_config=SimpleNamespace(num_hidden_layers=n),load_config=None,lora_backend=None,base_model=None)
    ad.initialize_weights_from_tensors(tensors)
    state=[]
    for li,layer in enumerate(ad.layers):
        if not getattr(layer,'weights',None):continue
        for k,v in layer.weights.items():
            if torch.is_tensor(v):state.append({'layer':li,'key':k,'shape':list(v.shape),'norm':float(torch.linalg.vector_norm(v.float())),'tensor':v.detach().float().cpu().numpy()})
    return {'runtime':f'sglang-{version}','scale':float(ad.scaling),'r':int(cfg.r),'alpha':float(cfg.lora_alpha),
            'use_rslora':bool(cfg.hf_config.get('use_rslora',False)),'state':state,'source_hashes':src}

def recover_qv_from_sglang(load,q_rank,v_rank,q_out,v_out,in_dim):
    As=[x for x in load['state'] if 'qkv_proj' in x['key'] and 'lora_A' in x['key'] and len(x['shape'])==2]
    Bs=[x for x in load['state'] if 'qkv_proj' in x['key'] and 'lora_B' in x['key'] and len(x['shape'])==2]
    if not As or not Bs:return None
    ar,br=As[0],Bs[0];A=np.asarray(ar['tensor']);B=np.asarray(br['tensor'])
    if A.shape[0] < q_rank+2*v_rank or A.shape[1] != in_dim or B.shape[0] < q_out+2*v_out:return None
    return {'Aq':A[:q_rank,:],'Av':A[q_rank+v_rank:q_rank+2*v_rank,:],
            'Bq':B[:q_out,:q_rank],'Bv':B[q_out+v_out:q_out+2*v_out,:v_rank],
            'A_key':ar['key'],'B_key':br['key'],'fused_A_shape':list(A.shape),'fused_B_shape':list(B.shape)}
