import math
from semapeft.peft import lora_scale

def test_plain_scale(): assert lora_scale({'lora_alpha':16,'use_rslora':False},'q_proj',8)==2.0
def test_rslora_scale(): assert abs(lora_scale({'lora_alpha':16,'use_rslora':True},'q_proj',16)-4.0)<1e-12
def test_alpha_pattern(): assert lora_scale({'lora_alpha':16,'use_rslora':False,'alpha_pattern':{'q_proj':64}},'q_proj',8)==8.0
