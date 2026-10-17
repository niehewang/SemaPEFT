#!/usr/bin/env python
import argparse,json
from semapeft.peft import lower_adapter_to_scale
ap=argparse.ArgumentParser(description='Exact scale lowering for a PEFT adapter.')
ap.add_argument('--adapter',required=True);ap.add_argument('--output',required=True);ap.add_argument('--target-scale',type=float,required=True)
a=ap.parse_args(); print(json.dumps(lower_adapter_to_scale(a.adapter,a.output,a.target_scale),indent=2))
