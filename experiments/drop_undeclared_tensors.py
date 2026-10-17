#!/usr/bin/env python
import argparse,json
from semapeft.peft import drop_undeclared_tensors
ap=argparse.ArgumentParser(description='Repair undeclared serialized LoRA tensors by retaining only target_modules.')
ap.add_argument('--adapter',required=True);ap.add_argument('--output',required=True)
a=ap.parse_args(); print(json.dumps({'kept_tensor_keys':drop_undeclared_tensors(a.adapter,a.output)},indent=2))
