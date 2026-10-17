from pathlib import Path
import ast, hashlib, json, os, sys, textwrap

def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def extract_node_source(path, pred):
    src=Path(path).read_text(encoding="utf-8",errors="ignore")
    tree=ast.parse(src)
    lines=src.splitlines()
    for n in tree.body:
        if pred(n):
            starts=[getattr(n,"lineno",1)]
            for dec in getattr(n,"decorator_list",[]):
                starts.append(dec.lineno)
            start=min(starts)-1
            end=getattr(n,"end_lineno",None)
            if end is None:
                return ast.get_source_segment(src,n)
            return "\n".join(lines[start:end])
    raise RuntimeError(f"AST node not found in {path}")

def extract_class(path,name):
    return extract_node_source(path,lambda n:isinstance(n,ast.ClassDef) and n.name==name)

def extract_func(path,name):
    return extract_node_source(path,lambda n:isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)

def find_py_with(root, symbol, preferred_substr=None):
    c=[]
    for p in Path(root).rglob("*.py"):
        if preferred_substr and preferred_substr not in str(p):
            continue
        s=p.read_text(encoding="utf-8",errors="ignore")
        if f"def {symbol}" in s or f"class {symbol}" in s:
            c.append(p)
    if not c and preferred_substr:
        return find_py_with(root,symbol,None)
    if not c:
        raise RuntimeError(f"Could not locate {symbol} under {root}")
    return sorted(c,key=lambda p:len(str(p)))[0]
