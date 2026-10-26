#!/usr/bin/env python3
"""Exercise the actual model loader and decoder using the existing tiny fixture."""
import json, os, pathlib, shutil, subprocess, sys
root=pathlib.Path(sys.argv[1]).resolve()
exe=root/'moe-trace'; model=root/'tiny.gguf'; work=root/'tiny-tests'
work.mkdir(exist_ok=True)
env={k:v for k,v in os.environ.items() if k not in ['CNM_EXPERT_MASK','CNM_EXPERT_BIAS','TRACE_GPU','TRACE_LONG','TRACE_TOKEN_FILE']}
env['CUDA_VISIBLE_DEVICES']=''
def run(name,plan=None,mask=None,expect=True):
    out=work/name;out.mkdir(exist_ok=True)
    e=dict(env)
    if plan is not None:
        path=work/(name+'.json');path.write_text(plan if isinstance(plan,str) else json.dumps(plan));e['CNM_EXPERT_BIAS']=str(path)
    if mask:e['CNM_EXPERT_MASK']=str(mask)
    with (out/'run.log').open('w') as f:p=subprocess.run([str(exe),str(model),str(out)],env=e,stdout=f,stderr=subprocess.STDOUT)
    assert (p.returncode==0)==expect,(name,p.returncode)
    if not expect:assert 'CNM_EXPERT_BIAS' in (out/'run.log').read_text(),name
    return out
hot=json.loads((root/'inputs/tiny-hard.json').read_text())
cold={il:sorted(set(range(32))-set(ids)) for il,ids in hot.items()}
base=run('base');zero=run('zero',{'bias':0,'layers':cold})
hard=run('hard',mask=root/'inputs/tiny-hard.json');inf=run('inf',{'bias':'-inf','layers':cold})
finite=run('finite',{'bias':-2,'layers':cold});per=run('per',{'layers':{il:{str(e):-2 for e in ids} for il,ids in cold.items()}})
def same(a,b):
    files=[p.name for p in a.iterdir() if p.name!='run.log']
    assert set(files)=={p.name for p in b.iterdir() if p.name!='run.log'}
    assert all((a/f).read_bytes()==(b/f).read_bytes() for f in files),(a,b)
same(base,zero);same(hard,inf);same(finite,per)
assert (base/'logits.f32').read_bytes()!=(finite/'logits.f32').read_bytes()
invalid=[{'bias':1,'layers':cold},{'bias':-1,'layers':{'0':[32]}},{'bias':-1,'layers':{'0':[-1]}},{'bias':-1,'layers':{'0':[1,1]}},{'layers':{'0':[1]}},{'layers':{'00':{'1':-1}}},{'bias':'-inf','layers':{'0':list(range(30))}},{'bias':-1e100,'layers':cold},{'layers':{}},{'layers':{'999':{'1':-1}}},{'layers':{'0':{'1':True}}},{'bias':'nan','layers':cold},{'bias':-1,'layers':cold,'extra':0},'{"layers":{"0":{"1":-1,"1":-2}}}']
for n,plan in enumerate(invalid):run('invalid-'+str(n),plan,expect=False)
run('mutually-exclusive',{'bias':-1,'layers':cold},mask=root/'inputs/tiny-hard.json',expect=False)
(work/'result.json').write_text(json.dumps({'zero_equals_unbiased':True,'negative_infinity_equals_hard_all_tensors_and_logits':True,'scalar_equals_per_expert':True,'finite_changes_logits':True,'invalid_plans_rejected':len(invalid)+1},indent=2)+'\n')
print((work/'result.json').read_text())
