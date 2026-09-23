import sys, numpy as np, ast
import simE3_aug as SE
job=int(sys.argv[1]); njobs=int(sys.argv[2])
done=set()
for l in open('simE3_rows0-3.log'):
    if l.strip():
        d=ast.literal_eval(l.strip()); done.add((d['mode'],d['B'],d['Kmean'],d['beta'],d['nu']))
settings=[]
for mode in [0,2,1]:
    for beta in [0.25,0.5,1.0,2.0]:
        for (B,Kmean) in [(64,20),(100,30)]:
            settings.append(dict(B=B,Kmean=Kmean,beta_true=np.array([beta]),mode=mode,nu=0.5,phi=0.5,s2=5.0,t2=0.5,sX=1.0,xfield=(0.3 if mode==2 else None),reps=24))
settings.append(dict(B=64,Kmean=20,beta_true=np.array([1.0]),mode=0,nu=1.5,phi=0.5,s2=5.0,t2=0.5,sX=1.0,xfield=None,reps=24))
settings.append(dict(B=12,Kmean=96,beta_true=np.array([0.5]),mode=1,nu=0.5,phi=0.5,s2=1.0,t2=1.0,sX=1.0,xfield=None,reps=24))
settings.append(dict(B=12,Kmean=96,beta_true=np.array([1.0]),mode=1,nu=0.5,phi=0.5,s2=1.0,t2=1.0,sX=1.0,xfield=None,reps=24))
todo=[st for st in settings if (st['mode'],st['B'],st['Kmean'],float(st['beta_true'][0]),st['nu']) not in done]
orig=SE.aug_newton
def fast(*a,**k):
    k.setdefault('n_iter',10); k.setdefault('n_samp',150); return orig(*a,**k)
SE.aug_newton=fast
for i,st in enumerate(todo):
    if i%njobs==job:
        print({k:(round(v,4) if isinstance(v,float) else v) for k,v in SE.run(**st).items()},flush=True)
