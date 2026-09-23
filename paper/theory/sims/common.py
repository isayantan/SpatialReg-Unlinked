import numpy as np, scipy.special as sp, itertools
def matern(d,nu,phi,s2):
    d=np.asarray(d,float); out=np.full(d.shape,s2); m=d>0; x=np.sqrt(2*nu)*d[m]/phi
    out[m]=s2*2**(1-nu)/sp.gamma(nu)*x**nu*sp.kv(nu,x); return out
def design(B,K,rng):
    g=int(round(np.sqrt(B))); pts=[];blk=[]
    for i in range(g):
        for j in range(g):
            p=rng.uniform(size=(K,2))/g+np.array([i,j])/g; pts.append(p); blk+=[i*g+j]*K
    return np.vstack(pts),np.array(blk)
def cov(pts,nu,phi,s2):
    D=np.linalg.norm(pts[:,None]-pts[None],axis=2); return matern(D,nu,phi,s2)
def perms(K): return np.array(list(itertools.permutations(range(K))))
def arctan_p(a): return np.arctan(1/a)/np.pi
