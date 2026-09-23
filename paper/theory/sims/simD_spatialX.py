# information with spatially correlated covariate: I_or = tr(Omega Sigma_g), I_agg = tr(V^-1 A' Sigma_g A) vs iid X (Sigma_g = I)
import numpy as np
from common import *
rng=np.random.default_rng(3)
B,K=49,6; s2,t2=5.0,0.5
pts,blk=design(B,K,rng); n=len(pts)
A=np.zeros((n,B)); A[np.arange(n),blk]=1
SW=cov(pts,0.5,0.5,s2); Sig=SW+t2*np.eye(n); Om=np.linalg.inv(Sig); V=A.T@Sig@A; Vi=np.linalg.inv(V)
def infos(Sg):
    return np.trace(Om@Sg), np.trace(Vi@A.T@Sg@A)
print("X field (sigma_g^2=1) | I_or | I_agg | I_or/I_agg")
for lab,Sg in [("iid",np.eye(n)),("exp range 0.1",cov(pts,0.5,0.1,1.0)),("exp range 0.5 (same as W)",cov(pts,0.5,0.5,1.0)),("exp range 2",cov(pts,0.5,2.0,1.0)),("Matern 2.5 range 0.5",cov(pts,2.5,0.5,1.0)),("0.5 iid + 0.5 exp0.5",0.5*np.eye(n)+0.5*cov(pts,0.5,0.5,1.0))]:
    a,b=infos(Sg); print(f"{lab:28s} {a:8.1f} {b:7.2f} {a/b:6.1f}")
# per-block v^g for nearest pairs: floors p^X with v^g
import itertools
for lab,Sg in [("exp range 0.5",cov(pts,0.5,0.5,1.0)),("Matern 2.5 range 0.5",cov(pts,2.5,0.5,1.0))]:
    fl=[]
    for b in range(B):
        idx=np.where(blk==b)[0]; best=0
        for i,j in itertools.combinations(idx,2):
            v=Sg[i,i]+Sg[j,j]-2*Sg[i,j]; best=max(best,np.arctan(np.sqrt(t2)/(2.0*np.sqrt(v/2)))/np.pi)  # beta=2
        fl.append(best)
    print(lab,"beta=2: mean over blocks of max-pair floor p^X =",round(np.mean(fl),3)," (iid X floor:",round(np.arctan(np.sqrt(t2)/2)/np.pi,3),")")
