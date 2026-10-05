# empirical re-linking rate of the Bayes/ML adversary vs the floors of Theorem 1, paper design B=49, K=6, exponential kernel
import numpy as np, itertools
from common import *
rng=np.random.default_rng(11)
B,K=49,6; s2,phi,t2,sX=5.0,0.5,0.5,1.0; tau=np.sqrt(t2)
P=perms(K); nP=len(P)
pts,blk=design(B,K,rng); SW=cov(pts,0.5,phi,s2); n=len(pts)
Sig=SW+t2*np.eye(n)
def run(beta,reps=200):
    # returns dict of empirical rates and floors
    outX_strong=[];outX_marg=[];outS_strong=[];misX=[];allX=[]; floorX=[];floorS=[];floor_mis=[]
    for r in range(reps):
        X=rng.normal(size=n)*sX; W=rng.multivariate_normal(np.zeros(n),SW); eps=rng.normal(size=n)*tau
        wrongXs=0;wrongXm=0;wrongS=0;mis=0
        for b in range(B):
            idx=np.where(blk==b)[0]; Xb=X[idx]; Wb=W[idx]; eb=eps[idx]
            piX=P[rng.integers(nP)]; piS=P[rng.integers(nP)]
            # covariate-key sub-case data (piS=I): Y = Xb[piX]*beta + Wb + eps
            Y=Xb[piX]*beta+Wb+eb
            XP=Xb[P]
            # strong adversary: knows W -> least squares over sigma
            d=((Y-Wb)[None,:]-XP*beta)**2; est=d.sum(1).argmin()
            wrongXs+= int(not np.array_equal(P[est],piX))
            mis+= int((P[est]!=piX).sum())
            # marginal adversary (does not know W): block-marginal Gaussian with C=SW_bb+t2 I
            C=SW[np.ix_(idx,idx)]+t2*np.eye(K); Ci=np.linalg.inv(C)
            R=Y[None,:]-XP*beta; q=-0.5*np.einsum('pi,ij,pj->p',R,Ci,R); estm=q.argmax()
            wrongXm+= int(not np.array_equal(P[estm],piX))
            # location-key sub-case: Y = Xb*beta + Wb[piS] + eps ; strong adversary knows W (values, not order)
            Y2=Xb*beta+Wb[piS]+eb
            d2=((Y2-Xb*beta)[None,:]-Wb[P])**2; est2=d2.sum(1).argmin()
            # ties: distinct W values a.s., so exact
            wrongS+= int(not np.array_equal(P[est2],piS))
            if r==0:
                # floors for this block (X-floor is analytic; S-floor depends on the pair variances)
                fS=0
                for i,j in itertools.combinations(range(K),2):
                    v=SW[idx[i],idx[i]]+SW[idx[j],idx[j]]-2*SW[idx[i],idx[j]]
                    fS=max(fS,arctan_p(np.sqrt(v)/(np.sqrt(2)*tau)))
                floorS.append(fS)
        outX_strong.append(wrongXs/B); outX_marg.append(wrongXm/B); outS_strong.append(wrongS/B); misX.append(mis/n)
        allX.append(int(wrongXs==0))
    pX=arctan_p(abs(beta)*sX/tau)
    return dict(beta=beta, pX=pX, X_strong=np.mean(outX_strong), X_marg=np.mean(outX_marg),
                mis_floor=2*pX/K, mis_emp=np.mean(misX), all_bound=(1-pX)**B, all_emp=np.mean(allX),
                pS=np.mean(floorS), pS_min=np.min(floorS), S_strong=np.mean(outS_strong))
res=[]
for beta in [0.25,0.5,1.0,2.0,4.0,8.0]:
    d=run(beta,reps=100); res.append(d)
    print({k:(round(v,4) if isinstance(v,float) else v) for k,v in d.items()},flush=True)
np.save('simB_res.npy',res,allow_pickle=True)
