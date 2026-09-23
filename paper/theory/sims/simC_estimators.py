# composite ML (exact, K=4) vs areal GLS vs oracle GLS, general case (both keys per block, independent)
import numpy as np
from common import *
from scipy.optimize import minimize_scalar
rng=np.random.default_rng(23)
K=4; s2,phi,t2,sX=5.0,0.5,0.5,1.0; tau=np.sqrt(t2)
P=perms(K); nP=len(P)
def one(B,beta,reps,case):
    pts,blk=design(B,K,rng); SW=cov(pts,0.5,phi,s2); n=len(pts); Sig=SW+t2*np.eye(n)
    A=np.zeros((n,B)); A[np.arange(n),blk]=1
    V=A.T@Sig@A; Vi=np.linalg.inv(V); Dm=np.diag([K]*B)
    Iagg=sX**2*np.trace(Vi@Dm)
    Om=np.linalg.inv(Sig); Ior=sX**2*np.trace(Om)
    # precompute block covariances for candidate piS (case general) or fixed
    blocks=[np.where(blk==b)[0] for b in range(B)]
    Cinv={}
    for b in range(B):
        idx=blocks[b]; Cb=SW[np.ix_(idx,idx)]
        if case=='X': Cinv[b]=[np.linalg.inv(Cb+t2*np.eye(K))]
        else: Cinv[b]=[np.linalg.inv(Cb[np.ix_(p,p)]+t2*np.eye(K)) for p in P]
    est={'areal':[],'oracle':[],'cl':[],'cl_local':[]}; Jcl=[]
    for r in range(reps):
        X=rng.normal(size=n)*sX; W=rng.multivariate_normal(np.zeros(n),SW); eps=rng.normal(size=n)*tau
        Y=np.zeros(n); Xtrue=X.copy(); order=np.arange(n)
        for b in range(B):
            idx=blocks[b]; piX=P[rng.integers(nP)]; piS=P[rng.integers(nP)] if case!='X' else np.arange(K)
            Y[idx]=X[idx][piX]*beta+W[idx][piS]+eps[idx]; Xtrue[idx]=X[idx][piX]; order[idx]=idx[piS]
        # areal GLS
        T=A.T@Y; S=A.T@X; est['areal'].append((S@Vi@T)/(S@Vi@S))
        # oracle GLS: Y - Xtrue*beta = W[order] + eps, covariance SW[order][:,order] + t2 I
        Oo=np.linalg.inv(SW[np.ix_(order,order)]+t2*np.eye(n))
        est['oracle'].append((Xtrue@Oo@Y)/(Xtrue@Oo@Xtrue))
        # composite likelihood: per block a,b,c over candidate (piX,piS)
        abc=[]
        for b in range(B):
            idx=blocks[b]; Yb=Y[idx]; XP=X[idx][P]
            aa=[];bb=[];cc=[]
            for Ci in Cinv[b]:
                aa.append(np.full(nP,Yb@Ci@Yb)); bb.append(XP@Ci@Yb); cc.append(np.einsum('pi,ij,pj->p',XP,Ci,XP))
            abc.append((np.concatenate(aa),np.concatenate(bb),np.concatenate(cc)))
        def negll(bt):
            tot=0
            for a,b_,c in abc:
                q=-0.5*(a-2*b_*bt+c*bt*bt); m=q.max(); tot+=m+np.log(np.exp(q-m).sum())
            return -tot
        grid=np.linspace(-4,4,161); vals=np.array([negll(g) for g in grid]); g0=grid[vals.argmin()]
        rres=minimize_scalar(negll,bounds=(g0-0.1,g0+0.1),method='bounded',options={'xatol':1e-5})
        est['cl'].append(rres.x)
        # local maximiser started at the areal estimate: search within 3 areal sd of it
        w=3/np.sqrt(Iagg); ar=est['areal'][-1]
        rloc=minimize_scalar(negll,bounds=(ar-w,ar+w),method='bounded',options={'xatol':1e-5})
        est['cl_local'].append(rloc.x)
        # observed composite information at true beta
        J=0
        for a,b_,c in abc:
            q=-0.5*(a-2*b_*beta+c*beta*beta); w=np.exp(q-q.max()); w/=w.sum(); Sc=b_-c*beta
            J+=(w*c).sum()-((w*Sc**2).sum()-(w*Sc).sum()**2)
        Jcl.append(J)
    out=dict(B=B,beta=beta,case=case,mse_areal=np.mean((np.array(est['areal'])-beta)**2),
             mse_oracle=np.mean((np.array(est['oracle'])-beta)**2),
             mse_cl=np.mean((np.array(est['cl'])-beta)**2),bias_cl=np.mean(est['cl'])-beta,
             mse_cl_local=np.mean((np.array(est['cl_local'])-beta)**2),bias_cl_local=np.mean(est['cl_local'])-beta,
             wrong_sign_cl=np.mean(np.sign(est['cl'])!=np.sign(beta)),
             Iagg=Iagg,Icl=np.mean(Jcl),pred_ratio=np.mean(Jcl)/Iagg,
             mse_bound_areal=(K*s2+t2)/(sX**2*(B-2)))
    out['emp_ratio']=out['mse_areal']/out['mse_cl']; out['emp_ratio_local']=out['mse_areal']/out['mse_cl_local']; out['Ior']=Ior
    return out
res=[]
for case in ['X','G']:
    for B in [25,49,100]:
        for beta in [0.5,1.0,2.0]:
            d=one(B,beta,reps=150,case=case); res.append(d)
            print({k:(round(v,4) if isinstance(v,float) else v) for k,v in d.items()},flush=True)
np.save('simC_res.npy',res,allow_pickle=True)
