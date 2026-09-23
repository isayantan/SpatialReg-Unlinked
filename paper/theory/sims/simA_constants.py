# constants tau_c^2, mu^2, delta_cov, D(R)/B for Matern designs; per-block composite info ratio (covariate-key, marginal over W_b)
import numpy as np, itertools, sys
from common import *
rng=np.random.default_rng(7)
B,K=49,6; s2=5.0; sX=1.0
pts,blk=design(B,K,rng)
P=perms(K); nP=len(P)
def hamming(p): return int((p!=np.arange(K)).sum())
def constants(nu,phi,t2):
    SW=cov(pts,nu,phi,s2); Sig=SW+t2*np.eye(len(pts)); Om=np.linalg.inv(Sig); n=len(pts)
    # tau_c^2
    tc2=0
    for b in range(B):
        idx=np.where(blk==b)[0]
        for a,c in itertools.combinations(idx,2):
            tc2=max(tc2,1/np.linalg.eigvalsh(Om[np.ix_([a,c],[a,c])])[0])
    # mu^2, delta_cov over common-key relabellings
    mu2=0; dmin=np.inf; Dmax=0
    for r in P:
        h=hamming(r)
        if h==0: continue
        idx=np.concatenate([np.where(blk==b)[0][r] for b in range(B)])  # R Sigma R^T = Sig[idx][:,idx]
        SigR=Sig[np.ix_(idx,idx)]
        M=Om@SigR; ev=np.linalg.eigvals(M).real
        mu2=max(mu2,ev.max()); D=0.5*(np.trace(M)-n); Dmax=max(Dmax,D/B); dmin=min(dmin,D/(h*B))
    return tc2,mu2,dmin,Dmax,SW
def comp_info_ratio(SW,t2,beta,reps=400):
    # total composite information (covariate-key sub-case, W_b marginalised per block) vs global areal GLS information tr(V^-1 D)
    n=len(pts); A=np.zeros((n,B)); A[np.arange(n),blk]=1; V=A.T@(SW+t2*np.eye(n))@A
    Iagg=sX**2*np.trace(np.linalg.inv(V)@np.diag([K]*B)); Ior=sX**2*np.trace(np.linalg.inv(SW+t2*np.eye(n)))
    tot=0
    for b in range(B):
        idx=np.where(blk==b)[0]; C=SW[np.ix_(idx,idx)]+t2*np.eye(K); Ci=np.linalg.inv(C)
        J=[]
        for r in range(reps):
            X=rng.normal(size=K)*sX; W=rng.multivariate_normal(np.zeros(K),C); pi=P[rng.integers(nP)]
            Y=X[pi]*beta+W   # pi applied: Y_i = X_{pi(i)} beta + W*_i  (equivalent up to relabelling)
            XP=X[P]                     # nP x K candidates
            c=np.einsum('pi,ij,pj->p',XP,Ci,XP); bb=XP@Ci@Y
            q=-0.5*(c*beta**2-2*bb*beta); q-=q.max(); w=np.exp(q); w/=w.sum()
            S=bb-c*beta
            J.append((w*c).sum()-((w*S**2).sum()-(w*S).sum()**2))
        tot+=np.mean(J)
    return tot/Iagg, tot/Ior
rows=[]
for nu in [0.5,1.5,2.5]:
    for phi in [0.25,0.5,1.0]:
        t2=0.5
        tc2,mu2,dmin,Dmax,SW=constants(nu,phi,t2)
        rs=[comp_info_ratio(SW,t2,beta,reps=150)[0] for beta in [0.5,1.0,2.0]]; ro=comp_info_ratio(SW,t2,1.0,reps=150)[1]
        rows.append((nu,phi,t2,tc2,mu2,dmin,Dmax,*rs,ro))
        print(f"nu={nu} phi={phi} t2={t2}: tau_c2={tc2:.2f} mu2={mu2:.1f} delta_cov={dmin:.3f} maxD/B={Dmax:.2f} Icl/Iagg(b=.5,1,2)={rs[0]:.2f},{rs[1]:.2f},{rs[2]:.2f} Icl/Ior(b=1)={ro:.2f}",flush=True)
for t2 in [0.1,2.0]:
    tc2,mu2,dmin,Dmax,SW=constants(0.5,0.5,t2)
    rs=[comp_info_ratio(SW,t2,beta,reps=150)[0] for beta in [0.5,1.0,2.0]]; ro=comp_info_ratio(SW,t2,1.0,reps=150)[1]
    rows.append((0.5,0.5,t2,tc2,mu2,dmin,Dmax,*rs,ro))
    print(f"nu=0.5 phi=0.5 t2={t2}: tau_c2={tc2:.2f} mu2={mu2:.1f} delta_cov={dmin:.3f} maxD/B={Dmax:.2f} Icl/Iagg={rs[0]:.2f},{rs[1]:.2f},{rs[2]:.2f} Icl/Ior(b=1)={ro:.2f}",flush=True)
np.save('simA_rows.npy',np.array(rows))
