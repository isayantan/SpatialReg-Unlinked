# Remedy for slow mixing at high SNR: block-Gibbs moves that re-sample the assignment of m=4 positions exactly
# (4! = 24 configurations per key), mixed with transpositions. Compare chains from random starts vs true start.
import numpy as np, itertools, time
from numba import njit
from check_mixing import make_block, chain_with_start
from common import matern
rng=np.random.default_rng(5)
PERMS4=np.array(list(itertools.permutations(range(4))),dtype=np.int64)

@njit(cache=True)
def chain_gibbs(Y, XB, Om, a, c, mode, n_burn, n_samp, thin, seed, PERMS):
    np.random.seed(seed); K=Y.shape[0]; m=4; nP=PERMS.shape[0]
    r=np.empty(K)
    for k in range(K): r[k]=Y[a[k]]-XB[c[k]]
    Omr=Om@r; lp=-0.5*(r@Omr); best=lp
    Pa=np.zeros((K,K)); Pc=np.zeros((K,K)); ns=0
    pos=np.empty(m,dtype=np.int64); vals=np.empty(m,dtype=np.int64); vals2=np.empty(m,dtype=np.int64); newr=np.empty(m); dr=np.empty(m); logw=np.empty(nP)
    for it in range(n_burn+n_samp*thin):
        for sweep in range(K//m+1):
            # choose which key to move
            which = 1 if mode==1 else (0 if mode==2 else np.random.randint(3))
            # sample m distinct positions
            cnt=0
            while cnt<m:
                p=np.random.randint(K); ok=True
                for q in range(cnt):
                    if pos[q]==p: ok=False
                if ok: pos[cnt]=p; cnt+=1
            for q in range(m): vals[q]= a[pos[q]] if which!=1 else c[pos[q]]
            for q in range(m): vals2[q]= c[pos[q]]
            # log-density change for each of the nP re-assignments (exact quadratic form update)
            for pi in range(nP):
                for q in range(m):
                    v=vals[PERMS[pi,q]]
                    if which==0: newr[q]=Y[v]-XB[c[pos[q]]]
                    elif which==1: newr[q]=Y[a[pos[q]]]-XB[v]
                    else: newr[q]=Y[v]-XB[vals2[PERMS[pi,q]]]
                    dr[q]=newr[q]-r[pos[q]]
                s=0.0
                for q in range(m):
                    s+=dr[q]*Omr[pos[q]]
                    for q2 in range(m): s+=0.5*Om[pos[q],pos[q2]]*dr[q]*dr[q2]
                logw[pi]=-s
            mx=logw.max(); tot=0.0
            for pi in range(nP): logw[pi]=np.exp(logw[pi]-mx); tot+=logw[pi]
            u=np.random.rand()*tot; acc=0.0; chosen=nP-1
            for pi in range(nP):
                acc+=logw[pi]
                if u<=acc: chosen=pi; break
            # apply
            for q in range(m):
                v=vals[PERMS[chosen,q]]
                if which==0: a[pos[q]]=v
                elif which==1: c[pos[q]]=v
                else: a[pos[q]]=v; c[pos[q]]=vals2[PERMS[chosen,q]]
                newr[q]=(Y[a[pos[q]]]-XB[c[pos[q]]]); dr[q]=newr[q]-r[pos[q]]
            dl=0.0
            for q in range(m):
                dl-=dr[q]*Omr[pos[q]]
                for q2 in range(m): dl-=0.5*Om[pos[q],pos[q2]]*dr[q]*dr[q2]
            for q in range(m): r[pos[q]]=newr[q]
            for mm in range(K):
                for q in range(m): Omr[mm]+=dr[q]*Om[mm,pos[q]]
            lp+=dl
            if lp>best: best=lp
        if it>=n_burn and (it-n_burn)%thin==0:
            ns+=1
            for k in range(K): Pa[k,a[k]]+=1.0; Pc[k,c[k]]+=1.0
    return Pa/ns, Pc/ns, best

print("K=5, beta=2: block-Gibbs chain vs exact enumeration")
Y,X,Om,at,ct=make_block(5,2.0); XB=X*2.0; K=5
perms=list(itertools.permutations(range(K))); logw=np.empty((len(perms),len(perms)))
for i,a in enumerate(perms):
    for j,c in enumerate(perms):
        r=Y[list(a)]-XB[list(c)]; logw[i,j]=-0.5*r@Om@r
w=np.exp(logw-logw.max()); w/=w.sum(); Pc_exact=np.zeros((K,K))
for i,a in enumerate(perms):
    for j,c in enumerate(perms):
        for k in range(K): Pc_exact[k,c[k]]+=w[i,j]
Pa,Pc,_=chain_with_start(Y,XB,Om,np.arange(K),np.arange(K),0,200,4000,2,1)
Pa2,Pc2,_=chain_gibbs(Y,XB,Om,np.arange(K),np.arange(K),0,200,4000,2,1,PERMS4)
print(f"   transposition chain: max|P_c-exact| = {np.abs(Pc-Pc_exact).max():.3f};  block-Gibbs chain: {np.abs(Pc2-Pc_exact).max():.3f}")
print("K=20/30, both keys: block-Gibbs chains, random starts vs true start")
for K in [20,30]:
    for beta in [1.0,2.0,4.0]:
        res=[]
        for rep in range(5):
            Y,X,Om,at,ct=make_block(K,beta); XB=X*beta
            t0=time.time()
            P1a,P1c,b1=chain_gibbs(Y,XB,Om,rng.permutation(K),rng.permutation(K),0,200,800,2,11+rep,PERMS4)
            P2a,P2c,b2=chain_gibbs(Y,XB,Om,rng.permutation(K),rng.permutation(K),0,200,800,2,97+rep,PERMS4)
            P3a,P3c,b3=chain_gibbs(Y,XB,Om,at.copy(),ct.copy(),0,200,800,2,5+rep,PERMS4)
            dt=(time.time()-t0)/3
            r=Y[at]-XB[ct]; lp_true=-0.5*r@Om@r
            ptrue=np.mean([P3c[k,ct[k]] for k in range(K)])
            res.append((np.abs(P1c-P2c).max(), np.abs(P1c-P3c).max(), b1-lp_true, ptrue, dt))
        res=np.array(res)
        print(f"   K={K} beta={beta}: max|P_c chain1-chain2| = {res[:,0].mean():.3f}, random-start vs true-start = {res[:,1].mean():.3f}, best logpost - true = {res[:,2].mean():+.1f}, P(record at true place) = {res[:,3].mean():.2f}, {res[:,4].mean():.2f}s per chain")
