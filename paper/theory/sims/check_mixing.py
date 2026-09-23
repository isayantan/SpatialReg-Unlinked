# Does the transposition chain mix on S_K x S_K at K=20?  (a) K=5: chain vs exact enumeration of 5!^2 keys.
# (b) K=20,30: two independent chains from random starts vs a chain started at the TRUE key: agreement of marginal
#     assignment matrices, and log-posterior of the chains' best sample vs the true key's.
import numpy as np, itertools, time
from numba import njit
from simE_realistic import block_mcmc
from common import matern
rng=np.random.default_rng(3)

@njit(cache=True)
def chain_with_start(Y, XB, Om, a, c, mode, n_burn, n_samp, thin, seed):
    # same as block_mcmc but returns best log-density seen and the marginal matrices for a AND c
    np.random.seed(seed); K=Y.shape[0]
    r=np.empty(K)
    for k in range(K): r[k]=Y[a[k]]-XB[c[k]]
    Omr=Om@r; lp=-0.5*(r@Omr); best=lp
    Pa=np.zeros((K,K)); Pc=np.zeros((K,K)); ns=0
    for it in range(n_burn+n_samp*thin):
        for p in range(K):
            k=np.random.randint(K); l=np.random.randint(K)
            if k==l: continue
            which = 1 if mode==1 else (0 if mode==2 else np.random.randint(3))
            if which==0: dk=Y[a[l]]-Y[a[k]]; dl=-dk
            elif which==1: dk=-(XB[c[l]]-XB[c[k]]); dl=-dk
            else: dk=(Y[a[l]]-XB[c[l]])-r[k]; dl=(Y[a[k]]-XB[c[k]])-r[l]
            dlog=-(dk*Omr[k]+dl*Omr[l]+0.5*(Om[k,k]*dk*dk+Om[l,l]*dl*dl+2.0*Om[k,l]*dk*dl))
            if np.log(np.random.rand())<dlog:
                if which==0 or which==2: t=a[k]; a[k]=a[l]; a[l]=t
                if which==1 or which==2: t=c[k]; c[k]=c[l]; c[l]=t
                r[k]+=dk; r[l]+=dl
                for m in range(K): Omr[m]+=dk*Om[m,k]+dl*Om[m,l]
                lp+=dlog
                if lp>best: best=lp
        if it>=n_burn and (it-n_burn)%thin==0:
            ns+=1
            for k in range(K): Pa[k,a[k]]+=1.0; Pc[k,c[k]]+=1.0
    return Pa/ns, Pc/ns, best

def make_block(K, beta, s2=5.0, phi=0.5, t2=0.5, mode=0):
    pts=rng.uniform(size=(K,2))*0.15   # one cell of a 7x7 grid on the unit square
    D=np.linalg.norm(pts[:,None]-pts[None],axis=2); SW=matern(D,0.5,phi,s2); Om=np.linalg.inv(SW+t2*np.eye(K))
    X=rng.normal(size=K); W=rng.multivariate_normal(np.zeros(K),SW); eps=rng.normal(size=K)*np.sqrt(t2)
    piX=rng.permutation(K); piS=(np.arange(K) if mode==1 else (piX if mode==2 else rng.permutation(K)))
    Y=X[piX]*beta+W[piS]+eps
    # true aligned key in (a,c) form: aligned position k = location k (true order); Y index with location k: inverse of piS ; X row at location k: piX[piS^-1[k]]
    inv_piS=np.argsort(piS); a_true=inv_piS.copy(); c_true=piX[inv_piS]
    return Y,X,Om,a_true,c_true

print("(a) K=5, both keys: chain vs exact enumeration of 5!^2 = 14400 keys (max abs error of marginal matrix P_c)")
for beta in [0.5,2.0]:
    Y,X,Om,at,ct=make_block(5,beta); XB=X*beta; K=5
    perms=list(itertools.permutations(range(K))); logw=np.empty((len(perms),len(perms)))
    for i,a in enumerate(perms):
        for j,c in enumerate(perms):
            r=Y[list(a)]-XB[list(c)]; logw[i,j]=-0.5*r@Om@r
    w=np.exp(logw-logw.max()); w/=w.sum()
    Pc_exact=np.zeros((K,K)); Pa_exact=np.zeros((K,K))
    for i,a in enumerate(perms):
        for j,c in enumerate(perms):
            for k in range(K): Pc_exact[k,c[k]]+=w[i,j]; Pa_exact[k,a[k]]+=w[i,j]
    Pa,Pc,_=chain_with_start(Y,XB,Om,np.arange(K),np.arange(K),0,200,4000,2,1)
    print(f"   beta={beta}: max|P_c(chain)-P_c(exact)| = {np.abs(Pc-Pc_exact).max():.3f}, max|P_a diff| = {np.abs(Pa-Pa_exact).max():.3f}  (entries are probabilities in [0,1])")

print("(b) K=20 and 30, both keys: agreement between independent random-start chains, and vs a chain started at the true key")
for K in [20,30]:
    for beta in [0.5,1.0,2.0,4.0]:
        res=[]
        for rep in range(5):
            Y,X,Om,at,ct=make_block(K,beta); XB=X*beta
            t0=time.time()
            P1a,P1c,b1=chain_with_start(Y,XB,Om,rng.permutation(K),rng.permutation(K),0,300,1500,2,11+rep)
            P2a,P2c,b2=chain_with_start(Y,XB,Om,rng.permutation(K),rng.permutation(K),0,300,1500,2,97+rep)
            P3a,P3c,b3=chain_with_start(Y,XB,Om,at.copy(),ct.copy(),0,300,1500,2,5+rep)
            dt=time.time()-t0
            r=Y[at]-XB[ct]; lp_true=-0.5*r@Om@r
            # per-record posterior probability of the true covariate placement (mean over k of P_c[k, c_true[k]])
            ptrue=np.mean([P3c[k,ct[k]] for k in range(K)])
            res.append((np.abs(P1c-P2c).max(), np.abs(P1c-P3c).max(), b1-lp_true, b2-lp_true, ptrue, dt))
        res=np.array(res)
        print(f"   K={K} beta={beta}: max|P_c chain1-chain2| = {res[:,0].mean():.3f}, max|P_c random-start - true-start| = {res[:,1].mean():.3f}, best logpost - logpost(true key) = {res[:,2].mean():+.1f}/{res[:,3].mean():+.1f}, P(record at true place) = {res[:,4].mean():.2f}, {res[:,5].mean()/3:.2f}s per chain")
