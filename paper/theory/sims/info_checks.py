# Three exact-enumeration checks (K=4 or 5 per block, W_b integrated out):
# (a) location key: information share and floor p^S do not depend on beta; cost of not knowing pi_S is small
# (b) multivariate X: the information share depends on beta only through Lambda = beta' Sigma_X beta, in every direction
# (c) X a spatial field independent of W: share and floors as functions of beta, iid vs range 0.3 vs range 0.1
import numpy as np, itertools, matplotlib, matplotlib.ticker
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import matern
rng=np.random.default_rng(4)
s2,phi,t2=5.0,0.5,0.5; tau=np.sqrt(t2)

def enum_info(K,beta,mode,d=1,SigX=None,xrange=None,reps=200,cell=0.15):
    """Fisher information matrix (d x d) of the block-marginal likelihood at beta, by enumeration; plus oracle and floors."""
    beta=np.atleast_1d(beta).astype(float); SigX=np.eye(d) if SigX is None else SigX
    P=np.array(list(itertools.permutations(range(K)))); nP=len(P)
    J=np.zeros((d,d)); Ior=np.zeros((d,d)); pS=[]; pX=[]
    for r in range(reps):
        pts=rng.uniform(size=(K,2))*cell; D=np.linalg.norm(pts[:,None]-pts[None],axis=2)
        SW=matern(D,0.5,phi,s2); C=SW+t2*np.eye(K)
        if xrange is None: X=rng.normal(size=(K,d))@np.linalg.cholesky(SigX).T
        else:
            Cg=matern(D,0.5,xrange,1.0); X=np.linalg.cholesky(Cg+1e-9*np.eye(K))@rng.normal(size=(K,d))@np.linalg.cholesky(SigX).T
        W=rng.multivariate_normal(np.zeros(K),SW); eps=rng.normal(size=K)*tau
        piX=(np.arange(K) if mode=='S' else P[rng.integers(nP)]); piS=(np.arange(K) if mode=='X' else (piX if mode=='XS' else P[rng.integers(nP)]))
        Y=X[piX]@beta+W[piS]+eps
        # candidate configurations
        if mode=='X': cands=[(p,np.arange(K)) for p in P]
        elif mode=='S': cands=[(np.arange(K),p) for p in P]
        elif mode=='XS': cands=[(p,p) for p in P]
        else: cands=[(p,q) for p in P for q in P]
        Cinv={}
        logq=np.empty(len(cands)); Sc=np.empty((len(cands),d)); Ac=np.empty((len(cands),d,d))
        for i,(p,q) in enumerate(cands):
            key=tuple(q)
            if key not in Cinv: Cq=SW[np.ix_(q,q)]+t2*np.eye(K); Cinv[key]=(np.linalg.inv(Cq),np.linalg.slogdet(Cq)[1])
            Ci,ld=Cinv[key]; Xp=X[p]; res=Y-Xp@beta
            logq[i]=-0.5*res@Ci@res-0.5*ld; Sc[i]=Xp.T@Ci@res; Ac[i]=Xp.T@Ci@Xp
        w=np.exp(logq-logq.max()); w/=w.sum()
        ES=(w[:,None]*Sc).sum(0); J+=(w[:,None,None]*Ac).sum(0)-((w[:,None,None]*np.einsum('ci,cj->cij',Sc,Sc)).sum(0)-np.outer(ES,ES))
        # oracle information given the true keys
        Ci0=np.linalg.inv(SW[np.ix_(piS,piS)]+t2*np.eye(K)); Xt=X[piX]; Ior+=Xt.T@Ci0@Xt
        # floors on this block
        Lam=beta@SigX@beta
        vS=max(SW[i,i]+SW[j,j]-2*SW[i,j] for i,j in itertools.combinations(range(K),2))
        pS.append(np.arctan(np.sqrt(2)*tau/np.sqrt(vS))/np.pi)
        if xrange is None: pX.append(np.arctan(tau/np.sqrt(Lam))/np.pi if Lam>0 else 0.5)
        else:
            vg=max(2*(1-matern(np.array([D[i,j]]),0.5,xrange,1.0)[0]) for i,j in itertools.combinations(range(K),2))
            pX.append(np.arctan(np.sqrt(2)*tau/np.sqrt(Lam*vg))/np.pi if Lam>0 else 0.5)
    return J/reps, Ior/reps, np.mean(pS), np.mean(pX)

betas=np.array([0.1,0.2,0.35,0.5,0.75,1,1.5,2,3,4])
fig,ax=plt.subplots(1,3,figsize=(14,3.9))
# (a) location key
K=4
shX=[];shS=[];shB=[];shXS=[];pSs=[]
for b in betas:
    J,I,pS,pX=enum_info(K,b,'X',reps=150); shX.append(J[0,0]/I[0,0])
    J,I,pS,pX=enum_info(K,b,'S',reps=300); shS.append(J[0,0]/I[0,0]); pSs.append(pS)
    J,I,pS,pX=enum_info(K,b,'B',reps=120); shB.append(J[0,0]/I[0,0])
    J,I,pS,pX=enum_info(K,b,'XS',reps=150); shXS.append(J[0,0]/I[0,0])
ax[0].plot(betas,shS,'-o',label='location key unknown ($\\pi_X$ known)'); ax[0].plot(betas,shX,'-s',label='covariate key unknown ($\\pi_S$ known)')
ax[0].plot(betas,shB,'-^',label='both keys unknown'); ax[0].plot(betas,shXS,'-v',label='single key ($\\pi_X=\\pi_S$)')
ax0b=ax[0].twinx(); ax0b.plot(betas,pSs,'k--',label='floor $p^S$'); ax0b.plot(betas,np.arctan(tau/betas)/np.pi,'k:',label='floor $p^X$'); ax0b.set_ylim(0,0.5); ax0b.set_ylabel('floor'); ax0b.legend(fontsize=7,loc='center right')
ax[0].set_title(f'(a) which key is unknown (K={K}, iid $X$)',fontsize=9); ax[0].legend(fontsize=7,loc='center left')
# (b) multivariate: share matrix eigenvalues vs Lambda, d=1,2,3
K=5
for d,c in [(1,'C0'),(2,'C1'),(3,'C2')]:
    SigX=np.eye(d)*(1.0 if d==1 else 1.0); 
    if d>1: SigX=0.5*np.eye(d)+0.5*np.ones((d,d))
    par=[];perp=[]
    for Lam in [0.05,0.1,0.25,0.5,1,2,4,8]:
        u=rng.normal(size=d); beta=u*np.sqrt(Lam/(u@SigX@u))
        J,I,_,_=enum_info(K,beta,'X',d=d,SigX=SigX,reps=120)
        Sh=np.linalg.solve(I,J)   # I^-1 J : share, direction-wise
        bt=beta/np.linalg.norm(beta); par.append(bt@Sh@bt)
        if d>1:
            Q=np.linalg.qr(np.column_stack([bt,rng.normal(size=(d,d-1))]))[0][:,1:]; perp.append(np.trace(Q.T@Sh@Q)/(d-1))
    Lams=[0.05,0.1,0.25,0.5,1,2,4,8]
    ax[1].plot(Lams,par,'-o',color=c,label=f'd={d}: along $\\beta$')
    if d>1: ax[1].plot(Lams,perp,'--s',color=c,label=f'd={d}: orthogonal to $\\beta$')
ax[1].set_xscale('log'); ax[1].set_xlabel('$\\Lambda=\\beta^\\top\\Sigma_X\\beta$'); ax[1].set_title(f'(b) several covariates (K={K}, covariate key)',fontsize=9); ax[1].legend(fontsize=7)
# (c) X a field independent of W
K=4
for xr,c,lab in [(None,'C0','iid $X$'),(0.3,'C1','$X$ field, range 0.3'),(0.1,'C2','$X$ field, range 0.1')]:
    sh=[];fl=[]
    for b in betas:
        J,I,pS,pX=enum_info(K,b,'XS',xrange=xr,reps=150); sh.append(J[0,0]/I[0,0]); fl.append(pX)
    ax[2].plot(betas,sh,'-o',color=c,label=lab)
    ax2b=ax[2].twinx() if xr is None else ax2b; ax2b.plot(betas,fl,'--',color=c)
ax2b.set_ylim(0,0.5); ax2b.set_ylabel('floor $p^X$ (dashed)')
ax[2].set_title('(c) single key, $X$ a field independent of $W$ (K=4)',fontsize=9); ax[2].legend(fontsize=7,loc='center left')
for a in ax:
    a.set_ylabel('composite / oracle information'); a.set_ylim(-0.02,1.05); a.grid(alpha=.3)
for a in [ax[0],ax[2]]: a.set_xscale('log'); a.set_xlabel(r'$\beta$'); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
plt.tight_layout(); plt.savefig('../Figures/info_checks.pdf'); plt.savefig('../Figures/info_checks.png',dpi=120); print('done')
