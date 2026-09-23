# Theory check: per-block composite information I_b(beta) relative to the block-sum information, exact enumeration
# (covariate key only, W_b integrated out), as a function of beta; with the two-point floor p^X on the same axis.
import numpy as np, itertools, matplotlib, matplotlib.ticker
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import matern
rng=np.random.default_rng(2)
s2,phi,t2,sX=5.0,0.5,0.5,1.0; tau=np.sqrt(t2)
def block_info(K,beta,reps=300):
    pts=rng.uniform(size=(K,2))*0.15; D=np.linalg.norm(pts[:,None]-pts[None],axis=2)
    C=matern(D,0.5,phi,s2)+t2*np.eye(K); Ci=np.linalg.inv(C); Om=Ci
    P=np.array(list(itertools.permutations(range(K)))); nP=len(P)
    Isum=K*sX**2/C.sum()                     # information in the block sum alone
    Ior=sX**2*np.trace(Om)                   # oracle information of the block (keys known)
    J=[]
    for r in range(reps):
        X=rng.normal(size=K)*sX; W=rng.multivariate_normal(np.zeros(K),C); pi=P[rng.integers(nP)]
        Y=X[pi]*beta+W; XP=X[P]
        c=np.einsum('pi,ij,pj->p',XP,Ci,XP); b=XP@Ci@Y
        q=-0.5*(c*beta**2-2*b*beta); q-=q.max(); w=np.exp(q); w/=w.sum()
        S=b-c*beta; J.append((w*c).sum()-((w*S**2).sum()-(w*S).sum()**2))
    # pair-conditional variance for SNR_c on this block
    tc2=max(1/np.linalg.eigvalsh(Om[np.ix_([i,j],[i,j])])[0] for i,j in itertools.combinations(range(K),2))
    return np.mean(J), Isum, Ior, tc2
betas=np.array([0.05,0.1,0.2,0.35,0.5,0.75,1,1.5,2,3,4,6])
fig,ax1=plt.subplots(1,1,figsize=(6,3.8)); ax=[ax1,ax1]
for K,c in [(4,'C0'),(6,'C1'),(8,'C2')]:
    g=[];o=[];snr=[]
    for beta in betas:
        Ib,Isum,Ior,tc2=block_info(K,beta,reps=500 if K<8 else 120)
        g.append(Ib/Isum); o.append(Ib/Ior); snr.append(beta**2*sX**2/tc2)
    ax[1].plot(betas,o,'-o',color=c,label=f'K={K}: $I_b(\\beta)/I_b^{{\\rm oracle}}$')
pX=np.arctan(tau/(np.abs(betas)*sX))/np.pi
ax2=ax[1].twinx(); ax2.plot(betas,pX,'k--',label='two-point floor $p^X$'); ax2.set_ylabel('$p^X$ (re-linking floor)'); ax2.set_ylim(0,0.5)
for a in [ax1]: a.set_xscale('log'); a.set_xlabel(r'$\beta$  ($\tau^2=0.5$, $\sigma_X=1$)'); a.grid(alpha=.3); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[1].set_ylabel('composite information / oracle information'); ax[1].set_title('share of the oracle information the composite likelihood retains,\nand the re-linking floor, as functions of the effect size',fontsize=9); ax[1].legend(fontsize=8,loc='center left'); ax2.legend(fontsize=8,loc='center right')
plt.tight_layout(); plt.savefig('../Figures/info_vs_beta.pdf'); plt.savefig('../Figures/info_vs_beta.png',dpi=120)
for K in [6]:
    for beta in [0.25,0.5,1,2,4]:
        Ib,Isum,Ior,tc2=block_info(K,beta,reps=200); print(f"K={K} beta={beta}: I_b/I_sum={Ib/Isum:.2f}  I_b/I_or={Ib/Ior:.2f}  SNR_c={beta**2/tc2:.2f}  p^X={np.arctan(tau/beta)/np.pi:.3f}")
