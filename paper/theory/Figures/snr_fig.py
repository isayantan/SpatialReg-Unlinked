import numpy as np, scipy.special as sp, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
rng=np.random.default_rng(1)
def matern(d,nu,phi,s2):
    d=np.asarray(d,float); out=np.full(d.shape,s2)
    m=d>0; x=np.sqrt(2*nu)*d[m]/phi
    out[m]=s2*2**(1-nu)/sp.gamma(nu)*x**nu*sp.kv(nu,x); return out
def design(B,K):
    g=int(round(np.sqrt(B))); pts=[];blk=[]
    for i in range(g):
        for j in range(g):
            p=rng.uniform(size=(K,2))/g+np.array([i,j])/g; pts.append(p); blk+= [i*g+j]*K
    return np.vstack(pts),np.array(blk)
def constants(pts,blk,nu,phi,s2,t2):
    D=np.linalg.norm(pts[:,None]-pts[None],axis=2); Sig=matern(D,nu,phi,s2)+t2*np.eye(len(pts))
    ev=np.linalg.eigvalsh(Sig); lam=ev[-1]; kap=ev[-1]/ev[0]
    Om=np.linalg.inv(Sig); tc2=0
    for b in np.unique(blk):
        idx=np.where(blk==b)[0]
        for a in range(len(idx)):
            for c in range(a+1,len(idx)):
                S=[idx[a],idx[c]]; O=Om[np.ix_(S,S)]; tc2=max(tc2,1/np.linalg.eigvalsh(O)[0])
    return lam,kap,tc2
beta,sX,s2,t2=2.0,1.0,5.0,0.5
fig,ax=plt.subplots(1,3,figsize=(12,3.4))
Bs=[16,36,64,100,144,196]; nus=[0.5,1.5,2.5]; cols=['C0','C1','C2']
for nu,c in zip(nus,cols):
    old=[];new=[]
    for B in Bs:
        pts,blk=design(B,6); lam,kap,tc2=constants(pts,blk,nu,0.5,s2,t2)
        old.append(beta**2/(lam*kap)); new.append(beta**2*sX**2/tc2)
    ax[0].semilogy([6*B for B in Bs],old,'--o',color=c,mfc='white',label=fr'old, $\nu={nu}$')
    ax[0].semilogy([6*B for B in Bs],new,'-o',color=c,label=fr'$\mathrm{{SNR}}_c$, $\nu={nu}$')
ax[0].set_xlabel('$n=6B$ (K=6, unit square)'); ax[0].set_ylabel('SNR at $\\beta=2$'); ax[0].legend(fontsize=6,ncol=2); ax[0].set_title('(a) infill: $n$ grows, domain fixed',fontsize=9)
phis=np.linspace(0.05,2,12)
pts,blk=design(49,6)
for nu,c in zip(nus,cols):
    for t,ls in zip([0.1,0.5,2.0],[':','-','--']):
        tc=[constants(pts,blk,nu,p,s2,t)[2] for p in phis]
        ax[1].plot(phis,tc,ls,color=c,label=fr'$\nu={nu},\ \tau^2={t}$')
ax[1].set_xlabel('range $\\phi$ ($B=49$, $K=6$)'); ax[1].set_ylabel('$\\tau_c^2$'); ax[1].legend(fontsize=6,ncol=3); ax[1].set_title('(b) $\\tau_c^2$ vs range and nugget ($\\sigma^2=5$)',fontsize=9)
ax[1].axhline(2*s2+t2,color='grey',lw=.5)
Ks=[2,4,6,10,15,20]
for nu,c in zip(nus,cols):
    old=[];new=[]
    for K in Ks:
        pts2,blk2=design(49,K); lam,kap,tc2=constants(pts2,blk2,nu,0.5,s2,t2)
        old.append(beta**2/(lam*kap)); new.append(beta**2*sX**2/tc2)
    ax[2].semilogy(Ks,old,'--o',color=c,mfc='white',label=fr'old, $\nu={nu}$'); ax[2].semilogy(Ks,new,'-o',color=c,label=fr'$\mathrm{{SNR}}_c$, $\nu={nu}$')
ax[2].set_xlabel('block size $K$ ($B=49$)'); ax[2].set_ylabel('SNR at $\\beta=2$'); ax[2].set_title('(c) block size',fontsize=9); ax[2].legend(fontsize=6,ncol=2,loc='center right')
plt.tight_layout(); plt.savefig('Figures/snr_comparison.pdf'); plt.savefig('Figures/snr_comparison.png',dpi=110)
# print table
for nu in nus:
    for B in [16,49,144]:
        pts,blk=design(B,6); lam,kap,tc2=constants(pts,blk,nu,0.5,s2,t2)
        print(f"nu={nu} B={B} n={6*B}: lam={lam:.0f} kappa={kap:.0f} oldSNR={beta**2/(lam*kap):.1e} tau_c2={tc2:.2f} SNR_c={beta**2/tc2:.2f}")
