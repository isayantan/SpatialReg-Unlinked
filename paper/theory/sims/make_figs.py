import numpy as np, ast, itertools, matplotlib, matplotlib.ticker
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from numba import njit
from check_mixing import make_block
from check_mixing2 import PERMS4
# ---------- Figure A: beta estimation at realistic K ----------
rows=[ast.literal_eval(l.strip()) for l in open('simE3_all.log') if l.strip()]
v2=[ast.literal_eval(l.strip()) for l in open('simE3_mode0_v2.log') if l.strip()]
# replace mode-0 rows by the corrected-sampler rerun where available
key=lambda d:(d['mode'],d['B'],d['Kmean'],d['beta'],d['nu'])
new={key(d):d for d in v2}
rows=[new.get(key(d),d) for d in rows]
names={0:'both keys, iid $X$',1:'covariate key only',2:'single key, $X$ a field (range 0.3)'}
fig,axes=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
for ax,mode in zip(axes,[0,1,2]):
    for (B,K),ls in [((64,20),'-'),((100,30),'--')]:
        rs=sorted([d for d in rows if d['mode']==mode and d['B']==B and d['Kmean']==K and d['nu']==0.5],key=lambda d:d['beta'])
        if not rs: continue
        b=[d['beta'] for d in rs]
        ax.plot(b,[d['mse_areal'] for d in rs],ls,marker='s',color='C3',label=f'areal GLS, B={B}, K̄={K}')
        ax.plot(b,[d['mse_aug'] for d in rs],ls,marker='o',color='C0',label=f'augmented composite, B={B}, K̄={K}')
        ax.plot(b,[d['mse_oracle'] for d in rs],ls,marker='^',color='C2',label=f'oracle GLS, B={B}, K̄={K}')
    ax.set_yscale('log'); ax.set_xscale('log'); ax.set_xticks([0.25,0.5,1,2]); ax.set_xticklabels(['0.25','0.5','1','2']); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r'$\beta$'); ax.set_title(names[mode],fontsize=10); ax.grid(alpha=.3)
axes[0].set_ylabel('MSE of $\\hat\\beta$'); axes[0].legend(fontsize=6.5)
plt.tight_layout(); plt.savefig('../Figures/beta_realistic.pdf'); plt.savefig('../Figures/beta_realistic.png',dpi=120)

# ---------- Figure B: key posteriors at K=20 ----------
@njit(cache=True)
def chain_pairs(Y, XB, Om, a, c, mode, n_burn, n_samp, thin, seed):
    np.random.seed(seed); K=Y.shape[0]
    r=np.empty(K)
    for k in range(K): r[k]=Y[a[k]]-XB[c[k]]
    Omr=Om@r
    Pa=np.zeros((K,K)); Pc=np.zeros((K,K)); Pair=np.zeros((K,K)); ns=0
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
        if it>=n_burn and (it-n_burn)%thin==0:
            ns+=1
            for k in range(K): Pa[k,a[k]]+=1.0; Pc[k,c[k]]+=1.0; Pair[a[k],c[k]]+=1.0
    return Pa/ns, Pc/ns, Pair/ns
rng=np.random.default_rng(21)
K=20
fig,axes=plt.subplots(2,4,figsize=(13,6.4))
for row,(mode,label) in enumerate([(0,'both keys'),(1,'covariate key only')]):
    for col,beta in enumerate([0.5,1.0,2.0,4.0]):
        rng2=np.random.default_rng(100+col)
        import check_mixing as CM; CM.rng=rng2
        Y,X,Om,at,ct=make_block(K,beta,mode=mode); XB=X*beta
        a0=at.copy() if mode==1 else rng2.permutation(K); c0=rng2.permutation(K)
        Pa,Pc,Pair=chain_pairs(Y,XB,Om,a0,c0,mode,500,6000,2,7+col)
        # pairing matrix in true order: rows = responses ordered by true location, cols = X rows ordered by true location
        M=Pair[np.ix_(at,ct)]   # M[i,j] = P(response at true location i is paired with the X row truly at location j); truth = identity
        ax=axes[row,col]; im=ax.imshow(M,vmin=0,vmax=1,cmap='Blues'); ax.set_title(f'{label}, $\\beta$={beta}\nP(true pairing) = {np.mean(np.diag(M)):.2f}',fontsize=9)
        ax.set_xlabel('X row (true location order)',fontsize=8); ax.set_ylabel('response (true location order)',fontsize=8); ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.6,label='posterior probability of the pairing')
plt.savefig('../Figures/key_posteriors_K20.pdf',bbox_inches='tight'); plt.savefig('../Figures/key_posteriors_K20.png',dpi=110,bbox_inches='tight')
# placement (location) posterior for the both-keys case, beta=4
fig,axes=plt.subplots(1,3,figsize=(11,3.4))
CM.rng=np.random.default_rng(103); Y,X,Om,at,ct=make_block(K,4.0,mode=0); XB=X*4.0
Pa,Pc,Pair=chain_pairs(Y,XB,Om,rng.permutation(K),rng.permutation(K),0,500,6000,2,3)
for ax,(Mx,ttl) in zip(axes,[(Pair[np.ix_(at,ct)],'pairing $Y_i\\leftrightarrow X_j$\n(truth = diagonal)'),(Pa[:,at],'location key: $Y$ at location $k$\n(truth = diagonal)'),(Pc[:,ct],'covariate key: $X$ row at location $k$\n(truth = diagonal)')]):
    im=ax.imshow(Mx,vmin=0,vmax=1,cmap='Blues'); ax.set_title(ttl+f'\nmean diagonal = {np.mean(np.diag(Mx)):.2f}',fontsize=9); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('Both keys unknown, K=20, $\\beta=4$: the pairing is identified, the placement is not',fontsize=10)
plt.tight_layout(); plt.savefig('../Figures/key_posterior_pairing_vs_placement.pdf'); plt.savefig('../Figures/key_posterior_pairing_vs_placement.png',dpi=110)
print('done')
