import numpy as np, ast, matplotlib, matplotlib.ticker
matplotlib.use('Agg'); import matplotlib.pyplot as plt
rows=[ast.literal_eval(l.strip()) for l in open('simE3_all.log') if l.strip()]
v2=[ast.literal_eval(l.strip()) for l in open('simE3_mode0_v2.log') if l.strip()]
key=lambda d:(d['mode'],d['B'],d['Kmean'],d['beta'],d['nu']); new={key(d):d for d in v2}; rows=[new.get(key(d),d) for d in rows]
names={0:'both keys, iid $X$',1:'covariate key only',2:'single key, $X$ a field'}
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
for mode,c in zip([0,1,2],['C0','C1','C2']):
    for (B,K),ls,mk in [((64,20),'-','o'),((100,30),'--','s')]:
        rs=sorted([d for d in rows if d['mode']==mode and d['B']==B and d['Kmean']==K and d['nu']==0.5],key=lambda d:d['beta'])
        b=[d['beta'] for d in rs]
        axes[0].plot(b,[d['mse_areal']/d['mse_aug'] for d in rs],ls,marker=mk,color=c,label=f'{names[mode]}, B={B}, K̄={K}')
        axes[1].plot(b,[d['mse_aug']/d['mse_oracle'] for d in rs],ls,marker=mk,color=c,label=f'{names[mode]}, B={B}, K̄={K}')
for ax,ttl,yl in [(axes[0],'gain over areal GLS','MSE(areal) / MSE(composite)'),(axes[1],'distance to the oracle','MSE(composite) / MSE(oracle)')]:
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks([0.25,0.5,1,2]); ax.set_xticklabels(['0.25','0.5','1','2']); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.axhline(1,color='grey',lw=.8); ax.set_xlabel(r'$\beta$'); ax.set_ylabel(yl); ax.set_title(ttl,fontsize=10); ax.grid(alpha=.3)
axes[0].legend(fontsize=6.5)
plt.tight_layout(); plt.savefig('../Figures/beta_ratio.pdf'); plt.savefig('../Figures/beta_ratio.png',dpi=120); print('ok')
