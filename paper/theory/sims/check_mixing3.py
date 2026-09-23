import numpy as np, time
from check_mixing import make_block, chain_with_start
from check_mixing2 import chain_gibbs, PERMS4
rng=np.random.default_rng(9)
print("Covariate key only (location key known), K=20: does the posterior concentrate on the truth, and do chains agree?")
for beta in [1.0,2.0,4.0]:
    res=[]
    for rep in range(5):
        Y,X,Om,at,ct=make_block(20,beta,mode=1); XB=X*beta; K=20
        P1a,P1c,b1=chain_gibbs(Y,XB,Om,at.copy(),rng.permutation(K),1,300,3000,2,11+rep,PERMS4)
        P2a,P2c,b2=chain_gibbs(Y,XB,Om,at.copy(),rng.permutation(K),1,300,3000,2,97+rep,PERMS4)
        P3a,P3c,b3=chain_gibbs(Y,XB,Om,at.copy(),ct.copy(),1,300,3000,2,5+rep,PERMS4)
        r=Y[at]-XB[ct]; lp_true=-0.5*r@Om@r
        res.append((np.abs(P1c-P2c).max(), np.abs(P1c-P3c).max(), b1-lp_true, np.mean([P3c[k,ct[k]] for k in range(K)]), np.mean([P1c[k,ct[k]] for k in range(K)])))
    res=np.array(res)
    print(f"   beta={beta}: chain1 vs chain2 = {res[:,0].mean():.3f}, random vs true start = {res[:,1].mean():.3f}, best logpost - true = {res[:,2].mean():+.1f}, P(record at true place): true-start {res[:,3].mean():.2f}, random-start {res[:,4].mean():.2f}")
print("Both keys, K=20, beta=4: longer chains (30000 samples) and 8-chain average vs single chain")
for rep in range(3):
    Y,X,Om,at,ct=make_block(20,4.0); XB=X*4.0; K=20
    Ps=[chain_gibbs(Y,XB,Om,rng.permutation(K),rng.permutation(K),0,1000,30000,1,100*rep+j,PERMS4)[1] for j in range(8)]
    avgA=np.mean(Ps[:4],axis=0); avgB=np.mean(Ps[4:],axis=0)
    print(f"   rep {rep}: single long chains max diff = {np.abs(Ps[0]-Ps[1]).max():.3f}; two 4-chain averages max diff = {np.abs(avgA-avgB).max():.3f}; P(true place) = {np.mean([avgA[k,ct[k]] for k in range(K)]):.2f}")
