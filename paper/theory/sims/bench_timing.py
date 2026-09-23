import numpy as np, time
from simE_realistic import block_mcmc
from simE3_aug import aug_newton
from common import matern
rng=np.random.default_rng(1)
def make(B,K,mode=0,d=1):
    g=int(np.ceil(np.sqrt(B))); pts=[];blk=[];b=0
    for (i,j) in [(i,j) for i in range(g) for j in range(g)][:B]:
        pts.append(rng.uniform(size=(K,2))/g+np.array([i,j])/g); blk+=[b]*K; b+=1
    pts=np.vstack(pts); blk=np.array(blk); n=len(pts); blocks=[np.where(blk==bb)[0] for bb in range(B)]
    D=np.linalg.norm(pts[:,None]-pts[None],axis=2); SW=matern(D,0.5,0.5,5.0); Sig=SW+0.5*np.eye(n)
    X=rng.normal(size=(n,d)); W=rng.multivariate_normal(np.zeros(n),SW); eps=rng.normal(size=n)*np.sqrt(0.5)
    Y=np.zeros(n)
    for idx in blocks:
        piX=rng.permutation(K); piS=rng.permutation(K); Y[idx]=X[idx][piX]@np.ones(d)+W[idx][piS]+eps[idx]
    A=np.zeros((n,B)); A[np.arange(n),blk]=1; V=A.T@Sig@A; S=A.T@X; T=A.T@Y
    Om=[np.linalg.inv(SW[np.ix_(idx,idx)]+0.5*np.eye(K)) for idx in blocks]
    return Y,X,blocks,Om,S,T,V
# warm-up JIT
Y,X,blocks,Om,S,T,V=make(4,10); block_mcmc(Y[blocks[0]],X[blocks[0]]@np.ones(1),Om[0],np.arange(10),np.arange(10),0,10,10,1,0)
print("per-block MCMC (100 burn + 300 samples x thin 2 = 700 sweeps of K proposals):")
for K in [10,20,30,50,100,200]:
    Y,X,blocks,Om,S,T,V=make(4,K)
    t0=time.perf_counter()
    for r in range(5): block_mcmc(Y[blocks[0]],X[blocks[0]]@np.ones(1),Om[0],np.arange(K),np.arange(K),0,100,300,2,r)
    dt=(time.perf_counter()-t0)/5
    print(f"  K={K:4d}: {dt*1000:8.1f} ms per block  ({dt*1e6/(700*K):.2f} us per proposal)")
print("full fit (areal start, damped Newton, 10 iterations x per-block MCMC), one core:")
for (B,K) in [(64,20),(100,30),(100,50),(400,30),(100,100)]:
    Y,X,blocks,Om,S,T,V=make(B,K)
    b0=np.linalg.solve(S.T@np.linalg.solve(V,S),S.T@np.linalg.solve(V,T))
    t0=time.perf_counter(); aug_newton(Y,X,blocks,Om,S,T,V,b0,0,n_iter=10,n_samp=150,seed=0); dt=time.perf_counter()-t0
    print(f"  B={B:4d} K={K:4d} n={B*K:6d}: {dt:6.1f} s  (block precisions: {B} inversions of {K}x{K})")
