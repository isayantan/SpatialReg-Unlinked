# Areal-augmented composite likelihood: l* = log f(T;beta) [GLS of block sums, full V] + sum_b log f_b(Y_b | T_b; beta)
#  = l_cl + [l_GLS(T) - sum_b log phi(T_b; S_b'beta, V_bb)].  Score/Hessian of the bracket are explicit; the l_cl part by per-block MCMC.
import numpy as np, sys, time
from simE_realistic import block_mcmc
from common import matern

def aug_newton(Y, X, blocks, Om_blocks, S, T, V, beta0, mode, n_iter=15, n_burn=100, n_samp=300, thin=2, seed=0):
    d=X.shape[1]; beta=beta0.copy(); Vi=np.linalg.inv(V); Vd=np.diag(V)
    for it in range(n_iter):
        Sc=np.zeros(d); A=np.zeros((d,d))
        for bi, idx in enumerate(blocks):
            Yb=Y[idx]; Xb=X[idx]; K=len(idx); XB=Xb@beta
            Pc,G,_,_=block_mcmc(Yb,XB,Om_blocks[bi],np.arange(K),np.arange(K),mode,n_burn,n_samp,thin,seed+1000*it+bi)
            Sc+=(G@Xb).sum(axis=0)
            EX=Pc@Xb; A+=EX.T@Om_blocks[bi]@EX
            for k in range(K):
                Exx=(Pc[k][:,None]*Xb).T@Xb; A+=Om_blocks[bi][k,k]*(Exx-np.outer(EX[k],EX[k]))
        # cross-block correction of the sums
        res=T-S@beta
        Sc+= S.T@Vi@res - S.T@(res/Vd)
        M = A + S.T@Vi@S   # PD preconditioner dominating the information of l*: damped Newton
        step=np.linalg.solve(M,Sc)
        beta=beta+step
        if it>=3 and np.linalg.norm(step)<1e-3*max(1.0,np.linalg.norm(beta)): break
    return beta

def run(B,Kmean,beta_true,mode,nu,phi,s2,t2,sX,xfield,reps,seed=0,d=1):
    rng=np.random.default_rng(seed); res={'areal':[],'oracle':[],'aug':[]}; tt=[]
    for rep in range(reps):
        g=int(np.ceil(np.sqrt(B))); pts=[];blk=[];b=0
        for (i,j) in [(i,j) for i in range(g) for j in range(g)][:B]:
            Kb=10+rng.poisson(Kmean-10); pts.append(rng.uniform(size=(Kb,2))/g+np.array([i,j])/g); blk+=[b]*Kb; b+=1
        pts=np.vstack(pts); blk=np.array(blk); n=len(pts); blocks=[np.where(blk==bb)[0] for bb in range(B)]
        D=np.linalg.norm(pts[:,None]-pts[None],axis=2); SW=matern(D,nu,phi,s2); Sig=SW+t2*np.eye(n)
        X=rng.normal(size=(n,d))*sX if xfield is None else (np.linalg.cholesky(matern(D,0.5,xfield,1.0)+1e-8*np.eye(n))@rng.normal(size=(n,d)))*sX
        W=rng.multivariate_normal(np.zeros(n),SW); eps=rng.normal(size=n)*np.sqrt(t2)
        Y=np.zeros(n); order=np.arange(n); Xtrue=X.copy()
        for idx in blocks:
            K=len(idx); piX=rng.permutation(K); piS=(np.arange(K) if mode==1 else (piX if mode==2 else rng.permutation(K)))
            Y[idx]=X[idx][piX]@beta_true+W[idx][piS]+eps[idx]; Xtrue[idx]=X[idx][piX]; order[idx]=idx[piS]
        A_ind=np.zeros((n,B)); A_ind[np.arange(n),blk]=1; V=A_ind.T@Sig@A_ind; Vi=np.linalg.inv(V); S=A_ind.T@X; T=A_ind.T@Y
        b_areal=np.linalg.solve(S.T@Vi@S,S.T@Vi@T)
        Oo=np.linalg.inv(SW[np.ix_(order,order)]+t2*np.eye(n)); b_or=np.linalg.solve(Xtrue.T@Oo@Xtrue,Xtrue.T@Oo@Y)
        Om_blocks=[np.linalg.inv(SW[np.ix_(idx,idx)]+t2*np.eye(len(idx))) for idx in blocks]
        t0=time.time(); b_aug=aug_newton(Y,X,blocks,Om_blocks,S,T,V,b_areal,mode,seed=seed*7919+rep); tt.append(time.time()-t0)
        res['areal'].append(b_areal); res['oracle'].append(b_or); res['aug'].append(b_aug)
    r={k:np.array(v) for k,v in res.items()}; mse=lambda a: float(np.mean(np.sum((a-beta_true)**2,axis=1)))
    return dict(B=B,Kmean=Kmean,beta=float(beta_true[0]),mode=mode,nu=nu,phi=phi,xfield=xfield,mse_areal=mse(r['areal']),mse_oracle=mse(r['oracle']),mse_aug=mse(r['aug']),
                bias_aug=float(np.mean(r['aug'][:,0])-beta_true[0]),wrong_sign=float(np.mean(np.sign(r['aug'][:,0])!=np.sign(beta_true[0]))),ratio=mse(r['areal'])/mse(r['aug']),
                oracle_over_aug=mse(r['oracle'])/mse(r['aug']),sec=float(np.mean(tt)))
if __name__=='__main__':
    settings=[]
    if len(sys.argv)>1 and sys.argv[1]=='test':
        for beta in [0.5,1.0]:
            print({k:(round(v,4) if isinstance(v,float) else v) for k,v in run(64,20,np.array([beta]),2,0.5,0.5,5.0,0.5,1.0,0.3,20).items()},flush=True)
        sys.exit()
    for mode in [0,2,1]:
        for beta in [0.25,0.5,1.0,2.0]:
            for (B,Kmean) in [(64,20),(100,30)]:
                settings.append(dict(B=B,Kmean=Kmean,beta_true=np.array([beta]),mode=mode,nu=0.5,phi=0.5,s2=5.0,t2=0.5,sX=1.0,xfield=(0.3 if mode==2 else None),reps=40))
    settings.append(dict(B=64,Kmean=20,beta_true=np.array([1.0]),mode=0,nu=1.5,phi=0.5,s2=5.0,t2=0.5,sX=1.0,xfield=None,reps=40))
    settings.append(dict(B=12,Kmean=96,beta_true=np.array([0.5]),mode=1,nu=0.5,phi=0.5,s2=1.0,t2=1.0,sX=1.0,xfield=None,reps=40))
    settings.append(dict(B=12,Kmean=96,beta_true=np.array([1.0]),mode=1,nu=0.5,phi=0.5,s2=1.0,t2=1.0,sX=1.0,xfield=None,reps=40))
    for st in settings:
        print({k:(round(v,4) if isinstance(v,float) else v) for k,v in run(**st).items()},flush=True)
