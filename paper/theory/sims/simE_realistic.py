# Composite-likelihood estimator at realistic block sizes: per-block MCMC over keys (transposition Metropolis),
# Newton/Fisher-scoring for beta started at areal GLS. Compared with areal GLS and oracle GLS.
import numpy as np, sys, time
from numba import njit
from common import matern

@njit(cache=True)
def block_mcmc(Y, XB, Om, a, c, mode, n_burn, n_samp, thin, seed):
    # Y: K responses, XB: K values of X beta (per record), Om: KxK precision; a,c: index arrays (aligned position -> Y index, -> X index)
    # mode 0: both keys free; 1: covariate key only (a fixed); 2: single key (a==c moved together)
    # returns: mean over samples of (X~^T Om r) needs X; we return the posterior mean of c-assignment moments via arrays:
    # Er[k] = E[r_k], and we accumulate Wm = E[ P_c ] (K x K one-hot expectation of c), plus E[(Om r)] and second moments for Var(S).
    np.random.seed(seed)
    K = Y.shape[0]
    r = np.empty(K)
    for k in range(K):
        r[k] = Y[a[k]] - XB[c[k]]
    Omr = Om @ r
    Pc = np.zeros((K, K))     # E[ one-hot(c) ]  (row k = aligned pos, col = X index)
    Omr_mean = np.zeros(K)    # E[Om r]
    # for the score S = X~^T Om r we need E[ sum_k X[c[k]] (Om r)_k ] : accumulate G[k, j] = E[ 1{c[k]=j} (Om r)_k ]
    G = np.zeros((K, K))
    G2 = np.zeros((K, K, 1))  # placeholder unused
    n_acc = 0; n_prop = 0
    total = n_burn + n_samp * thin
    ns = 0
    for it in range(total):
        for p in range(K):
            k = np.random.randint(K); l = np.random.randint(K)
            if k == l: continue
            if mode == 1:
                which = 1      # covariate key only: a fixed, move c
            elif mode == 2:
                which = 0      # single key: pi_S^T pi_X = I, so c fixed, move a
            else:
                which = np.random.randint(3)   # both keys: move a, move c, or move a matched (Y,X) pair between locations
            if which == 0:   # swap a[k], a[l]
                dk = Y[a[l]] - Y[a[k]]; dl = -dk
            elif which == 1: # swap c[k], c[l]
                dk = -(XB[c[l]] - XB[c[k]]); dl = -dk
            else:            # swap both
                dk = (Y[a[l]] - XB[c[l]]) - r[k]; dl = (Y[a[k]] - XB[c[k]]) - r[l]
            dlog = -(dk * Omr[k] + dl * Omr[l] + 0.5 * (Om[k, k] * dk * dk + Om[l, l] * dl * dl + 2.0 * Om[k, l] * dk * dl))
            n_prop += 1
            if np.log(np.random.rand()) < dlog:
                n_acc += 1
                if which == 0 or which == 2:
                    t = a[k]; a[k] = a[l]; a[l] = t
                if which == 1 or which == 2:
                    t = c[k]; c[k] = c[l]; c[l] = t
                r[k] += dk; r[l] += dl
                for m in range(K):
                    Omr[m] += dk * Om[m, k] + dl * Om[m, l]
        if it >= n_burn and (it - n_burn) % thin == 0:
            ns += 1
            for k in range(K):
                Pc[k, c[k]] += 1.0
                G[k, c[k]] += Omr[k]
                Omr_mean[k] += Omr[k]
    return Pc / ns, G / ns, Omr_mean / ns, n_acc / max(n_prop, 1)

def composite_newton(Y, X, blocks, Om_blocks, beta0, mode, n_iter=6, n_burn=100, n_samp=300, thin=2, seed=0, samples_for_var=True):
    d = X.shape[1]; beta = beta0.copy(); hist=[beta.copy()]
    for it in range(n_iter):
        S = np.zeros(d); A = np.zeros((d, d)); VarS = np.zeros((d,d))
        for bi, idx in enumerate(blocks):
            Yb = Y[idx]; Xb = X[idx]; K = len(idx); XB = Xb @ beta
            a = np.arange(K); c = np.arange(K)
            Pc, G, Omr_m, acc = block_mcmc(Yb, XB, Om_blocks[bi], a, c, mode, n_burn, n_samp, thin, seed + 1000*it + bi)
            # E[S_pi] = E[ X~^T Om r ] = sum_k sum_j 1{c[k]=j}(Om r)_k X[j]  = G^T-weighted:  S_b = sum_k sum_j G[k,j] X[j]
            S += (G.T @ np.ones(K)) @ Xb if False else (G @ Xb).sum(axis=0)
            # expected Hessian part: E[X~^T Om X~] with X~ = Pc-averaged? exact: E[X~^T Om X~] = sum_{k,l} Om[k,l] E[X[c[k]] X[c[l]]^T]; approximate by plug-in with E over samples is heavy; use
            # E[X~]^T Om E[X~] + diag correction (Jensen) -> we use the exact-in-expectation form via Pc for the diagonal and mean for off-diagonal (good enough for scoring steps)
            EX = Pc @ Xb                       # K x d, E[X~ rows]
            A += EX.T @ Om_blocks[bi] @ EX
            # diagonal correction: sum_k Om[k,k] (E[X X^T at k] - EX_k EX_k^T)
            for k in range(K):
                Exx = (Pc[k][:,None] * Xb).T @ Xb
                A += Om_blocks[bi][k,k] * (Exx - np.outer(EX[k], EX[k]))
        step = np.linalg.solve(A, S)
        beta = beta + step; hist.append(beta.copy())
    return beta, hist

def run_setting(B, Kmean, beta_true, mode, nu, phi, s2, t2, sX, xfield, reps, seed=0, d=1):
    rng = np.random.default_rng(seed)
    out = {'areal':[], 'oracle':[], 'cl':[], 'time':[]}
    for rep in range(reps):
        # design: sqrt(B) x sqrt(B) cells, K_b ~ 10 + Poisson(Kmean-10), points uniform in cell
        g = int(np.ceil(np.sqrt(B))); pts=[]; blk=[]; b=0
        cells = [(i,j) for i in range(g) for j in range(g)][:B]
        for (i,j) in cells:
            Kb = 10 + rng.poisson(Kmean-10)
            pts.append(rng.uniform(size=(Kb,2))/g + np.array([i,j])/g); blk += [b]*Kb; b+=1
        pts = np.vstack(pts); blk = np.array(blk); n=len(pts)
        blocks = [np.where(blk==bb)[0] for bb in range(B)]
        D = np.linalg.norm(pts[:,None]-pts[None],axis=2)
        SW = matern(D, nu, phi, s2); Sig = SW + t2*np.eye(n)
        if xfield is None:
            X = rng.normal(size=(n,d))*sX
        else:
            Cg = matern(D, 0.5, xfield, 1.0)
            L = np.linalg.cholesky(Cg + 1e-8*np.eye(n)); X = (L @ rng.normal(size=(n,d)))*sX
        W = rng.multivariate_normal(np.zeros(n), SW); eps = rng.normal(size=n)*np.sqrt(t2)
        Y = np.zeros(n); order = np.arange(n); Xtrue = X.copy()
        for idx in blocks:
            K = len(idx)
            if mode == 1:   piX = rng.permutation(K); piS = np.arange(K)
            elif mode == 2: piX = rng.permutation(K); piS = piX
            else:           piX = rng.permutation(K); piS = rng.permutation(K)
            Y[idx] = X[idx][piX] @ beta_true + W[idx][piS] + eps[idx]; Xtrue[idx] = X[idx][piX]; order[idx] = idx[piS]
        # areal GLS
        A_ind = np.zeros((n,B)); A_ind[np.arange(n), blk]=1
        V = A_ind.T @ Sig @ A_ind; Vi = np.linalg.inv(V); S = A_ind.T @ X; T = A_ind.T @ Y
        b_areal = np.linalg.solve(S.T@Vi@S, S.T@Vi@T)
        # oracle GLS
        Oo = np.linalg.inv(SW[np.ix_(order,order)] + t2*np.eye(n))
        b_or = np.linalg.solve(Xtrue.T@Oo@Xtrue, Xtrue.T@Oo@Y)
        # composite via MCMC-Newton from areal
        Om_blocks = [np.linalg.inv(SW[np.ix_(idx,idx)] + t2*np.eye(len(idx))) for idx in blocks]
        t0=time.time()
        b_cl, hist = composite_newton(Y, X, blocks, Om_blocks, b_areal, mode, seed=seed*7919+rep)
        out['time'].append(time.time()-t0)
        out['areal'].append(b_areal); out['oracle'].append(b_or); out['cl'].append(b_cl)
    res = {k: np.array(v) for k,v in out.items()}
    mse = lambda a: np.mean(np.sum((a-beta_true)**2,axis=1))
    return dict(B=B,Kmean=Kmean,beta=float(beta_true[0]),mode=mode,nu=nu,phi=phi,xfield=xfield,
                mse_areal=mse(res['areal']), mse_oracle=mse(res['oracle']), mse_cl=mse(res['cl']),
                bias_cl=float(np.mean(res['cl'][:,0])-beta_true[0]), ratio=mse(res['areal'])/mse(res['cl']), sec_per_rep=float(np.mean(res['time'])))

if __name__=='__main__':
    which = sys.argv[1] if len(sys.argv)>1 else 'quick'
    settings=[]
    if which=='quick':
        settings=[dict(B=36,Kmean=20,beta_true=np.array([1.0]),mode=0,nu=0.5,phi=0.5,s2=5.0,t2=0.5,sX=1.0,xfield=None,reps=10)]
    else:
        for mode in [0,2,1]:
            for beta in [0.5,1.0,2.0]:
                for (B,Kmean) in [(64,20),(100,30)]:
                    settings.append(dict(B=B,Kmean=Kmean,beta_true=np.array([beta]),mode=mode,nu=0.5,phi=0.5,s2=5.0,t2=0.5,sX=1.0,xfield=(0.3 if mode==2 else None),reps=40))
        # smoother field, and a plate-like natural instance (K~96, few blocks, weak field, covariate key only)
        settings.append(dict(B=64,Kmean=20,beta_true=np.array([1.0]),mode=0,nu=1.5,phi=0.5,s2=5.0,t2=0.5,sX=1.0,xfield=None,reps=40))
        settings.append(dict(B=12,Kmean=96,beta_true=np.array([0.5]),mode=1,nu=0.5,phi=0.5,s2=1.0,t2=1.0,sX=1.0,xfield=None,reps=40))
        settings.append(dict(B=12,Kmean=96,beta_true=np.array([1.0]),mode=1,nu=0.5,phi=0.5,s2=1.0,t2=1.0,sX=1.0,xfield=None,reps=40))
    for st in settings:
        r = run_setting(**st); print({k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()}, flush=True)
