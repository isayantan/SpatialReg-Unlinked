"""simF: joint maximisation over block-specific keys (profile estimator) vs the marginal-likelihood
estimator of beta.  Covariate key only, W known (conditional on W), K=6, tau^2=0.5, x ~ N(0,1),
keys uniform per block; all 720 keys enumerated per block, beta on a grid of step 0.01.
Output: mean and sd over replications of both estimators.  Used in Section 'Numerical checks'."""
import numpy as np, itertools, sys
from scipy.special import logsumexp
K=6; tau2=0.5; perms=np.array(list(itertools.permutations(range(K)))); grid=np.linspace(-1.5,1.5,301)
def estimators(R,X):
    prof=np.zeros_like(grid); marg=np.zeros_like(grid)
    for b in range(R.shape[0]):
        XP=X[b][perms]; a=(XP**2).sum(1); c=XP@R[b]
        ll=-(np.outer(grid**2,a)-2*np.outer(grid,c))/(2*tau2)       # log-lik(beta, key) up to a constant
        prof+=ll.max(1); marg+=logsumexp(ll,axis=1)
    return grid[prof.argmax()], grid[marg.argmax()]
def run(beta_true,B,reps,seed):
    rng=np.random.default_rng(seed); P=[]; M=[]
    for _ in range(reps):
        X=rng.standard_normal((B,K)); key=np.array([rng.permutation(K) for _ in range(B)])
        R=np.take_along_axis(X,key,1)*beta_true+rng.standard_normal((B,K))*np.sqrt(tau2)   # Y - W
        p,m=estimators(R,X); P.append(p); M.append(m)
    P=np.array(P); M=np.array(M)
    print(f"beta={beta_true:4.2f} B={B:4d}: joint-max mean {P.mean():6.3f} sd {P.std():.3f} mean|.| {np.abs(P).mean():.3f} | marginal mean {M.mean():6.3f} sd {M.std():.3f}", flush=True)
if __name__=='__main__':
    reps=int(sys.argv[1]) if len(sys.argv)>1 else 40
    for bt in [0.0,0.25,0.5,1.0]: run(bt,200,reps,1)
    for B in [50,200,800]: run(0.5,B,reps,2)
