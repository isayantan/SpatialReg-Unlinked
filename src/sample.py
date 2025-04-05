

def perm_to_P(perm):
    K = len(perm)
    P = np.zeros((K, K))
    P[np.arange(K), perm] = 1
    return P


def round_to_perm(P):
    N = P.shape[0]
    assert P.shape == (N, N)
    row, col = linear_sum_assignment(-P)
    P = np.zeros((N, N))
    P[row, col] = 1.0
    return P


def sinkhorn_logspace(logP, niters=10):
    for _ in range(niters):
        # Normalize columns and take the log again
        logP = logP - logsumexp(logP, axis=0, keepdims=True)
        # Normalize rows and take the log again
        logP = logP - logsumexp(logP, axis=1, keepdims=True)
    return logP


def sample_q(params, unpack_W, unpack_Ps, Cs, num_sinkhorn, temp=0.1):
    # Sample W
    mu_W, log_sigmasq_W, log_mu_Ps, log_sigmasq_Ps = params
    W_flat = mu_W + np.sqrt(np.exp(log_sigmasq_W)) * npr.randn(*mu_W.shape)
    W = unpack_W(W_flat)

    # Sample Ps: run sinkhorn to move mu close to Birkhoff
    Ps = []
    for log_mu_P, log_sigmasq_P, unpack_P, C in \
            zip(log_mu_Ps, log_sigmasq_Ps, unpack_Ps, Cs):
        # Unpack the mean, run sinkhorn, the pack it again
        log_mu_P = unpack_P(log_mu_P)
        log_mu_P = sinkhorn_logspace(log_mu_P - 1e8 * (1 - C), num_sinkhorn)
        log_mu_P = log_mu_P[C]

        log_sigmasq_P = log_sigmasq_P
        P = np.exp(log_mu_P) + \
            np.sqrt(np.exp(log_sigmasq_P)) * \
            npr.randn(*log_mu_P.shape)
        P = unpack_P(P)

        # Round to nearest permutation
        Phat = round_to_perm(P if isinstance(P, np.ndarray) else P.value)
        P = P * temp + (1 - temp) * Phat

        Ps.append(P)

    Ps = np.array(Ps)
    return W, Ps