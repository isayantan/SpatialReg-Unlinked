# Estimation target for the revision: areal-augmented composite likelihood

Prepared 22 Sep 2026 after reading the REPAIR code (`src/revised_VIGP_Unlinked.py`, `src/revised_elbo.py`, `src/Experiments/Diff_piX_piS/`, `src/GPArealModel.py`, `src/analysis_varyB_annealed.py`). Nothing in the repository was modified. Prototype and simulation scripts: `02_Resubmission/theory/sims/simE_realistic.py` (per-block MCMC over keys), `simE3_aug.py` (the estimator), logs `simE3*.log`.

## 1. What REPAIR targets today

The ELBO is a mean-field bound on the **full** joint model: a Gaussian q over the whole n-vector W with a full n×n covariance (`Sigma_W`, coupled across blocks through `mean_Rphi_inv`), Gaussian q for β, Gamma q's for σ² and τ², importance-sampled q for φ, and one relaxed-permutation q per key (Sinkhorn mean `MX`, variance `VX`, straight-through rounding, temperature annealing). In `revised_elbo.py` the key q's are shared across blocks (`block_diag(*[V_S_star]*n_blocks)`), i.e. the common-key model; `Diff_piX_piS` gives each block its own `vi_piX(n_locations, block_idx)` but keeps the full-GP q for W. The β update is
`mu_beta = sigmasq_beta * (lambda_a2/lambda_b2) * sum_b X_b' M_X' (Y_b - M_S mu_W,b)`, which at a flat key posterior (M_X ≈ 11'/K) becomes a block-sum regression with the within-block denominator `X V_X X` — the attenuation toward β/K diagnosed earlier.

The target of this scheme is the full marginal likelihood, for which the theory document proves only the conditional Theorem 4 (asymptotics under a score CLT and a local LLN that we cannot verify).

## 2. What is proved, and what it costs to use it

Theorem 2(iii) and Theorem 4 (composite version) give consistency and asymptotic normality with covariance I_cl⁻¹ for the maximiser of the **composite likelihood**
ℓ_cl(β) = Σ_b log Σ_{π_X,π_S} φ_{K_b}(Y_b; π_X X_b β, π_S (Σ_W)_bb π_S' + τ²I),
in which W_b is integrated analytically (a K_b×K_b Gaussian) and the only intractable object is the sum over K_b!² key configurations **within a block**. Two facts make this the natural target for the code:

- For ℓ_cl the factorisation q(π) = Π_b q_b(π^(b)) is **exact**, not a mean-field approximation: block b's term involves only block b's keys. The only approximation is inside a block.
- Per-block optimisation with pooled global parameters is what Referee 2 asked for, and it parallelises.

Costs: (i) ℓ_cl ignores between-block field correlation for β, so its information is I_cl < I_rel (47–71 % of the oracle on the paper's design at K=6; less at large K); (ii) it only sees within-block distances, so covariance parameters (in particular the range) must come from the areal stage, which uses all between-block covariances of block sums; (iii) at large K the block sums are dominated by the field, and ℓ_cl alone loses the **sign** of β: its within-block contrasts identify |β| but the sign information sits in the cross-block GLS of the sums. This last point is what the realistic simulations exposed (composite from the areal start: MSE 0.09 vs areal 0.13 at β=0.5, K≈20, wrong sign in 10–25 % of runs).

## 3. The estimator: areal-augmented composite likelihood

Factorise the block-marginal density as f_b(Y_b) = f_b(T_b) · f_b(Y_b | T_b) with T_b the block sum, which is key-invariant so f_b(T_b;β) = φ(T_b; S_b'β, V_bb) does not involve the keys. Replace the product of the independent sum densities by the joint GLS density of all sums (full V, between-block correlation included):

ℓ*(β) = log φ_B(T; Sβ, V) + Σ_b [ log f_b(Y_b;β) − log φ(T_b; S_b'β, V_bb) ]  =  ℓ_cl(β) + [ ℓ_GLS(T;β) − Σ_b ℓ_b(T_b;β) ].

This is a composite likelihood in Lindsay's sense (one marginal component, B conditional components), so Theorem 4's conditions hold by the same mixing argument, and its information is I* = I_cl + I_agg − Σ_b I_b^sum ≥ max(I_cl, I_agg). The bracket is explicit; only the ℓ_cl score needs the key posterior.

Algorithm (prototype `simE3_aug.py`):
1. Areal GLS on block sums for β̃ and (σ², φ, τ²) — this is `GPArealModel` as it stands.
2. For each block in parallel, given β: Metropolis over the two keys with transposition proposals (each proposal O(K) using the block precision Ω_b), yielding E[S_π] = E[X̃'Ω_b r] and E[X̃'Ω_b X̃] (X̃ = rows of X_b in the posterior key order, r = aligned residual). Exact by enumeration when K_b ≤ 5; MCMC otherwise. The existing Sinkhorn relaxation can play the same role, but the MCMC has no relaxation bias and its accuracy is checkable against enumeration.
3. Damped Newton step for β with score S* = Σ_b E[S_π] + S'V⁻¹(T−Sβ) − Σ_b S_b(T_b−S_b'β)/V_bb and positive-definite preconditioner Σ_b E[X̃'Ω_b X̃] + S'V⁻¹S, iterated to convergence (≈8 steps).
4. Report β̂*, and the per-record key posteriors from step 2 as the empirical re-identification risk.

Cost at K≈20–30, B≈64–100: about 1–2 s per fit in numpy/numba on one core.

## 4. Realistic-setting results (40 replications each; exponential field σ²=5, range 0.5, τ²=0.5, σ_X=1; blocks on a √B×√B grid with K_b ~ 10+Poisson(K̄−10), points uniform in the cell)

Full results (also Table 4 of the theory document):

| keys | B | K̄ | ν | β | areal MSE | oracle MSE | augmented MSE | areal/aug | oracle/aug | wrong sign |
|---|---|---|---|---|---|---|---|---|---|---|
| both keys | 64 | 20 | 0.5 | 0.25 | 0.1306 | 0.0007 | 0.0758 | 1.7 | 0.01 | 0.25 |
| both keys | 64 | 20 | 0.5 | 0.5 | 0.1306 | 0.0007 | 0.0898 | 1.5 | 0.01 | 0.10 |
| both keys | 64 | 20 | 0.5 | 1.0 | 0.1306 | 0.0007 | 0.0014 | 95.2 | 0.48 | 0.00 |
| both keys | 64 | 20 | 0.5 | 2.0 | 0.1306 | 0.0007 | 0.0010 | 133.4 | 0.67 | 0.00 |
| both keys | 100 | 30 | 0.5 | 0.25 | 0.0959 | 0.0002 | 0.0510 | 1.9 | 0.00 | 0.25 |
| both keys | 100 | 30 | 0.5 | 0.5 | 0.0959 | 0.0002 | 0.0753 | 1.3 | 0.00 | 0.07 |
| both keys | 100 | 30 | 0.5 | 1.0 | 0.0735 | 0.0002 | 0.0005 | 163.0 | 0.45 | 0.00 |
| both keys | 100 | 30 | 0.5 | 2.0 | 0.0735 | 0.0002 | 0.0003 | 266.5 | 0.74 | 0.00 |
| both keys | 64 | 20 | 1.5 | 1.0 | 0.0249 | 0.0005 | 0.0012 | 20.9 | 0.42 | 0.00 |
| covariate key | 64 | 20 | 0.5 | 0.25 | 0.0705 | 0.0006 | 0.0406 | 1.7 | 0.01 | 0.17 |
| covariate key | 64 | 20 | 0.5 | 0.5 | 0.0705 | 0.0006 | 0.0429 | 1.6 | 0.01 | 0.04 |
| covariate key | 64 | 20 | 0.5 | 1.0 | 0.0705 | 0.0006 | 0.0008 | 85.8 | 0.69 | 0.00 |
| covariate key | 64 | 20 | 0.5 | 2.0 | 0.0705 | 0.0006 | 0.0009 | 82.9 | 0.67 | 0.00 |
| covariate key | 100 | 30 | 0.5 | 0.25 | 0.0647 | 0.0002 | 0.0526 | 1.2 | 0.00 | 0.25 |
| covariate key | 100 | 30 | 0.5 | 0.5 | 0.0647 | 0.0002 | 0.0367 | 1.8 | 0.01 | 0.04 |
| covariate key | 100 | 30 | 0.5 | 1.0 | 0.0647 | 0.0002 | 0.0003 | 193.7 | 0.62 | 0.00 |
| covariate key | 100 | 30 | 0.5 | 2.0 | 0.0647 | 0.0002 | 0.0003 | 222.2 | 0.71 | 0.00 |
| covariate key | 12 | 96 | 0.5 | 0.5 | 1.8166 | 0.0009 | 0.3250 | 5.6 | 0.00 | 0.33 |
| covariate key | 12 | 96 | 0.5 | 1.0 | 1.8166 | 0.0009 | 0.5160 | 3.5 | 0.00 | 0.12 |
| single key, X field (range 0.3) | 64 | 20 | 0.5 | 0.25 | 0.0671 | 0.0061 | 0.0597 | 1.1 | 0.10 | 0.21 |
| single key, X field (range 0.3) | 64 | 20 | 0.5 | 0.5 | 0.0671 | 0.0061 | 0.0440 | 1.5 | 0.14 | 0.00 |
| single key, X field (range 0.3) | 64 | 20 | 0.5 | 1.0 | 0.0671 | 0.0061 | 0.0288 | 2.3 | 0.21 | 0.00 |
| single key, X field (range 0.3) | 64 | 20 | 0.5 | 2.0 | 0.0671 | 0.0061 | 0.0207 | 3.2 | 0.29 | 0.00 |
| single key, X field (range 0.3) | 100 | 30 | 0.5 | 0.25 | 0.0541 | 0.0043 | 0.0495 | 1.1 | 0.09 | 0.25 |
| single key, X field (range 0.3) | 100 | 30 | 0.5 | 0.5 | 0.0541 | 0.0043 | 0.0580 | 0.9 | 0.07 | 0.08 |
| single key, X field (range 0.3) | 100 | 30 | 0.5 | 1.0 | 0.0541 | 0.0043 | 0.0296 | 1.8 | 0.14 | 0.00 |
| single key, X field (range 0.3) | 100 | 30 | 0.5 | 2.0 | 0.0541 | 0.0043 | 0.0134 | 4.0 | 0.32 | 0.00 |

Reading: at Λ/τ_c² ≳ 0.5 (β ≥ 1 here) the estimator is 20–360 times more precise than areal GLS and reaches 60–70 % of the oracle at β=2; at β ≤ 0.5 the within-block key posterior is nearly flat, the gain shrinks to 1.4–1.9× and the sign is wrong in 10–25 % of runs, where the areal estimator itself has standard error 0.3–0.4. That low-SNR regime is exactly where the between-block coupling of the key posteriors, which only the full likelihood uses, would matter; no estimator with a proved theory reaches it today.

The exposure-surface case (single key, covariate a smooth field with range 0.3) gives modest gains (1–4×) and stays at 7–32 % of the oracle, as Section 8 of the theory predicts for a covariate that is nearly constant within a block. The covariate-key-only case behaves like the both-keys case (86–220× at β ≥ 1). The plate-like rows (B=12, K≈96) show the limit of few blocks: with 12 sums the sign is poorly identified by any block-sum estimator and the augmented estimator inherits it (wrong sign 12–33 %); the real plate-mix-up regime, in which only a minority of wells are mislabelled, is easier because the sign is never in doubt.

## 5. Privacy and motivation: unaffected

Theorem 1's floors bound every adversary and say nothing about the analyst's estimator. The release-mechanism story (independent keys per block) is what makes the block terms decouple, so it is if anything more natural for the composite likelihood than for the common-key REPAIR. The natural-instance examples do not depend on the likelihood. The efficiency claim becomes "the augmented composite estimator beats areal GLS by I*/I_agg, proved and checked", instead of "REPAIR approximates something we cannot analyse". Theorem 3 (common key) is about the maximum-likelihood keys and is unaffected. One sentence for the Discussion: the full likelihood recovers the remaining information, its asymptotics are the open item (Theorem 4 states what is missing), and the composite estimator is a standard object (Lindsay 1988; Varin, Reid and Firth 2011) whose loss is quantified in Tables 1 and 4.

## 6. What to do in the code (not done; the repository is untouched)

- Add a per-block key sampler (transposition Metropolis; enumeration for K_b ≤ 5) that returns E[X̃'Ω_b r] and E[X̃'Ω_b X̃]; per-block, parallel, unequal K_b, with the option π_X = π_S.
- Replace the β update by the damped Newton step of §3, started at `GPArealModel`'s estimate; take σ², φ, τ² from the areal stage (or profile them on ℓ* afterwards).
- Keep the Sinkhorn relaxation as an alternative sampler and check it against the MCMC/enumeration at K=5 and K=20.
- Simulation design for the paper: the realistic grid of §4 (unequal K_b of 20–30, B of 64–100, Matérn ν∈{0.5,1.5}, both keys / single key with a covariate field / covariate key only, plate-like K≈96), reporting areal, augmented composite, oracle, and the per-record re-identification rate next to the floors.
