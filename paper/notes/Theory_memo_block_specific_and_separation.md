# Theory memo: what changes under block-specific permutations, unequal blocks and the three sub-cases, and how to prove the separation

Prepared 18 Sep 2026 from a reading of the proofs in the submitted supplement (`doubly_unlinked_supplement.tex`, Sections 2.1–2.3) together with the simulation design. Notation as in the paper: Π₁ = Π_Sᵀ Π_X, Π₂ = Π_Sᵀ, Π₂Y = Π₁Xβ + W*, W* ~ N(0, Σ), Σ = σ²R + τ²I; for a candidate pair (Π̂₁, Π̂₂) the two quantities that drive every bound are

- h₂ := d_H(Π̂₂, Π₂) (how far the estimated spatial permutation is from the truth), and
- h₁₂ := d_H(Π̂₁₂, I) with Π̂₁₂ := Π̂₂Π₂ᵀΠ₁Π̂₁ᵀ (the *relative* misalignment between the estimated X-alignment and the estimated Y-alignment).

Section 1 records what the current proofs actually depend on; Sections 2–4 give the answers for the three generalisations; Section 5 is the separation theorem I propose; Section 6 is the SNR redefinition, with numbers; Section 7 is the work plan.

---

## 1. Anatomy of the current proofs

**Theorem 1 (permutation recovery).** Union bound over all wrong candidates. For each candidate the event {Δ ≤ 0} is split into E₁ (signal: β²‖T‖² plus a Gaussian cross term), E₂₁ (difference of two χ²₁), E₂₂ (a Hanson–Wright quadratic form with rank ≤ 2h₂), and the event {‖T‖² < t*}. The four propositions give per-candidate tails of the form exp(−c·SNR·t), exp(−c·SNR·[t − h₂/SNR]) and 6·exp(−(h₁₂/10)·ψ(h₁₂/t)), valid for any t ∈ [h₂/SNR, h₁₂]. **None of these propositions uses the "same permutation in every block" structure.** They use only (i) the two full-n Hamming distances h₂, h₁₂, (ii) λ_max(Σ), κ(Σ), and (iii) X ~ N(0, I_n) (through Pananjady's Lemma 4 for ‖T‖²). The block structure enters in exactly three places:

1. h₂ = k₂B and h₁₂ = k₁₂B, so the smallest nonzero Hamming distance of a wrong candidate is 2B. This is what makes every exponent linear in B.
2. The candidate count: at most K^{k₂+k₁₂} candidates with given (k₂, k₁₂), summed over 2 ≤ k₂, k₁₂ ≤ K (K!² candidates in all).
3. The choice t = k₁₂(h₂ + log SNR)/SNR, which has to sit below h₁₂ = k₁₂B; this is satisfied iff SNR ≥ K + (log SNR)/B, and is the *only* source of the requirement SNR = Ω(K^α). It comes from the E₂₂ term, whose trace bound |tr A| ≤ √2·h₂·κ(Σ) grows with h₂: misaligning W* by Π̂₂ perturbs the noise quadratic form by an amount proportional to the number of moved coordinates, and the signal must dominate that.

**Theorem 2 (β consistency).** Writes β̂ − β = β·[X̃ᵀΣ⁻¹(Π̂₁₂ − I)X̃]/[X̃ᵀΣ⁻¹X̃] + (noise term), then bounds the first term by |β|·κ(Σ)·‖Π̂₁₂ − I‖₂ and shows P(Π̂₁₂ ≠ I) ≤ c₁K²e^{−2Bφ(SNR)}. Two things to notice. First, Theorem 2 does **not** avoid permutation recovery; it avoids recovering Π₁ and Π₂ *separately* and requires only the relative permutation Π̂₁₂ = I, i.e. that the estimator is wrong about X and wrong about Y in the same way. Its weaker SNR condition (log SNR > 1 instead of SNR ≥ K^α) is because comparing (Π̂₁, Π̂₂) against (Π̂₁, Π̂₁Π₁ᵀΠ₂) keeps Π̂₁ fixed, so the h₂-dependent E₂₂ term never appears; only h₁₂ does. Second, the bound ‖Π̂₁₂ − I‖₂ ≤ 2 is a spectral-norm bound and is O(1) even when a single pair of coordinates is misaligned; it cannot deliver "β is fine when a small fraction of positions is wrong". A Frobenius/Hamming version of the same step (Section 3) gives bias = O(|β|·κ·h₁₂/n), which is what one needs.

**Where the SNR normalisation comes from.** Two crude steps: in Prop. prob-T-less-t, ‖v‖²_{Σ⁻¹} ≥ λ_min(Σ⁻¹)‖v‖² = ‖v‖²/λ_max(Σ), and in Prop. E1, ‖M‖ ≤ √κ(Σ) with M = Σ^{−1/2}Π̂₂Π₂ᵀΣ^{1/2}. Both replace a quantity that depends only on the h moved coordinates by a global spectral bound; under strong dependence λ_max grows like n and κ like n² (Section 6).

---

## 2. Block-specific permutations (Referee 2, and every real instance)

Let Π₁ = bdiag(π₁⁽¹⁾,…,π₁⁽ᴮ⁾), Π₂ = bdiag(π₂⁽¹⁾,…,π₂⁽ᴮ⁾), block b of size K_b (Section 3 for unequal). Then h₂ = Σ_b k₂⁽ᵇ⁾ and h₁₂ = Σ_b k₁₂⁽ᵇ⁾ and **all four propositions hold verbatim** with these h's. What changes is items 1–3 of Section 1:

- Smallest nonzero Hamming distance is now 2 (one transposition in one block), not 2B.
- Number of candidates with total distance h is at most K^h·(number of compositions of h into ≤ B parts each ≥ 2) ≤ K^h·(2e(B/h + 1))^{h/2}.
- The per-candidate bound for a candidate at distance h is ≈ exp(−c·h·φ(SNR)) with φ(SNR) ≍ log SNR (Theorem 2's choice of t) and does not involve B at all.

Consequence for **permutation recovery**: the union bound is dominated by its smallest-h term,
P(some block wrong) ≲ Σ_{h≥2} [K·(2e(B/h+1))^{1/2}]^h e^{−c h φ(SNR)} ≍ B·K²·SNR^{−2c},
which does not vanish as B → ∞ at fixed SNR. This is not a weakness of the bound: with independent keys, each block is a separate K-sized shuffled-regression problem with no pooling, so P(all B blocks correct) ≤ (1 − p_K(SNR))^B → 0 for any fixed SNR, where p_K(SNR) > 0 is the per-block error probability (strictly positive for Gaussian data at any finite SNR). Exact recovery of all keys requires SNR growing polynomially in B, i.e. Theorem 1 becomes an *impossibility* statement at fixed SNR. That is the risk half of the separation, and it is the opposite of the common-key case where pooling gives e^{−cB}.

Consequence for **β via Theorem 2's route**: P(Π̂₁₂ ≠ I) no longer vanishes either (same union), so the plug-in profile MLE β̂(Π̂) is not shown consistent by that route. Two fixes, in increasing strength:

(a) *Hamming-fraction bias bound.* Replace ‖Π̂₁₂ − I‖₂ by a bound on the ratio of quadratic forms directly: the numerator X̃ᵀΣ⁻¹(Π̂₁₂ − I)X̃ involves only the h₁₂ moved coordinates and is O_p(h₁₂·‖Σ⁻¹‖_{loc}), the denominator is Ω_p(n/λ_max), so |bias| ≲ |β|·κ_loc·(h₁₂/n). With independent keys h₁₂/n converges to a constant ≤ (per-block error rate)·(typical distance/K), so β̂(Π̂) is consistent up to a bias proportional to the block error rate: bounded, not vanishing.

(b) *Marginal (mixture) likelihood, which is what REPAIR with per-block q(π_b) approximates.* Under independent keys the model for block b given (X_b, s_b) is a finite mixture over (π_X, π_S) ∈ P_{K_b} × P_{K_b} of N(π_X X_b β, π_S Σ_{b|·} π_Sᵀ + τ²I) with β shared across blocks. Identifiability of β is immediate because block sums are permutation-invariant: 1ᵀY_b = (1ᵀX_b)β + 1ᵀW_b + 1ᵀε_b for every (π_X, π_S). Hence the block-mean GLS estimator (ArealGP) is consistent for β at rate √B for **any** within-block permutation mechanism, any K_b, any SNR > 0, and the mixture MLE is consistent by standard M-estimation (identifiable, i.i.d. across blocks up to the between-block dependence of W, which is handled by the GP likelihood) with asymptotic variance between the oracle's and ArealGP's. This is the utility half of the separation and it needs no SNR condition at all.

So under block-specific keys the clean statement is: β is √B-consistent at any SNR (efficiency ladder oracle ≥ mixture MLE ≥ block means), while the probability of exactly recovering the keys tends to zero and the expected fraction of correctly re-linked records is bounded away from one. The common-key model is the case where pooling breaks the risk half; it stays in the paper as the privacy-weakest bound.

---

## 3. Unequal block sizes

Per-candidate bounds: unchanged (they only see h₂, h₁₂, Σ). Counting: replace K by K_max in the candidate counts; the mixture/aggregation consistency in 2(b) needs only B → ∞ with K_b bounded (or K_b growing slowly enough that the union bound over per-block candidates is controlled). The common-key model cannot even be defined with unequal blocks, which is a second reason the applied model must be block-specific. One disclosure remark worth a sentence: the block *size* is public and can identify the block (a county with 37 tracts), but it carries no information about the within-block correspondence, which is what the mechanism hides. The five Meuse points dropped to make equal blocks become unnecessary.

---

## 4. The three sub-cases, and where the separation lives

In the paper's parametrisation Π₁ = Π_SᵀΠ_X, Π₂ = Π_Sᵀ.

**Π_S = I (attribute shuffling; plate mix-ups; linkage error).** Π₂ = I is known, Π₁ = Π_X. Then h₂ = 0 for every candidate, the E₂₂ term vanishes identically, the candidate set is P_K (not P_K²), and the valid choice of t is t = h·log SNR/SNR for any SNR > 1. Theorem 1's condition collapses from SNR = Ω(K^α) to log SNR > 0 with B ≳ log K/log SNR. **The polynomial-in-K SNR requirement is entirely due to the spatial permutation Π_S**: this is the classical shuffled-regression-with-correlated-errors problem and behaves like Pananjady's iid case restricted to blocks.

**Π_X = I (geography swapping; Census/ONS; Slide-seq).** Π₁ = Π₂ = Π_Sᵀ, so the estimator is constrained to π̂₁ = π̂₂ and then Π̂₁₂ = Π̂₂Π₂ᵀΠ₂Π̂₂ᵀ = I *for every candidate*. The bias term in Theorem 2 is identically zero: β̂(Π̂) is unbiased for any candidate, and consistent with no SNR condition (this is the formal reason agencies observe that swapping "barely affects regression coefficients"). Recovery of Π_S comes only from Case 2 of Theorem 1 (the W* covariance terms, exp(−c·k₂B) under a common key), so under a common key Π_S is recoverable regardless of β, and under independent keys the per-block error is constant. This sub-case gives the sharpest separation and the simplest theorem.

**Π_X = Π_S (exposure surface; DHS/Cook County with X read at the true address).** Π₁ = I is known and Π₂ = Π_Sᵀ is the only unknown; h₁₂ = h₂. Here Π̂₁₂ = I iff Π̂₂ = Π₂, so Theorem 2's route needs *exact* recovery and gives no separation. The separation must come from 2(b): block sums are still permutation-invariant (1ᵀY_b = (1ᵀX_b)β + …), so β is consistent by aggregation and by the mixture MLE, while per-block recovery has bounded error under independent keys. Note that in this sub-case the signal does help recover Π_S (unlike Π_X = I), which is why the Cook County risk curves will differ between the two mechanisms; both should be reported.

**General two-permutation case.** As in the paper, plus 2(b). The role of the general model is to nest the three sub-cases in one likelihood and one algorithm.

---

## 5. Proposed separation theorem (block-specific keys, unequal blocks)

Setting: blocks b = 1..B of sizes K_b ≤ K_max; independent unknown (π_X⁽ᵇ⁾, π_S⁽ᵇ⁾); W a GP with Matérn covariance and nugget τ² > 0 on a fixed domain; X either iid or an exposure surface (Π_X = Π_S).

**Theorem A (utility).** Let β̂_agg be the GLS estimator from block means and β̂_mix the maximiser of the marginal likelihood over β, Γ with the per-block permutations integrated out. Then β̂_agg − β = O_p(B^{−1/2}) for every SNR > 0 and every mechanism in the class, and β̂_mix is consistent with Var(β̂_mix) ≤ Var(β̂_agg), with equality to the oracle variance in the limit where the per-block permutation posteriors concentrate. Proof: permutation invariance of block sums for β̂_agg (three lines); identifiability + standard M-estimation under a stationary mixing GP for β̂_mix (the only technical point is the between-block dependence, handled by the GP likelihood or by a blocking CLT). No condition on K or SNR.

**Theorem B (risk).** For any estimator of the keys, the per-block error probability is at least p(K, SNR_loc) > 0, where p is given by a two-point (Le Cam) bound comparing the truth with a within-block transposition of the two closest locations: p ≥ ¼·exp(−KL), KL = β²(X_i − X_j)²/(2·Var(W*_i − W*_j | rest)) for the Π_S = I direction and the analogous covariance-KL for the Π_S direction; a Fano version over the K_b! within-block permutations gives p bounded away from zero whenever SNR_loc ≲ log K. Consequently P(all keys recovered) ≤ (1 − p)^B → 0 and E[fraction of records correctly re-linked] ≤ 1 − p·(2/K_max). Under a common key, the same quantities tend to 1 and 0 respectively (existing Theorem 1), which is the statement that a common key is the weakest mechanism.

Together: at fixed SNR and any K, β is estimable at the parametric rate while the correspondences are not recoverable — a two-sided separation with a converse, which the current paper does not have. The existing Theorems 1–2 become the common-key corollary.

---

## 6. Redefining the SNR (Referee 1's fixed-domain point, and the reason the theory does not describe the experiments)

The paper's SNR is β²/(λ_max(Σ)κ(Σ)). On the simulation design (unit square, √B×√B cells, K uniform points per cell, σ² = 5, range 0.5, τ² = 0.5) it is between 10⁻⁵ and 10⁻⁸ in every cell, because λ_max ≈ 600–4,900 and κ ≈ 1,100–9,700 grow with n. No simulation satisfies log SNR > 1, yet REPAIR recovers both permutations almost everywhere. The bound is loose exactly where the two crude steps of Section 1 replace local quantities by global spectral ones.

The quantity the proofs actually need, for a candidate that moves only coordinates inside blocks, is the covariance of the moved coordinates *conditional on the rest of the field*, i.e. the Schur complement Σ_{b|−b} = (Σ⁻¹_{bb})⁻¹. On the same design:

| B | K | n | λ_max(Σ) | κ(Σ) | max_b λ_max(Σ_{b|−b}) | κ of the worst block Schur complement | SNR_paper (β=2 / 8) | SNR_cond := β²/max_b λ_max(Σ_{b|−b}) (β=2 / 8) |
|---|---|---|---|---|---|---|---|---|
| 49 | 6 | 294 | 592 | 1,110 | 7.46 | ≤ 14 | 6e-6 / 1e-4 | 0.54 / 8.6 |
| 49 | 20 | 980 | 1,988 | 3,912 | 13.46 | ≤ 26 | 5e-7 / 8e-6 | 0.30 / 4.8 |
| 121 | 6 | 726 | 1,469 | 2,880 | 3.75 | ≤ 7 | 9e-7 / 2e-5 | 1.07 / 17.1 |
| 121 | 20 | 2,420 | 4,882 | 9,730 | 8.57 | ≤ 17 | 8e-8 / 1e-6 | 0.47 / 7.5 |

Three properties of SNR_cond: it is O(1)–O(10) where the paper's SNR is 10⁻⁶; it *increases* with B (denser sampling pins W down: the screening effect), whereas the paper's SNR decreases with n; and it orders the empirical difficulty exactly as the recovery heatmap does (β = 2 harder than 8, K = 20 harder than 6, B = 121 easier than 49). It is also bounded uniformly in n under fixed-domain asymptotics for Matérn with a nugget (the nugget is essential; without it the conditional variance is not bounded below and the Schur complement degenerates), which turns the increasing-domain/Toeplitz paragraph on p. 11 into a screening-effect paragraph (Stein's work on the screening effect is the reference) and answers Referee 1's fixed-domain comment with a definition rather than a discussion.

How to carry it through the proofs: condition on W* outside the moved set S (a union of within-block subsets); on S the relevant covariance is Σ_{S|S^c}; for the small-h candidates that dominate the union bound, |S| ≤ 2K and λ_max(Σ_{S|S^c}) ≤ max_b λ_max(Σ_{b|−b}); for large-h candidates keep the crude bounds (their exponents are large anyway). The four propositions go through with (λ_max, κ) replaced by their conditional versions on the dominant terms.

---

## 7. What has to be done, in order

1. Theorem A: write the block-mean invariance argument and the mixture-MLE consistency (short; the GP between-block dependence is the only care point). This is the utility side for both applications and for all three sub-cases.
2. Theorem B: two-point and Fano lower bounds per block; the (1 − p)^B and expected-fraction corollaries. This is the converse the paper currently lacks and the risk side of the R-U map.
3. Restate Theorems 1–2 as the common-key corollary with SNR_cond in place of SNR, redoing Props. E1 and prob-T-less-t with the Schur complement on small-h candidates. This is what makes the theory describe the experiments.
4. Sub-case corollaries: Π_S = I (SNR condition collapses to log SNR > 0), Π_X = I (bias identically zero), Π_X = Π_S (separation only via Theorem A).
5. Method: promote `Experiments/Diff_piX_piS` to the main algorithm (per-block q(π_b), pooled globals), allow K_b, add the Π_X = Π_S constraint as an option (one permutation variable instead of two).
6. Simulations: common vs independent keys; report SNR_cond for every cell; report per-record re-identification rate alongside exact recovery; the low-SNR and Matérn-ν sweeps Referee 1 asked for come for free once SNR_cond is the design variable.
