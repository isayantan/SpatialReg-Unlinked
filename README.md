# REPAIR: Doubly-Unlinked Regression for Dependent Data

This repository contains the code for the paper

> **Doubly-Unlinked Regression for Dependent Data**
> Anik Burman¹, Sayantan Choudhury², Debangan Dey³
> ¹Department of Biostatistics, Johns Hopkins University
> ²Department of Statistics and Data Science, MBZUAI
> ³Department of Statistics, Texas A&M University

**REPAIR** (**RE**gression with **P**ermutation **A**lignment via var**I**ational infe**R**ence) is a variational Bayes method for spatial regression when both the covariate–outcome linkage and the response–location linkage within spatial blocks are unknown.

---

## Problem

Consider $N = B \times K$ observations divided into $B$ spatial blocks of size $K$. Within each block $i$, the model is

$$\mathbf{Y}_i = \boldsymbol{\pi}_X \mathbf{X}_i \beta + \boldsymbol{\pi}_S \mathbf{W}_i + \boldsymbol{\varepsilon}_i, \qquad i = 1, \ldots, B,$$

where $\boldsymbol{\pi}_X, \boldsymbol{\pi}_S \in \mathcal{S}_K$ are **latent permutation matrices** shared across all blocks, $\mathbf{W} \sim \mathrm{GP}(0, \sigma^2 C(\cdot;\phi))$ is a latent spatial field with Matérn ($\nu = 1/2$) covariance, and $\boldsymbol{\varepsilon}_i \sim N(\mathbf{0}, \tau^2 \mathbf{I}_K)$. Neither $\boldsymbol{\pi}_X$ nor $\boldsymbol{\pi}_S$ is observed.

REPAIR jointly recovers $(\boldsymbol{\pi}_X, \boldsymbol{\pi}_S, \beta, \phi, \sigma^2, \tau^2)$ via mean-field variational inference with Gumbel–softmax relaxation of the permutation variables and temperature annealing.

---

## Vignette

A self-contained walkthrough is provided in [`vignette.ipynb`](vignette.ipynb). It covers:

1. Generating synthetic doubly-unlinked data for any rectangular $B = n_\text{rows} \times n_\text{cols}$ block grid
2. Fitting the **FullGP** oracle baseline (true links known)
3. Fitting the **ArealGP** baseline (block-averaged data)
4. Running **REPAIR** and inspecting ELBO convergence
5. Evaluating permutation recovery accuracy
6. Comparing parameter estimates across all three methods

---

## Manuscript

The LaTeX source of the paper lives in this repository so that it can be edited on Overleaf and on GitHub alike: `doubly_unlinked.tex` / `doubly_unlinked_supplement.tex` at the root are the working revision, and [`paper/`](paper/README.md) holds the submitted version, the theory document, and working notes. See `paper/README.md` for the layout and the Overleaf sync workflow.

---

## Repository Structure

```
SpatialReg-Unlinked/
├── vignette.ipynb                  # ← start here
├── src/
│   ├── helper_functions.py         # clean API (generate data, train models, evaluate)
│   ├── revised_VIGP_Unlinked.py    # core REPAIR implementation
│   ├── revised_elbo.py             # variational distributions q(π_X), q(π_S)
│   ├── utils.py                    # Sinkhorn, Hungarian, GP kernel utilities
│   ├── GPModel.py                  # FullGP (oracle) likelihood model
│   ├── GPArealModel.py             # ArealGP (block-averaged) likelihood model
│   ├── analysis_varyB_annealed.py  # simulation study script (vary B, K, φ, seed)
│   ├── data_analysis_2.ipynb       # real-data analysis (dataset 1)
│   ├── data_analysis_3.ipynb       # real-data analysis (dataset 2)
│   └── generate_data_new.ipynb     # data generation notebook
├── Figures/                        # paper figures
├── data/                           # simulation inputs and results (not tracked)
└── notebooks/                      # exploratory notebooks
```

---

## Quick Start

```python
import sys
sys.path.insert(0, 'src')

from helper_functions import (
    generate_synthetic_data,
    train_oracle_gp,
    train_areal_gp,
    run_vigp_unlinked,
    permutation_accuracy,
    summarise_results,
)

# Generate doubly-unlinked data — any rectangular grid works
# B=6 → 2×3, B=12 → 3×4, B=9 → 3×3 (auto-detected)
data = generate_synthetic_data(B=9, n_i=5, beta_true=1.5, phi_true=2.0, seed=42)

# Fit all three methods
oracle = train_oracle_gp(s=data["s"], x=data["x"], y=data["y"])
areal  = train_areal_gp(s_perm=data["s_perm"],
                         region_assignments=data["region_assignments"],
                         x_perm_flat=data["x_perm_flat"], y=data["y"])
repair = run_vigp_unlinked(data=data, n_iter=50, seed=521)

# Permutation recovery
print(permutation_accuracy(repair["est_perm_x"], data["perm_matrix_x"]))

# Parameter comparison table (True / FullGP / ArealGP / REPAIR)
print(summarise_results(oracle, areal, repair, data))
```

---

## Simulation Study

The simulation script varies the number of blocks $B$, block size $K$, spatial range $\phi$, and random seed:

```bash
cd src
python analysis_varyB_annealed.py <B> <K> <seed> [phi]
# e.g.: python analysis_varyB_annealed.py 9 5 1 2.0
```

Results are saved to `data/results/vary_B2_annealed/B_{B}_n_{K}/phi_{phi}/results_seed_{seed}.pt`.

---

## Dependencies

```
python >= 3.10
torch
numpy
scipy
pandas
matplotlib
tqdm
```

Install with:

```bash
pip install torch numpy scipy pandas matplotlib tqdm
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{burman2025doubly,
  title   = {Doubly-Unlinked Regression for Dependent Data},
  author  = {Burman, Anik and Choudhury, Sayantan and Dey, Debangan},
  journal = {},
  year    = {2025}
}
```
