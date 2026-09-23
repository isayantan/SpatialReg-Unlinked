# Manuscript: *Doubly-Unlinked Regression for Dependent Data*

Burman, Choudhury, Dey — under revision after a first round of review (Sept 2026).

## Where things are

| Path | What |
|---|---|
| `/doubly_unlinked.tex`, `/doubly_unlinked_supplement.tex` (repo root) | **Working revision** of the main paper and supplement. They sit at the repo root because Overleaf compiles from the project root; `macros_du.tex`, `doubly_unlinked.bib`, `oup-authoring-template.cls`, `latexmkrc`, `Figures/`, `assets/` are their inputs. |
| `paper/submitted_2026-04/` | The manuscript and supplement as submitted on 15 Apr 2026 (the main PDF is rebuilt from the submitted source with a neutral running header). The matching source is git tag **`submitted-2026-04-15`**: `git diff submitted-2026-04-15 -- doubly_unlinked.tex` shows every change made since submission. |
| `paper/theory/` | Theory document for the revision (`separation_theorems.tex/.pdf`): separation floors, efficiency, common-key recovery, full-likelihood section, multivariate and spatially varying covariates. `sims/` holds the numerical checks behind its tables; `Figures/` its figures. |
| `paper/notes/` | Working memos: estimator design (composite vs full likelihood), literature scan of real-world unlinked instances, decision tables for applications, theory memo. |
| `paper/private/` | **Git-ignored, local only**: referee reports, decision letter, cover letters, letterhead, the earlier submission package, the original submitted PDF. Nothing in here reaches GitHub or Overleaf. Keep a backup elsewhere. |

## Overleaf ⇄ GitHub workflow

The Overleaf project was created by importing this repository, so the whole repo is mirrored there.

* **Main document in Overleaf:** `doubly_unlinked.tex` (Menu → Main document). Switch to `doubly_unlinked_supplement.tex` to build the supplement, or to `paper/theory/separation_theorems.tex` for the theory doc (its `\graphicspath` handles compiling from the root).
* **Edits made on Overleaf → GitHub:** in Overleaf, Menu → GitHub → *Push Overleaf changes to GitHub* (creates a commit on `main`). Then `git pull` locally.
* **Edits made locally → Overleaf:** `git push`, then in Overleaf Menu → GitHub → *Pull GitHub changes into Overleaf*.
* Neither direction is automatic; if both sides changed, Overleaf asks you to resolve the conflict, so pull before starting a session on either side.
* Build products (`.aux`, `.bbl`, root `.log`/`.pdf`, …) are git-ignored; compiled PDFs of the revision live only in Overleaf until we freeze a submission copy into `paper/submitted_<date>/`.
* `\journaltitle{Preprint}` in `doubly_unlinked.tex` keeps the running header neutral; set it to the journal's name only in the copy that is actually submitted.

## Local compile

```bash
latexmk -pdf doubly_unlinked.tex             # main paper (latexmkrc is picked up automatically)
latexmk -pdf doubly_unlinked_supplement.tex  # supplement
cd paper/theory && latexmk -pdf separation_theorems.tex
```
