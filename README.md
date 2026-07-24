# 3D RAIL (Python)

Python port of a MATLAB RAIL integrator for 3D convection-diffusion-type PDEs,
extended to 3D via **Tucker decomposition**. RAIL ("rank-adaptive
implicit-explicit") keeps the solution in a **low-rank Tucker format** the
whole time, so a full `N x N x N` grid never has to be stored densely.

Reference: Nakao, Ceruti, Einkemmer, *A low-rank, high-order implicit-explicit
integrator for three-dimensional convection-diffusion equations*, Computer
Physics Communications 325 (2026) 110163.

This README is written for someone who knows the math but is new to the code.
It covers the concepts shared by every solver in this repo: how a Tucker
tensor is stored, the K-step / S-step algorithm that advances one time step,
and the MATLAB-vs-NumPy index-ordering gotcha. Each solver's own README covers
only what's specific to it (the exact PDE, file map, and CLI usage).

---

## Requirements

- Python 3.13+
- numpy, scipy, matplotlib
- [`pyTensorlab`](https://pytensorlab.net) (imported as `tl`) — provides
  `mlsvd` (multilinear SVD / Tucker decomposition), `tmprod` (n-mode product),
  and `tens2mat` (tensor unfolding).

```bash
pip install -r requirements.txt
```

## Repository layout

| Folder | PDE | Status |
|---|---|---|
| [`linear_advection_diffusion/`](linear_advection_diffusion/README.md) | linear advection + diffusion + source | Verified against MATLAB (see below) |
| [`linear_reaction_diffusion/`](linear_reaction_diffusion/README.md) | linear advection + diffusion + linear reaction (`+beta*u`) + source | Extension beyond the MATLAB reference |
| [`viscous_burgers/`](viscous_burgers/README.md) | nonlinear (Burgers) advection + diffusion + source | Extension beyond the MATLAB reference |

Each folder is self-contained and runnable on its own (`python main.py ...`);
none of them import from one another.

---

## 1. Tucker format: how the solution is stored

A full 3D solution on an `N x N x N` grid has `N^3` numbers — too many to keep
dense every step. Instead we store a **Tucker tensor**: three thin "factor
matrices" plus a small "core".

```
A Tucker tensor is a pair (U, G):
    U = [U1, U2, U3]   factor matrices, shapes (N1,r1), (N2,r2), (N3,r3)
    G                  core tensor,     shape (r1, r2, r3)

Full tensor:
    T[i,j,k] = sum over a,b,c of  G[a,b,c] * U1[i,a] * U2[j,b] * U3[k,c]
```

The triple `(r1, r2, r3)` is the **multilinear rank** (MLR). If the ranks stay
small, we store `N*r1 + N*r2 + N*r3 + r1*r2*r3` numbers instead of `N^3` — a
massive saving. The whole point of RAIL is to keep the ranks small and
**adapt** them as the solution evolves.

> **Convention used everywhere in this code:** a Tucker tensor is the Python
> tuple `(factors, core)` where `factors = [U1, U2, U3]` (a list of NumPy
> arrays) and `core` is a 3D NumPy array.

---

## 2. The three pieces of one time step

Diffusion (and, where present, a linear reaction term) is treated
*implicitly*; advection is treated *explicitly*. Every implicit stage, at any
order, is built from the same three ideas:

1. **K-steps — update the factor matrices.**
   For each mode `m` (x, y, z) we solve a **Sylvester equation** for a new
   factor matrix, then orthonormalize it with a QR factorization. One
   Sylvester solve per mode, so three per stage.

2. **Reduced augmentation — make the ranks compatible.**
   The core solver (Simoncini) needs all three ranks equal, `r1 = r2 = r3 = R`.
   We stack the new bases next to the old ones, orthonormalize, and keep the
   `R` most significant directions (`red_aug_S`). Higher-order stages also do a
   *per-axis* augmentation before the K-steps (`red_aug_K`).

3. **S-step — update the core.**
   With the factor matrices fixed, the core `G` satisfies a third-order tensor
   Sylvester equation. `simoncini.py` solves it directly (V. Simoncini,
   *Numerical solution of a class of third order tensor linear equations*,
   BUMI 2020, Algorithm T3-sylv).

Then we **truncate** (`nonconstrun`): an HOSVD of the new core throws away
tiny singular values, shrinking the rank back down. This is the
"rank-adaptive" part of RAIL.

Higher-order schemes (`imex222`, `imex443`) are multi-stage DIRK methods built
from the same machinery: stage 1 is an `imex111` sub-step, and each later
stage assembles its right-hand side as an explicit **list of terms** —
`(coeff, kind, tensor, flux_dir)` tuples where `kind` is `'identity'`
(transport/source), `'diffusion'`, or `'flux'` (one advection flux). Two
helpers, `k_rhs_piece`/`s_rhs_piece`, turn each term into a K-step matrix or
S-step tensor contribution; this mirrors the MATLAB line-for-line so the two
codebases can be read side by side. Each solver's `imex.py` documents its own
PDE-specific term list.

---

## 3. Index-ordering gotcha (read this before debugging tensors)

MATLAB stores arrays **column-major** (Fortran order); NumPy defaults to
**row-major** (C order). This bites whenever you unfold/reshape a tensor:

- `pytensorlab.tens2mat` and `tmprod` use NumPy C-order. They are *internally
  consistent*, so as long as we build a right-hand side with `tmprod` and then
  matricize it with `tens2mat`, the column ordering always matches itself.
  That is why this code never hand-writes a Kronecker product for the RHS.
- The one place we *do* write a Kronecker product is the Sylvester `B` matrix
  in a K-step. There we use the verified rule: for mode `m`, the other two
  modes pair in **ascending** order (smaller index outer, larger index inner).
- `simoncini.py` reshapes with `order='F'` on purpose, to match the MATLAB
  `tens2mat(B,1)` exactly. Do not "simplify" those to C-order.

---

## 4. Gotchas that apply across all three solvers

- **HOSVD ranks can differ by 1-2 between MATLAB and Python** near the
  truncation threshold / large CFL even when the L1 error matches to ~1e-8 —
  this is LAPACK tie-breaking on a near-degenerate singular value, not a port
  bug.
- **`tkr` vs `tkron` column order.** `flux_product` builds each product
  factor with `tkr` (row-wise Khatri-Rao) and the core with `tkron`
  (`np.kron`). These must use a *consistent* column ordering for the product
  to be correct. The numbers reproduce the expected convergence orders, which
  is good evidence the ordering is right, but this is the most
  ordering-sensitive spot in the port.

## License

MIT — see [LICENSE](LICENSE).
