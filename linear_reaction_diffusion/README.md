# 3D RAIL — Linear Advection-Diffusion-Reaction Solver (Python)

This folder is a copy of `3DRAIL_py/linear_advection_diffusion/` (itself a
Python port of the MATLAB code in
`3dRAIL/rail3d_sourcecode/rail3d_linear_advection_diffusion/`) extended with a
linear reaction term `+beta*u`, per
`docs/3DRAIL_linear_reaction_term_1.md`. It solves a 3D linear
advection-diffusion-**reaction** equation while keeping the solution in a
**low-rank Tucker format** the whole time, using the **RAIL** integrator (a
"rank-adaptive implicit-explicit" scheme).

The reaction term is folded entirely into the diffusion matrices
(`main.py::build_diff_matrices`, `F_tilde = F + (beta/3)*I`) at setup; nothing
else in `imex.py`/`helpers.py`/`simoncini.py` changes. `beta` is hardcoded per
test case as `TestParameters.beta` (not a run()/CLI argument, since each
test's exact solution/source is derived for one specific beta) -- positive
means growth, negative means decay, and `beta=0.0` (the default) reproduces
the original solver's behavior exactly. See `main.py`'s docstring for the
`dt*beta < 1` growth-stability constraint. LoMaC's conserved reference moments
(`lomac.py`) are rescaled by `exp(beta*t)` rather than held fixed at the
initial condition, so LoMaC truncation composes correctly with the reaction
term.

This README is written for an engineering student who knows the math but is new
to large code bases.  It explains (1) what problem we solve, (2) how a Tucker
tensor is stored, (3) what each file does and which MATLAB file it came from,
and (4) pseudocode walkthroughs of the time step.

---

## 1. The PDE

On the cube `[-L/2, L/2]^3` with periodic boundary conditions we solve

```
u_t + (a1 u)_x + (a2 u)_y + (a3 u)_z  =  d1 u_xx + d2 u_yy + d3 u_zz  +  phi
        \___________________________/      \____________________________/     \_/
                advection (explicit)              diffusion (implicit)        source
```

- `(a1, a2, a3)` is a (possibly space- and time-dependent) **flow field**.
- `(d1, d2, d3)` are constant **diffusion coefficients**.
- `phi` is a known **source term**.

"IMEX" = **IM**plicit-**EX**plicit.  Diffusion is stiff, so we treat it
*implicitly* (solve a linear system each step).  Advection is not stiff, so we
treat it *explicitly* (just evaluate it).  Three IMEX schemes are available,
of increasing order of accuracy: **IMEX111** (1st), **IMEX222** (2nd),
**IMEX443** (3rd).

---

## 2. Tucker format: how the solution is stored

A full 3D solution on an `N x N x N` grid has `N^3` numbers.  That is huge.
Instead we store a **Tucker tensor**: three thin "factor matrices" plus a small
"core".

```
A Tucker tensor is a pair (U, G):
    U = [U1, U2, U3]   factor matrices, shapes (N1,r1), (N2,r2), (N3,r3)
    G                  core tensor,     shape (r1, r2, r3)

Full tensor:
    T[i,j,k] = sum over a,b,c of  G[a,b,c] * U1[i,a] * U2[j,b] * U3[k,c]
```

The triple `(r1, r2, r3)` is the **multilinear rank** (MLR).  If the ranks stay
small, we store `N*r1 + N*r2 + N*r3 + r1*r2*r3` numbers instead of `N^3` — a
massive saving.  The whole point of RAIL is to keep the ranks small and
**adapt** them as the solution evolves.

> **Convention used everywhere in this code:** a Tucker tensor is the Python
> tuple `(factors, core)` where `factors = [U1, U2, U3]` (a list of NumPy
> arrays) and `core` is a 3D NumPy array.

---

## 3. File map (Python <-> MATLAB)

| Python file            | Role                                              | MATLAB origin                          |
|------------------------|---------------------------------------------------|----------------------------------------|
| `main.py`              | Driver: build grids, time loop, plots             | `main.m`                               |
| `test_parameters.py`   | Test case (IC, flow fields, source, exact)        | `test_parameters.m`                    |
| `imex.py`              | All IMEX steps (1st/2nd/3rd order) + dispatcher   | `IMEX111.m`, `IMEX222.m`, `IMEX443.m`  |
| `helpers.py`           | Small tensor utilities                            | `TKR.m`, `tkron.m`, `red_aug_*.m`, `nonconstrun.m` |
| `simoncini.py`         | Direct solver for the core (S-step) equation      | `simoncini_direct_solver.m`            |



## 4. The three pieces of one time step

Every IMEX step (any order) is built from the same three ideas.  For a single
implicit stage they are:

1. **K-steps — update the factor matrices.**
   For each mode `m` (x, y, z) we solve a **Sylvester equation** for a new
   factor matrix, then orthonormalize it with a QR factorization.  One Sylvester
   solve per mode, so three per stage.

2. **Reduced augmentation — make the ranks compatible.**
   The core solver (Simoncini) needs all three ranks equal, `r1 = r2 = r3 = R`.
   We stack the new bases next to the old ones, orthonormalize, and keep the
   `R` most significant directions (`red_aug_S`).  Higher-order stages also do a
   *per-axis* augmentation before the K-steps (`red_aug_K`).

3. **S-step — update the core.**
   With the factor matrices fixed, the core `G` satisfies a third-order tensor
   Sylvester equation.  `simoncini.py` solves it directly.

Then we **truncate** (`nonconstrun`): an HOSVD of the new core throws away tiny
singular values, shrinking the rank back down.  This is the "rank-adaptive"
part of RAIL.

---

## 5. Pseudocode walkthroughs

### 5.1 `main.py`

```
read command-line options (order, test number, lambda sweep, Tf, N, tol)
p     <- TestParameters(test, Tf, N, N, N)        # grids, IC, flow, source, exact soln
diff  <- build differentiation matrices (spectral, periodic)   # Dx,Dy,Dz, Dxx,Dyy,Dzz

for each lambda value:
    dt    <- lambda / (CFL bound)                 # time-step size from a CFL rule
    tvals <- 0, dt, 2dt, ..., Tf                  # last step trimmed to land on Tf

    (U, G) <- initial Tucker tensor from p
    record initial multilinear rank

    for n = 1 .. Nt-1:                            # march in time
        (U, G, MLR) <- imex_step(order, U, G, ..., diff, tol)
        record MLR

    if an exact solution exists:
        L1err <- dx*dy*dz * sum |full(U,G) - u_exact|

plot rank-vs-time  (and error-vs-lambda for a sweep)
```

The two spectral differentiation matrices come straight from Trefethen's
*Spectral Methods in MATLAB* (a skew-symmetric Toeplitz matrix for the first
derivative, a symmetric one for the second).

### 5.2 `imex.py` — one first-order step (IMEX111)

IMEX111 is the general `k_step` / `s_step` machinery (see 5.3) run for a single
implicit stage with diagonal coefficient `a_diag = 1` and star basis equal to
the current basis `U_n` — no reduced-augmentation against a predictor or
earlier stages is needed, because there aren't any.

```
function imex111(U_n, G_n, MLR_n, A, B, C, P, tn, dtn, diff, tol):
    solution <- (U_n, G_n)

    terms <- [ (1.0, 'identity', solution, -) ]              # transport u^n
    terms += flux_terms(-dtn, flow(tn), solution)             # -dt * (a_i u) fluxes at t^n
    terms += [ (dtn, 'identity', P(tn+dtn), -) ]              # +dt * source at t^{n+1}

    V_star <- U_n                                  # in IMEX111, star basis = current basis

    # --- K-steps: one Sylvester solve per mode ---
    for m in (x, y, z):
        V_ddagger[m] <- k_step(m, a_diag=1, dtn, diff, V_star, terms)

    # --- reduced augmentation: make r1=r2=r3=R ---
    (V_new, R) <- red_aug_S(V_ddagger, U_n)

    # --- S-step: update the core ---
    S_nn <- s_step(a_diag=1, dtn, diff, V_new, R, terms)

    # --- truncate back down ---
    return nonconstrun(V_new, S_nn, tol)
```

### 5.3 `imex.py` — second/third order

The higher-order schemes are **multi-stage** DIRK methods.  Stage 1 is just an
IMEX111 sub-step.  Each later stage builds a right-hand side out of the
already-computed stages and then runs the same K-step / augment / S-step / truncate
machinery, with the implicit diffusion scaled by the DIRK diagonal coefficient.

To keep the code readable *and* close to the MATLAB, each stage's right-hand
side is written as an explicit **list of terms**.  A term is a tuple

```
(coeff, kind, tensor, flux_dir)
```

where `kind` is one of:

- `'identity'`  — transport `u^n` or a source `phi` (no spatial derivative);
- `'diffusion'` — `d1 u_xx + d2 u_yy + d3 u_zz` of an earlier stage (explicit
  part: expands into three second-derivative pieces);
- `'flux'`      — one advection flux `d/dx_d (a_d u)` (`flux_dir` says which
  direction `d`).

Each term in the list corresponds directly to one `B = B + lmlragen(...)` line
in `IMEX222.m` / `IMEX443.m`.  For example, IMEX222 stage 2:

```
terms = [
    ( 1.0,            'identity',  u^n,          - )    # transport
    ( (1-gamma)*dt,   'diffusion', u^(1),        - )    # diffusion of stage 1
    ( (1-gamma)*dt,   'identity',  P(t1),        - )    # source at node c1
    ( gamma*dt,       'identity',  P(t^{n+1}),   - )    # source at node c2
]
terms += flux_terms(-delta*dt,     flow(t^n), u^n)      # advection at t^n
terms += flux_terms(-(1-delta)*dt, flow(t1),  u^(1))    # advection at stage 1
```

The two helper functions `k_rhs_piece` and `s_rhs_piece` turn each term into the
matrix (for a K-step) or tensor (for the S-step) contribution.  They are the
Python equivalents of the repeated MATLAB patterns:

```
K-step piece (MATLAB):  F1 * tens2mat(core, m) * kron( P3'*Vstar , P2'*Vstar )
S-step piece (MATLAB):  lmlragen({ Vx'*F1 , Vy'*F2 , Vz'*F3 }, core)
```

One implicit stage (`implicit_stage`) then does:

```
function implicit_stage(U_n, terms, a_diag, dt, diff, tol, predictor, previous_U):
    V_RK    <- [ previous stages ... , U_n ]                 # stacked bases
    V_star  <- red_aug_K(predictor, V_RK)                    # augment for K-steps
    for m in (x,y,z):
        V_ddagger[m] <- k_step(m, a_diag, dt, diff, V_star, terms)
    (V_new, R) <- red_aug_S(V_ddagger, V_RK)                 # equal ranks for S-step
    S_nn       <- s_step(a_diag, dt, diff, V_new, R, terms)  # Simoncini core solve
    return nonconstrun(V_new, S_nn, tol)                     # truncate
```

### 5.4 `helpers.py`

| Function       | What it does                                                       | MATLAB |
|----------------|-------------------------------------------------------------------|--------|
| `tucker_full`  | Rebuild the dense tensor from `(factors, core)` (n-mode product)  | `lmlragen` (built-in) |
| `tkr`          | Transpose (row-wise) Khatri-Rao product of two matrices           | `TKR.m` |
| `tkron`        | Order-3 tensor Kronecker product (just `np.kron`)                 | `tkron.m` |
| `flux_product` | Pointwise product of two Tucker tensors (stays Tucker)            | the `E_i` lines |
| `red_aug_K`    | Reduced augmentation before K-steps (per-axis ranks)              | `red_aug_K.m` |
| `red_aug_S`    | Reduced augmentation before the S-step (equal rank `R`)           | `red_aug_S.m` |
| `nonconstrun`  | HOSVD truncation at tolerance `tol`                               | `nonconstrun.m` |

### 5.5 `simoncini.py`

Direct solver for the third-order tensor Sylvester equation that the S-step
produces.  Reference: V. Simoncini, *Numerical solution of a class of third
order tensor linear equations*, BUMI 2020 (Algorithm T3-sylv).  This file is a
careful translation of `simoncini_direct_solver.m`; the comments flag the
spots where MATLAB column-major (`order='F'`) reshapes must be matched exactly.

---

## 6. Running it

```bash
# single run, first order, test 1 (beta=1.0 is hardcoded in test_parameters.py)
python main.py --order 1 --test 1

# third-order run on a smaller grid, shorter time
python main.py --order 3 --test 1 --N 32 --Tf 1.0

# order-of-accuracy study: sweep lambda and plot the L1 error
python main.py --order 2 --test 1 --sweep 1
```

Options: `--order {1,2,3}`, `--test {1}`, `--sweep {0,1}`, `--Tf`, `--N`,
`--tol`.  Outputs `rank_vs_time.png` (always) and `l1_error.png` (sweep only).

The one test case lives in `test_parameters.py`; its docstring describes the
PDE (including the linear reaction term), initial condition, and exact
solution.

---

## 7. Index-ordering gotcha (read this before debugging tensors)

MATLAB stores arrays **column-major** (Fortran order); NumPy defaults to
**row-major** (C order).  This bites whenever you unfold/reshape a tensor:

- `pytensorlab.tens2mat` and `tmprod` use NumPy C-order.  They are *internally
  consistent*, so as long as we build a right-hand side with `tmprod` and then
  matricize it with `tens2mat`, the column ordering always matches itself.  That
  is why this code never hand-writes a Kronecker product for the RHS.
- The one place we *do* write a Kronecker product is the Sylvester `B` matrix in
  a K-step.  There we use the verified rule: for mode `m`, the other two modes
  pair in **ascending** order (smaller index outer, larger index inner).
- `simoncini.py` reshapes with `order='F'` on purpose, to match the MATLAB
  `tens2mat(B,1)` exactly.  Do not "simplify" those to C-order.

---

## 8. Notes / open questions for the user

- **LoMaC / conservation.** The conservative `lomac_0`/`lomac_01`/`lomac_012`
  truncations (preserving mass / mass+momentum / mass+momentum+energy) are
  ported in `lomac.py` and selectable via `--truncation`; default is the plain
  `nonconstrun`. With the linear reaction term, LoMaC's reference moments are
  rescaled by `exp(beta*t)` rather than held fixed at the initial condition, so
  LoMaC and reaction compose correctly (verified to ~1e-15).
- **`tkr` vs `tkron` column order.** `flux_product` builds each product factor
  with `tkr` (row-wise Khatri-Rao) and the core with `tkron` (`np.kron`).  These
  must use a *consistent* column ordering for the product to be correct.  The
  numbers reproduce the expected convergence orders, which is good evidence the
  ordering is right, but this is the most ordering-sensitive spot in the port —
  worth a second look if a future test ever disagrees with MATLAB.
