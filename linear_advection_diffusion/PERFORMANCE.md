# Performance notes — `linear_advection_diffusion`

This document records the optimizations made to the Python port of the 3D RAIL
IMEX advection–diffusion solver, why they work, and how much they helped.  The
audience is an engineering student: the goal is to understand *where the time
goes* and *why the fixes are correct*, not just that they are faster.

All numbers below are for the **Test 2 λ-sweep** (`python main.py --test 2
--sweep 1 --order 1`): a grid of `N = 100` per dimension, final time `Tf = 10`,
sweeping the CFL factor λ over `0.5 … 2.0` (16 runs).  MATLAB runs the same
sweep in ≈ 35 s.

## Bottom line

| Version | Test-2 sweep | Single order-1 run |
|---|---|---|
| Original port | **75 s** | 6.0 s |
| + cached K-step Schur | 34.3 s | 2.7 s |
| + `np.kron` and cached Simoncini Schur | **24.7 s** | 1.75 s |

End result: **≈ 3× faster than the original port, and ≈ 1.4× faster than
MATLAB**.  Every per-λ L¹ error is **bit-identical** to the original — these are
pure speed changes, not approximations.

## How we found the bottleneck

Profiling a single run with `cProfile` showed that **~68 % of the runtime was
`scipy.linalg.solve_sylvester`**, and inside it almost all the cost was the
**Schur decomposition** (`scipy.linalg.schur`).  This makes sense: the solver
uses the Bartels–Stewart algorithm, which Schur-decomposes its two coefficient
matrices and then does a triangular solve.  The Schur step is the expensive
part, and we were doing it tens of thousands of times.

The key realization is that **most of those Schur decompositions are of matrices
that never change.**  Recomputing the same factorization every time step is the
waste; caching it is the fix.

## Optimization 1 — cache the K-step Schur factorization

**Where:** `imex111._k_step` and `imex.k_step`.

Each K-step solves a Sylvester equation `A K + K B = Q` whose left matrix is

```
A = I − θ · D″_m        (θ = stage-coefficient × dt, D″_m the mode-m diffusion matrix)
```

Within a single run, `dt` and `D″_m` are fixed, so **`A` is identical for almost
every time step**.  Yet `solve_sylvester` re-Schur-decomposed this dense
`N×N` (100×100) matrix on every call — about 530 steps × 3 modes ≈ 1600 times
per λ.

**Fix:** `DiffMatrices.schur_of_A(m, θ)` memoizes the real Schur factorization,
keyed by `(mode, θ)`.  The companion helper `solve_sylvester_cached` is a
line-for-line copy of `scipy.linalg.solve_sylvester` (same Bartels–Stewart
conventions) that accepts the precomputed Schur of `A` instead of recomputing
it.  We verified it is **bit-identical** to `solve_sylvester` (max difference
`0.0`).

A subtlety: `dt` is *meant* to be constant, but `main.py` forms each `dtn` as a
difference of `numpy.arange` values, so consecutive steps differ in the last
bit or two (~1e-16). Keying directly on the raw float gave **33** cache entries
instead of the ideal handful.  Rounding the key to 12 significant figures
collapses that floating-point noise (the change in `A` is far below machine
precision) to **6** entries (3 modes × {interior step, final shortened step}),
while keeping genuinely different θ values — distinct DIRK stage coefficients,
or the final step — separate.

**Effect:** single run 6.0 s → 2.7 s; sweep 75 s → 34.3 s.

## Optimization 2 — `tl.kron` → `np.kron` in the K-step `B` matrix

**Where:** `imex111._k_step` and `imex.k_step`.

The Sylvester `B` matrix is built from two Kronecker products.  pytensorlab's
`tl.kron` is a generic N-way implementation with significant Python overhead
(~0.5 s of the profile).  For two matrices it produces exactly the same result
as `numpy.kron` (verified, max difference `0.0`), so we just call `np.kron`,
which is compiled C.

**Effect:** ~0.45 s off a single run.

## Optimization 3 — cache the Schur factorization inside Simoncini's solver

**Where:** `simoncini.simoncini`.

After the K-step Schur was cached, the new bottleneck was the **S-step**
(Simoncini's direct tensor-Sylvester solver).  Its inner loop runs `R` times
(`R` = the *augmented* rank from `red_aug_S`, which reaches **~25** before
truncation), and each iteration solves a Sylvester equation

```
P · X + X · Q_syl = F
```

with `P = M2 \ A1`.  Crucially, **`P` does not depend on the loop index `k`** —
only `Q_syl` (through one scalar) and `F` change.  But `solve_sylvester`
re-Schur-decomposed `P` on every iteration.

**Fix:** Schur-decompose `P` **once** before the loop and reuse it (the same
cached-Bartels–Stewart pattern as Optimization 1), leaving only the small
per-iteration Schur of `Q_syl`.  This is fully general — it assumes nothing
about the coefficient matrices — and preserves the carefully-debugged
column-major (`order='F'`) reshapes that this solver depends on.  Results are
unchanged.

**Effect:** single run 2.7 s → 1.75 s; sweep 34.3 s → 24.7 s.

## On "unnecessarily large matrices"

A deliberate check for oversized allocations found **none in the time-stepping
loop**:

- Every per-step quantity is projected down to the working rank `R` (≤ ~25): the
  K-step right-hand side is `(N, R²)`, and the S-step operates on `R×R×R` cores.
  No `N×N×N` tensors are formed inside the loop.
- The only `N×N×N` array (100³ = 1 M entries) is the final L¹-error
  reconstruction `tucker_full(U, G)` compared against `u_exact` — built once per
  λ, outside the loop, and unavoidable for measuring the error.
- Simoncini's *full residual operator* `L_op` has size `R³ × R³` — up to
  **15625 × 15625** when `R = 25`.  It is only built under `check_res=True`,
  which the port leaves `False`, so it is never materialized.  (The MATLAB
  reference also has this line commented out.)

So the remaining cost is genuinely small dense linear algebra (Schur / SVD / QR
on `R×R` and `N×R` matrices) plus pytensorlab's per-call overhead on the many
small `tmprod` projections.

## Why MATLAB was *not* faster despite the same algorithm

MATLAB and Python use the same Bartels–Stewart Sylvester solve and the same
Simoncini algorithm; the original 2× gap was constant-factor (MATLAB's JIT loop
and LAPACK calls have lower per-call overhead than scipy's `solve_sylvester`,
which validates inputs in Python on each of ~10⁴ calls).  The caching above is
an *algorithmic* saving that neither codebase originally exploited, which is why
the optimized Python now runs faster than the MATLAB it was ported from.

## Possible further work (not done)

The largest remaining single cost is the per-iteration `schur(Q_syl)` inside
Simoncini (~10⁴ small Schur calls).  Because this application *always* passes
identity matrices for `M1, M2, H1, H3`, `Q_syl` reduces to `A2ᵀ + R[k,k]·I` — a
pure diagonal shift — so `schur(A2)` could also be computed once and merely
shifted each iteration (estimated sweep ≈ 21 s).  This was intentionally **not**
implemented: it would specialize the general Simoncini solver to the
identity-coefficient case, trading generality and a little readability for a
modest gain we did not need (Python already beats MATLAB).
