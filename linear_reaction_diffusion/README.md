# 3D RAIL — Linear Advection-Diffusion-Reaction Solver (Python)

A copy of `linear_advection_diffusion/`, extended with a linear reaction term
`+beta*u` (see [`docs/3dRAIL_linear_reaction_term_1.md`](docs/3dRAIL_linear_reaction_term_1.md)
for the full derivation). This is an extension beyond the MATLAB reference,
not a straight port.

> See the [top-level README](../README.md) for the shared concepts: the
> Tucker storage format, the K-step/S-step time-stepping algorithm, and the
> MATLAB-vs-NumPy index-ordering gotcha. This README covers only what's
> specific to this solver.

---

## The PDE

On the cube `[-L/2, L/2]^3` with periodic boundary conditions:

```
u_t + (a1 u)_x + (a2 u)_y + (a3 u)_z  =  d1 u_xx + d2 u_yy + d3 u_zz  +  beta*u  +  phi
```

`beta` is a scalar linear reaction coefficient: positive means growth,
negative means decay. It's hardcoded per test case as `TestParameters.beta`
(not a CLI argument, since each test's exact solution/source is derived for
one specific beta), and `beta=0.0` (the default) reproduces the original
solver's behavior exactly.

The reaction term needs **no changes** to the K-step/S-step machinery: it
folds entirely into the diffusion matrices at setup
(`main.py::build_diff_matrices`, `F_tilde = F + (beta/3)*I`), so
`imex.py`/`helpers.py`/`simoncini.py` are otherwise identical to
`linear_advection_diffusion/`. See `main.py`'s docstring for the
`dt*beta < 1` growth-stability constraint, and the derivation doc above for
why the `1/3` split works.

LoMaC's conserved reference moments (`lomac.py`) are rescaled by `exp(beta*t)`
rather than held fixed at the initial condition, so LoMaC truncation composes
correctly with the reaction term.

## File map (Python <-> MATLAB)

| Python file            | Role                                              | MATLAB origin                          |
|------------------------|---------------------------------------------------|----------------------------------------|
| `main.py`              | Driver: build grids, time loop, plots             | `main.m`                               |
| `test_parameters.py`   | Test case (IC, flow fields, source, exact)        | `test_parameters.m`                    |
| `imex.py`              | All IMEX steps (1st/2nd/3rd order) + dispatcher   | `IMEX111.m`, `IMEX222.m`, `IMEX443.m`  |
| `helpers.py`           | Small tensor utilities                            | `TKR.m`, `tkron.m`, `red_aug_*.m`, `nonconstrun.m` |
| `simoncini.py`         | Direct solver for the core (S-step) equation      | `simoncini_direct_solver.m`            |

## Running it

```bash
# single run, first order, test 1 (beta=1.0 is hardcoded in test_parameters.py)
python main.py --order 1 --test 1

# third-order run on a smaller grid, shorter time
python main.py --order 3 --test 1 --N 32 --Tf 1.0

# order-of-accuracy study: sweep lambda and plot the L1 error
python main.py --order 2 --test 1 --sweep 1
```

Options: `--order {1,2,3}`, `--test {1}`, `--sweep {0,1}`, `--Tf`, `--N`,
`--tol`. Outputs `rank_vs_time.png` (always) and `l1_error.png` (sweep only).

The one test case lives in `test_parameters.py`; its docstring describes the
PDE (including the linear reaction term), initial condition, and exact
solution.

## Notes / open questions

- **LoMaC / conservation.** The conservative `lomac_0`/`lomac_01`/`lomac_012`
  truncations (preserving mass / mass+momentum / mass+momentum+energy) are
  ported in `lomac.py` and selectable via `--truncation`; default is the plain
  `nonconstrun`. With the reaction term, LoMaC's reference moments are
  rescaled by `exp(beta*t)` rather than held fixed at the initial condition,
  so LoMaC and reaction compose correctly (verified to ~1e-15).
