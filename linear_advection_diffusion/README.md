# 3D RAIL — Linear Advection-Diffusion Solver (Python)

Python port of the MATLAB code in
`3dRAIL/rail3d_sourcecode/rail3d_linear_advection_diffusion/`. Solves a 3D
linear advection-diffusion equation with the RAIL integrator.

> See the [top-level README](../README.md) for the shared concepts: the
> Tucker storage format, the K-step/S-step time-stepping algorithm, and the
> MATLAB-vs-NumPy index-ordering gotcha. This README covers only what's
> specific to this solver.

---

## The PDE

On the cube `[-L/2, L/2]^3` with periodic boundary conditions:

```
u_t + (a1 u)_x + (a2 u)_y + (a3 u)_z  =  d1 u_xx + d2 u_yy + d3 u_zz  +  phi
        \___________________________/      \____________________________/     \_/
                advection (explicit)              diffusion (implicit)        source
```

- `(a1, a2, a3)` is a (possibly space- and time-dependent) **flow field**.
- `(d1, d2, d3)` are constant **diffusion coefficients**.
- `phi` is a known **source term**.

Three IMEX schemes are available, of increasing order of accuracy:
**IMEX111** (1st), **IMEX222** (2nd), **IMEX443** (3rd).

## File map (Python <-> MATLAB)

| Python file            | Role                                              | MATLAB origin                          |
|------------------------|---------------------------------------------------|----------------------------------------|
| `main.py`              | Driver: build grids, time loop, plots             | `main.m`                               |
| `test_parameters.py`   | All 8 test cases (IC, flow fields, source, exact) | `test_parameters.m`                    |
| `imex.py`              | All IMEX steps (1st/2nd/3rd order) + dispatcher   | `IMEX111.m`, `IMEX222.m`, `IMEX443.m`  |
| `helpers.py`           | Small tensor utilities                            | `TKR.m`, `tkron.m`, `red_aug_*.m`, `nonconstrun.m` |
| `simoncini.py`         | Direct solver for the core (S-step) equation      | `simoncini_direct_solver.m`            |

## Running it

```bash
# single run, first order, test 1
python main.py --order 1 --test 1

# third-order run on a smaller grid, shorter time
python main.py --order 3 --test 1 --N 32 --Tf 1.0

# order-of-accuracy study: sweep lambda and plot the L1 error
python main.py --order 2 --test 1 --sweep 1
```

Options: `--order {1,2,3}`, `--test {1..8}`, `--sweep {0,1}`, `--Tf`, `--N`,
`--tol`. Outputs `rank_vs_time.png` (always) and `l1_error.png` (sweep only).

The eight test cases live in `test_parameters.py`; their docstrings describe
the PDE, initial condition, and (where available) exact solution for each.

## Notes / open questions

- **LoMaC / conservation not ported.** The MATLAB integrators truncate with
  the conservative `lomac_012`, which preserves mass/momentum/energy; this
  port uses the plain `nonconstrun` and does not track the macroscopic
  quantities. For accuracy tests (tests 1, 2, 3, 6) the results match expected
  orders, but if you need exact mass/energy conservation (especially the
  Dougherty-Fokker-Planck test 8), the LoMaC truncation would still need to be
  ported. (`linear_reaction_diffusion/` does include a ported LoMaC.)
