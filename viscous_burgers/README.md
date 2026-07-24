# 3D RAIL — Viscous Burgers Solver (Python)

Modifies `linear_advection_diffusion/` to replace the linear advection flux
with the nonlinear Burgers flux `(u^2/2)`. This is an extension beyond the
MATLAB reference, not a straight port.

> See the [top-level README](../README.md) for the shared concepts: the
> Tucker storage format, the K-step/S-step time-stepping algorithm, and the
> MATLAB-vs-NumPy index-ordering gotcha. This README covers only what's
> specific to this solver.

---

## The PDE

On the cube `[-L/2, L/2]^3` with periodic boundary conditions:

```
u_t + (u^2/2)_x + (u^2/2)_y + (u^2/2)_z  =  d1 u_xx + d2 u_yy + d3 u_zz  +  phi
```

Diffusion is still treated implicitly; the nonlinear advection flux and
source are explicit. The only algorithmic change from
`linear_advection_diffusion/` is in `helpers.py::flux_product`: the flux is
`flux_product(solution, solution)` (`u` times itself) instead of
`flux_product(flow, solution)` (a separate flow-field tensor times `u`).

Since `flux_product` multiplies Tucker ranks mode-by-mode
(`rank(a*b) = rank(a) * rank(b)`), squaring `u` against itself grows the rank
faster than a typically-low-rank flow field would in the linear solver — worth
knowing if you see the rank climb faster here than in
`linear_advection_diffusion/` on a similar problem.

## File map

| Python file            | Role                                              |
|------------------------|-----------------------------------------------------|
| `main.py`              | Driver: build grids, time loop, plots               |
| `test_parameters.py`   | Test cases (IC, source, exact)                      |
| `imex.py`              | All IMEX steps (1st/2nd/3rd order) + dispatcher     |
| `helpers.py`           | Small tensor utilities (`flux_product` uses `u*u`)  |
| `simoncini.py`         | Direct solver for the core (S-step) equation        |

## Running it

```bash
# single run, first order, test 1
python main.py --order 1 --test 1

# rank-growth test (no exact solution)
python main.py --order 2 --test 2

# order-of-accuracy study: sweep lambda and plot the L1 error
python main.py --order 2 --test 1 --sweep 1
```

Options: `--order {1,2,3}`, `--test {1,2}`, `--sweep {0,1}`.

- `--test 1`: accuracy test, rank-(2,2,2) initial condition, manufactured
  exact solution.
- `--test 2`: rank-growth test, no source, no exact solution.

Outputs `rank_vs_time.png` (always) and `l1_error.png` (sweep only).
