# Adding a Linear Reaction Term to 3d-RAIL

**Spec for implementation. Derivation, modified equations, change points, and verification tests.**

Reference: Nakao, Ceruti, Einkemmer, *A low-rank, high-order implicit-explicit integrator for three-dimensional convection-diffusion equations*, Computer Physics Communications 325 (2026) 110163. Equation numbers below refer to that paper.

> **Sign convention.** The reaction term sits on the right-hand side as $+\beta u$. Therefore **positive $\beta$ means growth, negative $\beta$ means decay.** Every formula in this document follows that convention.

---

## 0. TL;DR for the implementer

The target PDE gains a term $+\beta u$ on the right-hand side. That term is a scalar multiple of the identity in every mode. It introduces no spatial derivative, couples no dimensions, and adds no rank.

**The entire change is a shift of the three diffusion matrices:**

```
Fx_tilde = Fx + (beta/3) * I_Nx
Fy_tilde = Fy + (beta/3) * I_Ny
Fz_tilde = Fz + (beta/3) * I_Nz
```

Substitute `F -> F_tilde` at every site where the implicit operator is assembled. Every K equation, every G equation, every `W` accumulation, and the Simoncini solver all remain structurally identical. Sections 1 through 4 prove this and write out the resulting equations explicitly. Section 5 lists the exact code change points. Section 7 gives the verification tests.

Do **not** modify `Dx, Dy, Dz` (the first-derivative matrices used by the explicit flux terms). The reaction term is unrelated to the flux.

---

## 1. Problem statement

Original PDE (Eq. 1.1 in the paper, written out):

```
u_t + (a1*u)_x + (a2*u)_y + (a3*u)_z = d1*u_xx + d2*u_yy + d3*u_zz + c(x,y,z,t)
```

New PDE with a linear reaction term:

```
u_t + (a1*u)_x + (a2*u)_y + (a3*u)_z = d1*u_xx + d2*u_yy + d3*u_zz + beta*u + c(x,y,z,t)
```

`beta` is a scalar constant. **Positive `beta` means growth; negative `beta` means decay.**

Rearranged into IMEX form:

$$u_t \;=\; \underbrace{d_1u_{xx}+d_2u_{yy}+d_3u_{zz} \;+\; \beta u}_{\text{implicit}} \;-\; \underbrace{\big[(a_1u)_x+(a_2u)_y+(a_3u)_z\big]}_{\text{explicit}} \;+\; \underbrace{c(x,y,z,t)}_{\text{implicit (as in the paper)}}$$

### Why the reaction term belongs on the implicit side

Its discrete eigenvalue is $+\beta$, independent of mesh size, so it introduces no stiffness and could legitimately be treated explicitly. Treating it implicitly costs nothing because it is linear and diagonal, and it produces the cleaner algebra shown below. This spec assumes **implicit** treatment. Section 4.5 describes the explicit alternative.

---

## 2. Derivation of the semi-discrete form

### 2.1 Existing structure

Recall the Tucker ansatz,

$$\mathcal{U}(t) = \mathcal{G}(t)\times_1\mathbf{V}_x(t)\times_2\mathbf{V}_y(t)\times_3\mathbf{V}_z(t)$$

and the paper's spatial discretization of the diffusion operator (Eq. 3.2):

$$\big(d_1\partial_x^2 + d_2\partial_y^2 + d_3\partial_z^2\big)u \;\longleftrightarrow\; \mathcal{G}\times_1\mathbf{F}_x\mathbf{V}_x\times_2\mathbf{V}_y\times_3\mathbf{V}_z \;+\; \mathcal{G}\times_1\mathbf{V}_x\times_2\mathbf{F}_y\mathbf{V}_y\times_3\mathbf{V}_z \;+\; \mathcal{G}\times_1\mathbf{V}_x\times_2\mathbf{V}_y\times_3\mathbf{F}_z\mathbf{V}_z$$

Here $\mathbf{F}_x, \mathbf{F}_y, \mathbf{F}_z$ discretize $d_1\partial_x^2$, $d_2\partial_y^2$, $d_3\partial_z^2$ respectively. In the MATLAB reference these are the variables `Dxx, Dyy, Dzz`, with the diffusion coefficients already folded in.

### 2.2 Discretizing the reaction term

Evaluating $+\beta u$ on the tensor-product grid is pointwise scaling:

$$\beta u \;\longleftrightarrow\; \beta\,\mathcal{U} \;=\; \beta\,\mathcal{G}\times_1\mathbf{V}_x\times_2\mathbf{V}_y\times_3\mathbf{V}_z$$

This does not yet look like a diffusion term, so it cannot be absorbed directly. Use $\mathbf{I}_{N_x}\mathbf{V}_x = \mathbf{V}_x$ and split the scalar three ways, pushing one third into each mode:

$$\beta\,\mathcal{U} \;=\; \mathcal{G}\times_1\!\Big(\tfrac{\beta}{3}\mathbf{I}_{N_x}\mathbf{V}_x\Big)\times_2\mathbf{V}_y\times_3\mathbf{V}_z \;+\; \mathcal{G}\times_1\mathbf{V}_x\times_2\!\Big(\tfrac{\beta}{3}\mathbf{I}_{N_y}\mathbf{V}_y\Big)\times_3\mathbf{V}_z \;+\; \mathcal{G}\times_1\mathbf{V}_x\times_2\mathbf{V}_y\times_3\!\Big(\tfrac{\beta}{3}\mathbf{I}_{N_z}\mathbf{V}_z\Big)$$

Each summand now has exactly the shape of a diffusion term. Adding these to Eq. (3.2) and collecting mode by mode:

$$\boxed{\;\tilde{\mathbf{F}}_x := \mathbf{F}_x + \tfrac{\beta}{3}\mathbf{I}_{N_x}, \qquad \tilde{\mathbf{F}}_y := \mathbf{F}_y + \tfrac{\beta}{3}\mathbf{I}_{N_y}, \qquad \tilde{\mathbf{F}}_z := \mathbf{F}_z + \tfrac{\beta}{3}\mathbf{I}_{N_z}\;}$$

### 2.3 The modified tensor differential equation

Equation (3.7) becomes, with only $\mathbf{F}\to\tilde{\mathbf{F}}$ changed:

$$\frac{d}{dt}\Big(\mathcal{G}\times_1\mathbf{V}_x\times_2\mathbf{V}_y\times_3\mathbf{V}_z\Big) = \Big\{\mathcal{G}\times_1\tilde{\mathbf{F}}_x\mathbf{V}_x\times_2\mathbf{V}_y\times_3\mathbf{V}_z + \mathcal{G}\times_1\mathbf{V}_x\times_2\tilde{\mathbf{F}}_y\mathbf{V}_y\times_3\mathbf{V}_z + \mathcal{G}\times_1\mathbf{V}_x\times_2\mathbf{V}_y\times_3\tilde{\mathbf{F}}_z\mathbf{V}_z\Big\}$$
$$-\;\Big\{\mathcal{G}^e_1\times_1\mathbf{D}_x\mathbf{E}_{1,x}\times_2\mathbf{E}_{1,y}\times_3\mathbf{E}_{1,z} + \mathcal{G}^e_2\times_1\mathbf{E}_{2,x}\times_2\mathbf{D}_y\mathbf{E}_{2,y}\times_3\mathbf{E}_{2,z} + \mathcal{G}^e_3\times_1\mathbf{E}_{3,x}\times_2\mathbf{E}_{3,y}\times_3\mathbf{D}_z\mathbf{E}_{3,z}\Big\}$$
$$+\;\mathcal{G}^c\times_1\mathbf{C}_x\times_2\mathbf{C}_y\times_3\mathbf{C}_z$$

Structurally identical to Eq. (3.7). **Every downstream derivation in the paper therefore applies verbatim.**

### 2.4 Note on the choice of split

The thirds are bookkeeping, not modeling. Dumping the whole $+\beta\mathbf{I}$ into $\tilde{\mathbf{F}}_x$ and leaving $\mathbf{F}_y,\mathbf{F}_z$ untouched yields algebraically identical K and G equations, which is a useful correctness check (see Section 3.1). The symmetric split is preferred because it keeps each $\tilde{\mathbf{F}}_i$ symmetric negative **definite** in the decay regime $\beta<0$ (rather than merely semi-definite), which strengthens the stability result cleanly and keeps the three modes balanced.

---

## 3. The modified K equations

### 3.1 Collapsing the shift

Start from Eq. (3.23) with $\mathbf{F}\to\tilde{\mathbf{F}}$, where $a_{kk}$ is the diagonal DIRK coefficient at stage $k$:

$$\big(\mathbf{I}_{N_x} - a_{kk}\Delta t\,\tilde{\mathbf{F}}_x\big)\mathbf{K}_1^{(k)} \;-\; a_{kk}\Delta t\,\mathbf{K}_1^{(k)}\Big[(\tilde{\mathbf{F}}_z\mathbf{V}_z^{\star,(k)})^T\mathbf{V}_z^{\star,(k)} \;\oplus\; (\tilde{\mathbf{F}}_y\mathbf{V}_y^{\star,(k)})^T\mathbf{V}_y^{\star,(k)}\Big] = \mathbf{W}_1^{(k-1)}$$

where $\oplus$ is the Kronecker sum, $\mathbf{A}\oplus\mathbf{B} = \mathbf{A}\otimes\mathbf{I} + \mathbf{I}\otimes\mathbf{B}$.

Now expand. Write $\mathbf{M}_y := (\mathbf{F}_y\mathbf{V}_y^{\star,(k)})^T\mathbf{V}_y^{\star,(k)}$ and $\mathbf{M}_z := (\mathbf{F}_z\mathbf{V}_z^{\star,(k)})^T\mathbf{V}_z^{\star,(k)}$. Because the projection bases are orthonormal,

$$(\tilde{\mathbf{F}}_y\mathbf{V}_y^{\star})^T\mathbf{V}_y^{\star} = \mathbf{M}_y + \tfrac{\beta}{3}(\mathbf{V}_y^\star)^T\mathbf{V}_y^\star = \mathbf{M}_y + \tfrac{\beta}{3}\mathbf{I}_{r_y^\star}$$

and likewise in $z$. The Kronecker sum of two shifted matrices collapses:

$$\Big(\mathbf{M}_z + \tfrac{\beta}{3}\mathbf{I}\Big)\oplus\Big(\mathbf{M}_y + \tfrac{\beta}{3}\mathbf{I}\Big) = \mathbf{M}_z\oplus\mathbf{M}_y \;+\; \tfrac{2\beta}{3}\mathbf{I}_{r_z^\star r_y^\star}$$

Collecting the $\tfrac{\beta}{3}$ from mode 1 and the $\tfrac{2\beta}{3}$ from modes 2 and 3 gives a single scalar shift totalling $\beta$. It enters the identity block with a minus sign, because every occurrence is multiplied by $-a_{kk}\Delta t$.

### 3.2 K1 equation (x basis update)

$$\boxed{\;\Big[\big(1 - a_{kk}\Delta t\,\beta\big)\mathbf{I}_{N_x} \;-\; a_{kk}\Delta t\,\mathbf{F}_x\Big]\mathbf{K}_1^{(k)} \;-\; a_{kk}\Delta t\,\mathbf{K}_1^{(k)}\big[\mathbf{M}_z \oplus \mathbf{M}_y\big] \;=\; \mathbf{W}_1^{(k-1)}\;}$$

The reaction term appears solely as a scalar multiplier on the identity block. Nothing else in the operator changes. Note that the result is independent of how the split was chosen, as it must be.

### 3.3 K2 and K3 equations

Following the ordering conventions of Eqs. (3.9b) and (3.9c):

$$\Big[\big(1 - a_{kk}\Delta t\beta\big)\mathbf{I}_{N_y} - a_{kk}\Delta t\,\mathbf{F}_y\Big]\mathbf{K}_2^{(k)} \;-\; a_{kk}\Delta t\,\mathbf{K}_2^{(k)}\big[\mathbf{M}_z \oplus \mathbf{M}_x\big] = \mathbf{W}_2^{(k-1)}$$

$$\Big[\big(1 - a_{kk}\Delta t\beta\big)\mathbf{I}_{N_z} - a_{kk}\Delta t\,\mathbf{F}_z\Big]\mathbf{K}_3^{(k)} \;-\; a_{kk}\Delta t\,\mathbf{K}_3^{(k)}\big[\mathbf{M}_y \oplus \mathbf{M}_x\big] = \mathbf{W}_3^{(k-1)}$$

with $\mathbf{M}_x := (\mathbf{F}_x\mathbf{V}_x^{\star,(k)})^T\mathbf{V}_x^{\star,(k)}$.

### 3.4 Mapping to a Sylvester solver

For `scipy.linalg.solve_sylvester(A, B, Q)` solving $\mathbf{A}\mathbf{X} + \mathbf{X}\mathbf{B} = \mathbf{Q}$, the K1 step uses

```
A = (1 - a_kk*dt*beta) * I_Nx - a_kk*dt*Fx
B = -a_kk*dt * (kron(Mz, I_ry) + kron(I_rz, My))
Q = W1
```

Equivalently and preferably, build `A = I_Nx - a_kk*dt*Fx_tilde` and `B = -a_kk*dt*(kron(Mz_tilde, I_ry) + kron(I_rz, My_tilde))` from the shifted matrices. Both produce the same numbers.

### 3.5 The right-hand side W

**This is the piece that is easy to miss.** The DIRK sum $\sum_{\ell=1}^{k}a_{k\ell}$ over implicit terms splits into the $\ell=k$ part (which went into the Sylvester operator above) and the $\ell<k$ part (which lives in $\mathbf{W}$). The reaction term is implicit, so it contributes to **both**.

Define the mode-1 projected previous-stage solution:

$$\mathbf{P}_1^{(\ell)} := \mathbf{V}_x^{(\ell)}\mathbf{G}_{(1)}^{(\ell)}\Big[(\mathbf{V}_z^{(\ell)})^T\mathbf{V}_z^{\star,(k)}\otimes(\mathbf{V}_y^{(\ell)})^T\mathbf{V}_y^{\star,(k)}\Big]$$

Equation (3.24) then reads, with the new group marked:

$$\mathbf{W}_1^{(k-1)} = \mathbf{V}_x^n\mathbf{G}_{(1)}^n\Big[(\mathbf{V}_z^n)^T\mathbf{V}_z^{\star,(k)}\otimes(\mathbf{V}_y^n)^T\mathbf{V}_y^{\star,(k)}\Big]$$

$$+\;\Delta t\sum_{\ell=1}^{k-1}a_{k\ell}\Big\{(\mathbf{F}_x\mathbf{V}_x^{(\ell)})\mathbf{G}_{(1)}^{(\ell)}\big[(\mathbf{V}_z^{(\ell)})^T\mathbf{V}_z^{\star}\otimes(\mathbf{V}_y^{(\ell)})^T\mathbf{V}_y^{\star}\big]$$
$$\qquad\qquad + \;\mathbf{V}_x^{(\ell)}\mathbf{G}_{(1)}^{(\ell)}\big[(\mathbf{V}_z^{(\ell)})^T\mathbf{V}_z^{\star}\otimes(\mathbf{F}_y\mathbf{V}_y^{(\ell)})^T\mathbf{V}_y^{\star}\big]$$
$$\qquad\qquad + \;\mathbf{V}_x^{(\ell)}\mathbf{G}_{(1)}^{(\ell)}\big[(\mathbf{F}_z\mathbf{V}_z^{(\ell)})^T\mathbf{V}_z^{\star}\otimes(\mathbf{V}_y^{(\ell)})^T\mathbf{V}_y^{\star}\big] \;\;\underbrace{+\;\beta\,\mathbf{P}_1^{(\ell)}}_{\textbf{NEW}}\Big\}$$

$$-\;\Delta t\sum_{\ell=1}^{k}\tilde a_{k+1,\ell}\big\{\text{flux terms, unchanged}\big\} \;+\; \Delta t\sum_{\ell=1}^{k}a_{k\ell}\big\{\text{source terms, unchanged}\big\}$$

$\mathbf{P}_1^{(\ell)}$ is already assembled as part of the diffusion group, so the added cost is one scaled matrix add per stage.

**If you implement via the $\tilde{\mathbf{F}}$ substitution, this term appears automatically.** Verify: replacing $\mathbf{F}_x,\mathbf{F}_y,\mathbf{F}_z$ by $\tilde{\mathbf{F}}$ in the three diffusion lines above contributes $+\tfrac{\beta}{3}\mathbf{P}_1^{(\ell)}$ from each, summing to $+\beta\mathbf{P}_1^{(\ell)}$.

---

## 4. The modified G equation

### 4.1 Notation

Let $\hat{\mathbf{V}}_x^{(k)}, \hat{\mathbf{V}}_y^{(k)}, \hat{\mathbf{V}}_z^{(k)}$ be the reduced-augmented bases (all of common rank $\hat r$ after the reduction of Eq. 3.16), and define the projected operators

$$\hat{\mathbf{F}}_x := (\hat{\mathbf{V}}_x^{(k)})^T\mathbf{F}_x\hat{\mathbf{V}}_x^{(k)}, \qquad \hat{\mathbf{F}}_y := (\hat{\mathbf{V}}_y^{(k)})^T\mathbf{F}_y\hat{\mathbf{V}}_y^{(k)}, \qquad \hat{\mathbf{F}}_z := (\hat{\mathbf{V}}_z^{(k)})^T\mathbf{F}_z\hat{\mathbf{V}}_z^{(k)}$$

### 4.2 Vectorized form

Substituting $\tilde{\mathbf{F}}$ into Eq. (3.26) and collapsing the shifts exactly as in Section 3.1:

$$\boxed{\;\Big\{\Big[\big(1-a_{kk}\Delta t\beta\big)\mathbf{I}_{\hat r} - a_{kk}\Delta t\,\hat{\mathbf{F}}_z\Big]\otimes\mathbf{I}_{\hat r}\otimes\mathbf{I}_{\hat r} \;+\; \mathbf{I}_{\hat r}\otimes\big[-a_{kk}\Delta t\,\hat{\mathbf{F}}_y\big]\otimes\mathbf{I}_{\hat r} \;+\; \mathbf{I}_{\hat r}\otimes\mathbf{I}_{\hat r}\otimes\big[-a_{kk}\Delta t\,\hat{\mathbf{F}}_x\big]\Big\}\,\mathrm{vec}\big(\hat{\mathcal{G}}^{(k)}\big) = \mathrm{vec}\big(\mathcal{B}^{(k-1)}\big)\;}$$

The identity is attached to the $z$ factor, matching the paper's convention and the MATLAB reference (where the scalar identity lives in `A2`).

### 4.3 Tensor form

Matching Eq. (3.42):

$$\hat{\mathcal{G}}^{(k)}\times_1\Big[\big(1-a_{kk}\Delta t\beta\big)\mathbf{I} - a_{kk}\Delta t\,\hat{\mathbf{F}}_x\Big] \;-\; a_{kk}\Delta t\,\hat{\mathcal{G}}^{(k)}\times_2\hat{\mathbf{F}}_y \;-\; a_{kk}\Delta t\,\hat{\mathcal{G}}^{(k)}\times_3\hat{\mathbf{F}}_z \;=\; \mathcal{B}^{(k-1)}$$

(Attaching the identity to mode 1 or mode 3 is equivalent. Follow whichever convention the existing solver call uses.)

### 4.4 The right-hand side B

Equation (3.27) gains one group inside the $\ell<k$ implicit sum:

$$\mathcal{B}^{(k-1)} \;\mathrel{+}=\; +\,\beta\,\Delta t\sum_{\ell=1}^{k-1}a_{k\ell}\;\mathcal{G}^{(\ell)}\times_1(\hat{\mathbf{V}}_x^{(k)})^T\mathbf{V}_x^{(\ell)}\times_2(\hat{\mathbf{V}}_y^{(k)})^T\mathbf{V}_y^{(\ell)}\times_3(\hat{\mathbf{V}}_z^{(k)})^T\mathbf{V}_z^{(\ell)}$$

Again, the $\tilde{\mathbf{F}}$ substitution produces this automatically.

### 4.5 Explicit-treatment alternative

If the reaction term is instead treated explicitly, remove it from the Sylvester and G operators entirely (the identity coefficient returns to $1$) and move it into the $\tilde a_{k+1,\ell}$ sums alongside the flux terms, with sign $+\beta$ times the stage solution. This is a valid scheme with no stiffness penalty, but it is more code churn for no gain. Implicit is recommended.

---

## 5. Implementation checklist

### 5.1 The single change point

Build the shifted operators once at setup, immediately after the spectral differentiation matrices are constructed:

```python
# after Fx, Fy, Fz (MATLAB: Dxx, Dyy, Dzz) are built
beta = params.beta        # linear reaction coefficient, scalar
                          # beta > 0 -> growth, beta < 0 -> decay

Fx_t = Fx + (beta / 3.0) * np.eye(Nx)
Fy_t = Fy + (beta / 3.0) * np.eye(Ny)
Fz_t = Fz + (beta / 3.0) * np.eye(Nz)
```

Then pass `Fx_t, Fy_t, Fz_t` wherever `Fx, Fy, Fz` were passed before.

### 5.2 Sites that must receive the shifted matrices

Mirroring the MATLAB reference layout (`IMEX111.m`, `IMEX222.m`, `IMEX443.m`):

| Site | What changes |
|---|---|
| K1 Sylvester, argument 1 | `speye(Nx) - a_kk*dt*Dxx` → uses `Dxx_t` |
| K1 Sylvester, argument 2 | both `(Dzz*Vz_star)'*Vz_star` and `(Dyy*Vy_star)'*Vy_star` → use `Dzz_t`, `Dyy_t` |
| K2 Sylvester, arguments 1 and 2 | same pattern with `Dyy_t`, `Dzz_t`, `Dxx_t` |
| K3 Sylvester, arguments 1 and 2 | same pattern with `Dzz_t`, `Dyy_t`, `Dxx_t` |
| K step RHS, previous-stage diffusion terms | in IMEX222/443, terms carrying `(Dxx*Vx_1)`, `(Dyy*Vy_1)`, `(Dzz*Vz_1)` etc. with coefficients like `(1-gamma)*dt` → use shifted matrices |
| G step operators `A1, A2, A3` | `-dt*Vy'*(Dyy*Vy)`, `speye(R)-dt*Vz'*(Dzz*Vz)`, `-dt*Vx'*(Dxx*Vx)` → use shifted matrices |
| `IMEX111` prediction calls | the prediction that builds `V_dagger` also receives `Dxx, Dyy, Dzz`; passing the shifted versions propagates automatically |

### 5.3 What does NOT change

- `Dx, Dy, Dz` (first-derivative matrices for the flux). Leave untouched.
- `simoncini_direct_solver` / `simoncini.py`. The shift $\mathbf{M}\to\mathbf{M}+\tfrac{\beta}{3}\mathbf{I}$ leaves eigenvectors unchanged and translates eigenvalues, so the direct solver works as-is. Symmetry is preserved, so any symmetry assumption in the solver still holds.
- `red_aug_K`, `red_aug_S`. Reduced augmentation is purely basis bookkeeping.
- The flux assembly (`TKR`, `tkron`, the `E1, E2, E3` construction).
- The Butcher tableaus.
- Rank adaptivity logic. See 5.5.

### 5.4 Test parameters and manufactured solutions

If a manufactured solution is used to verify order of accuracy, the source term must be updated. For a fixed target solution $u_{\text{exact}}$, adding $+\beta u$ to the right-hand side requires

$$c_{\text{new}}(x,y,z,t) = c_{\text{old}}(x,y,z,t) - \beta\,u_{\text{exact}}(x,y,z,t)$$

Since $u_{\text{exact}}$ is low-rank by construction in every test case, $\beta u_{\text{exact}}$ is low-rank, so append it as extra Tucker components to the source `P`. Alternatively, leave `c` alone and change the exact solution: for a source-free problem with constant $\beta$, if $u_0$ solves the original PDE then $u_0 e^{\beta t}$ solves the new one. The second option is cleaner for the existing test suite.

### 5.5 Rank behavior

$\beta\mathcal{U}$ has the same multilinear rank as $\mathcal{U}$. The reaction term never inflates the augmented bases and never enters the Hadamard-product machinery that drives rank to $(r^2, r^2, r^2)$ for the flux terms. No change to rank-adaptivity thresholds is needed.

### 5.6 Conservation and LoMaC

Mass is no longer conserved in time. For constant $\beta$ and zero source, total mass scales as $e^{\beta t}$: it grows for $\beta>0$ and decays for $\beta<0$.

LoMaC remains **valid and unchanged**. It preserves the moments of the *pre-truncation* tensor $\hat{\mathcal{U}}^{(k)}$, and that tensor already carries the correct scaled mass from the evolution. The change is only in how results are checked:

- Old check: `mass(n) / mass(0) == 1` to machine precision.
- New check: `mass(n) / mass(0) == exp(beta * t_n)` to machine precision.

Update any diagnostic in the time-stepping loop that asserts constant mass, momentum, or energy. With a linear reaction, all three macroscopic quantities scale by the same factor $e^{\beta t}$.

One practical caution for $\beta>0$ over long runs: the solution grows exponentially while the truncation tolerance $\varepsilon$ is absolute. Either scale $\varepsilon$ with the current solution norm or use a relative tolerance, otherwise the effective rank will creep upward as the magnitude grows.

---

## 6. Stability

Repeat the Theorem 3.3 argument with $\mathbf{A} := \Delta t(\hat{\mathbf{V}}_x^{n+1})^T\mathbf{F}_x\hat{\mathbf{V}}_x^{n+1}$ and analogues in $y, z$, diagonalized to $\mathbf{D}, \mathbf{E}, \mathbf{F}$. Equation (3.50) becomes

$$\tilde G^{n+1}_{ij} = \alpha_{ij}\tilde G^n_{ij}, \qquad \alpha_{ij} = \frac{1}{1 - \Delta t\,\beta - D_{ii} - E_{jj} - F_{jj}}$$

**Case $\beta < 0$ (decay).** With $\mathbf{F}_x, \mathbf{F}_y, \mathbf{F}_z$ symmetric negative semi-definite, every $D_{ii}, E_{jj}, F_{jj} \le 0$, and $-\Delta t\beta > 0$, so the denominator exceeds 1 and $\alpha_{ij} < 1$ strictly. The scheme is unconditionally stable and now strictly contractive. This matches the physics.

**Case $\beta > 0$ (growth).** Require $\Delta t\,\beta < 1$ to keep the denominator positive and avoid a sign flip in the amplification factor. The constraint is mesh-independent and in practice far weaker than the convection CFL condition of Eq. (4.1). Add an assertion at setup:

```python
assert beta <= 0 or dt * beta < 1.0, \
    "backward-Euler growth constraint violated: need dt < 1/beta"
```

Note that $\alpha_{ij} > 1$ in this regime is correct and expected. The solution genuinely grows. The constraint only prevents the discrete amplification factor from turning negative or blowing up through the singularity at $\Delta t\beta = 1$.

---

## 7. Verification plan

Run these in order. Each isolates one piece.

### Test A: pure reaction

Set `a1 = a2 = a3 = 0`, `d1 = d2 = d3 = 0`, `c = 0`, `beta != 0`. Start from a rank-one initial condition, for example $u_0 = \sin x \sin y \sin z$.

- Exact solution: $u(x,y,z,t) = u_0(x,y,z)\,e^{\beta t}$.
- Multilinear rank must stay at $(1,1,1)$ for all time.
- Error should sit at the truncation tolerance, and should show the expected temporal order of the chosen IMEX tableau against the exponential.
- Run once with `beta > 0` and once with `beta < 0`. The growth case is the one that catches a sign error, since a flipped sign produces visible decay instead of visible growth.
- This test isolates the identity shift with no other operator active.

### Test B: reaction plus diffusion

Set `a1 = a2 = a3 = 0`, `d1 = d2 = d3 = d`, `c = 0`, `beta != 0`, initial condition $\sin(x+y+z)$ expressed in Tucker form.

- Exact solution: $u = e^{(\beta - 3d)t}\sin(x+y+z)$.
- Confirm the expected order of accuracy for IMEX111 (first), IMEX222 (second), IMEX443 (third).
- This test confirms the shift composes correctly with the real diffusion operator and that the G step is right.
- Useful edge case: choose $\beta = 3d$ so the exact solution is steady in time. Any drift is a pure sign or scaling bug in the shift.

### Test C: full convection-diffusion-reaction

Take existing `testnumber == 1` (constant coefficients, rank-one accuracy test) and set `beta != 0` with the exact solution multiplied by $e^{\beta t}$.

- Confirm order of accuracy is preserved for all three tableaus.
- This is the test that exercises the previous-stage `W` and `B` terms in IMEX222 and IMEX443. If the order drops to first for the high-order tableaus, the bug is almost certainly a missing reaction contribution in the $\ell < k$ implicit sums (Section 3.5 or 4.4).

### Test D: regression

Set `beta = 0` and confirm bitwise-comparable output against the pre-change code on any existing test. The shift becomes the zero matrix and nothing should move beyond floating-point noise from the added `eye` scaling.

### Test E: conservation diagnostic

Run a LoMaC variant (`lomac_0`, `lomac_01`, or `lomac_012`) with `beta != 0` and confirm

```
mass(t_n) / mass(0)  ==  exp(beta * t_n)     to ~1e-15
```

rather than the old constant-mass check.

---

## 8. Summary of new terms

Convention reminder: $+\beta u$ on the right-hand side, so $\beta>0$ is growth and $\beta<0$ is decay.

| Object | Original | With reaction |
|---|---|---|
| Mode operators | $\mathbf{F}_x, \mathbf{F}_y, \mathbf{F}_z$ | $\mathbf{F}_i + \tfrac{\beta}{3}\mathbf{I}$ |
| K$_i$ Sylvester, left factor | $\mathbf{I} - a_{kk}\Delta t\,\mathbf{F}_i$ | $(1-a_{kk}\Delta t\beta)\mathbf{I} - a_{kk}\Delta t\,\mathbf{F}_i$ |
| K$_i$ Sylvester, right factor | $-a_{kk}\Delta t(\mathbf{M}\oplus\mathbf{M})$ | unchanged after collapse |
| K$_i$ RHS, $\ell<k$ implicit sum | 3 diffusion terms | $+\;\beta\,\mathbf{P}_i^{(\ell)}$ |
| G operator, identity factor | $\mathbf{I}_{\hat r} - a_{kk}\Delta t\,\hat{\mathbf{F}}_z$ | $(1-a_{kk}\Delta t\beta)\mathbf{I}_{\hat r} - a_{kk}\Delta t\,\hat{\mathbf{F}}_z$ |
| G RHS $\mathcal{B}$, $\ell<k$ implicit sum | 3 diffusion terms | $+\;\beta\,\Delta t\,a_{k\ell}\,\Pi\mathcal{G}^{(\ell)}$ |
| Amplification factor | $1/(1-D-E-F)$ | $1/(1-\Delta t\beta-D-E-F)$ |
| Mass | conserved | scales as $e^{\beta t}$ |
| Time-step constraint | none from reaction | $\Delta t\,\beta < 1$ when $\beta>0$ |
| Rank impact | n/a | none |
| Simoncini solver | n/a | no change |
