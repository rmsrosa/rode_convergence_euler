# Linear RODE with non-homogenous Wiener term

Here, we consider the Euler approximation of the specific RODE
```math
  \begin{cases}
    \displaystyle \frac{\mathrm{d}X_t}{\mathrm{d} t} = - X_t + W_t, \qquad 0 \leq t \leq T, \\
    \left. X_t \right|_{t = 0} = X_0,
  \end{cases}
```
This is one of the simplest examples and has the explicit solution
```math
X_t = e^{-t}X_0 + \int_0^t e^{-(t - s)} W_s \;\mathrm{d}s.
```

For estimating the order of convergence, we compute a number of numerical approximations of pathwise solutions and the average of their absolute differences.

The computed solution is calculated for sample values $W_{t_j}(\omega_k)$, for samples $\omega_1, \ldots, \omega_K$ and on the mesh points $t_0, \ldots, t_n$. We cannot compute the integrals $\int_0^{t_i} e^{-(t_i - s)}W_s\;\mathrm{d}s$ exactly just from the values on the mesh points, but we can compute their exact expectation via integration by parts
```math

```

