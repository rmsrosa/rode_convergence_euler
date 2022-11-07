# Linear RODE with non-homogenous Wiener noise

Here, we consider the Euler approximation of the following linear random ordinary differential equation with the noise in the nonhomogeneous term, in the form of a Wiener process:
```math
  \begin{cases}
    \displaystyle \frac{\mathrm{d}X_t}{\mathrm{d} t} = - X_t + W_t, \qquad 0 \leq t \leq T, \\
    \left. X_t \right|_{t = 0} = X_0.
  \end{cases}
```
This is one of the simplest examples of RODEs and has the explicit solution
```math
X_t = e^{-t}X_0 + \int_0^t e^{-(t - s)} W_s \;\mathrm{d}s.
```

For estimating the order of convergence, we use the Monte Carlo method, computing a number of numerical approximations of pathwise solutions and taking the average of their absolute differences.

The computed solution is calculated from realizations $W_{t_j}(\omega_k)$, for samples paths $\{W_t(\omega_k)\}_{t\geq 0}$, with $k = 1, \ldots, K,$ and on the mesh points $t_0, \ldots, t_n$. We cannot compute the integral $\int_0^{t_j} e^{-(t_j - s)}W_s\;\mathrm{d}s$ exactly just from the values on the mesh points, but we can compute its expectation. First we break down the sum into parts:
```math
\int_0^{t_j} e^{-(t_j - s)}W_s\;\mathrm{d}s = \sum_{i = 0}^{j-1} \int_{t_i}^{t_{i+1}} e^{-(t_j - s)}W_s\;\mathrm{d}s.
```
On each mesh interval, we use that
```math
B_t = W_t - W_{t_i} - \frac{t - t_i}{t_{i+1}-t_i}(W_{t_{i+1}} - W_{t_i})
```
is a Brownian bridge on the interval $[t_i, t_{i+1}]$, independent of $\{W_t\}_{t\geq 0}$. Notice that
```math
\mathrm{d}W_t = \mathrm{d}B_t + \frac{W_{t_{t+1}}-W_{t_i}}{t_{i+1}-t_i}\;\mathrm{d}t.
```

Thus,
```math
\begin{align*}
\mathrm{d}(e^{-(t_j-t)}W_t) & = e^{-(t_j-t)}W_t\;\mathrm{d}t + e^{-(t_j-t)}\;\mathrm{d}W_t \\
& = e^{-(t_j-t)}W_t\;\mathrm{d}t + e^{-(t_j-t)} \left(\mathrm{d}B_t + \frac{W_{t_{t+1}}-W_{t_i}}{t_{i+1}-t_i}\;\mathrm{d}t\right),
\end{align*}
```
so that
```math
\begin{align*}
\int_{t_i}^{t_{i+1}} e^{-(t_j - s)}W_s\;\mathrm{d}s & = e^{-(t_j-t_{i+1})}W_{t_{i+1}} - e^{-(t_j-t_i)}W_{t_i} + \int_{t_i}^{t_{i+1}} e^{-(t_j - s)}\;\mathrm{d}B_s \\
& \qquad + \frac{W_{t_{t+1}}-W_{t_i}}{t_{i+1}-t_i}\int_{t_i}^{t_{i+1}} e^{-(t_j - s)}\;\mathrm{d}s.
\end{align*}
```

Taking the expectation, using that the expectation of an Itô integral with respect to a Brownian bridge with zero endpoints is zero, and using that the values at the mesh points are given, we find that
```math
\mathbb{E}\left[ \int_{t_i}^{t_{i+1}} e^{-(t_j - s)}W_s\;\mathrm{d}s\right] = e^{-(t_j-t_{i+1})}W_{t_{i+1}} - e^{-(t_j-t_i)}W_{t_i} + \frac{W_{t_{t+1}}-W_{t_i}}{t_{i+1}-t_i}\left( e^{-(t_j - t_{i+1})} - e^{-(t_j - t_i)}\right).
```

Hence, given the realizations of a Wiener noise on the mesh points,
```math
\begin{align*}
\mathbb{E}[X_{t_j}] & = e^{-t_j}X_0 + \sum_{i=0}^{j-1} \left(e^{-(t_j-t_{i+1})}W_{t_{i+1}} - e^{-(t_j-t_i)}W_{t_i}\right) \\
& \qquad + \sum_{i=0}^{j-1} \frac{W_{t_{t+1}}-W_{t_i}}{t_{i+1}-t_i}\left( e^{-(t_j - t_{i+1})} - e^{-(t_j - t_i)}\right).
\end{align*}
```

The first summation telescopes out and, since $W_0 = 0$, we are left with
```math
\mathbb{E}[X_{t_j}] = e^{-t_j}X_0 + W_{t_j} - \sum_{i=0}^{j-1} \frac{W_{t_{t+1}}-W_{t_i}}{t_{i+1}-t_i}\left( e^{-(t_j - t_{i+1})} - e^{-(t_j - t_i)}\right).
```


