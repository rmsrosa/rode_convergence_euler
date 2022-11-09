# Homogenous linear RODE with sine of Wiener noise coefficient

Now we consider a homogeneous linear equation in which the coefficient depends on the sine of a Wiener process.

## The equation

More precisely, we consider the RODE
```math
  \begin{cases}
    \displaystyle \frac{\mathrm{d}X_t}{\mathrm{d} t} = (\mu + \sigma \sin W_t) X_t, \qquad 0 \leq t \leq T, \\
    \left. X_t \right|_{t = 0} = X_0,
  \end{cases}
```
where $\{W_t\}_{t\geq 0}$ is a Wiener process.

The explicit solution
```math
X_t = e^{\int_0^t (\mu + \sigma \sin(W_s)) \;\mathrm{d}s} X_0.
```

## Computing the exact solution

As before, we cannot compute the integral $\int_0^{t_j} \sin(W_s)\;\mathrm{d}s$ exactly, just from the values $W_{t_j}$ of the noise, on the mesh points, but we can compute its expectation. We break down the sum into parts:
```math
\int_0^{t_j} \sin(W_s)\;\mathrm{d}s = \sum_{i = 0}^{j-1} \int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}s.
```

On each mesh interval, we use that
```math
B_t = W_t - W_{t_i} - \frac{t - t_i}{t_{i+1}-t_i}(W_{t_{i+1}} - W_{t_i})
```
is a Brownian bridge on the interval $[t_i, t_{i+1}]$, independent of $\{W_t\}_{t\geq 0}$.

According to Itô's formula, for a smooth function $f=f(w)$, we have
```math
\mathrm{d}f(W_t) = \frac{1}{2}f''(W_t)\;\mathrm{d}t + f'(W_t)\;\mathrm{d}W_t.
```
Considering $f(w) = \sin(w)$, we find
```math
\mathrm{d}\sin(W_t) = -\frac{1}{2}\sin(W_t)\;\mathrm{d}t + \cos(W_t)\;\mathrm{d}W_t.
```

Since
```math
\mathrm{d}W_t = \mathrm{d}B_t + \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\;\mathrm{d}t,
```
we obtain
```math
\mathrm{d}\sin(W_t) = -\frac{1}{2}\sin(W_t)\;\mathrm{d}t + \cos(W_t)\left(\mathrm{d}B_t + \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\;\mathrm{d}t\right),
```
which can be written as
```math
\sin(W_t)\;\mathrm{d}t = -2\mathrm{d}\sin(W_t) + 2\cos(W_t)\left(\mathrm{d}B_t + 2\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\;\mathrm{d}t\right).
```

Thus,
```math
\begin{align*}
\int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}s & = -2\sin(W_{t_{i+1}}) + 2\sin(W_{t_i}) - 2\int_{t_i}^{t_{i+1}} \cos(W_s)\;\mathrm{d}B_s \\
& \qquad - 2\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\int_{t_i}^{t_{i+1}} \cos(W_s)\;\mathrm{d}s.
\end{align*}
```

Analogously,
```math
\begin{align*}
\int_{t_i}^{t_{i+1}} \cos(W_s)\;\mathrm{d}s & = -2\cos(W_{t_{i+1}}) + 2\cos(W_{t_i}) + 2\int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}B_s \\
& \qquad + 2\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}s.
\end{align*}
```

Substituting,
```math
\begin{align*}
\int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}s & = -2\sin(W_{t_{i+1}}) + 2\sin(W_{t_i}) - 2\int_{t_i}^{t_{i+1}} \cos(W_s)\;\mathrm{d}B_s \\
& \qquad - 4\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\left(-\cos(W_{t_{i+1}}) + \cos(W_{t_i})\right) \\
& \qquad - 4\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i} \int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}B_s\\
& \qquad - 4\left(\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\right)^2 \int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}s.
\end{align*}
```

Therefore,
```math
\begin{align*}
\left(1 + 4\left(\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\right)^2 \right)\int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}s & = -2\sin(W_{t_{i+1}}) + 2\sin(W_{t_i}) - 2\int_{t_i}^{t_{i+1}} \cos(W_s)\;\mathrm{d}B_s \\
& \qquad - 4\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\left(-\cos(W_{t_{i+1}}) + \cos(W_{t_i})\right) \\
& \qquad - 4\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i} \int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}B_s.
\end{align*}
```

Thus, we can write that
```math
\begin{align*}
\int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}s & = \left(1 + 4\left(\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\right)^2 \right)^{-1} \\
& \qquad \left( -2\sin(W_{t_{i+1}}) + 2\sin(W_{t_i}) - 4\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\left(-\cos(W_{t_{i+1}}) + \cos(W_{t_i})\right)\right) \\
& \qquad + Z_i,
\end{align*}
```
where
```math
Z_i = \left(1 + 4\left(\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\right)^2 \right)^{-1}\left( - 2\int_{t_i}^{t_{i+1}} \cos(W_s)\;\mathrm{d}B_s - 4\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i} \int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}B_s\right).
```

Hmm, $Z_i$ averages to zero, but without knowing more about it, we cannot get a more explicit form for the expected exact solution, which envolves the exponential of the above integral. This example will have to be on hold, for now.
