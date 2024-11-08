# Main results

Consider the random ordinary differential equation
```math
  \begin{cases}
    \displaystyle \frac{\mathrm{d}X_t}{\mathrm{d} t} = f(t, X_t, Y_t), \qquad 0 \leq t \leq T, \\
    \left. X_t \right|_{t = 0} = X_0,
  \end{cases}
```
on a time interval $I=[0, T]$, with $T > 0$, and where the noise $\{Y_t\}_{t\in I}$ is a given stochastic process. The sample space is denoted by $\Omega$.

The Euler method for solving this initial value problem consists in approximating the solution on a uniform time mesh $t_j = j\Delta t_N$, $j = 0, \ldots, N$, with fixed time step $\Delta t_N = T/N$, for a given $N\in \mathbb{N}$. In such a mesh, the Euler scheme takes the form
```math
  X_{t_j}^N = X_{t_{j-1}}^N + \Delta t_N f(t_{j-1}, X_{t_{j-1}}^N, Y_{t_{j-1}}), \qquad j = 1, \ldots, N,
```
with the initial condition
```math
  X_0^N = X_0.
```

When the noise $\{Y_t\}_{t\in I}$ has $\theta$-Hölder continuous sample paths, it is show in [Grune & Kloeden (2001)](https://link.springer.com/article/10.1023/A:1021995918864), under suitable regularity conditions on $f=f(t, x, y)$, that the Euler scheme converges pathwise with order at least $\theta$ with respect to the time step. Similarly, under suitable global conditions, the convergence holds also in the strong sense, i.e. there exists a constant $C \geq 0$ such that
```math
    \max_{j=0, \ldots, N}\mathbb{E}\left[ \left\| X_{t_j} - X_{t_j}^N \right\| \right] \leq C \Delta t_N^\theta, \qquad \forall N \in \mathbb{N},
```
where $\mathbb{E}[\cdot]$ indicates the expectation of a random variable on the underlying probability space.

We show, in the paper, that, in many classical examples, it is possible to exploit further conditions that yield in fact a strong order 1 convergence, with the sample paths still being Hölder continuous or even discontinuous, i.e.
```math
    \max_{j=0, \ldots, N}\mathbb{E}\left[ \left\| X_{t_j} - X_{t_j}^N \right\| \right] \leq C \Delta t_N, \qquad \forall N \in \mathbb{N},
```

We show that this holds essentially for any semi-martingale noise. This includes point processes, transport processes, and Itô diffusion process.

For a fractional Brownian motion process with Hurst parameter $H$, which is not a semi-martingale when $H\neq 1/2,$ we still get strong order 1 convergence for $1/2 \leq H < 1,$ while for $0 < H < 1/2$ the strong order of convergence drops to $H + 1/2,$ which, however, is still higher than the previously estimated order of $H$, which is the Hölder regularity of the pathwise solutions.
