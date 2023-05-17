# Main idea

The improvement in the convergence estimates relies on a novel approach with **four main points:**

1. Estimate the accumulated global error instead of the local error;
2. Write the global error as an iterated integral over the large and the small mesh scales;
3. Use Fubini Theorem to switch the order of integration, moving the critical regularity from the small to the large scale;
4. Assume either a control of the total variation of the sample paths of the noise (as in many point processes and transport process) or use the Itô isometry (when the noise is an It\^o process, such as Wiener, Ornstein-Uhlenbeck, and Geometric Brownian motion processes) in order to bound the large scale.

Let us go over them with more details.

## The first main idea: consider a global error

The first main idea of the proof is to not estimate the local error and, instead, work with an explicit formula for the global error, namely
```math
\begin{align*}
    X_{t_j} - X_{t_j}^N & = X_0 - X_0^N \\
    & \qquad + \int_0^{t_j} \left( f(s, X_s, Y_s) - f(s, X_{\tau^N(s)}, Y_s) \right)\;\mathrm{d}s  \\ 
    & \qquad + \int_{0}^{t_j} \left( f(s, X_{\tau^N(s)}, Y_s) - f(s, X_{\tau^N(s)}^N, Y_s) \right)\;\mathrm{d}s \\
    & \qquad + \int_0^{t_j} \left( f(s, X_{\tau^N(s)}^N, Y_s) - f(\tau^N(s), X_{\tau^N(s)}^N, Y_{\tau^N(s)}) \right)\;\mathrm{d}s,
\end{align*}
```
for $j = 1, \ldots, N,$ where $\tau^N$ is a piecewise constant function with jumps at the mesh points $t_j$, given by
```math
\tau^N(t) = \max_j\{j\Delta t_N; \; j\Delta t_N \leq t\} = \left[\frac{t}{\Delta t_N}\right]\Delta t_N = \left[\frac{tN}{T}\right]\frac{T}{N}.
```

Assuming that $f=f(t, x, y)$ is uniformly globally Lipschitz continous in $x$, with Lipschitz constant $L_x\geq 0$, the first two integrals can be handled in such a way that we obtain the following basic bound for the global error
```math
\begin{align*}
    |X_{t_j} - X_{t_j}^N| & \leq \left( |X_0 - X_0^N| + L_X \int_0^{t_j} |X_s - X_{\tau^N(s)}| \;\mathrm{d}s \right. \\
    & \qquad \left. \left|\int_0^{t_j} \left( f(s, X_{\tau^N(s)}^N, Y_s) - f(\tau^N(s), X_{\tau^N(s)}^N, Y_{\tau^N(s)}) \right)\;\mathrm{d}s\right|\right) e^{L_X t_j}.
\end{align*}
```

The first term vanishes due to the initial condition $X_0^N = X_0$. The second term only depends on the solution and can be easily estimated to be of order $\Delta t_N$ under natural regularity conditions on the term $f=f(t, x, y)$. The only problematic, noise-sensitive term is the last one. The classical analysis is to use an assumed $\theta$-Hölder regularity of the noise sample paths and estimate the local error as
```math
    \mathbb{E}\left[\left|f(s, X_{\tau^N(s)}^N, Y_s) - f(\tau^N(s), X_{\tau^N(s)}^N, Y_{\tau^N(s)})\right|\right] \leq C\Delta t^{\theta}.
```

Instead, we estimate the global error.

## Second main idea: global error in the form of an iterated integral

The second main idea is to consider the whole global error and assume that the steps of the process given by $F_t = f(t, X_{\tau^N(t)}^N, Y_t)$ can be controlled in a suitable way, in the form of an integral of some sort:
```math
    F_s - F_\tau = \int_\tau^s \;\mathrm{d}F_\xi.
```
This can be either in the sense of a Riemann-Stieltjes integral or of an Itô integral. The first sense fits the case of noises with bounded total variation, while the second one fits the case of an Itô noise. In this way, we write the global error as an iterated integral:
```math
    \int_0^{t_j} \left( f(s, X_{\tau^N(s)}^N, Y_s) - f(\tau^N(s), X_{\tau^N(s)}^N, Y_{\tau^N(s)}) \right)\;\mathrm{d}s = \int_0^{t_j} \int_{\tau^N(s)}^s \;\mathrm{d}  F_\xi\;\mathrm{d}s.
```

## Third main idea: use Fubini to change the critical regularity to the large scale

The third main idea is to use Fubini's Theorem to switch the order of integration, making the lower regularity (acting on the variable $\xi$) vary on the larger scale (on the interval $[0, t_j]$) instead of on the small scale of the time-step (on $[\tau^N(s), s]$). In this way, we obtain
```math
\begin{align*}
    \int_0^{t_j} \left( f(s, X_{\tau^N(s)}^N, Y_s) - f(\tau^N(s), X_{\tau^N(s)}^N, Y_{\tau^N(s)}) \right)\;\mathrm{d}s & = \int_0^{t_j} \int_{\tau^N(s)}^s \;\mathrm{d}  F_\xi\;\mathrm{d}s \\
    & = \int_0^{t_j} \int_{\xi}^{\tau^N(\xi) + \Delta t_N} \;\mathrm{d}s \;\mathrm{d} F_\xi \\
    & = \int_0^{t_j} (\tau^N(\xi) + \Delta t_N - \xi) \;\mathrm{d} F_\xi.
\end{align*}
```

## Fourth idea: exploit the noise to estimate the error in the global scale

The final, fourth idea is to assume some global estimate to bound
```math
    \mathbb{E}\left[\left| \int_0^{t_j} \left( f(s, X_{\tau^N(s)}^N, Y_s) - f(\tau^N(s), X_{\tau^N(s)}^N, Y_{\tau^N(s)}) \right)\;\mathrm{d}s\right|\right] \leq \Delta t_N \mathbb{E}\left[\int_0^{t_j} \;\mathrm{d} F_\xi\right] \leq C \Delta t_N,
```
which yields the strong order 1 convergence provided the remaining expectation is finite. The way this is done depends on the noise.

In the case of an Itô noise, we have something like
```math
\;\mathrm{d} F_\xi\ = A_t\;\mathrm{d}t + B_t\;\mathrm{d}W_t,
```
for suitable processes $\{A_t\}_t$ and $\{B_t\}_t$, and then we use the *Itô isometry* and suitable global assumptions on $f=f(t, x,, y)$ in order to bound
```math
\mathbb{E}\left[\int_0^{t_j} \;\mathrm{d} F_\xi\right] \leq \int_0^T\mathbb{E}\left[\|A_t\|\right] \;\mathrm{d}t + \left( \int_0^T\mathbb{E}\left[\|B_t\|^2\right] \;\mathrm{d}t\right)^{1/2} < \infty.
```

In the case of noises with sample paths of bounded variation, we can relax the above condition and work not with $\{F_t\}_{t\in I}$ itself but with a bound on the step of the form
```math
    \|f(s, X_{\tau^N(s)}^N, Y_s) - f(\tau^N(s), X_{\tau^N(s)}^N, Y_{\tau^N(s)})\| \leq \bar F_s - \bar F_{\tau^N(s)}.
```
Only this bounding process $\{\bar F_t\}_{t\in I}$ is required to have sample paths of bounded variation, which is usually easier to check, and so that
```math
\mathbb{E}\left[\int_0^{t_j} \;\mathrm{d} F_\xi\right] \leq \mathbb{E}\left[V(F_\xi; 0, T)\right]\infty.
```

The case of fractional Brownian motion is more delicate, but follows a similar idea, except the sample paths satisfy
```math
    F_s - F_\tau = \int_\tau^s (s - \xi)^{H-1/2}\;\mathrm{d}F_\xi + \mathcal{O}(\Delta t_N),
```
which eventually leads to $\Delta t_N^{H + 1/2}$ order, for $0 < H < 1/2.$
