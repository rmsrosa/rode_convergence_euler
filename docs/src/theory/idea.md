## Main idea

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
Instead, we look at the whole global error and assume that the steps of the process given by $F_t = f(t, X_{\tau^N(t)}^N, Y_t)$ can be controlled in a suitable way. In order to give the main idea, let us assume for the moment that the sample paths of $\{F_t\}_{t\in I}$ satisfy
```math
    F_s - F_\tau = \int_\tau^s \;\mathrm{d}F_\xi,
```
either in the sense of a Riemann-Stieltjes integral or of an Itô integral. The first sense fits the case of noises with bounded total variation, while the second one fits the case of an Itô noise. In any case, we bound the global error term using the Fubini Theorem,
```math
\begin{align*}
    \int_0^{t_j} \left( f(s, X_{\tau^N(s)}^N, Y_s) - f(\tau^N(s), X_{\tau^N(s)}^N, Y_{\tau^N(s)}) \right)\;\mathrm{d}s & = \int_0^{t_j} \int_{\tau^N(s)}^s \;\mathrm{d}  F_\xi\;\mathrm{d}s \\
    & = \int_0^{t_j} \int_{\xi}^{\tau^N(\xi) + \Delta t_N} \;\mathrm{d}s \;\mathrm{d} F_\xi \\
    & = \int_0^{t_j} (\tau^N(\xi) + \Delta t_N - \xi) \;\mathrm{d} F_\xi.
\end{align*}
```

Then, we find that
```math
    \mathbb{E}\left[\left| \int_0^{t_j} \left( f(s, X_{\tau^N(s)}^N, Y_s) - f(\tau^N(s), X_{\tau^N(s)}^N, Y_{\tau^N(s)}) \right)\;\mathrm{d}s\right|\right] \\
    \leq \Delta t_N \mathbb{E}\left[\int_0^{t_j} \;\mathrm{d} F_\xi\right],
```
which yields the strong order 1 convergence provided the remaining expectation is finite.

In the case of an Itô integral, this is exactly what we assume, because the Itô integral is not order preserving; the bound on the remaining expectation is obtained via Itô isometry. In the case of bounded variation, however, we can relax the above condition and work not with $\{F_t\}_{t\in I}$ itself but with a bound on the step of the form
```math
    |f(s, X_{\tau^N(s)}^N, Y_s) - f(\tau^N(s), X_{\tau^N(s)}^N, Y_{\tau^N(s)})| \leq \bar F_s - \bar F_{\tau^N(s)}.
```
Only this bounding process $\{\bar F_t\}_{t\in I}$ is required to have sample paths of bounded variation, which is usually easier to check.

The conditions above are not readily verifiable, but more explicit conditions for each of the two cases are given. Essentially, $f=f(t, x, y)$ is required to have minimal regularity in the sense of differentiability and growth conditions and the noise $\{Y_t\}_{t\in I}$ is either required to have sample paths of bounded variation or to be an Itô noise.
