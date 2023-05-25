# Extras

## A discrete Gronwall Lemma

For the estimate of the global error, we use the following discrete version of the Grownwall Lemma, which is a particular case of the result found in \cite{GiraultRaviart1981} (see also \cite{Clark1987}).

**Discrete Gronwall Lemma:** Let $(e_j)_{j}$ be a (finite or infinite) sequence of positive numbers starting at $j=0$ and satisfying
```math
    e_j \leq a \sum_{i=0}^{j-1} e_i + b,
```
for every $j$ with $e_0 = 0$, and where $a, b \geq 0$. Then,
```math
    e_j \leq b e^{aj}, \qquad \forall j.
```

This follows, more precisely, from Lemma V.2.4 in [Girault & Raviart (1981)](https://link.springer.com/book/10.1007/BFb0063447) by taking $n = j$, $a_n = e_j$, $b_n = 0$, $c_n = b$, and $\lambda = a$.

For the sake of completenes, we present a simple proof valid for this particular case.
    
The result is trivially true for $j=0$. Suppose, by induction, that the result is true up to $j-1$. Then, using the estimate and the induction hypothesis, we find
```math
   e_j \leq a \sum_{i=0}^{j-1} be^{ai} + b = b \left(a \sum_{i=0}^{j-1} e^{ai} + 1\right).
```

Using that $1 + a \leq e^a$, we have $a \leq e^a - 1$, hence
```math
    e_j \leq b\left((e^a - 1)\sum_{i=0}^{j-1} e^{ia} + 1\right).
```

Using that $\sum_{i=0}^{j-1} \alpha^i = (\alpha^j - 1)(\alpha - 1)$, with $\alpha = e^a$, we see that
```math
    (e^a - 1)\sum_{i=0}^{j-1} e^{ia} \leq e^{ja} - 1,
```
so that
```math
    e_j \leq be^{ja}.
```
This completes the induction.

## Integral calculations for the fBm convergence

Here, we detail some calculations of deterministic integrals left out of the article.

At some point, we arrived at the following formula for the global error associated with the noise:

```math
    \begin{align*}
        \int_0^{t_j} & \left( f(s, X_{\tau^N(s)}^N, Y_s) - f(\tau^N(s), X_{\tau^N(s)}^N, Y_{\tau^N(s)}) \right)\;\mathrm{d}s \\
        & = \frac{1}{\Gamma(H + 1/2)}\int_0^{t_j} \int_{-\infty}^{\tau^N(s)} \left( (s-\xi)^{H-1/2} - (\tau^N(s)-\xi)^{H-1/2}\right) \;\mathrm{d}W_\xi \;\mathrm{d}s \\
        & \qquad + \frac{1}{\Gamma(H + 1/2)}\int_0^{t_j} \int_{\tau^N(s)}^s (s - \xi)^{H-1/2} \;\mathrm{d}W_\xi \;\mathrm{d}s \\
        & = \frac{1}{\Gamma(H + 1/2)}\int_{-\infty}^{0} \int_{0}^{t_j} \left( (s-\xi)^{H-1/2} - (\tau^N(s)-\xi)^{H-1/2}\right) \;\mathrm{d}s \;\mathrm{d}W_\xi \\
        & \qquad + \frac{1}{\Gamma(H + 1/2)}\int_{0}^{t_j} \int_{\tau^N(\xi)+\Delta t_N}^{t_j} \left( (s-\xi)^{H-1/2} - (\tau^N(s)-\xi)^{H-1/2}\right)  \;\mathrm{d}s \;\mathrm{d}W_\xi\\
        & \qquad + \frac{1}{\Gamma(H + 1/2)}\int_0^{t_j} \int_\xi^{\tau^N(\xi) + \Delta t_N} (s - \xi)^{H-1/2} \;\mathrm{d}s \;\mathrm{d}W_\xi \\
    \end{align*}
```

### First term

For the first term, notice $\sigma \mapsto 1/(\sigma - \xi)^{H-1/2}$ is continuously differentiable on the interval $\sigma > \xi$, so that
```math
    (s-\xi)^{H-1/2} - (\tau^N(s)-\xi)^{H-1/2} = - (H-1/2)\int_{\tau^N(s)}^s (\sigma - \xi)^{H - 3/2} \;\mathrm{d}\sigma.
```
Thus,
```math
    \int_{0}^{t_j} \left( (s-\xi)^{H-1/2} - (\tau^N(s)-\xi)^{H-1/2}\right) \;\mathrm{d}s = (H-1/2)\int_{0}^{t_j} \int_{\tau^N(s)}^s (\sigma - \xi)^{H - 3/2} \;\mathrm{d}\sigma \;\mathrm{d}s.
```
Exchanging the order of integration yields
```math
\begin{align*}
    \int_{0}^{t_j} \left( (s-\xi)^{H-1/2} \right. & \left. - (\tau^N(s)-\xi)^{H-1/2} \right) \;\mathrm{d}s \\
    & = (H-1/2)\int_{0}^{t_j} \int_{\sigma}^{\tau^N(\sigma) + \Delta t_N} (\sigma - \xi)^{H - 3/2} \;\mathrm{d}s \;\mathrm{d}\sigma \\
    & = (H-1/2)\int_{0}^{t_j} \left(\tau^N(\sigma) + \Delta t_N - \sigma\right) (\sigma - \xi)^{H - 3/2} \;\mathrm{d}\sigma.
\end{align*}
```
Hence,
```math
    \left|\int_{0}^{t_j} \left( (s-\xi)^{H-1/2} - (\tau^N(s)-\xi)^{H-1/2} \right) \;\mathrm{d}s\right| \\
    \leq (1/2 - H)\int_{0}^{t_j} \Delta t_N (\sigma - \xi)^{H - 3/2} \;\mathrm{d}\sigma.
```
Now, using the Lyapunov inequality and the Itô isometry, and using the same trick as above,

```math
\begin{align*}
    & \mathbb{E}\left[\left|\int_{-\infty}^{0} \int_{0}^{t_j} \left( (s-\xi)^{H-1/2} - (\tau^N(s)-\xi)^{H-1/2}\right) \;\mathrm{d}s \;\mathrm{d}W_\xi \right|\right] \\
    & \qquad\qquad \leq \left(\int_{-\infty}^{0} \left(\int_{0}^{t_j} \left( (s-\xi)^{H-1/2} - (\tau^N(s)-\xi)^{H-1/2}\right) \;\mathrm{d}s \right)^2 \;\mathrm{d}\xi \right)^{1/2} \\
    & \qquad\qquad \leq \Delta t_N \left(\int_{-\infty}^{0} \left( (1/2 - H)\int_0^{t_j} (\sigma - \xi)^{H-3/2} \;\mathrm{d}\sigma \right)^2 \;\mathrm{d}\xi \right)^{1/2} \\
    & \qquad\qquad \leq (1/2 - H)\Delta t_N \left(\int_{-\infty}^{0} \left(\int_0^T (\sigma - \xi)^{H-3/2} \;\mathrm{d}\sigma \right)^2 \;\mathrm{d}\xi \right)^{1/2}.
\end{align*}
```
Therefore,
```math
    \frac{1}{\Gamma(H + 1/2)}\Delta t_N \mathbb{E}\left[\left|\int_{-\infty}^{0} \int_{0}^{t_j} \left( (s-\xi)^{H-1/2} - (\tau^N(s)-\xi)^{H-1/2}\right) \;\mathrm{d}s \;\mathrm{d}W_\xi \right|\right] \\
    \leq C_H^{(1)}\Delta t_N,
```
for a suitable constant $C_H^{(1)}$. We see this term is of order 1 in $\Delta t_N.$

### Second term

The second term is similar,
```math
\begin{align*}
    \int_{\tau^N(\xi)+\Delta t_N}^{t_j} & \left( (s-\xi)^{H-1/2} - (\tau^N(s)-\xi)^{H-1/2}\right) \;\mathrm{d}s \\ 
    & = (H-1/2)\int_{\tau^N(\xi)+\Delta t_N}^{t_j} \int_{\tau^N(s)}^s (\sigma - \xi)^{H - 3/2} \;\mathrm{d}\sigma \;\mathrm{d}s \\
    & = (H-1/2)\int_{\tau^N(\xi)+\Delta t_N}^{t_j} \int_\sigma^{\tau^N(\sigma) + \Delta t_N} (\sigma - \xi)^{H - 3/2} \;\mathrm{d}s \;\mathrm{d}\sigma \\
    & = (H-1/2)\int_{\tau^N(\xi)+\Delta t_N}^{t_j} \left(\tau^N(\sigma) + \Delta t_N - \sigma\right) (\sigma - \xi)^{H - 3/2} \;\mathrm{d}\sigma.
\end{align*}
```
Thus,
```math
    \left| \int_{\tau^N(\xi)+\Delta t_N}^{t_j} \left( (s-\xi)^{H-1/2} - (\tau^N(s)-\xi)^{H-1/2}\right) \;\mathrm{d}s \right| \\
    \leq (1/2 - H)\Delta t_N \int_{\tau^N(\xi)+\Delta t_N}^{t_j} (\sigma - \xi)^{H - 3/2} \;\mathrm{d}\sigma.
```
Hence,
```math
\begin{align*}
    & \mathbb{E}\left[\left|\int_{0}^{t_j} \int_{\tau^N(\xi)+\Delta t_N}^{t_j} \left( (s-\xi)^{H-1/2} - (\tau^N(s)-\xi)^{H-1/2}\right) \;\mathrm{d}s \;\mathrm{d}W_\xi\right|\right] \\
    & \qquad\qquad \leq \left(\int_{0}^{t_j} \left(\int_{\tau^N(\xi)+\Delta t_N}^{t_j} \left( (s-\xi)^{H-1/2} - (\tau^N(s)-\xi)^{H-1/2}\right) \;\mathrm{d}s \right)^2 \;\mathrm{d}\xi \right)^{1/2} \\
    & \qquad\qquad \leq \Delta t_N (1/2 - H)\left(\int_{0}^{t_j} \left( \int_{\tau^N(\xi)+\Delta t_N}^{T} (\sigma - \xi)^{H-3/2} \;\mathrm{d}\sigma \right)^2 \;\mathrm{d}\xi \right)^{1/2}.
\end{align*}
```
Therefore,
```math
    \frac{1}{\Gamma(H + 1/2)}\mathbb{E}\left[\left|\int_{0}^{t_j} \int_{\tau^N(\xi)+\Delta t_N}^{t_j} \left( (s-\xi)^{H-1/2} - (\tau^N(s)-\xi)^{H-1/2}\right)  \;\mathrm{d}s \;\mathrm{d}W_\xi\right|\right] \\
    \leq C_H^{(2)}\Delta t_N,
```
for a possibly different constant $C_H^{(2)}$. This term is also of order 1.

### Third term

For the last term, we have
```math
    0 \leq \int_\xi^{\tau^N(\xi) + \Delta t_N} (s - \xi)^{H-1/2} \;\mathrm{d}s = \frac{1}{H + 1/2} (\tau^N(\xi) + \Delta t_N - \xi)^{H + 1/2} \\
    \leq \frac{1}{H + 1/2} \Delta t_N^{H + 1/2}.
```
so that, using the Lyapunov inequality and the Itô isometry
```math
    \mathbb{E}\left[\left|\int_0^{t_j} \int_\xi^{\tau^N(\xi) + \Delta t_N} (s - \xi)^{H-1/2} \;\mathrm{d}s \;\mathrm{d}W_\xi\right|\right] \\
    \leq \left( \int_0^{t_j} \left(\int_\xi^{\tau^N(\xi) + \Delta t_N} (s - \xi)^{H-1/2} \;\mathrm{d}s\right)^2 \;\mathrm{d}\xi\right)^{1/2} \\ 
    \leq \left( \int_0^{t_j} \Delta t_N^{2H + 1} \;\mathrm{d}\xi\right)^{1/2} \leq t_j^{1/2} \Delta t_N^{H + 1/2}.
```
Therefore,
```math
    \frac{1}{\Gamma(H + 1/2)}\mathbb{E}\left[\left|\int_0^{t_j} \int_\xi^{\tau^N(\xi) + \Delta t_N} (s - \xi)^{H-1/2} \;\mathrm{d}s \;\mathrm{d}W_\xi\right|\right] \leq C_H^{(3)} \Delta t_N^{H + 1/2},
```
for a third constant $C_H^{(3)}$.

### Putting them together

Putting the three estimates together the noise error becomes
```math
    \mathbb{E}\left[\left|\int_0^{t_j} \left( f(s, X_{\tau^N(s)}^N, Y_s) - f(\tau^N(s), X_{\tau^N(s)}^N, Y_{\tau^N(s)}) \right)\;\mathrm{d}s\right|\right] \\
    \leq C_H^{(4)} \Delta t_N + C_H^{(3)} \Delta t_N^{H + 1/2},
```
where $C_H^{(4)} = C_H^{(1)} + C_H^{(2)}$. 

This eventually leads to a strong error of the form
```math
\max_{j=0, \ldots, N}\mathbb{E}\left[ \left| X_{t_j} - X_{t_j}^N \right| \right] \leq C_1 \Delta t_N + C_2 \Delta t_N^{H + 1/2}, \qquad \forall N \in \mathbb{N},
```
for suitable constants $C_1, C_2\geq 0.$ This proves that the error is of the order of
```math
\textrm{strong error} \lesssim \max\{H + 1/2, 1\}.
```