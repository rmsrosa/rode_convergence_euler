# Extras

## A discrete Gronwall Lemma

For the estimate of the global error, we use the following discrete version of the Grownwall Lemma, which is a particular case of the result found in \cite{GiraultRaviart1981} (see also \cite{Clark1987}).

**Discrete Gronwall Lemma:** Let $(e_j)_{j=0, \ldots, N}$ be a (finite or infinite, $N\leq \infty$) sequence of positive numbers satisfying
```math
    e_j \leq a \sum_{i=0}^{j-1} e_i + b,
```
for every $j=1, \ldots,$ with $e_0 = 0$, and where $a, b > 0$. Then,
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
