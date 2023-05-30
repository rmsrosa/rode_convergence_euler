# API

Here we include the docstrings of the methods implemented in `RODEConvergence.jl`, including [noise processes](#noises), [solver methods](#solver-methods), and [convergence estimate tools](#error-estimation).

## Noises

```@docs
WienerProcess
```

```@docs
OrnsteinUhlenbeckProcess
```

```@docs
RODEConvergence.HomogeneousLinearItoProcess
```

```@docs
GeometricBrownianMotionProcess
```

```@docs
CompoundPoissonProcess
```

```@docs
PoissonStepProcess
```

```@docs
ExponentialHawkesProcess
```

```@docs
TransportProcess
```

```@docs
FractionalBrownianMotionProcess
```

```@docs
ProductProcess
```

```@docs
RODEConvergence.UnivariateProcess{T}
```

```@docs
RODEConvergence.MultivariateProcess{T}
```

```@docs
RODEConvergence.rand!
```

## Solver methods

```@docs
RandomEuler
```

```@docs
RandomHeun
```

```@docs
RODEConvergence.CustomMethod
```

## Error estimation

```@docs
RODEConvergence.ConvergenceSuite
```

```@docs
RODEConvergence.ConvergenceResult
```

```@docs
RODEConvergence.solve
```

## Output

```@docs
generate_error_table
```

```@docs
RODEConvergence.plot_convergence
```

```@docs
RODEConvergence.plot_suite
```

## Extras

```@docs
AbstractProcess
```

```@docs
RODEConvergence.RODEMethod
```

```@docs
RODEConvergence.solve!
```

```@docs
RODEConvergence.calculate_trajerrors!
```
