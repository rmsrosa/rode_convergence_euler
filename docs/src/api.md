# API

Here we include the docstrings of the functions implemented in `RODEConvergence.jl`.

## Noises

```@docs
WienerProcess
```

```@docs
OrnsteinUhlenbeckProcess
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

## Methods

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

```@docs
RODEConvergence.solve!
```

```@docs
RODEConvergence.calculate_trajerrors!
```

## Output

```@docs
RODEConvergence.plot_suite
```

```@docs
RODEConvergence.plot_convergence
```

```@docs
generate_error_table
```

## Extras

```@docs
AbstractProcess
```

```@docs
RODEConvergence.RODEMethod
```

```@docs
RODEConvergence.CompoundPoissonProcessAlt
```
