# API

Here we include the docstrings of the functions implemented in `RODEConvergence.jl`.

## Noises

```@docs
WienerProcess
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
RODEConvergence.solve!(xt, t0, tf, x0, f, yt, method::RandomEuler)
```

```@docs
RODEConvergence.solve!(xt, t0, tf, x0, f, yt, method::RandomHeun)
```

```@docs
RODEConvergence.calculate_trajerrors!
```

```@docs
calculate_trajerrors!
```

## Output

```@docs
generate_error_table
```

## Extras

These are non-exported.

```@docs
RODEConvergence.CompoundPoissonProcessAlt
```
