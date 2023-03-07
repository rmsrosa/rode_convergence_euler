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
CustomUnivariateMethod
```

```@docs
CustomMultivariateMethod
```

## Error estimation

```@docs
ConvergenceSuite
```

```@docs
ConvergenceResult
```

```@docs
RODEConvergence.solve
```

```@docs
RODEConvergence.solve!
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

```@docs
RODEConvergence.CustomMethod
```
