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
MultiProcess
```

## Solvers

```@docs
solve_euler!
```

```@docs
solve_heun!
```

## Error estimation

```@docs
prepare_variables
```

```@docs
calculate_errors
```

```@docs
calculate_errors!
```

## Output

```@docs
plot_sample_approximations
```

```@docs
generate_error_table
```

```@docs
plot_dt_vs_error
```

```@docs
plot_t_vs_errors
```

## Extras

These are non-exported.

```@docs
RODEConvergence.CompoundPoissonProcessAlt
```