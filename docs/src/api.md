# API

Here we include the docstrings of the functions implemented in `RODEConvergence.jl`, starting with the exported functions and ending with some internal and other functions used for tests.

## Noises

```@docs
Wiener_noise
```

```@docs
gBm_noise
```

```@docs
CompoundPoisson_noise
```

```@docs
StepPoisson_noise
```

```@docs
Transport_noise
```

```@docs
fBm_noise
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

## Internal or extra functions

```@docs
RODEConvergence.CompoundPoisson_noise_alt
```

```@docs
RODEConvergence.fG_daviesharte
```

```@docs
RODEConvergence.fG_daviesharte_naive
```

```@docs
RODEConvergence.fBm_daviesharte
```
