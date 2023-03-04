struct RODEIVP{T, F, R, S}
    f::F
    noise::R
    x0law::S
    t0::T
    tf::T
    function RODEIVP(f::F, noise::R, x0law::S, t0::T, tf::T) where {T, F, R, S}
        x0law isa ContinuousDistribution || error(
            "`x0law` must be a Univariate or Multivariate Continuous Distribution"
        )
        noise isa AbstractProcess || error(
            "`noise` must be a Univariate or Multivariate AbstractProcess"
        )
        eltype(x0law) == T || error(
            "eltypes of `x0law` and `noise` must coincide with the types of `t0` and `tf`"
        )
        return new{T, F, R, S}(f, noise, x0law, t0, tf)
    end
end

