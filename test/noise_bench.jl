## Benchmarks attempting to get a non-allocating multi-noise, but inference is having trouble with it

function fff1!(rng::AbstractRNG, Y::Vector{<:RODEConvergence.AbstractProcess}, yt::AbstractMatrix)
    axes(eachcol(yt)) == axes(Y) || throw(
        DimensionMismatch("Columns of `yt` and vector of noises `Y` must match indices.")
    )
    for (i, yti) in enumerate(eachcol(yt))
        rand!(rng, Y[i], yti)
    end
end

function fff2!(rng::AbstractRNG, Y::Vector{<:RODEConvergence.AbstractProcess}, yt::AbstractMatrix)
    axes(eachcol(yt)) == axes(Y) || throw(
        DimensionMismatch("Columns of `yt` and vector of noises `Y` must match indices.")
    )
    for i in eachindex(Y)
        rand!(rng, Y[i], view(yt, :, i))
    end
end

function fff3!(rng::AbstractRNG, Y::Vector{<:RODEConvergence.AbstractProcess}, yt::AbstractMatrix)
    axes(eachcol(yt)) == axes(Y) || throw(
        DimensionMismatch("Columns of `yt` and vector of noises `Y` must match indices.")
    )

    ntuple(i -> rand!(rng, Y[i], view(yt, :, i)), length(Y))
    nothing
end

function fff4!(rng::AbstractRNG, Y::Vector{<:RODEConvergence.AbstractProcess}, yt::AbstractMatrix)
    axes(eachcol(yt)) == axes(Y) || throw(
        DimensionMismatch("Columns of `yt` and vector of noises `Y` must match indices.")
    )
    for (Yi, yti) in zip(Y, eachcol(yt))
        rand!(rng, Yi, yti)
    end
end


function ggg1!(rng::AbstractRNG, Y::Tuple{Vararg{RODEConvergence.AbstractProcess}}, yt::AbstractMatrix)
    axes(eachcol(yt)) == axes(Y) || throw(
        DimensionMismatch("Columns of `yt` and of tuple `Y` of noises must match indices.")
    )
    for (i, yti) in enumerate(eachcol(yt))
        rand!(rng, Y[i], yti)
    end
end

function ggg2!(rng::AbstractRNG, Y::Tuple{Vararg{RODEConvergence.AbstractProcess}}, yt::AbstractMatrix)
    axes(eachcol(yt)) == axes(Y) || throw(
        DimensionMismatch("Columns of `yt` and of tuple `Y` of noises must match indices.")
    )
    for i in eachindex(Y)
        rand!(rng, Y[i], view(yt, :, i))
    end
end

function ggg3!(rng::AbstractRNG, Y::Tuple{Vararg{RODEConvergence.AbstractProcess}}, yt::AbstractMatrix)
    axes(eachcol(yt)) == axes(Y) || throw(
        DimensionMismatch("Columns of `yt` and of tuple `Y` of noises must match indices.")
    )
    ntuple(i -> rand!(rng, Y[i], view(yt, :, i)), length(Y))
    nothing
end

function ggg4!(rng::AbstractRNG, Y::Tuple{Vararg{RODEConvergence.AbstractProcess}}, yt::AbstractMatrix)
    axes(eachcol(yt)) == axes(Y) || throw(
        DimensionMismatch("Columns of `yt` and of tuple `Y` of noises must match indices.")
    )
    for (Yi, yti) in zip(Y, eachcol(yt))
        rand!(rng, Yi, yti)
    end
end

@info "fff1"
@btime fff1!($rng, $Y, $YMt)

@info "fff2"
@btime fff2!($rng, $Y, $YMt)

@info "fff3"
@btime fff3!($rng, $Y, $YMt)

@info "fff4"
@btime fff4!($rng, $Y, $YMt)

@info "ggg1"
@btime ggg1!($rng, $Ytup, $YMt)

@info "ggg2"
@btime ggg2!($rng, $Ytup, $YMt)

@info "ggg3"
@btime ggg3!($rng, $Ytup, $YMt)

@info "ggg4"
@btime ggg4!($rng, $Ytup, $YMt)

for i in 1:length(Y)
    @info "fff2 $i"
    @btime fff2!($rng, $(Y[1:i]), $(YMt[:, 1:i]))

    @info "ggg2 $i"
    @btime ggg2!($rng, $(Ytup[1:i]), $(YMt[:, 1:i]))
end
