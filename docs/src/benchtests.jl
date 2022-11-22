using BenchmarkTools

function mysum1(u)
    r = u[1]
    @inbounds for i in 2:length(u)
        r += u[i]
    end
    r
end

function mysum2(u)
    i = firstindex(u)
    r = u[i]
    i = nextind(u, i)
    @inbounds while i ≤ lastindex(u)
        r += u[i]
        i = nextind(u, i)
    end
    r
end

function mysum3(u)
    i = firstindex(u)
    r = u[i]
    @inbounds while (i = nextind(u, i)) ≤ lastindex(u)
        r += u[i]
    end
    r
end

function mysum4(u)
    i = firstindex(u)
    r = u[i]
    idxs = filter(!=(firstindex(u)), eachindex(u))
    @inbounds for i in idxs
        r += u[i]
    end
    r
end

function mysum5(u)
    i = firstindex(u)
    r = u[i]
    @inbounds for i in filter(!=(firstindex(u)), eachindex(u))
        r += u[i]
    end
    r
end

function mysum6(u)
    r = u[1]
    @inbounds for i in nextind(u, firstindex(u)):lastindex(u)
        r += u[i]
    end
    r
end

function mysum7(u)
    r = first(u)
    @inbounds for i in eachindex(u)
        if i > firstindex(u)
            r += u[i]
        end
    end
    r
end

function mysum8(u)
    i1, rst = Iterators.peel(eachindex(u))
    r = u[i1]
    @inbounds for i in rst
        r += u[i]
    end
    r
end

function mysum9(u)
    r = first(u)
    @inbounds for i in Iterators.drop(eachindex(u), 1)
        r += u[i]
    end
    r
end

u = rand(1000)

mysum1(u) ≈ mysum2(u) ≈ mysum3(u) ≈ mysum4(u) ≈ sum(u)
mysum5(u) ≈ mysum6(u) ≈ mysum7(u) ≈ mysum8(u) ≈ sum(u)
mysum9(u) ≈ sum(u)

@info :sum
@btime sum($u)

@info :mysum1
@btime mysum1($u)

@info :mysum2
@btime mysum2($u)

@info :mysum3
@btime mysum3($u)

@info :mysum4
@btime mysum4($u)

@info :mysum5
@btime mysum5($u)

@info :mysum6
@btime mysum6($u)

@info :mysum7
@btime mysum7($u)

@info :mysum8
@btime mysum8($u)

@info :mysum9
@btime mysum9($u)