using StaticArrays

#= To do ==================================================

==========================================================#

struct Polynomial{N, T}
    θ :: SVector{N, T}
end

import Base.getindex
import Base.*
import Base.+
import Base.-
import Distributions.partype

getindex(p::Polynomial, ind::Integer) = p.θ[ind+1]
partype(::Type{Polynomial{N,T}}) where {T,N} = T
partype(p::Polynomial) = partype(typeof(p))

powers(::Type{Polynomial{N}}) where {N} = SVector{N}((1:N) .- 1)
powers(::Type{Polynomial{N,T}}) where {N,T} = powers(Polynomial{N})
powers(p::Polynomial) = powers(typeof(p))

Polynomial(x::AbstractVector{T}) where T = Polynomial{length(x),T}(x)
Polynomial{N}(x::AbstractVector{T}) where {N,T} = Polynomial{N,T}(x)
Polynomial{N}(p::Polynomial{N0,T}) where {N,N0,T} = Polynomial{N,T}(p)

Polynomial{N}(x::T) where {N,T<:Real} = Polynomial{N,T}(@SVector fill(x, N))
Polynomial{N,T}(x::Real) where {N,T}  = Polynomial{N,T}(@SVector fill(x, N))

function Polynomial{N,T}(p::Polynomial{N0}) where {N, N0, T}
    if N < N0
        error("Cannot convert a polynomial to a lower order")
    elseif N == N0
        return p
    else
        Δ = N - N0
        return Polynomial{N,T}([p.θ; SVector{Δ}(zeros(T, Δ))])
    end
end

function differential(p::Polynomial{N}) where N
    ind = SVector{N-1}(2:N)
    return Polynomial{N-1}(p.θ[ind].*powers(p)[ind])
end

function integral(p::Polynomial{N}) where N
    return Polynomial{N+1}([0; p.θ./(powers(p).+1)])
end


"Shortcut to multiply a polynomial p(x) by x (essentially bumping the order)"
times_x(p::Polynomial{N}) where {N} = Polynomial{N+1}([0; p.θ])

#Functor (see link which ironically, is about polynomials) https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects)
"This uses Horner's method of polynomial evaluation"
function (p::Polynomial{N,T})(x::Real) where {N,T}
    ex = promote_type(T, typeof(x))(p.θ[end])

    for ii in (N-1):(-1):1
        ex = muladd(x, ex, p.θ[ii])
    end
    return ex
end


# ==================================================================================
# Operators for polynomials
# ==================================================================================
*(p::Polynomial{N}, x::Real) where N = Polynomial{N}(x*p.θ)
*(x::Real, p::Polynomial{N}) where N = Polynomial{N}(x*p.θ)
*(p::Polynomial{N}, u::Polynomial{2}) where N = u[0]*p + u[1]*times_x(p)

+(p1::Polynomial{N}, p2::Polynomial{N}) where N = Polynomial{N}(p1.θ + p2.θ)
-(p1::Polynomial{N}, p2::Polynomial{N}) where N = Polynomial{N}(p1.θ - p2.θ)

function +(p1::Polynomial{N1}, p2::Polynomial{N2}) where {N1,N2}
    N = max(N1, N2)
    return Polynomial{N}(p1) + Polynomial{N}(p2)
end

function -(p1::Polynomial{N1}, p2::Polynomial{N2}) where {N1,N2}
    N = max(N1, N2)
    return Polynomial{N}(p1) - Polynomial{N}(p2)
end

+(p1::Polynomial{N}, p2::Real) where N = Polynomial{N}([p1.θ[1]+p2; p1.θ[SVector{N-1}(2:N)]])
+(p1::Real, p2::Polynomial{N}) where N = p2 + p1
-(p1::Polynomial{N}, p2::Real) where N = Polynomial{N}([p1.θ[1]-p2; p1.θ[SVector{N-1}(2:N)]])
-(p1::Real, p2::Polynomial{N}) where N = Polynomial{N}([p1-p2.θ[1]; p2.θ[SVector{N-1}(2:N)]])


"""
Substitutes x in a polynomial with u=(mx+k) raised to the appropriate power and collects terms
p  = ax^2 + bx + c
pt = a(mx+k)^2 + b(mx+k) + c(mx+k)^0 -> (collect x terms)
"""
function substitute(p::Polynomial{N}, u::Polynomial{2}) where N
    u_expanded = expansions(Polynomial{N}, u)
    return substitute_expansions(p, u_expanded)
end

"""
Substitutes polynomial terms with expansions obtained by the expansions function;
every polynomial parameter will be multiplied by a substitution u=(mx+k) raised to the appropriate power
p  = ax^2 + bx + c
pt = a(mx+k)^2 + b(mx+k) + c(mx+k)^0 
if intermediate results do not need to be reused, use substitute(p::Polynomial, u::Polynomial{2}) instead
"""
function substitute_expansions(p::Polynomial{N}, u_expanded::SVector{N, <:Polynomial{N}}) where N
    return mapreduce((pθ, ue)-> pθ*ue, +, p.θ, u_expanded)
end

"Expands u=(mx+k) to a power to produce a Polynomial{N}, returning all intermediate expansions"
function expansions(::Type{Polynomial{N}}, u::Polynomial{2,T}) where {N,T}
    p = Polynomial{1,T}(SVector(1))
    return SVector{N}([Polynomial{N}(p); _recursive_expansion(Polynomial{N,T}, p, u)])
end

function _recursive_expansion(::Type{Polynomial{N,T}}, p::Polynomial{K}, u::Polynomial{2}) where {N, K, T}
    PT  = Polynomial{N,T}
    p⁺  = p*u
    return SVector{N-K, PT}([PT(p⁺); _recursive_expansion(PT, p⁺, u)])
end

function _recursive_expansion(::Type{Polynomial{N,T}}, p::Polynomial{N}, u::Polynomial{2}) where {N,T}
    return SVector{0,Polynomial{N,T}}()
end