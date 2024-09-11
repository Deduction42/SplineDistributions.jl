#= To do ==================================================

==========================================================#
include("__imports.jl")

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
        return Polynomial{N,T}([p.θ; @SVector zeros(T, Δ)])
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
-(p::Polynomial{N}) where N = Polynomial{N}(-p.θ)
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
Substitutes a linear function u(x) = b + ax inside a polynomial p(x) and collects the terms
"""
function substitute(p::Polynomial{N,T1}, u::Polynomial{2,T2}) where {N,T1,T2}
    T = promote_type(T1,T2)
    pk  = p.θ[end]*u
    ind = SVector{N-1}(1:(N-1)) 
    return Polynomial{N,T}(horner_expansion(pk, u, p.θ[ind]))
end

"""
Recursively applies Hornner's method to expand polynomial terms of p(x) given a linear substitution u(x) = b + ax
"""
function horner_expansion(p::Polynomial, u::Polynomial{2}, θ::SVector{N,<:Real}) where N
    pk  = (p + θ[end])*u
    ind = SVector{N-1}(1:(N-1))
    return horner_expansion(pk, u, θ[ind])
end

function horner_expansion(p::Polynomial, u::Polynomial{2}, θ::SVector{1,<:Real})
    return p + θ[1]
end
