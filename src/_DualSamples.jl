import Interpolations as Itp

"""
Object that contains vector samples f(x) and ∂f(x) evaluated at x, useful for constructing cubic splines
"""
@kwdef struct DualSamples{T}
    x  :: StepRangeLen{Float64, Float64, Float64, Int64}
    y  :: Vector{T}
    ∂y :: Vector{T}
    DualSamples{T}(x,y,∂y) where T = length(x)==length(y)==length(∂y) ? new{T}(x,y,∂y) : error("Arguments to DualSamples must all have the same length")
end
DualSamples(x::StepRangeLen, y::AbstractVector{T1}, ∂y::AbstractVector{T2}) where {T1<:Real, T2<:Real} = DualSamples{promote_type(T1,T2)}(x, y, ∂y)
Base.firstindex(s::DualSamples) = firstindex(s.x)
Base.lastindex(s::DualSamples)  = lastindex(s.x)

"""
Object that contins a single sample f(x) and ∂f(x) evaluated at x, useful for constructing cubic splines
"""
@kwdef struct DualSample{T}
    x  :: Float64
    y  :: T
    ∂y :: T
end
Base.getindex(s::DualSamples{T}, ii::Integer) where T = DualSample{T}(s.x[ii], s.y[ii], s.∂y[ii])

"""
Constructs a DualSamples object without derivative information,
uses Interpolations.jl to calculate derivatives using natural boundary conditions
"""
function DualSamples{T}(x, y::AbstractVector) where T
    #Use interpolations to calculate derivatives
    f0 = Itp.interpolate(y, Itp.BSpline(Itp.Cubic(Itp.Natural(Itp.OnGrid()))))
    fx = Itp.scale(f0, x)
    ∂y = map(xi->Itp.gradient(fx, xi)[1], x)

    return DualSamples{T}(x, y, ∂y)
end
DualSamples(x, y::AbstractVector{T}) where T = DualSamples{T}(x, y)

"""
Constructs a DualSamples object from the basis "x" and two functions: f(x) and ∂f(x)
"""
function DualSamples(x, f::Function, ∂f::Function)
    y = f.(x)
    ∂y = ∂f.(x)
    T = promote_type(eltype(y), eltype(∂y))
    return DualSamples{T}(x, y, ∂y)
end


# =================================================================================
# Arithmetic oparators on DualSamples (with derivative rules)
# =================================================================================
import Base.*
import Base.+

function *(s1::DualSamples, s2::DualSamples)
    if s1.x != s2.x
        error("The two DualSamples objects must have the same sample points")
    end
    y  = s1.y.*s2.y
    ∂y = s1.y.*s2.∂y .+ s1.∂y.*s2.y
    return DualSamples(s1.x, y, ∂y)
end

function +(s1::DualSamples, s2::DualSamples)
    if s1.x != s2.x
        error("The two DualSamples objects must have the same sample points")
    end
    return DualSamples(s1.x, (s1.y .+ s2.y), (s1.∂y .+ s2.∂y))
end