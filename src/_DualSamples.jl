import Interpolations as Itp

"""
Object that contins samples f(x) and ∂f(x) evaluated at x, useful for constructing cubic splines
"""
@kwdef struct DualSamples{T}
    x  :: StepRangeLen{Float64, Float64, Float64, Int64}
    y  :: Vector{T}
    ∂y :: Vector{T}
end

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