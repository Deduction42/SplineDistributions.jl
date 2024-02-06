import Interpolations as Itp

@kwdef struct SplineSamples{T}
    x  :: StepRangeLen{Float64, Float64, Float64, Int64}
    y  :: Vector{T}
    ∂y :: Vector{T}
end

"""
Constructs a SplineSamples object without derivative information,
uses Interpolations.jl to calculate derivatives using natural boundary conditions
"""
function SplineSamples{T}(x, y::AbstractVector) where T
    #Use interpolations to calculate derivatives
    f0 = Itp.interpolate(y, Itp.BSpline(Itp.Cubic(Itp.Natural(Itp.OnGrid()))))
    fx = Itp.scale(f0, x)
    ∂y = map(xi->Itp.gradient(fx, xi)[1], x)

    return SplineSamples{T}(x, y, ∂y)
end
SplineSamples(x, y::AbstractVector{T}) where T = SplineSamples{T}(x, y)

# =================================================================================
# Arithmetic oparators on SplineSamples (with derivative rules)
# =================================================================================
import Base.*
import Base.+

function *(s1::SplineSamples, s2::SplineSamples)
    if s1.x != s2.x
        error("The two SplineSamples objects must have the same sample points")
    end
    y  = s1.y.*s2.y
    ∂y = s1.y.*s2.∂y .+ s1.∂y.*s2.y
    return SplineSamples(s1.x, y, ∂y)
end

function +(s1::SplineSamples, s2::SplineSamples)
    if s1.x != s2.x
        error("The two SplineSamples objects must have the same sample points")
    end
    return SplineSamples(s1.x, (s1.y .+ s2.y), (s1.∂y .+ s2.∂y))
end