include("_ShiftedGamma.jl")

import Interpolations as Itp
import FiniteDifferences as FteDiff

"""
Object that contains vector samples f(x) and ∂f(x) evaluated at x, useful for constructing cubic splines
"""
@kwdef struct DualSamples{T}
    x  :: StepRangeLen{Float64, Float64, Float64, Int64}
    y  :: Vector{T}
    dy :: Vector{T}
    DualSamples{T}(x,y,dy) where T = length(x)==length(y)==length(dy) ? new{T}(x,y,dy) : error("Arguments to DualSamples must all have the same length")
end
DualSamples(x::StepRangeLen, y::AbstractVector{T1}, dy::AbstractVector{T2}) where {T1<:Real, T2<:Real} = DualSamples{promote_type(T1,T2)}(x, y, dy)
Base.firstindex(s::DualSamples) = firstindex(s.x)
Base.lastindex(s::DualSamples)  = lastindex(s.x)
Base.length(s::DualSamples)     = length(s.x)

"""
Object that contins a single sample f(x) and ∂f(x) evaluated at x, useful for constructing cubic splines
"""
@kwdef struct DualSample{T}
    x  :: Float64
    y  :: T
    dy :: T
end
Base.getindex(s::DualSamples{T}, ii::Integer) where T = DualSample{T}(s.x[ii], s.y[ii], s.dy[ii])

"""
Constructs a DualSamples object without derivative information,
uses Interpolations.jl to calculate derivatives using natural boundary conditions
"""
function DualSamples{T}(x, y::AbstractVector) where T
    #Use interpolations to calculate derivatives
    f0 = Itp.interpolate(y, Itp.BSpline(Itp.Cubic(Itp.Natural(Itp.OnGrid()))))
    fx = Itp.scale(f0, x)
    dy = map(xi->Itp.gradient(fx, xi)[1], x)

    return DualSamples{T}(x, y, dy)
end
DualSamples(x, y::AbstractVector{T}) where T = DualSamples{T}(x, y)

"""
Constructs a DualSamples object from the basis "x" and a tuple of two functions: f(x) and ∂f(x)
"""
function DualSamples(x, ft::Tuple{<:Function, <:Function})
    y  = ft[1].(x)
    dy = ft[2].(x)
    T = promote_type(eltype(y), eltype(dy))
    return DualSamples{T}(x, y, dy)
end

"""
Constructs a DualSamples object from the basis "x" and a single function: f(x)
Derivatives are estimated using FiniteDifferences.jl
"""
function DualSamples(x, f::Function)
    ∂ = FteDiff.central_fdm(5,1) 
    df(x) = ∂(f, x) 
    return DualSamples(x, (f, df))
end


"""
Constructs a DualSamples object from a continuous univariate distribution
"""
function DualSamples(x::StepRangeLen, d::Distribution)
    y  = pdf.(d,x)
    dy = dpdf.(d,x)
    T  = promote_type(eltype(y), eltype(dy))
    return DualSamples{T}(x, y, dy)
end

"""
Constructs a DualSample object from a continuous univariate distribution
"""
function DualSample(x::Real, d::Distribution)
    y  = pdf(d, x)
    dy = dpdf(d, x)
    T  = promote_type(typeof(y), typeof(dy))
    return DualSample{T}(x, y, dy)
end

"""
Updates a DualSamples object with the contents of another
"""
function update!(s1::DualSamples, s2::DualSamples)
    if s1.x != s2.x
        error("Cannot update a DualSamples object if the input has different sample points")
    end
    s1.y  .= s2.y
    s1.dy .= s2.dy
    return s1
end

# =================================================================================
# Arithmetic oparators on DualSamples (with derivative rules)
# =================================================================================
import Base.*
import Base.+
import Base.exp
import Base.log

+(s1::DualSamples, x::Real) = DualSamples(s1.x, s1.y.+x, s1.dy)
+(x::Real, s1::DualSamples) = s1 + x
*(s1::DualSamples, x::Real) = DualSamples(s1.x, s1.y.*x, s1.dy*x)
*(x::Real, s1::DualSamples) = s1 * x

+(s1::DualSample, x::Real) = DualSample(s1.x, s1.y+x, s1.dy)
+(x::Real, s1::DualSample) = s1 + x
*(s1::DualSample, x::Real) = DualSample(s1.x, s1.y*x, s1.dy*x)
*(x::Real, s1::DualSample) = s1 * x

function *(s1::DualSamples, s2::DualSamples)
    if s1.x != s2.x
        error("The two DualSamples objects must have the same sample points")
    end
    y  = s1.y.*s2.y
    dy = s1.y.*s2.dy .+ s1.dy.*s2.y
    return DualSamples(s1.x, y, dy)
end

exp(s1::DualSamples) = exp!(deepcopy(s1))
log(s1::DualSamples) = log!(deepcopy(s1))

function exp!(s1::DualSamples)
    for ii in eachindex(s1.y)
        exps1 = exp(s1[ii])
        s1.y[ii]  = exps1.y
        s1.dy[ii] = exps1.dy 
    end
    return s1
end

function log!(s1::DualSamples)
    for ii in eachindex(s1.y)
        logs1 = log(s1[ii])
        s1.y[ii]  = logs1.y
        s1.dy[ii] = logs1.dy 
    end
    return s1
end

function +(s1::DualSamples, s2::DualSamples)
    if s1.x != s2.x
        error("The two DualSamples objects must have the same sample points")
    end
    return DualSamples(s1.x, (s1.y .+ s2.y), (s1.dy .+ s2.dy))
end

function *(s1::DualSample, s2::DualSample)
    if s1.x != s2.x
        error("The two DualSample objects must have the same sample point x")
    end
    y  = s1.y*s2.y
    dy = s1.y*s2.dy + s1.dy*s2.y
    return DualSample(s1.x, y, dy)
end

function +(s1::DualSample, s2::DualSample)
    if s1.x != s2.x
        error("The two DualSamples objects must have the same sample point x")
    end
    return DualSample(s1.x, (s1.y + s2.y), (s1.dy + s2.dy))
end

function exp(s1::DualSample)
    y  = exp(s1.y)
    dy = y*s1.dy 
    return DualSample(s1.x, y, dy)
end

function log(s1::DualSample)
    y  = log(s1.y)
    dy = s1.dy/s1.y
    return DualSample(s1.x, y, dy)
end