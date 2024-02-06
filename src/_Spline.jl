#= to do =====================================================
use Interpolations.jl to build a spline from points
differentiate
integrate
==============================================================#
@kwdef struct Spline{N,T}
    vertices :: StepRangeLen{Float64, Float64, Float64, Int64}
    segments :: Vector{Polynomial{N,T}}
end

"""
Splines are functors and can be evaluated
"""
function (s::Spline{N,T})(x::Real) where {N,T}
    if !(s.vertices[begin] <= x <= s.vertices[end])
        return promote_type(T, typeof(x))(NaN)
    end

    i0 = ceil(Int64, (x-s.vertices[begin])/s.vertices.step)
    ic = clamp(i0, firstindex(s.segments), lastindex(s.segments))
    return s.segments[ic](x)
end
polytype(::Type{Spline{N}}) where N = Polynomial{N}
polytype(p::Spline{N}) where N = Polynomial{N}

"""
CubicSplines are special cases of Splines
"""
const CubicSpline{T} = Spline{4,T} where T

CubicSpline(x::StepRangeLen, y::AbstractVector, dy::AbstractVector) = CubicSpline(x, Vector(y), Vector(dy)) 

function CubicSpline(s::SplineSamples{T}) where T
    return CubicSpline(s.x, s.y, s.∂y)
end

function CubicSpline(x::StepRangeLen, y::AbstractVector{T}) where T
    s = SplineSamples{promote_type(T,Float64)}(x, y)
    return CubicSpline(s.x, s.y, s.∂y)
end

function CubicSpline(x::StepRangeLen, y::Vector, dy::Vector)
    if length(x) != length(y) != length(dy)
        error("Input arguments must all have the same length: x=>$(length(x)), y=>$(length(y)), dy=>$(length(dy))")
    end

    function fit_segment(ii)
        ind = SVector(ii, ii+1)
        return fit_cubic_segment(x[ind], y[ind], dy[ind])
    end
    segments = [fit_segment(ii) for ii in 1:(length(x)-1)]

    T = partype(eltype(segments))
    return CubicSpline{T}(x, segments)
end


# ===============================================================================
# Differentiation and integration of splines
# ===============================================================================
"""
Create a new spline that is the derivative of the old one
"""
function differential(s::Spline{N,T}) where {N,T}
    return Spline{N-1,T}(s.vertices, differential.(s.segments))
end

"""
Create a new spline that is the integral of the old one, and it set to 0 at the first vertex
"""
function integral(s::Spline{N,T}, c=0) where {N,T}
    C  = T(c)
    x  = s.vertices

    #Create a set of improper integrals
    ∫s = Spline{N+1,T}(s.vertices, integral.(s.segments))

    #Add constants to all integrals to make them proper
    for (k, ∫f) in enumerate(∫s.segments)
        F0 = ∫f(x[k]) 
        F1 = ∫f(x[k+1]) 
        ∫s.segments[k] = ∫f + (C-F0)
        C = C + (F1-F0)
    end

    #Return the proper integral
    return ∫s
end

# ===============================================================================
# Fitting methods (Cubic Splines only)
# ===============================================================================
"""
Fit a cubic spline segment using two points for x, with corresponding values of y and derivatives dy_dx 
"""
function fit_cubic_segment(x::SVector{2}, y::SVector{2}, dy_dx::SVector{2})
    #Solve the system of equations in the unit basis
    dx_du = x[2]-x[1]
    pu = _fit_standard_cubic_segment(y, dy_dx*dx_du)

    #Include how to translate the original basis x into u 
    px = _unit2domain(pu, x)

    return px
end

"""
Convert a polynomial in the unit-space u=[0,1] to the domain space x=[x1,x2]
"""
function _unit2domain(pu::Polynomial{4}, x::SVector)
    dx_du = x[2]-x[1]
    uₓ = Polynomial(SVector(-x[1]/dx_du, 1/dx_du))
    return substitute(pu, uₓ)
end

"""
Fits two data points and two dervivatives to a standard cubic segment u=[0,1]
"""
function _fit_standard_cubic_segment(y::SVector{2}, dy_du::SVector{2})
    Yᵤ = [y; dy_du]

    #Inverse domain basis for unit range u = [0,1]
    U⁻¹ = inv(@SArray [     
        1.0  0.0  0.0  0.0; #y[1]   = θ₁0⁰ +  θ₂0¹ +  θ₃0² + θ₄0³ 
        1.0  1.0  1.0  1.0; #y[2]   = θ₁1⁰ +  θ₂1¹ +  θ₃1² + θ₄1³ 
        0.0  1.0  0.0  0.0; #dy[1]  = 0θ₁  + 1θ₂0⁰ + 2θ₃0¹ + 3θ₄0² 
        0.0  1.0  2.0  3.0; #dy[1]  = 0θ₁  + 1θ₂1⁰ + 2θ₃1¹ + 3θ₄1² 
    ])

    return Polynomial{4}(U⁻¹*Yᵤ)
end



#=
# Testing cubic segment ==========================================================
using Plots
ω = 1.0
fy(x) = sin.(ω.*x)
fdy_dx(x) = cos.(ω.*x)
x = SVector(0,2)

y = fy.(x)
dy_dx = fdy_dx.(x)

px = fit_cubic_segment(x, y, dy_dx)

vx = LinRange(x[1], x[2], 100)
plot(vx, [fy.(vx), px.(vx)] )
# ================================================================================
=#


# Testing cubic splines ==========================================================
using Plots
ω = 1.0
fy(x)  = sin(ω*x)
∂fy(x) = cos(ω*x)
∫fy(x) = -cos(ω*x)
xs = 0:1.0:10
ys = fy.(xs)
dys = ∂fy.(xs)

s = CubicSpline(xs, ys, dys)
∂s = differential(s)
∫s = integral(s, -1)

x = 0:0.01:10
scatter(xs, ∫fy.(xs))
plot!(x, ∫fy.(x))
plot!(x, ∫s.(x))

