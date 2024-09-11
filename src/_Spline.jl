#= to do =====================================================

==============================================================#
include("_DualSamples.jl")

@kwdef struct Spline{N,T}
    vertices :: StepRangeLen{Float64, Float64, Float64, Int64}
    segments :: Vector{Polynomial{N,T}}
    function Spline{N,T}(x,s) where {N,T} 
        if length(x) == (length(s)+1) 
            return new{N,T}(x,s)
        else 
            error("Length of vertices must be the length of segments plus 1")
        end
    end
end
Base.length(s::Spline) = 1
Broadcast.broadcastable(s::Spline) = Ref(s)
-(s::Spline{N,T}) where {N,T} = Spline{N,T}(s.vertices, -s.segments)

"""
Splines are functors and can be evaluated, if out of range, an extrapolation will be used (robust against roundoff error)
"""
function (s::Spline{N,T0})(x::Real) where {N,T0}
    return s.segments[segment_index(s, x)](x)
end

"""
Find the matching spline segment for real input "x"
"""
function segment_index(s::Spline, x::Real)
    ind = ceil(Int64, (x-s.vertices[begin])/s.vertices.step)
    return clamp(ind, firstindex(s.segments), lastindex(s.segments)) 
end

polytype(::Type{Spline{N}}) where N = Polynomial{N}
polytype(p::Spline{N}) where N = Polynomial{N}
getbounds(s::Spline) = extrema((s.vertices[begin], s.vertices[end]))

function inbounds(s::Spline, x::Real)
    bounds = getbounds(s)
    return bounds[1] <= x <= bounds[2]
end

"""
Shortcut method for obtaining DualSamples from a CubicSpline using the (f, ∂f) method
"""
function DualSamples(s::Spline)
    ∂s    = differential(s)
    f(x)  = s(x)
    ∂f(x) = ∂s(x)
    return DualSamples(s.vertices, (f, ∂f))
end


"""
CubicSplines are special cases of Splines
"""
const CubicSpline{T} = Spline{4,T} where T
CubicSpline(x::StepRangeLen, segments::AbstractVector{Polynomial{4,T}}) where T = CubicSpline{T}(x, segments)

"""
CubicSpline constructors from a DualSamples object
"""
function CubicSpline(s::DualSamples{T}) where T
    segments = [fit_cubic_segment(s[ii], s[ii+1]) for ii in firstindex(s):(lastindex(s)-1)]
    return CubicSpline{T}(s.x, segments)
end

function CubicSpline(x::StepRangeLen, y::AbstractVector{<:Real})
    return CubicSpline(DualSamples(x, y))
end

function CubicSpline(x::StepRangeLen, y::AbstractVector{<:Real}, dy::AbstractVector{<:Real})
    return CubicSpline(DualSamples(x, y, dy))
end

"""
Updates an existing CubicSpline with a DualSamples object (to avoid allocation)
"""
function update!(f::CubicSpline, s::DualSamples)
    if f.vertices != s.x
        error("Cannot update a CubicSpline with DualSamples if their domain bases are different")
    end
    for ii in firstindex(s):(lastindex(s)-1)
        f.segments[ii] = fit_cubic_segment(s[ii], s[ii+1])
    end
    return f
end

"""
Updates an existing DualSamples object with a CubicSpline (to avoid allocation)
"""
function update!(s::DualSamples, f::CubicSpline)
    for (ii, xi) in enumerate(s.x)
        s.y[ii]  = f(xi)
        s.dy[ii] = derivative(f, xi)
    end
    return s
end

"""
Performs a linear substitution of a spline function, returning a spline with a linearly-transformed domain
"""
function substitute(s::Spline{N,T}, u::Polynomial{2,<:Any}) where {N,T}
    vertices = (s.vertices .- u.θ[1])./u.θ[2]
    segments = map(p->substitute(p,u), s.segments)
    return Spline{N,T}(vertices, segments)
end


"""
Fills a spline with polynomial values
"""
function fillspline(x::StepRangeLen, p::Polynomial{N,T}) where {N,T}
    return Spline{N,T}(x, fill(p, length(x)-1))
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
Overwrite an existing spline ∫s that is the integral of spline s (to avoid allocation)
"""
function integral!(∫s::Spline{N,T}, s::Spline; c=0) where {N,T}
    if ∫s.vertices != s.vertices
        error("Cannnot update integral spline because vertices are different")
    end
    C  = T(c)
    x  = s.vertices
    for (k, sk) in enumerate(s.segments)
        ∫sk = integral(sk)
        
        (F0, F1) = (∫sk(x[k]), ∫sk(x[k+1]))
        ∫s.segments[k] = ∫sk + (C-F0)

        C  = C + (F1-F0)
    end
    return ∫s
end


"""
Create a new spline ∫s that is the integral of spline s
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

"""
Retrieve derivative for a single input value x
"""
function derivative(s::Spline, x::Real)
    if !(s.vertices[begin] <= x <= s.vertices[end]) #Return NaN if out of range
        return promote_type(T, typeof(x))(NaN)
    end
    return differential(s.segments[segment_index(s, x)])(x)
end

"""
Calculate definite integral over entire domain, avoids some allocation
"""
function integrate(s::Spline{N,T}) where {N,T}
    C = zero(T)
    x = s.vertices

    for (k, sk) in enumerate(s.segments)
        ∫sk = integral(sk)
        F0 = ∫sk(x[k])
        F1 = ∫sk(x[k+1])
        C  = C + (F1-F0)
    end

    return C
end

"""
Normalizes a spline so that its integral is 1 (useful for representing densities)
"""
function normalize!(s::Spline)
    K = 1/integrate(s)
    s.segments .= s.segments .* K
    return s
end

# ===============================================================================
# Fitting methods (Cubic Splines only)
# ===============================================================================
"""
Fit a cubic spline with two dual samples, each containing [x, y, dy]
"""
function fit_cubic_segment(s1::DualSample, s2::DualSample)
    x  = SVector(s1.x, s2.x)
    y  = SVector(s1.y, s2.y)
    dy = SVector(s1.dy, s2.dy)
    return fit_cubic_segment(x, y, dy)
end

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
#=
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
=#
