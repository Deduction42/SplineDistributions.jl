#= To Do ===================================================================
(1) Separate out the cdfBasis as a separate input so that it can be reused
(2) Test against a normal convolution benchmark
(3) Test incremental convolutions against a single equivalent
===========================================================================#

@kwdef struct SplineDensity{T} <: Distribution{Univariate, Continuous}
    pdf :: Spline{4,T}
    cdf :: Spline{5,T}
end

function SplineDensity(spline::CubicSpline{T}; normalize=true) where T
    density = SplineDensity{T}(
        pdf = spline,
        cdf = integral(spline)
    )
    if normalize
        return normalize!(density)
    else
        return density
    end
end

function SplineDensity(s::SplineSamples{T}; normalize=true) where T
    spline  = CubicSpline(s)
    return SplineDensity(spline, normalize=normalize)
end

function normalize!(d::SplineDensity)
    #Scale all polynomials so that the domain integral is 1
    K = 1/cdf(d, Inf)
    d.pdf.segments .= d.pdf.segments .* K
    d.cdf.segments .= d.cdf.segments .* K
    return d
end

function pdf(d::Spline{N,T0}, x::Real) where {T0, N}
    T = promote_type(T0, typeof(x))
    supported = d.vertices[begin] <= x <= d.vertices[end]
    result = ifelse(supported, d(x), zero(T))
    
    return ifelse(isnan(x), T(NaN), result)
end

function cdf(d::Spline{N,T0}, x::Real) where {T0, N}
    Δ = (d.vertices[begin], d.vertices[end])

    return d(clamp(x, Δ[1], Δ[2]))
end

pdf(d::SplineDensity, x::Real)  = pdf(d.pdf, x)
cdf(d::SplineDensity, x::Real)  = cdf(d.cdf, x)



