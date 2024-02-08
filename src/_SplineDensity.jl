#= To Do ===================================================================
(1) Separate out the cdfBasis as a separate input so that it can be reused
(2) Test against a normal convolution benchmark
(3) Test incremental convolutions against a single equivalent
===========================================================================#

@kwdef struct SplineDensity{T} <: Distribution{Univariate, Continuous}
    pdf :: Spline{4,T}
    cdf :: Spline{5,T}
    samples :: DualSamples
end

function SplineDensity(pdf_spline::CubicSpline{T}; integrate=true, normalize=true) where T
    ∂pdf_spline(z) = derivative(pdf_spline, z)
    x  = pdf_spline.vertices
    cdf_spline = (integrate|normalize) ? integral(pdf_spline) : fillspline(x, Polynomial{5,T}(NaN))

    density = SplineDensity{T}(
        pdf = pdf_spline,
        cdf = cdf_spline,
        samples = DualSamples(x, pdf_spline.(x), ∂pdf_spline.(x))
    )

    if normalize
        return normalize!(density)
    else
        return density
    end
end

function SplineDensity(s::DualSamples{T}; normalize=true) where T
    spline  = CubicSpline(s)
    return SplineDensity(spline, normalize=normalize)
end

function normalize!(d::SplineDensity)
    #Scale all polynomials so that the domain integral is 1
    K = 1/cdf(d, Inf)
    d.pdf.segments .= d.pdf.segments .* K
    d.cdf.segments .= d.cdf.segments .* K
    d.samples.y    .= d.samples.y .* K
    d.samples.∂y   .= d.samples.∂y .* K
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



