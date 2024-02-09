#= To Do ===================================================================

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
    return ifelse(inbounds(d, x), d(x), zero(T))
end

function cdf(d::Spline{N,T0}, x::Real) where {T0, N}
    bounds = getbounds(d)
    return d(clamp(x, bounds[1], bounds[2]))
end

pdf(d::SplineDensity, x::Real)  = pdf(d.pdf, x)
cdf(d::SplineDensity, x::Real)  = cdf(d.cdf, x)



