#= To Do ===================================================================

===========================================================================#
include("_Spline.jl")

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


"""
Sychronize a SplineDensity object from its samples
"""
function sync_from_samples!(d::SplineDensity; sync_cdf=true, normalize=false)
    update!(d.pdf, d.samples)
    if sync_cdf | normalize
        integral!(d.cdf, d.pdf)
    end
    return normalize ? normalize!(d) : d
end

"""
Sychronize a SplineDensity object from its probability density function (pdf)
"""
function sync_from_pdf!(d::SplineDensity; sync_cdf=true, sync_samples=true, normalize=false)
    if sync_samples | normalize
        update!(d.samples, d.pdf)
    end
    if sync_cdf | normalize
        integral!(d.cdf, d.pdf)
    end
    return normalize ? normalize!(d) : d
end

"""
Evaluate the cdf at +Inf to get the normalization constant, and then normalize to insure the density integrates to 1
"""
function normalize!(d::SplineDensity)
    #Scale all polynomials so that the domain integral is 1
    K = 1/cdf(d, Inf)
    d.pdf.segments .= d.pdf.segments .* K
    d.cdf.segments .= d.cdf.segments .* K
    d.samples.y    .= d.samples.y .* K
    d.samples.dy   .= d.samples.dy .* K
    return d
end