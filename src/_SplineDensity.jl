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

function SplineDensity(samples::DualSamples{T}; integrate=true, normalize=true) where T
    spline  = CubicSpline(samples)
    return SplineDensity(spline, integrate=integrate, normalize=normalize)
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
Shortcut method for obtaining DualSamples from a CubicSpline using the f and ∂f method
"""
function DualSamples(d::SplineDensity)
    return DualSamples(d.pdf)
end

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

"""
Substitute a linear polynomial in the pdf
"""
function substitute(d::SplineDensity, u::Polynomial{2})
    pdfsub = substitute(d.pdf, u)
    
    if pdfsub.vertices.step < 0 #If vertices are negative steps, rearrange
        xs = reverse(pdfsub.vertices)
        ps = reverse!(pdfsub.segments)
        return SplineDensity(CubicSpline(xs, ps))

    else #Keep things as-is (elements are in order)
        return SplineDensity(pdfsub)
    end
end


function mean(d::SplineDensity{T}) where T
    μ = zero(promote_type(T,Float64))

    for (ii, ply) in enumerate(d.pdf.segments)
        ∫xply = integral(times_x(ply))
        x = d.pdf.vertices[SVector(ii, ii+1)]
        μ  += (∫xply(x[2]) - ∫xply(x[1]))
    end

    return μ
end

function var(d::SplineDensity{T}) where T
    μ  = mean(d)
    σ² = zero(promote_type(T,Float64))
    ε  = Polynomial{2}(SVector(-μ, 1.0))

    for (ii, ply) in enumerate(d.pdf.segments)
        ∫ε²ply = integral(ply*ε*ε)
        x   = d.pdf.vertices[SVector(ii, ii+1)]
        σ² += (∫ε²ply(x[2]) - ∫ε²ply(x[1]))
    end

    return σ²
end




