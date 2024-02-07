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

@kwdef struct SplineConvolutionBasis{D<:Distribution, N, T<:Real}
    distribution :: D
    basis :: Vector{SVector{N,T}}
end

#const CubicConvolutionBasis{D, T} = SplineConvolutionBasis{D, 4, T} where {D,T}


function normalize!(d::SplineDensity)
    #Scale all polynomials so that the domain integral is 1
    K = 1/d.cdf(d.cdf.vertices[end]) 
    d.pdf.segments .= d.pdf.segments .* K
    d.cdf.segments .= d.cdf.segments .* K
    return d
end

pdf(d::Spline, x::Real) = d(x)

pdf(d::SplineDensity, x::Real)  = d.pdf(x)
cdf(d::SplineDensity, x::Real)  = d.cdf(x)
ccdf(d::SplineDensity, x::Real) = 1-d.cdf(x)


"""
subtract gamma distribution from spline density random variable
"""
function random_var_subtract(fh::Union{<:SplineDensity,<:CubicSpline}, fw::Gamma)
    return random_var_subtract(fh, SplineConvolutionBasis(fh, fw))
end
function random_var_subtract(fh::SplineDensity, fw::SplineConvolutionBasis)
    return random_var_subtract(fh.pdf, fw)
end
function random_var_subtract(fh::CubicSpline, fw::SplineConvolutionBasis)
    return CubicSpline(random_var_subtract_samples(fh, fw))
end

function SplineConvolutionBasis(s::SplineDensity, d::Distribution)
    return SplineConvolutionBasis(s.pdf, d)
end

function SplineConvolutionBasis(s::Spline{N,T1}, dG::Gamma{T2}) where {N, T1, T2}
    T = promote_type(T1, T2)
    PolyType = polytype(CubicSpline)
    vx = s.vertices

    cdfBasis   = map(x->∫xᵏgammapdf_basis(PolyType, dG, x), vx)
    convBasis  = @views cdfBasis[(begin+1):end] .- cdfBasis[begin:(end-1)]

    return SplineConvolutionBasis{Gamma, N, T}(dG, convBasis)
end

function random_var_subtract_samples(fh::CubicSpline{T1}, fw::SplineConvolutionBasis{Gamma, 4, T2}) where {T1, T2}
    T = promote_type(T1, T2)
    vx = fh.vertices
    x0 = fh.vertices[1]
    vp = fh.segments
    basis = fw.basis
    
    vpdf  = zeros(T, length(vx))
    v∂pdf = zeros(T, length(vx))
    v∂pdf[end] = NaN

    #ix is the index on the x-axis to calculate the convolution for
    Np = length(vp)
    for ix in 1:Np
        ux = Polynomial{2}(SVector{2}(vx[ix]-x0, 1)) #Shift-transformation

        #indg is the gamma lag indices, while indp is the polynomial lag indices
        indp = ix:Np
        indg = 1:length(indp)

        #Perform the discreteized convolutions over the lag intervals
        for (ip, ig) in zip(indp, indg)
            p  = substitute(vp[ip], ux)
            ∂p = substitute(differential(vp[ip]), ux) #derivative of ux is 1
            vpdf[ix]  += dot(p.θ, basis[ig])
            v∂pdf[ix] += dot(∂p.θ, basis[ig][SVector{3}(1:3)])
        end
    end

    #Set the final derivative to zero which is the limit if NaN appears
    if isnan(v∂pdf[end])
        v∂pdf[end] = 0.0
    end

    return SplineSamples{T}(
        x = vx,
        y = vpdf,
        ∂y= v∂pdf
    )
end






"""
Convolve xᵏ with a gamma distribution for all terms in Polynomial{N} dG from 0 to x
Final integral can be obtained by multiplying corresponding terms with polynomial instance
"""
function ∫xᵏgammapdf_basis(::Type{Polynomial{N}}, dG::Gamma, x) where N
    return map(k->∫xᵏgammapdf(k, dG, x), powers(Polynomial{N}))
end


"Integrate xᵏ times gamma distribution dG from 0 to x"
function ∫xᵏgammapdf(k::Real, dG::Gamma, x::Real)
    if k == 0 #Shortcut
        return cdf(dG, x)
    end

    #The convolution is θ^k*(Γ(α+k)/Γ(α))*cdf(Gamma(α+k,θ), x)
    #This uses log-exponential form to reduce odds of overflow error (because Gamma functions can get big)
    (α, θ) = (dG.α, dG.θ)
    gammaConv = exp(k*log(abs(θ)) + loggamma(α+k)-loggamma(α)) * cdf(Gamma(α+k,θ), x)
    
    #In cases where θ is allowed to be negative
    #sgn = ifelse(isodd(k), sign(θ), sign(abs(θ)))
    #return flipsign(gammaConv, sgn)
    return gammaConv
end

