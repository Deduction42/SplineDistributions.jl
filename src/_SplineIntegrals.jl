include("_SplineDensity.jl")

@kwdef struct SplineIntegrals{D<:Distribution, N, T<:Real}
    distribution :: D
    vertices :: StepRangeLen{Float64, Float64, Float64, Int64}
    samples  :: Vector{SVector{N,T}}
end

const CubicIntegrals{D, T} = SplineIntegrals{D, 4, T} where {D,T}


"""
Construct polynomial integrals with pdf functions of d (∫ p*fd dx) 
Polynomial terms are based off the spline-based pdf of s
Integral samples are obtained at the vertices of s
"""
function SplineIntegrals(s::SplineDensity, d::Distribution)
    return SplineIntegrals(s.pdf, d)
end

"""
Construct polynomial integrals with pdf functions of d (∫ p*fd dx) 
Polynomial terms are based off the spline s
Integral samples are obtained at the vertices of s
"""
function SplineIntegrals(s::Spline{N,T1}, dG::Gamma{T2}) where {N, T1, T2}
    T = promote_type(T1, T2)
    PolyType = polytype(CubicSpline)
    samples  = map(x->∫xᵏgammapdf_basis(PolyType, dG, x), s.vertices)

    return SplineIntegrals{Gamma, N, T}(dG, s.vertices, samples)
end

"""
Subtract a gamma random variable (fw) from a spline density function (fh) and overwrite the samples of fh
By default it also updates the pdf and the cdf. It does not normalize so the integral might be slightly less than 1.
"""
function random_var_subtract!(fh::SplineDensity, fw::Gamma, update_pdf=true, update_cdf=true)
    ∫fw = SplineIntegrals(fh, fw)
    return random_var_subtract!(fh, ∫fw, update_pdf=update_pdf, update_cdf=update_cdf)
end

"""
Subtract the SplineIntegrals of a gamma random variable (fw) from a spline density function (fh) and overwrite the samples of fh
SplineIntegrals are used as a standin for the original distribution allowing for reuse, avoiding repeaded cdf calcualtions
By default it also updates the pdf and the cdf. It does not normalize so the integral might be slightly less than 1.
"""
function random_var_subtract!(fh::SplineDensity{T}, fw::CubicIntegrals{Gamma}; update_pdf=true, update_cdf=true) where T
    if fh.pdf.vertices != fw.vertices
        error("SplineDensity and SplineIntegral must be evaluated at the same vertices")
    end

    vx = fh.pdf.vertices
    x0 = fh.pdf.vertices[1]
    polys = fh.pdf.segments
    intervalpolyterms = diff(fw.samples) #Polynomial integral terms over the intervals (set at SplineIntegrals)

    #ix is the index on the x-axis to calculate the convolution for
    Np = length(polys)
    for ix in 1:Np
        ux = Polynomial{2}(SVector{2}(vx[ix]-x0, 1)) #Shift-transformation

        #indg is the gamma lag indices, while indp is the polynomial lag indices
        indp = ix:Np
        indg = 1:length(indp)
        iy  = zero(T)
        idy = zero(T)

        #Perform the discreteized convolutions over the lag intervals
        for (ip, ig) in zip(indp, indg)
            poly  = substitute(polys[ip], ux)
            ∂poly = substitute(differential(polys[ip]), ux) #derivative of ux is 1
            polyterms = intervalpolyterms[ig]
            iy  +=  dot(poly.θ, polyterms)
            idy +=  dot(∂poly.θ, polyterms[SVector(1,2,3)])
        end
        fh.samples.y[ix]  = iy
        fh.samples.dy[ix] = idy 
    end
    #Set the final samples to zero, as this is the limit
    fh.samples.y[end]  = zero(T)
    fh.samples.dy[end] = zero(T)

    if update_pdf
        update!(fh.pdf, fh.samples)
    end
    if update_cdf
        integral!(fh.cdf, fh.pdf)
    end

    return fh
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
