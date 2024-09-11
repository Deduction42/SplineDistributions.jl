include("_SplineDensity.jl")

@kwdef struct SplineConvolution{D<:Distribution, N, T<:Real}
    distribution :: D
    vertices :: StepRangeLen{Float64, Float64, Float64, Int64}
    samples  :: Vector{SVector{N,T}}
end

const CubicSplineConvolution{D, T} = SplineConvolution{D, 4, T} where {D,T}


"""
Construct polynomial integrals with pdf functions of d (∫ p*fd dx) 
Polynomial terms are based off the spline-based pdf of s
Integral samples are obtained at the vertices of s
"""
function SplineConvolution(s::SplineDensity, d::Distribution)
    return SplineConvolution(s.pdf, d)
end

"""
Construct polynomial integrals with pdf functions of d (∫ p*fd dx) 
Polynomial terms are based off the spline s
Integral samples are obtained at the vertices of s
"""
function SplineConvolution(s::Spline{N,T1}, dG::Gamma{T2}) where {N, T1, T2}
    T = promote_type(T1, T2)
    PolyType = polytype(CubicSpline)
    samples  = map(x->∫xᵏpdf_basis(PolyType, dG, x), s.vertices .- s.vertices[1])

    return SplineConvolution{Gamma, N, T}(dG, s.vertices, samples)
end

"""
Subtract a gamma random variable (fw) from a spline density function (fh) and overwrite the samples of fh
By default it also updates the pdf and the cdf. It does not normalize so the integral might be slightly less than 1.
"""
function convolve_minus!(fh::SplineDensity, fw::Gamma; update_pdf=true, update_cdf=true)
    ∫fw = SplineConvolution(fh, fw)
    return convolve_minus!(fh, ∫fw, update_pdf=update_pdf, update_cdf=update_cdf)
end

"""
Subtract the SplineConvolution of a gamma random variable (fw) from a spline density function (fh) and overwrite the samples of fh
SplineConvolution are used as a standin for the original distribution allowing for reuse, avoiding repeaded cdf calcualtions
By default it also updates the pdf and the cdf. It does not normalize so the integral might be slightly less than 1.
"""
function convolve_minus!(fh::SplineDensity{T}, fw::CubicSplineConvolution{Gamma}; update_pdf=true, update_cdf=true) where T
    if fh.pdf.vertices != fw.vertices
        error("SplineDensity and SplineIntegral must be evaluated at the same vertices")
    end

    vx = fh.pdf.vertices
    polys = fh.pdf.segments
    intervalpolyterms = diff(fw.samples) #Polynomial integral terms over the intervals (set at SplineConvolution)

    #ix is the index on the x-axis to calculate the convolution for
    Np = length(polys)
    for ix in 1:Np
        ux = Polynomial{2}(SVector{2}(vx[ix], 1)) #Shift-transformation

        #indg is the gamma lag indices, while indp is the polynomial lag indices
        indp = ix:Np
        indg = 1:length(indp)
        iy  = zero(T)
        i∂y = zero(T)

        #Perform the discreteized convolutions over the lag intervals
        for (ip, ig) in zip(indp, indg)
            poly  = substitute(polys[ip], ux)
            ∂poly = differential(poly)*ux[1] #Chain rule
            polyterms = intervalpolyterms[ig]
            iy  +=  dot(poly.θ, polyterms)
            i∂y +=  dot(∂poly.θ, polyterms[SVector(1,2,3)])
        end
        fh.samples.y[ix]  = iy
        fh.samples.dy[ix] = i∂y 
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
Add a gamma random variable (fw) to a spline density function (fh) (that starts at zero) and overwrite the samples of fh
By default it also updates the pdf and the cdf. It does not normalize so the integral might be slightly less than 1.
"""
function convolve!(fh::SplineDensity, fw::Gamma; update_pdf=true, update_cdf=true)
    ∫fw = SplineConvolution(fh, fw)
    return convolve!(fh, ∫fw, update_pdf=update_pdf, update_cdf=update_cdf)
end

"""
Add the SplineConvolution of (fw) from a spline density function (fh) (that starts at zero) and overwrite the samples of fh
SplineConvolution are used as a standin for the original distribution allowing for reuse, avoiding repeaded cdf calcualtions
By default it also updates the pdf and the cdf. It does not normalize so the integral might be slightly less than 1.
"""
function convolve!(fh::SplineDensity{T}, fw::CubicSplineConvolution{Gamma}; update_pdf=true, update_cdf=true) where T
    if fh.pdf.vertices != fw.vertices
        error("SplineDensity and SplineIntegral must be evaluated at the same vertices")
    end

    vx = fh.pdf.vertices
    polys = fh.pdf.segments
    intervalpolyterms = diff(fw.samples) #Polynomial integral terms over the intervals (set at SplineConvolution)

    #ix is the index on the x-axis to calculate the convolution for
    Np = length(polys)
    for ix in 1:Np
        ux = Polynomial{2}(SVector{2}(vx[ix], -1)) #Shift-transformation

        #indg is the gamma lag indices, while indp is the polynomial lag indices
        indp = ix:-1:1
        indg = 1:length(indp)
        iy  = zero(T)
        i∂y = zero(T)

        #Perform the discreteized convolutions over the lag intervals
        for (ip, ig) in zip(indp, indg)
            poly  =  substitute(polys[ip], ux)
            ∂poly =  differential(poly)*ux[1]
            polyterms = intervalpolyterms[ig]
            iy  +=  dot(poly.θ, polyterms)
            i∂y +=  dot(∂poly.θ, polyterms[SVector(1,2,3)])
        end
        fh.samples.y[ix+1]  = iy
        fh.samples.dy[ix+1] = i∂y 
    end

    #Set the final samples to zero, as this is the limit
    fh.samples.y[1]  = zero(T)
    fh.samples.dy[1] = zero(T)

    if update_pdf
        update!(fh.pdf, fh.samples)
    end
    if update_cdf
        integral!(fh.cdf, fh.pdf)
    end

    return fh
end

"""
Convolve xᵏ with a distribution d from 0 to x over all terms in Polynomial{N} 
Final integral can be obtained by multiplying corresponding terms with polynomial instance
"""
function ∫xᵏpdf_basis(::Type{Polynomial{N}}, d::Distribution, x) where N
    return map(k->∫xᵏpdf(k, d, x), powers(Polynomial{N}))
end


"Integrate xᵏ times gamma distribution dG from 0 to x"
function ∫xᵏpdf(k::Real, dG::Gamma, x::Real)
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
