@kwdef struct SplineConvolutionBasis{D<:Distribution, N, T<:Real}
    distribution :: D
    basis :: Vector{SVector{N,T}}
end

const CubicConvolutionBasis{D, T} = SplineConvolutionBasis{D, 4, T} where {D,T}




"""
construct a spline convolution basis
"""
function SplineConvolutionBasis(s::SplineDensity, d::Distribution)
    return SplineConvolutionBasis(s.pdf, d)
end

function SplineConvolutionBasis(s::Spline{N,T1}, dG::Gamma{T2}) where {N, T1, T2}
    T = promote_type(T1, T2)
    PolyType = polytype(CubicSpline)
    vx = s.vertices

    cdfBasis   = map(x->∫xᵏgammapdf_basis(PolyType, dG, x), vx)
    convBasis  = diff(cdfBasis)

    return SplineConvolutionBasis{Gamma, N, T}(dG, convBasis)
end


function random_var_subtract!(fh::SplineDensity{T}, fw::SplineConvolutionBasis{Gamma, 4}; update_pdf=true, update_cdf=true) where T
    vx = fh.pdf.vertices
    x0 = fh.pdf.vertices[1]
    vpoly = fh.pdf.segments
    basis = fw.basis
    

    #ix is the index on the x-axis to calculate the convolution for
    Np = length(vpoly)
    for ix in 1:Np
        ux = Polynomial{2}(SVector{2}(vx[ix]-x0, 1)) #Shift-transformation

        #indg is the gamma lag indices, while indp is the polynomial lag indices
        indp = ix:Np
        indg = 1:length(indp)
        iy  = zero(T)
        i∂y = zero(T)

        #Perform the discreteized convolutions over the lag intervals
        for (ip, ig) in zip(indp, indg)
            poly  = substitute(vpoly[ip], ux)
            ∂poly = substitute(differential(vpoly[ip]), ux) #derivative of ux is 1
            ibasis = basis[ig]
            iy  +=  dot(poly.θ, ibasis)
            i∂y +=  dot(∂poly.θ, ibasis[SVector(1,2,3)])
        end
        fh.samples.y[ix]  = iy
        fh.samples.∂y[ix] = i∂y 
    end
    #Set the final samples to zero, as this is the limit
    fh.samples.y[end]  = zero(T)
    fh.samples.∂y[end] = zero(T)

    if update_pdf
        update!(fh.pdf, fh.samples)
    end
    if update_cdf
        integral!(fh.cdf, fh.pdf)
    end
    return fh
end


#=
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
    return CubicSpline(_random_var_subtract(fh, fw))
end


function _random_var_subtract(fh::CubicSpline{T1}, fw::SplineConvolutionBasis{Gamma, 4, T2}) where {T1, T2}
    T = promote_type(T1, T2)
    vx = fh.vertices
    x0 = fh.vertices[1]
    vp = fh.segments
    basis = fw.basis
    
    vpdf  = zeros(T, length(vx))
    v∂pdf = zeros(T, length(vx))

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

    return DualSamples{T}(
        x = vx,
        y = vpdf,
        ∂y= v∂pdf
    )
end
=#





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
