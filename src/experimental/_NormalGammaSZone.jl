# =============================================================================================================================
# NormalGamma convolutions required for the CDF can be conveniently split into three zones
#   (1) The block zone (where we can just integrate the Normal component)
#   (2) The tail zone (where we can approximate the zone as a Normal PDF and integrate)
#   (3) The SZone (where we need to approximate either the Gamma CDF or Normal PDF as a polynomial and perform a convolution trick)
# Much of the code here revolves around proper selection of the SZone integration strategy in order of preference:
#   (1) Approximate the Normal PDF as a polynomial (as long as the Normal distribution isn't too skinny)
#   (2) Approximate the Gamma CDF as a polynomial (if the alpha term of the Gamma distribution is greater than 1)
#   (3) Perform a mixed approximation with small alpha (due to very sharp drop in the CDF function near zero)
#       (where the near-zero section approximates the Normal component, and the edge apprximates the Gamma)
# =============================================================================================================================


abstract type NormalGammaSZone{T} end

Base.@kwdef struct SZoneNormalPoly{T} <: NormalGammaSZone{T}
    Δs :: Tuple{T,T}
    β  :: Vector{T}
    dG :: SGamma{T}
    dN :: Normal{T}
end

Base.@kwdef struct SZoneGammaPoly{T} <: NormalGammaSZone{T}
    Δs :: Tuple{T,T}
    β  :: Vector{T}
    dG :: SGamma{T}
    dN :: Normal{T}
end

Base.@kwdef struct SZoneMixedPoly{T} <: NormalGammaSZone{T}
    Δs :: Tuple{T,T}
    si :: T
    aN :: SZoneNormalPoly{T}
    aG :: SZoneGammaPoly{T}
end

import StatsBase.standardize
const SQRT2 = sqrt(2)
standardize(d::Normal, x::Real)   = (x-d.μ)/(d.σ*SQRT2)
standardize(d::SGamma, x::Real)   = x/d.θ
unstandardize(d::Normal, x::Real) = x*d.σ*SQRT2 + d.μ
unstandardize(d::SGamma, x::Real) = x*d.θ

gamma_cdf_segments(dG::Gamma{T})  where T = (T(0), max(mean(dG),std(dG)))
gamma_cdf_segments(dG::SGamma{T}) where T = extrema(flipsign.(gamma_cdf_segments(Gamma(dG)), dG.θ))

piecewise_ccdf(d::NormalGamma, z::Real) = piecewise_cdf(mirror(d), -z)

function piecewise_cdf(d::NormalGamma, z::Real)
    #Initialize output 
    (dN, dG) = (d.normal, d.gamma)
    dNe = convolution_normal_equivalent(dN, z)

    #Segmentation
    Δs  = gamma_cdf_segments(dG)

    #Tail fitting parameters (which mirror a gaussian pdf)
    xc = flipsign(maximum(abs.(Δs)), dG.θ)
    (dNt, kt) = normal_gamma_conv_tail_approx(dNe, dG, xc)
        
    #Tail-zone integral
    log∫tail = dG.θ>0 ? logccdf(dNt, Δs[2]) : logcdf(dNt, Δs[1])
    ∫tail = exp(kt + log∫tail)

    #S-zone integral
    sZone  = build_szone(d, z)
    ∫szone = integrate(sZone)

    #Block-zone integral
    ∫block = ccdf(dNe, Δs[2])

    return ∫szone + ∫block - flipsign(∫tail, dG.θ)
end

function integrate(sZone::SZoneGammaPoly)
    (β, Δs, dNe) = (sZone.β, sZone.Δs, sZone.dN)
    sqrtπ = sqrt(π)
    Δz = standardize.(dNe, Δs)
    return poly_conv_gaussian(β, Δz)/sqrtπ
end

function integrate(sZone::SZoneNormalPoly)
    (β, Δs, dG) = (sZone.β, sZone.Δs, sZone.dG, sZone.dN)

    Δz = standardize.(dG, Δs)
    if dG.θ > 0
        return  poly_conv_scaled_lower_gamma(β, dG.α, Δz)*dG.θ
    else
        return  poly_conv_scaled_upper_gamma(β, dG.α, Δz)*dG.θ
    end
end

function integrate(sZone::SZoneMixedPoly)
    return integrate(sZone.aN) + integrate(sZone.aG)
end

function build_szone(d::NormalGamma, z::Real)
    (dN, dG) = (d.normal, d.gamma)
    dNe = convolution_normal_equivalent(dN, z)
    Δs  = gamma_cdf_segments(dG)

    #Approximate Normal distriubtion if the S-Zone is smaller than 8 standard deviations of dN
    if abs(Δs[2] - Δs[1]) < (dN.σ*8)
        return build_szone_normal_approx(dNe, dG, Δs)
    
    #Approximate Gamma over entire s-zone if α>1
    elseif dG.α > 1 
        return build_szone_gamma_approx(dNe, dG, Δs)

    #Otherwise use mixed approach
    else
        return build_szone_mixed_approx(dNe, dG, Δs)
    end
end

function build_szone_normal_approx(dNe::Normal, dG::SGamma, Δs::Tuple{<:Real,<:Real})
    T  = promote_type(eltype(dNe), eltype(dG))
    Δz = standardize.(dG, Δs)
    β  = fit_normal_pdf(dNe, Δz, x->unstandardize(dG, x))
    return SZoneNormalPoly{T}(Δs=Δs, β=β, dN=dNe, dG=dG)
end

function build_szone_gamma_approx(dNe::Normal, dG::SGamma, Δs::Tuple{<:Real,<:Real})
    T  = promote_type(eltype(dNe), eltype(dG))
    r  = 1.3
    Δsr = ifelse(dG.θ>0, (Δs[1],Δs[2]*r), (Δs[1]*r, Δs[2]) )
    Δz  = standardize.(dNe, Δsr)
    β  = fit_sgamma_cdf(dG, Δz, x->unstandardize(dNe,x))
    return SZoneGammaPoly{T}(Δs=Δs, β=β, dN=dNe, dG=dG)
end

function build_szone_mixed_approx(dNe::Normal{TN}, dG::SGamma{TG}, Δs::Tuple{<:Real,<:Real}) where {TN,TG}
    T  = promote_type(TN, TG)
    isPos = (dG.θ > 0) 

    #si  = 0.02*ifelse(isPos, Δs[2], Δs[1])     #Boundary based off gamma CDF
    si   = flipsign(dNe.σ/6, dG.θ)             #Boundary based off normal PDF

    #Make sure normal component isn't excessively skinny
    #dNe = Normal{TN}(dNe.μ, max(abs(si/8), dNe.σ))

    ΔsN = ifelse(isPos, (0.0, si), (si, 0.0))       #Innter (normal) segment
    ΔsG = ifelse(isPos, (si, Δs[2]), (Δs[1], si))   #Outer (gamma) segment

    aN = build_szone_normal_approx(dNe, dG, ΔsN)
    aG = build_szone_gamma_approx(dNe, dG, ΔsG)

    return SZoneMixedPoly{T}(Δs=Δs, si=si, aN=aN, aG=aG)
end


function normal_gamma_conv_tail_approx(dNe::Normal, dG::SGamma, xc::Real)
    δx  = min(dNe.σ, std(dG))
    Δx  = (dG.θ > 0) ? (xc, max(xc, dNe.μ)+δx) : (min(xc, dNe.μ)-δx, xc) 
    β = if dG.θ > 0 
        logf2gaussian(x-> logpdf(dNe,x) + logccdf(dG,x), Δx)
    else
        logf2gaussian(x-> logpdf(dNe,x) + logcdf(dG,x), Δx)
    end

    return expquad2normal(β)
end


function logf2gaussian(logf::Function, Δx::Tuple{<:Real,<:Real})
    a0 = -1e-3
    β = SVector{3}(fast_n_polynomial(logf, 3, Δx))

    if β[3] < a0 #Second order term must be negative for a Gaussian shape
        return β
    else #Second order is very flat, use a flat-ish distribution
        logf2(x) = logf(x) + a0*x^2
        return SVector{3}([fast_n_polynomial(logf, 2, Δx); a0])
    end
end

#Equivalent normal distribution of fe(x) = f(z-x)
function convolution_normal_equivalent(dN::Normal, z::Real)
    return Normal(z-dN.μ, dN.σ)
end


#Test code to produce the approximation
function normal_gamma_conv_approx(d::NormalGamma{T}, z::Real, vx::AbstractVector{<:Real}) where T<: Real
    #Initialize output 
    vy  = zero.(vx)
    (dN, dG) = (d.normal, d.gamma)
    dNe = convolution_normal_equivalent(dN, z)

    #Segmentation
    Δs  = gamma_cdf_segments(dG)

    #Tail fitting parameters (which mirror a gaussian pdf)
    xc = flipsign(maximum(abs.(Δs)), dG.θ)
    (dNt, kt) = normal_gamma_conv_tail_approx(dNe, dG, xc)
    
    #S-Segment fitting parameters
    sZone = build_szone(d, z)


    #Putting it all together
    for (ii, x) in enumerate(vx)
        if dG.θ > 0
            if x <= Δs[1]
                vy[ii] = 0
            elseif x <= Δs[2]
                vy[ii] = conv_approx(sZone, x)
            else
                vy[ii] = pdf(dNe,x) - exp(kt + logpdf(dNt, x))
            end
        else
            if x < Δs[1]
                vy[ii] = exp(kt + logpdf(dNt, x))
            elseif x <= Δs[2]
                vy[ii] = conv_approx(sZone, x)
            else
                vy[ii] = pdf(dNe, x)
            end
        end
    end
    
    return vy
end

#S-segment plots
function conv_approx(szone::SZoneNormalPoly, x::Real)
    (dG, β) = (szone.dG, szone.β)
    zi = standardize(dG, x)
    if dG.θ > 0
        return  cdf(Gamma(dG.α, 1), zi)*evalpoly(zi, β)
    else
        return ccdf(Gamma(dG.α, 1), zi)*evalpoly(zi, β)
    end
end

function conv_approx(szone::SZoneGammaPoly, x::Real)
    (dNe, β) = (szone.dN, szone.β)
    zi = standardize(dNe, x)
    return 1/(sqrt(2*π)*dNe.σ) * exp(-zi^2)*evalpoly(zi, β)
end

function conv_approx(szone::SZoneMixedPoly, x::Real)
    if szone.aN.Δs[1] <= x <= szone.aN.Δs[2]
        return conv_approx(szone.aN, x)
    else
        return conv_approx(szone.aG, x)
    end
end
