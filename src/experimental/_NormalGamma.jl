# ===============================================================================
# NormalGamma represents the addition of a Gamma and Normal random variable
# ===============================================================================
Base.@kwdef struct NormalGamma{T}  <: Distribution{Univariate, Continuous}
    normal :: Normal{T}
    gamma  :: SignedGamma{T}
end

NormalGamma(d1::Normal, d2::Gamma) = NormalGamma(d1, SignedGamma(d2))
NormalGamma{T}(d1::Normal, d2::Gamma) where T = NormalGamma{T}(d1, SignedGamma(d2))
NormalGamma(d::ShiftedGamma) = NormalGamma(Normal(d.x₀,0.0), d.gamma)

# =============================================================================================
# Common distribution functions
# =============================================================================================
mean(d::NormalGamma) = mean(d.normal) + mean(d.gamma)
var(d::NormalGamma)  = var(d.normal) + var(d.gamma)
skewness(d::NormalGamma) = skewness(d.gamma)*(var(d.gamma)/var(d))^(3/2)
kurtosis(d::NormalGamma) = kurtosis(d.gamma)*(var(d.gamma)/var(d))^2


# ===============================================================================
# PDF evaluation based off quadrature points
# For repeated evaluations, it's best to calcualte the quadrature points once
# ===============================================================================
function pdf(d::NormalGamma, z::Real)
    if abs(skewness(d)) < 0.3 #Normal case, approximately symmetric
        dN = Normal(mean(d.normal) + mean(d.gamma), sqrt(var(d.normal)+var(d.gamma)))
        return pdf(dN, z)

    elseif iszero(d.normal.σ) #Gamma case, prevent breakage due to σ == 0
        dS = ShiftedGamma(d.gamma, d.normal.μ)
        return pdf(dS, z)

    else
        return normalgamma_pdf(d, z)
    end
end

#PDF has analytical solution
#https://stats.stackexchange.com/questions/330858/what-is-the-convolution-of-a-normal-distribution-with-a-gamma-distribution
function normalgamma_pdf(d::NormalGamma, s::Real)
    θ = d.gamma.θ

    if isinf(s)
        return zero(s)
    elseif iszero(θ)
        return pdf(d.normal, s)
    end

    #Flip the argument if θ is negative
    x = flipsign(s, θ)

    #Parameters
    α = d.gamma.α
    β = flipsign(1/θ, θ) #Ensyre it is positive
    μ = flipsign(d.normal.μ, θ) #μ flips sign if θ is negative
    σ = d.normal.σ

    innerTerm = (β*σ^2 + μ - x)
    hyperTerm = 0.5 * (innerTerm/σ)^2
    logScale  = -0.5*α*log(2) + α*log(β) + (α-2)*log(σ) - 0.5*((x-μ)/σ)^2

    logHyper1 = log₁F₁(0.5*α, 0.5, hyperTerm) - log(sqrt(2)) - loggamma(0.5*(α+1))
    logHyper2 = log₁F₁(0.5*(α+1), 1.5, hyperTerm) - loggamma(0.5*α)

    #Scale the difference so that neither term is infinite (scale is reversed later)
    dLogScale = max(logHyper1, logHyper2)
    hyperDiff = σ*exp(logHyper1-dLogScale) - innerTerm*exp(logHyper2-dLogScale)

    return exp(logScale + dLogScale)*max(0, hyperDiff)
end

#Confluent hypergeometric function logarithm
function log₁F₁(a,b,x)    
    if abs(x) <= 200
        rawVal = raw₁F₁(a,b,x)
        if rawVal >= 0
            return log(rawVal)
        else
            println("Warning: Confluent hypergeometric function failed at a=$(a), b=$(b), x=$(x)")
            return log₁F₁_asymptotic(a,b,x)
        end
    else #Hypergeometric function gets too big, use asymptotic form
        return log₁F₁_asymptotic(a,b,x)
    end
end

#Different tools to calculate hypergeometric functions
raw₁F₁(a,b,x) = _₁F₁(a,b,x)

#Asymptotic confluent hypergeometric function https://en.wikipedia.org/wiki/Confluent_hypergeometric_function
function log₁F₁_asymptotic(a,b,z)
    if z >= 0
        return loggamma(abs(b)) + z + (a-b)*log(z) - loggamma(abs(a))
    else
        return loggamma(abs(b)) - a*log(-z) - loggamma(abs(b-a))
    end
end



# ===============================================================================
# CDF/CCDF using the piecewise method (see _NormalGammaSZone.jl)
# ===============================================================================
cdf(d::NormalGamma, x::Real)  = piecewise_cdf(d, x)
ccdf(d::NormalGamma, x::Real) = piecewise_ccdf(d, x)

#Mirrored NormalGamma distribution
mirror(d::NormalGamma) = NormalGamma(Normal(-d.normal.μ, d.normal.σ), SignedGamma(d.gamma.α, -d.gamma.θ))


#It is sometimes useful to ensure the NormalGamma has no spikes by adding a small normal variance
function no_spike(d::NormalGamma{T}) where T 
    if d.gamma.α > 1
        return d
    else 
        return NormalGamma{T}(
            normal = Normal(d.normal.μ, max(0.01, d.normal.σ)),
            gamma  = d.gamma    
        )
    end
end



# ===============================================================================
# CDF/CCDF based off Simpson's Rule (old, but useful for validation)
# ===============================================================================
#=
#NormalGamma CDF using Simpson's rule, based on colvolving a Gamma CDF with a Normal PDF (no spike solution)
simpson_ccdf(d::NormalGamma, x::Real) = simpson_cdf(mirror(d), -x)

function simpson_cdf(d::NormalGamma{T}, z::Real) where T
    n  = 24
    (dN, dG) = (d.normal, d.gamma)
    (μ, σ) = getproperty.( dN, (:μ,:σ) )
    (α, θ) = getproperty.( dG, (:α,:θ) )

    #Shifted normal distribution
    dNz = Normal(z-μ, σ)

    #Integral function 
    fi(x) = cdf(dG,x)*pdf(dNz, x)

    #Generate sample ranges from both distributions
    ΔN = (quantile(dNz, 1e-9), quantile(dNz, 1-1e-9))
    ΔG = (θ < 0) ? (quantile(dG, 1e-9), 0.0) : (0.0, quantile(dG, 1-1e-9))

    #Generate samples, Normal distributions like even sampling, while Gamma CDFs work better with quantile-based sampling
    samples = sort!([LinRange(ΔN[1],ΔN[2],n); unique!(quantile_samples(dG, ceil(Int64, 0.75*n)))])

    if θ < 0
        #Integrate the right tail and remove samples
        ∫right = ccdf(dNz, 0)
        filter!(x-> x<=0, samples)
        ∫left = length(samples)<=2 ? zero(∫right) : simpson³⁸rule(fi, samples)
        return ∫left + ∫right
    else
        #Ignore the "zero zone" on the normal distribution
        xMax = min(ΔG[2], ΔN[2])
        ∫right = ccdf(dNz, xMax)
        filter!(x-> 0<=x<=xMax, samples)
        ∫left  = length(samples)<=1 ? zero(∫right) : simpson³⁸rule(fi, samples)
        return ∫left + ∫right
    end
end
=#


