using Distributions

#Signed Gamma distribution that allows for θ values to be negative
struct SignedGamma{T} <: Distribution{Univariate, Continuous}
    α :: T
    θ :: T
    SignedGamma{T}(α, θ) where T = if (α <= 0)
        error("α must be greater than 0")
    elseif iszero(θ)
        error("θ must not be zero")
    else
        new()
    end
end

import Distributions.Gamma
SignedGamma(d::Gamma) = SignedGamma(d.α, d.θ)
Gamma(d::SignedGamma) = Gamma(d.α, abs(d.θ))

# ============================================================================================
# PDF/CDF functions
# ============================================================================================
pdf(d::SignedGamma, x::Real)    = pdf(Gamma(d), flipsign(x, d.θ))
logpdf(d::SignedGamma, x::Real) = logpdf(Gamma(d), flipsign(x, d.θ))

function cdf(d::SignedGamma, x::Real)
    dG = Gamma(d)
    return d.θ >= 0 ? cdf(dG, x) : ccdf(dG, -x)
end

function logcdf(d::SignedGamma, x::Real)
    dG = Gamma(d)
    return d.θ >= 0 ? logcdf(dG, x) : logccdf(dG, -x)
end

function ccdf(d::SignedGamma, x::Real)
    dG = Gamma(d)
    return d.θ >= 0 ? ccdf(dG, x) : cdf(dG, -x)
end

function logccdf(d::SignedGamma, x::Real)
    dG = Gamma(d)
    return d.θ >= 0 ? logccdf(dG, x) : logcdf(dG, -x)
end

# ============================================================================================
# Moments functions
# ============================================================================================
mean(d::SignedGamma) = d.α*d.θ
var(d::SignedGamma)  = d.α*d.θ^2
std(d::SignedGamma)  = sqrt(d.α)*abs(d.θ)
skewness(d::SignedGamma) = flipsign(2/sqrt(d.α), d.θ)
kurtosis(d::SignedGamma) = 6/d.α
