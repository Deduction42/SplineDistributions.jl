# ===============================================================================
# Gamma distribution that is simply shifted/rotated 
# (special case where NormalGamma has a normal variance of 0)
# ===============================================================================
Base.@kwdef struct ShiftedGamma{T} <: Distribution{Univariate, Continuous}
    gamma :: SignedGamma{T}
    x₀    :: T
end

#Subtract the shifting value to allow interior CDF evaluation
shift(d::ShiftedGamma, x::Real) = (x-d.x₀)


# ============================================================================================
# Moments functions
# ============================================================================================
pdf(d::ShiftedGamma, x::Real)   = pdf(d.gamma, shift(d,x))
cdf(d::ShiftedGamma, x::Real)   = cdf(d.gamma, shift(d,x))
ccdf(d::ShiftedGamma, x::Real)  = ccdf(d.gamma, shift(d,x))

logpdf(d::ShiftedGamma, x::Real)  = logpdf(d.gamma, shift(d,x))
logcdf(d::ShiftedGamma, x::Real)  = logcdf(d.gamma, shift(d,x))
logccdf(d::ShiftedGamma, x::Real) = logccdf(d.gamma, shift(d,x))

# ============================================================================================
# Moments functions
# ============================================================================================
mean(d::ShiftedGamma) = d.x₀ + mean(d.gamma)
var(d::ShiftedGamma) = var(d.gamma)
std(d::ShiftedGamma) = std(d.gamma)
skewness(d::ShiftedGamma) = skewness(d.gamma)
kurtosis(d::ShiftedGamma) = kurtosis(d.gamma)
