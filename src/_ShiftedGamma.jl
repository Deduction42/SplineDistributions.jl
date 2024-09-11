include("_SignedGamma.jl")

# ===============================================================================
# Gamma distribution that is simply shifted/rotated 
# (special case where NormalGamma has a normal variance of 0)
# ===============================================================================
Base.@kwdef struct ShiftedGamma{T} <: Distribution{Univariate, Continuous}
    α   :: T
    θ   :: T
    x₀  :: T
    function ShiftedGamma{T}(α, θ, x₀) where T
        if (α <= 0)
            return error("α must be greater than 0")
        elseif iszero(θ)
            return error("θ must not be zero")
        else
            return new{T}(α, θ, x₀)
        end
    end
end

function ShiftedGamma(α::T1, θ::T2, x₀::T3) where {T1,T2,T3}
    T = promote_type(T1,T2,T3)
    return ShiftedGamma{T}(α, θ, x₀)
end

function ShiftedGamma(d::SignedGamma{T1}, x₀::T2) where {T1,T2}
    T = promote_type(T1,T2)
    return ShiftedGamma{T}(d.α, d.θ, x₀)
end

function ShiftedGamma(d::Gamma{T1}, x₀::T2) where {T1,T2}
    T = promote_type(T1,T2)
    return ShiftedGamma{T}(d.α, d.θ, x₀)
end

#Subtract the shifting value to allow interior CDF evaluation
shift(d::ShiftedGamma, x::Real) = (x-d.x₀)
scale(d::ShiftedGamma, x::Real) = (x-d.x₀)/d.θ
SignedGamma(d::ShiftedGamma) = SignedGamma(d.α, d.θ)

"""
Create a ShiftedGamma distribution by substituting a linear polynomial expressionf or x
"""
function substitute(d::Gamma{T1}, u::Polynomial{2,T2}) where {T1,T2}
    T  = promote_type(T1,T2)
    θ  = d.θ/u[1]
    x₀ = -u[0]/u[1]
    return ShiftedGamma{T}(d.α, θ, x₀)
end

"""
Creates a (Gamma distribution, linear polynomial) that when combined, produces the input distribution d
"""
function unsubstitute(d::ShiftedGamma)
    u1 = sign(d.θ)
    u0 = -d.x₀*u1
    return (Gamma(d.α, abs(d.θ)), Polynomial{2}(SVector(u0, u1)))
end


# ============================================================================================
# Moments functions
# ============================================================================================
mean(d::ShiftedGamma) = mean(SignedGamma(d)) + d.x₀
var(d::ShiftedGamma)  = var(SignedGamma(d))
std(d::ShiftedGamma)  = std(SignedGamma(d))
skewness(d::ShiftedGamma) = skewness(SignedGamma(d))
kurtosis(d::ShiftedGamma) = kurtosis(SignedGamma(d))

# ============================================================================================
# Density functions
# ============================================================================================
pdf(d::ShiftedGamma, x::Real)   = pdf(SignedGamma(d), shift(d,x))
cdf(d::ShiftedGamma, x::Real)   = cdf(SignedGamma(d), shift(d,x))
ccdf(d::ShiftedGamma, x::Real)  = ccdf(SignedGamma(d), shift(d,x))

logpdf(d::ShiftedGamma, x::Real)  = logpdf(SignedGamma(d), shift(d,x))
logcdf(d::ShiftedGamma, x::Real)  = logcdf(SignedGamma(d), shift(d,x))
logccdf(d::ShiftedGamma, x::Real) = logccdf(SignedGamma(d), shift(d,x))
