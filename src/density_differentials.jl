include("_SplineConvolution.jl")

# =======================================================================================
# Derivatives of various density functions
# Required for creating DualSamples of distributions
# =======================================================================================
function dpdf(d::Gamma, x::Real)
    (α, θ) = (d.α, d.θ)
    pdfx = pdf(d,x)
    if (α <= 1)
        return ((α-1)*(θ/x)*pdfx - pdfx)/θ
    else
        return (pdf(Gamma(α-1, θ), x) - pdfx)/θ
    end
end

function dpdf(d::Normal, x::Real)
    z  = (x - d.μ)/d.σ
    dz = 1/d.σ
    u  = -0.5*(z)^2
    du = -z
    k  = 1/(sqrt(2*π)*d.σ)
    return k*exp(u)*du*dz
end

function dpdf(d::ShiftedGamma, x::Real)
    α  = d.α
    θ  = d.θ
    return flipsign(dpdf(Gamma(α, abs(θ)), flipsign(x-d.x₀, θ)), θ)
end

function dpdf(d::Spline{N,T0}, x::Real) where {T0, N}
    T = promote_type(T0, typeof(x))    
    return ifelse(inbounds(d, x), derivative(d,x), zero(T))
end

dpdf(d::SplineDensity, x::Real) = dpdf(d.pdf, x)




