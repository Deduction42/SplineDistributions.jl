
#Lower and upper gamma functions with better handling of large numbers
gamma_l(α,x) = exp(loggamma(α) + log(gamma_inc(α,x)[1]))
gamma_u(α,x) = exp(loggamma(α) + log(gamma_inc(α,x)[2]))

function poly_conv_scaled_lower_gamma(β::Vector{<:Real}, α::Real, Δz::Tuple{<:Real,<:Real})
    logScale = -loggamma(α)

    #Fast version of stepping through lower gamma 
    Γα  = 1
    γ1 = stepped_lower_gamma(α, Δz[1], length(β)+1, logscale=logScale)
    γ2 = stepped_lower_gamma(α, Δz[2], length(β)+1, logscale=logScale)

    #Slower version to verify faster version
    #Γα = exp(-logScale)
    #γ1 = [ gamma_l(α+k, Δz[1]) for k in 0:length(β) ]
    #γ2 = [ gamma_l(α+k, Δz[2]) for k in 0:length(β) ]

    vk = 1:length(β)
    ∫γ = (Δz[2].^vk.*γ2[1] .- Δz[1].^vk.*γ1[1])
    ∫γ .+= @views γ1[2:end] .-  γ2[2:end]
    ∫γ .= ∫γ ./ vk

    return dot(β, ∫γ)/Γα
end

function poly_conv_scaled_upper_gamma(β::Vector{<:Real}, α::Real, Δz::Tuple{<:Real,<:Real})
    logScale = -loggamma(α)

    #Fast version of stepping through lower gamma 
    Γα  = 1
    γ1 = stepped_lower_gamma(α, Δz[1], length(β)+1, logscale=logScale)
    γ2 = stepped_lower_gamma(α, Δz[2], length(β)+1, logscale=logScale)

    #Slower version to verify faster version
    #Γα = exp(-logScale)
    #γ1 = [ gamma_l(α+k, Δz[1]) for k in 0:length(β) ]
    #γ2 = [ gamma_l(α+k, Δz[2]) for k in 0:length(β) ]

    vk  = 1:length(β)
    ∫γ  = Δz[2].^vk.*(Γα-γ2[1]) .- Δz[1].^vk.*(Γα-γ1[1])
    ∫γ  .+= @views γ2[2:end] .- γ1[2:end]
    ∫γ  .= ∫γ ./ vk

    return dot(β, ∫γ)/Γα
end


function poly_conv_gaussian(β::Vector{<:Real}, Δz)
    #Fast version of stepping through lower gamma 
    γ1 = stepped_lower_gamma(0.5, Δz[1]^2, length(β), step_denom=2)
    γ2 = stepped_lower_gamma(0.5, Δz[2]^2, length(β), step_denom=2)

    #Slower version to verify faster version
    #γ1 = [ gamma_l(0.5*k, Δz[1]^2)[1] for k in 1:length(β) ]
    #γ2 = [ gamma_l(0.5*k, Δz[2]^2)[1] for k in 1:length(β) ]

    vk = 1:length(β)
    ∫k = 0.5.*((sign(Δz[2]).^vk).*γ2 .- (sign(Δz[1]).^vk).*γ1)

    return dot(β, ∫k)
end



function stepped_lower_gamma(α⁰::Real, x::Real, n::Integer; step_denom=1, logscale=0)
    γₙ   = zeros(promote_type(typeof(x),Float64), n)
    αₙ   = α⁰ .+ (0:(n-1))./step_denom
    logx = log(x)

    #Initialize the gamma function
    scaledgamma(ii,x) = exp(logscale + loggamma(αₙ[ii]) + log(gamma_inc(αₙ[ii],x)[1]))
    for ii in n:-1:(n-step_denom)
        γₙ[ii] = scaledgamma(ii,x)
    end

    #Build recursive lower gamma function (from the end first)
    for i0 in (n-step_denom-1):-1:1
        i1 = i0+step_denom
        γₙ[i0] = (γₙ[i1] + exp(αₙ[i0]*logx + logscale - x))/αₙ[i0]
    end

    return γₙ
end

# ===========================================================================================================
# Legacy code specifically for cubic polynomials convolved with Gamma distributions
# ===========================================================================================================
#Convolve Gamma distribution with cubic approximation of "f" over the interval Δx
function cubic_gamma_conv(f::Function, dG::SGamma, Δx::Tuple{Real,Real})
    p = fast_cubic_approx(f, Δx)
    return cubic_gamma_conv(p, dG, Δx)
end

#Gamma cubic convolution from parameters
function cubic_gamma_conv(p::NamedTuple{(:a,:b,:c,:d)}, dG::SGamma, Δx::Tuple{Real,Real})
    terms = polynomial_gamma_conv.(3:-1:0, dG, Ref(Δx))
    return sum(values(p).*terms)
end

#Convolve x^k with a gamma distribution over Δx
function polynomial_gamma_conv(k, dG::Union{SGamma,Gamma}, Δx::Tuple{Real,Real})
    return polynomial_gamma_conv(k, dG, Δx[2]) - polynomial_gamma_conv(k, dG, Δx[1])
end

#Convolve x^k with a gamma distribution from 0 to x
function polynomial_gamma_conv(k, dG::Union{SGamma,Gamma}, x::Real)
    (α, θ) = (dG.α, dG.θ)
    gammaConv = exp(k*log(abs(θ)) + loggamma(α+k)-loggamma(α)) * cdf(SGamma(α+k,θ), x)
    
    sgn = ifelse(isodd(k), sign(θ), sign(abs(θ)))
    return flipsign(gammaConv, sgn)
end

#Fast cubic approximations (that can be applied to a gamma function)
function cubic_function(p::NamedTuple{(:a,:b,:c,:d)}, x::Real)
    return p.a*x^3 + p.b*x^2 + p.c*x + p.d 
end

function fast_cubic_approx(f::Function, Δx::Tuple{<:Real,<:Real})
    s = 1/(Δx[2]-Δx[1]) # scale factor
    r = Δx[1]    
    p = UNIFORM_REGRESSION[4]*f.(LinRange(Δx[1],Δx[2],4))


    a = s^3*(    p[1] )
    b = s^2*( -3*p[1]*r   +   p[2])
    c =   s*(  3*p[1]*r^2 - 2*p[2]*r   + p[3])
    d =     (   -p[1]*r^3 +   p[2]*r^2 - p[3]*r  + p[4])

    return (a=a, b=b, c=c, d=d)
end







#Test lower gamma polynomial convolution
#=
k = 1:length(β)

pf(x::Real, k::Integer)   = β[k]*x^(k-1)*gamma_inc(α, x)[1]
∫pf0(x::Real, k::Integer) = 1/(k*exp(loggamma(α)))*(x^k*gamma_l(α, x) - gamma_l(α+k, x))
∫pf(x::Real, k::Integer)  = β[k]/(k*exp(loggamma(α)))*(x^k*gamma_l(α, x) - gamma_l(α+k, x))


vx  = collect(LinRange(Δx[1], Δx[2], 100))

∫f1 = simpson³⁸rule(x->sum(pf.(x, k)), vx)
∫f2 = sum( ∫pf.(Δx[2], k) .- ∫pf.(Δx[1], k))
∫f3 = poly_conv_scaled_lower_gamma(β, α, Δx)
vx  = collect(LinRange(Δx[1], Δx[2], 100))
plot(vx, sum(pf.(vx, k'), dims=2))
display((∫f1, ∫f2, ∫f3))
=#




#α = 0.5
#β = one.(k)
#Δx  = (0,2)


#Test upper gamma polynomial convolution
#=
k = 1:length(β)

pf(x::Real, k::Integer)   = β[k]*x^(k-1)*gamma_inc(α, x)[2]
∫pf0(x::Real, k::Integer) = 1/(k*exp(loggamma(α)))*(x^k*gamma_u(α, x) - gamma_u(α+k, x))
∫pf(x::Real, k::Integer)  = β[k]/(k*exp(loggamma(α)))*(x^k*gamma_u(α, x) - gamma_u(α+k, x))


vx  = collect(LinRange(Δx[1], Δx[2], 100))

∫f1 = simpson³⁸rule(x->sum(pf.(x, k)), vx)
∫f2 = sum( ∫pf.(Δx[2], k) .- ∫pf.(Δx[1], k))
∫f3 = poly_conv_scaled_upper_gamma(β, α, Δx)
vx  = collect(LinRange(Δx[1], Δx[2], 100))
plot(vx, sum(pf.(vx, k'), dims=2))
display((∫f1, ∫f2, ∫f3))
=#


#Test gaussain polynomial convolution
#=
#β = ones(16)
#Δx  = (-2,5)

k = 1:length(β)
pf(x::Real,  α::Integer)  = β[α]*x^(α-1)*exp(-x^2)
∫pf(x::Real, α::Integer)  = β[α]*0.5*(sign(x)^α)*gamma_l(α/2, x^2)



vx  = collect(LinRange(Δx[1], Δx[2], 100))

∫f1 = simpson³⁸rule(x->sum(pf.(x, k)), vx)
∫f2 = sum( ∫pf.(Δx[2], k) .- ∫pf.(Δx[1], k) )
∫f3 = poly_conv_gaussian(β, Δx)

plot(vx, sum(pf.(vx, k'), dims=2))
display((∫f1, ∫f2, ∫f3))
=#