(@isdefined UNIFORM_REGRESSION) || (const UNIFORM_REGRESSION = Vector{Matrix{Float64}}())

const MAX_TERMS = 30

# ================================================================================
# Polynomial approximations of Normal PDFs and Gamma CDFs
# ================================================================================

#Polynomial parameter vector for standard Normal distribution, focusing on range Δz
function fit_normal_pdf(dN::Normal, Δz::Tuple{<:Real,<:Real}, ft::Function = x->x)
    #Max points if range is 8σ
    Δs = ft.(Δz)
    n  = ceil(Int64, (30/(dN.σ*8))*abs(Δs[2]-Δs[1]))
    return fast_n_polynomial(x->pdf(dN, ft(x)), clamp(n,4,MAX_TERMS), Δz)
end

#Polynomial parameter vector for standard Gamma cdf/ccdf, focusing on range Δz
function fit_sgamma_cdf(dG::SGamma, Δz::Tuple{<:Real,<:Real}, ft::Function = x->x)
    #Find suitable number of polynomial terms and then fit
    Δs = ft.(Δz)
    n  = gamma_cdf_poly_num(dG, Δs) 
    return fast_n_polynomial(x->cdf(dG, ft(x)), clamp(n,4,MAX_TERMS), Δz)
end

function fit_sgamma_ccdf(dG::SGamma, Δz::Tuple{<:Real,<:Real}, ft::Function = x->x)
    #Find suitable number of polynomial terms and then fit
    n  = gamma_cdf_poly_num(dG, ft.(Δz)) 
    return fast_n_polynomial(x->ccdf(dG, ft(x)), clamp(n,4,MAX_TERMS), Δz)
end

#Number of polynomial terms for the cdf or ccdf
function gamma_cdf_poly_num(dG::SGamma, Δs::Tuple{<:Real,<:Real})
    if dG.α < 1 #K times the ratio difference for the ccdf
        r  = 5*ccdf(dG, Δs[1])/ccdf(dG, Δs[2])
        n0 = r*dG.α^(-0.4)
        return ceil(Int64, min(MAX_TERMS, n0))

    else #Max points range is greater than 3α
        n0 = 30*abs(Δs[2]-Δs[1])/(4*std(dG))
        return ceil(Int64, min(MAX_TERMS, n0))
    end
end


# ================================================================================
# Conversions between Normal Distribtions and Exponents of quadratic polynomials
# ================================================================================
function normal2expquad(dN::Normal)
    r2π = √(2π)
    k = -1/(2*dN.σ^2)
    β = k .* SA[dN.μ^2-log(dN.σ*r2π)/k, -2*dN.μ, 1]
    return β
end

function expquad2normal(β::SVector{3,<:Real})
    (a, b, c) = (β[3], β[2], β[1])
    r2π = √(2*π)

    σ  = sqrt(-1/(2*a))
    μ  = b*σ^2
    k  = 0.5*(μ/σ)^2 + c + log(σ*r2π)
    return (Normal(μ,σ), k)
end

# =================================================================================
# General fast polynomial regression methods
# =================================================================================

#Fast polynomial parameters for function f evaluated at n evenly-spaced intervals in Δx
function fast_n_polynomial(f::Function, n::Integer, Δx::Tuple{<:Real,<:Real})
    vf = [f(x) for x in LinRange(Δx[1], Δx[2], n)]
    return uniform_polynomial_regression(vf, Δx)
end


function uniform_polynomial_regression(vf::AbstractVector{<:Real}, Δx::Tuple{<:Real,<:Real})
    #Regression based off (x-xL)/(xU-xL) = (x-xL)/s
    n  = length(vf)
    s  = Δx[2]-Δx[1]
    ϕ  = UNIFORM_REGRESSION[n]*vf

    #Scale β0 so that we have (x-xL)
    sᵏ = one(s)
    for ii in eachindex(ϕ)
        ϕ[ii] = ϕ[ii]/sᵏ
        sᵏ = sᵏ*s
    end

    #Calculate reference values to subtract
    xL = polynomial_vector(-Δx[1], n)

    #Use Pascal's triangle to collect terms and produce unscaled parameters β
    pascalRow = ones(n)
    β = copy(ϕ)
    for ii in 1:n
        βii = zero(β[ii])
        Δii = ii-1

        for jj in 1:(n-Δii)
            βii = βii + pascalRow[jj]*ϕ[jj+Δii]*xL[jj]
        end
        β[ii] =  βii 
        pascal_iteration!(pascalRow, n-ii)
    end
    
    return β
end


function polynomial_matrix(x::AbstractVector, n)
    if n <= 1
        return one.(x)
    end

    X = ones(length(x),n)
    X[:,2] = x

    for c in 3:n
        for (r, xr) in enumerate(x)
            X[r,c] = X[r,c-1]*xr
        end
    end

    return X
end


function polynomial_vector(x::Real, n::Integer)
    if n <= 1
        return one.(x)
    else
        return pushfirst!(cumprod(fill(x,n-1)), 1)
    end
end


function pascals_triangle(n::Int)
    U = UpperTriangular{Int64}(ones(n,n))
    for c in 2:n
        U[2:c,c] .= @views U[1:(c-1), c-1] .+ U[2:c, c-1]
    end
    return U
end


function pascal_iteration!(x::AbstractVector{<:Real})
    return pascal_iteration!(x, lastindex(x))
end

function pascal_iteration!(x::AbstractVector{<:Real}, n::Integer)
    xk0 = zero(x[begin])
    for k in firstindex(x):min(lastindex(x),n)
        x[k] = x[k] + xk0
        xk0  = x[k]
    end
    return x
end




#Fill out the polynomial matrix

for ii in 1:MAX_TERMS
    M = if (ii == 1)
        [1.0;;]
    else
        local x, X, R
        x = LinRange(0,1,ii)
        X = polynomial_matrix(x, ii)
        R = X'*X
        Hermitian(0.5*(R+R'))\X'
    end
    if length(UNIFORM_REGRESSION) >= ii
        UNIFORM_REGRESSION[ii] = M
    else
        push!(UNIFORM_REGRESSION, M)
    end
end

