using Plots
include(joinpath(@__DIR__, "__assembly.jl"))

xs  = 0.0:0.01:1.0

dN0 = Normal(0.9, 0.03)
dS0 = SplineDensity(CubicSpline(xs, pdf.(dN0, xs)), normalize=false)

#Normal Distribution Shift Test (isn't exact but should be approximate)
#=
dG1 = Gamma(10, 0.05)
dN1 = Normal(mean(dG1), std(dG1))

dN2 = Normal(mean(dN0)-mean(dN1), sqrt(var(dN0)+var(dN1)))
dS2 = random_var_subtract!(deepcopy(dS0), SplineConvolutionBasis(dS0, dG1))

plot(xs, pdf.(dN2,xs))
plot!(xs, pdf.(dS2,xs))
=#


#Repeated shift test, measures loss of information, should be very close
#dG1 = Gamma(1, 0.05)

N   = 100
dG1 = SplineConvolutionBasis(dS0, Gamma(10/N, 0.05))
dGN = SplineConvolutionBasis(dS0, Gamma(10, 0.05))

function subtract_n_times!(dH, dW, N)
    for ii in 1:(N)
        random_var_subtract!(dH, dW)
    end
    return dH
end

dSN = subtract_n_times!(deepcopy(dS0), dGN, 1)
@time dS1 = subtract_n_times!(deepcopy(dS0), dG1, N)

#@profview dS1 = subtract_n_times!(deepcopy(dS0), dG1, N);

plot(xs, pdf.(dSN, xs))
plot!(xs, pdf.(dS1, xs))
