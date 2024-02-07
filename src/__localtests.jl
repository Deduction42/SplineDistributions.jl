using Plots
include(joinpath(@__DIR__, "__assembly.jl"))

xs  = 0.0:0.01:1.0

dN0 = Normal(0.9, 0.03)
dS0 = SplineDensity(SplineSamples(xs, pdf.(dN0, xs)), normalize=false)

#Normal Distribution Shift Test (isn't exact but should be approximate)
#=
dG1 = Gamma(10, 0.05)
dN1 = Normal(mean(dG1), std(dG1))

dN2 = Normal(mean(dN0)-mean(dN1), sqrt(var(dN0)+var(dN1)))
dS2 = SplineDensity(random_var_subtract(dS0, dG1))

plot(xs, pdf.(dN2,xs))
plot!(xs, pdf.(dS2,xs))
=#


#Repeated shift test, measures loss of information, should be very close
#dG1 = Gamma(1, 0.05)
#=
N   = 10
dG1 = SplineConvolutionBasis(dS0, Gamma(10/N, 0.05))
dGN = Gamma(10, 0.05)


rS1 = Ref(random_var_subtract(dS0, dG1))
for ii in 1:(N-1)
    rS1[] = random_var_subtract(rS1[], dG1)
end
dS1 = rS1[]
dSN = random_var_subtract(dS0, dGN)

plot(xs, pdf.(dSN,xs))
plot!(xs, pdf.(dS1,xs))
=#