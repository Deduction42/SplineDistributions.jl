using Revise
using SplineDistributions
using Distributions
using Plots

xs  = 0.0:0.01:1.0

dN0 = Normal(0.9, 0.03)
dS0 = SplineDensity(CubicSpline(xs, pdf.(dN0, xs)), normalize=false)

sanity_test = false
consistency_test = true

#Normal Distribution Shift Test (isn't exact but should be approximate)
if sanity_test
    dG1 = Gamma(10, 0.05)
    dN1 = Normal(mean(dG1), std(dG1))

    @time dN2 = Normal(mean(dN0)-mean(dN1), sqrt(var(dN0)+var(dN1)))
    @time dS2 = convolution_minus!(deepcopy(dS0), SplineConvolutions(dS0, dG1))

    plot(xs, pdf.(dN2,xs))
    plot!(xs, pdf.(dS2,xs))
end


#Repeated shift test, measures loss of information, should be very close
#dG1 = Gamma(1, 0.05)

if consistency_test
    N   = 1000
    dG1 = SplineConvolutions(dS0, Gamma(10/N, 0.05))
    dGN = SplineConvolutions(dS0, Gamma(10, 0.05))

    function subtract_n_times!(dH, dW, N)
        for ii in 1:(N)
            convolution_minus!(dH, dW)
        end
        return dH
    end

    dSN = subtract_n_times!(deepcopy(dS0), dGN, 1)
    @time dS1 = subtract_n_times!(deepcopy(dS0), dG1, N-1)
    @profview subtract_n_times!(deepcopy(dS0), dG1, N)

    plot(xs, pdf.(dSN, xs))
    plot!(xs, pdf.(dS1, xs))
end