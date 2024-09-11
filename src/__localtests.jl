using Plots
include(joinpath(@__DIR__, "__assembly.jl"))

xs  = 0.0:0.01:1.0

dN0 = Normal(0.9, 0.03)
dS0 = SplineDensity(CubicSpline(xs, pdf.(dN0, xs)), normalize=false)

sanity_test = false
consistency_test = true
substitute_test = false

#Normal Distribution Shift Test (isn't exact but should be approximate)
if sanity_test
    xs  = 0.0:0.01:1.0
    dN0 = Normal(0.9, 0.03)
    dS0 = SplineDensity(CubicSpline(xs, pdf.(dN0, xs)), normalize=false)
    dG1 = Gamma(10, 0.05)
    dN1 = Normal(mean(dG1), std(dG1))

    @time dN2 = Normal(mean(dN0)-mean(dN1), sqrt(var(dN0)+var(dN1)))
    @time dS2 = convolve_minus!(deepcopy(dS0), SplineConvolution(dS0, dG1))

    #@time dN2 = Normal(mean(dN0)+mean(dN1), sqrt(var(dN0)+var(dN1)))
    #@time dS2 = convolve!(deepcopy(dS0), SplineConvolution(dS0, dG1))

    vx = 0.0:0.001:1.0 #Smaller sample size to reveal derivative oddities
    plot(vx, pdf.(dN2,vx))
    plot!(vx, pdf.(dS2,vx))
end


#Repeated shift test, measures loss of information, should be very close
#dG1 = Gamma(1, 0.05)

if consistency_test
    N   = 1000
    dN0 = Normal(0.9, 0.03)
    dS0 = SplineDensity(CubicSpline(xs, pdf.(dN0, xs)), normalize=false)
    dG1 = SplineConvolution(dS0, Gamma(10/N, 0.05))
    dGN = SplineConvolution(dS0, Gamma(10, 0.05))

    function subtract_n_times!(dH, dW, N)
        for ii in 1:(N)
            convolve_minus!(dH, dW)
        end
        return dH
    end

    dSN = subtract_n_times!(deepcopy(dS0), dGN, 1)
    @time dS1 = subtract_n_times!(deepcopy(dS0), dG1, N)
    #@profview subtract_n_times!(deepcopy(dS0), dG1, N)

    plot(xs, pdf.(dSN, xs), label="$(N) steps")
    plot!(xs, pdf.(dS1, xs), label="single step")
end

if substitute_test
    dObs = SplineDensity(CubicSpline(xs, ccdf.(Gamma(2,0.1), xs)))
    u = Polynomial{2}(SVector(1.0,-1))
    dObsu = substitute(dObs, u)
    plot(xs, pdf.(dObs, xs))
    plot!(xs, pdf.(dObsu, xs))
end
