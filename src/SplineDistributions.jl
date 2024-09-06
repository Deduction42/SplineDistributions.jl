module SplineDistributions
    include(joinpath(@__DIR__,"__assembly.jl"))

    export 
        SplineDensity, 
        ShiftedGamma,
        SignedGamma,
        Polynomial,
        DualSample,
        Spline,
        CubicSpline,
        SplineIntegrals, 
        CubicIntegrals, 
        pdf,
        cdf,
        normalize!,
        random_var_subtract!
end
