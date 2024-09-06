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
        SplineConvolutions, 
        CubicConvolutions, 
        pdf,
        cdf,
        normalize!,
        convolution_minus!
end
