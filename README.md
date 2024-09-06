# SplineDistributions
This package enables the user to represent a distribution as a cubic spline (a set of cubic polynomials) density function. The main benefit of doing this is to build a flexible basis that can represent multiple types of distributions with little loss of information (much lower than histograms) while allowing for easier convolutions and integrals. This package is an extension to Distributions.jl and at maturity is expected to follow its API.  

A SplineDensity object can be constructed as follows:

Step (1): Define your sampling basis
```
xs  = 0.0:0.01:1.0
```

Step (2): Define your sampled values
```
dN0 = Normal(0.9, 0.03)
fs  = pdf.(dN0, xs)
```

Step (3): Build your distribution from a cubic spline
```
dS0 = SplineDensity(CubicSpline(xs, fs), normalize=false)
```
