using SpecialFunctions
using Distributions
using LinearAlgebra
using Interpolations
using StaticArrays

import Distributions.pdf
import Distributions.cdf
import Distributions.ccdf
import Distributions.logpdf
import Distributions.logcdf
import Distributions.logccdf
import Distributions.mean
import Distributions.var
import Distributions.std
import Distributions.skewness
import Distributions.kurtosis

include(joinpath(@__DIR__,"_SignedGamma.jl"))
include(joinpath(@__DIR__,"_ShiftedGamma.jl"))
include(joinpath(@__DIR__,"_Polynomial.jl"))
include(joinpath(@__DIR__,"_DualSamples.jl"))
include(joinpath(@__DIR__,"_Spline.jl"))
include(joinpath(@__DIR__,"_SplineDensity.jl"))
include(joinpath(@__DIR__,"_SplineIntegrals.jl"))