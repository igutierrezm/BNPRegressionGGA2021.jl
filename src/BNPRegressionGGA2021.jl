module BNPRegressionGGA2021

using BayesNegativeBinomial
using Distributions
using HypergeometricFunctions
using LinearAlgebra
using Random
using SpecialFunctions
using StatsBase: mean_and_std
using Walker2020Sampling
import Distributions: pdf, logpdf

const BNB = BayesNegativeBinomial

# All DGPM models are subtypes of AbstractModel:
abstract type AbstractModel end

# All subtypes of AbstractModel must extend (by composition) the Skeleton type:
include("skeleton.jl")

# All subtypes of AbstractModel must implement the following interface:
include("interface.jl")

# The methods available for any AbstractModel are described here:
include("methods.jl")

# The specific models are implemented here:
include("dgsbpnormaldependent.jl")
include("dgsbpnormal.jl")
include("dgpmerlang.jl")
include("dgpmbeta.jl")

end
