using BNPRegressionGGA2021
using Random
using Test

@testset "BNPRegressionGGA2021.jl" begin
    rng = MersenneTwister(1)
    N1, N0, D0 = 50, 100, 1
    y0 = randn(rng, N0)
    X0 = randn(rng, N0, D0)
    y1 = LinRange(-3, 3, N1) |> collect
    X1 = zeros(N1, D0)
    sampler = BNPRegressionGGA2021.Sampler(; y0, X0, y1, X1)
    BNPRegressionGGA2021.m(sampler)
    BNPRegressionGGA2021.update_suffstats!(sampler)
    BNPRegressionGGA2021.update_χ!(rng, sampler)
    BNPRegressionGGA2021.update_d!(rng, sampler)
    BNPRegressionGGA2021.update_r!(rng, sampler)
    BNPRegressionGGA2021.update_ϕ!(sampler)
    BNPRegressionGGA2021.update_β!(rng, sampler)
    BNPRegressionGGA2021.update_f!(sampler)
    sampler = BNPRegressionGGA2021.Sampler(; y0, X0, y1, X1)
    BNPRegressionGGA2021.step!(rng, sampler)
    chain = BNPRegressionGGA2021.sample(rng, sampler)
end
