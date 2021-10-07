using BNPRegressionGGA2021
using Random
using Test

@testset "BNPRegressionGGA2021.jl" begin
    rng = MersenneTwister(1)
    N0, N, D = 50, 100, 1
    y0 = randn(rng, N)
    X0 = randn(rng, N, D)
    y1 = LinRange(-3, 3, N0) |> collect
    X1 = zeros(N0, D)
    data = BNPRegressionGGA2021.Data(y0, X0, y1, X1)
    pa = BNPRegressionGGA2021.Parameters(; N, D)
    hp = BNPRegressionGGA2021.HyperParameters(; D)
    gq = BNPRegressionGGA2021.GeneratedQuantities(N = N0)
    smpl = BNPRegressionGGA2021.Sampler(y0, X0, y1, X1)
    BNPRegressionGGA2021.update_suffstats!(smpl)
    BNPRegressionGGA2021.update_χ!(rng, smpl)
    BNPRegressionGGA2021.update_d!(rng, smpl)
    BNPRegressionGGA2021.update_r!(rng, smpl)
    BNPRegressionGGA2021.update_θ!(rng, smpl)
    BNPRegressionGGA2021.update_w!(smpl)
    BNPRegressionGGA2021.update_f!(smpl)
    smpl = BNPRegressionGGA2021.Sampler(y0, X0, y1, X1)
    BNPRegressionGGA2021.step!(rng, smpl)
    chain = BNPRegressionGGA2021.sample(rng, smpl)
end
