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
    data = BNPRegressionGGA2021.Data(; y0, X0, y1, X1)
    pa = BNPRegressionGGA2021.Parameters(; N0, D0)
    tp = BNPRegressionGGA2021.TransformedParameters(; N0, D0)
    gq = BNPRegressionGGA2021.GeneratedQuantities(; N1)
    smpl = BNPRegressionGGA2021.Sampler(data)
    BNPRegressionGGA2021.update_suffstats!(smpl)
    BNPRegressionGGA2021.update_χ!(rng, smpl)
    BNPRegressionGGA2021.update_d!(rng, smpl)
    BNPRegressionGGA2021.update_r!(rng, smpl)
    BNPRegressionGGA2021.update_θ!(rng, smpl)
    # BNPRegressionGGA2021.update_β!(rng, smpl)
    BNPRegressionGGA2021.update_f!(smpl)
    smpl = BNPRegressionGGA2021.Sampler(data)
    BNPRegressionGGA2021.step!(rng, smpl)
    chain = BNPRegressionGGA2021.sample(rng, smpl)
end