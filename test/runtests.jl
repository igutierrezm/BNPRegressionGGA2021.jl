using BNPRegressionGGA2021
using Random
using Test

const DGSBP = BNPRegressionGGA2021

@testset "Skeleton - constructor" begin
    N1, N0, D0 = 2, 10, 2
    Random.seed!(1)
    y0 = randn(N0)
    y1 = randn(N1)
    X0 = randn(N0, D0)
    X1 = zeros(N1, D0)
    skl = DGSBP.Skeleton(; y0, X0, y1, X1)
end

@testset "Skeleton - accessors" begin
    N1, N0, D0 = 2, 10, 2
    Random.seed!(1)
    y0 = randn(N0)
    y1 = randn(N1)
    X0 = randn(N0, D0)
    X1 = zeros(N1, D0)
    skl = DGSBP.Skeleton(; y0, X0, y1, X1)
    @test DGSBP.ygrid(skl) === y1
    @test DGSBP.Xgrid(skl) === X1
    @test DGSBP.Ngrid(skl) == N1
    @test DGSBP.y(skl) === y0
    @test DGSBP.X(skl) === X0
    @test DGSBP.N(skl) == N0
    @test DGSBP.d(skl) == ones(N0)
    @test DGSBP.r(skl) == ones(N0)
    @test DGSBP.f(skl) == zeros(N1)
    @test DGSBP.rmax(skl) == 1
end

@testset "DGSBPNormal - constructor" begin
    N1, N0, D0 = 2, 10, 2
    Random.seed!(1)
    y0 = randn(N0)
    y1 = randn(N1)
    X0 = randn(N0, D0)
    X1 = zeros(N1, D0)
    skl = DGSBP.Skeleton(; y0, X0, y1, X1)
    m = DGSBP.DGSBPNormal(; skl)
end

@testset "DGSBPNormal - methods" begin
    N1, N0, D0 = 2, 10, 2
    Random.seed!(1)
    y0 = randn(N0)
    y1 = randn(N1)
    X0 = randn(N0, D0)
    X1 = zeros(N1, D0)
    skl = DGSBP.Skeleton(; y0, X0, y1, X1)
    m = DGSBP.DGSBPNormal(; skl)
    @test DGSBP.skeleton(m) === skl
    @test DGSBP.kernel_pdf(m, 0.0, 1) ≈ 1 / √(2 * π)
    @test_throws MethodError DGSBP.kernel_pdf(m, 0, 1)
end

# @testset "Skeleton.jl" begin
#     N1, N0, D0 = 2, 10, 2
#     Random.seed!(1)
#     y0 = randn(N0)
#     X0 = randn(N0, D0)
#     y1 = LinRange(-3, 3, N1) |> collect
#     X1 = zeros(N1, D0)

#     # sampler = BNPRegressionGGA2021.Sampler(; y0, X0, y1, X1)
#     # BNPRegressionGGA2021.m(sampler)
#     # BNPRegressionGGA2021.update_suffstats!(sampler)
#     # BNPRegressionGGA2021.update_χ!(rng, sampler)
#     # BNPRegressionGGA2021.update_d!(rng, sampler)
#     # BNPRegressionGGA2021.update_r!(rng, sampler)
#     # BNPRegressionGGA2021.update_ϕ!(sampler)
#     # BNPRegressionGGA2021.update_β!(rng, sampler)
#     # BNPRegressionGGA2021.update_f!(sampler)
#     # sampler = BNPRegressionGGA2021.Sampler(; y0, X0, y1, X1)
#     # BNPRegressionGGA2021.step!(rng, sampler)
#     # chain = BNPRegressionGGA2021.sample(rng, sampler)
# end
