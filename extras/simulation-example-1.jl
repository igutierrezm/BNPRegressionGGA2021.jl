# Simulation example 1

using CSV
using DataFrames
using Cairo
using Revise
using BayesNegativeBinomial
using BNPRegressionGGA2021
using Distributions
using Gadfly
using Random
using Statistics
using Test
using LinearAlgebra

function simulate_sample(rng, N0, N1, D)
    dϵ = MixtureModel(Normal, [(1.0, 1.0), (-1.0, 0.25)], [0.5, 0.5])
    x0 = [-0.5, 0.5]
    X0 = [ones(N0) rand(rng, x0, N0, D)]
    y0 = X0 * [ones(2); zeros(D - 1)] + rand(rng, dϵ, N0)
    y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 4)
    x1 = kron(x0[[1, 1, 2, 2]], ones(N1))
    x2 = kron(x0[[1, 2, 1, 2]], ones(N1))
    X1 = [ones(4 * N1) x1 x2 zeros(4 * N1, D - 2)]
    return y0, X0, y1, X1
end

# Estimate the density for one realization
p = begin 
    N0, N1, D = 1000, 50, 10
    rng = MersenneTwister(1);
    y0, X0, y1, X1 = simulate_sample(rng, N0, N1, D);
    smpl = BNPRegressionGGA2021.NormalSampler(; y0, X0, y1, X1);
    chainf, chainβ = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 10000, burnin = 5000);

    fb = mean(chainf);
    plot(x = y0, color = X0[:, 2], Geom.histogram(density = true, bincount = 50))
    p = plot(x = y1, y = fb, color = string.(X1[:, 2]), xgroup = string.(X1[:, 3]),  Geom.subplot_grid(Geom.line))
    img = SVG("figures/fig01.svg", 14cm, 8cm)
    draw(img, p)
    p
end

# Estimate p(\beta[j] | y) for each j, averaged over 100 samples
function get_γchains(R = 1)
    N0, N1, D = 1000, 50, 10
    γchains = zeros(D + 1, 100)
    rng = MersenneTwister(1);
    for i in 1:R
        y0, X0, y1, X1 = simulate_sample(rng, N0, N1, D);
        smpl = BNPRegressionGGA2021.NormalSampler(; y0, X0, y1, X1);
        chainf, chainβ = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 5000, burnin = 2500);
        γchains[:, i] .= 1 .- mean([iszero.(chainβ[it]) for it in 1:length(chainβ)])
        println(i)
    end
    return γchains
end
γchains = get_γchains(100)
df = DataFrame(γchains', :auto)
CSV.write("data/data.csv", df)
