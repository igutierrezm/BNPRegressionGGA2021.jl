using LinearAlgebra

struct Data
    # Training
    y0::Vector{Float64}
    X0::Matrix{Float64}
    # Prediction
    y1::Vector{Float64}
    X1::Matrix{Float64}
end

Base.@kwdef struct Parameters
    N::Int
    D::Int
    # Main variables
    r::Vector{Int} = ones(Int, N)
    d::Vector{Int} = ones(Int, N)
    β::Vector{Float64} = zeros(D)
    μ::Vector{Float64} = zeros(N)
    τ::Vector{Float64} = ones(N)
    # Auxiliary variables
    n::Vector{Int} = zeros(Int, N)
    ȳ::Vector{Float64} = zeros(N)
    s::Vector{Float64} = zeros(N)
end

Base.@kwdef struct HyperParameters
    D::Int
    m0β::Vector{Float64} = zeros(D)
    Σ0β::Matrix{Float64} = 10 * I(D)
    m0μ::Float64 = 0.0
    c0μ::Float64 = 2.0
    a0τ::Float64 = 1.0
    b0τ::Float64 = 1.0
    a0θ::Float64 = 1.0
    b0θ::Float64 = 1.0
    s0r::Int = 2
end

Base.@kwdef struct GeneratedQuantities
    N::Int
    f::Vector{Float64} = zeros(N)
end

struct Sampler
    data::Data
    pa::Parameters
    hp::HyperParameters
    gq::GeneratedQuantities
    function Sampler(
        ytrain::Vector{Float64}, 
        Xtrain::Matrix{Float64}, 
        ypred::Vector{Float64}, 
        Xpred::Matrix{Float64}; 
        hp::HyperParameters
    )
        N1, D = size(Xpred)
        N0, D = size(Xtrain)
        pa = Parameters(N = N0, D = D)
        gq = GeneratedQuantities(N = N1)
        data = Data(ytrain, Xtrain, ypred, Xpred)
        new(data, pa, hp, gq)
    end
end