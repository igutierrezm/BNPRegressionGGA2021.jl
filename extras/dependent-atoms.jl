begin
    using Distributions
    using LinearAlgebra
    using Random 
end;
 
Base.@kwdef struct Foo
    # Data
    y::Vector{Float64}
    X::Matrix{Float64}
    d::Vector{Int}
    # Transformed data
    N::Int = size(X, 1)
    K::Int = size(X, 2)
    J::Int = length(unique(d))
    Xvec::Vector{Vector{Float64}} = [X[i, :] for i in 1:N]
    # Hyperparameters
    B0::Symmetric{Float64, Matrix{Float64}} = Symmetric(9.0 * I(K))
    m0::Vector{Float64} = zeros(K)
    a0::Float64 = 0.01
    b0::Float64 = 0.01
    # Parameters 
    β::Vector{Vector{Float64}} = [zeros(K) for _ in 1:J]
    v::Vector{Float64} = [1.0 for _ in 1:J]
    # Transformed parameters 
    n::Vector{Int} = [sum(d .== j) for j in 1:J]
    XX::Vector{Matrix{Float64}} = [zeros(K, K) for _ in 1:J]
    Xy::Vector{Vector{Float64}} = [zeros(K) for _ in 1:J]
    yy::Vector{Float64} = [0.0 for _ in 1:J]
end;

function step!(model::Foo)
    (; N, J, d, y, Xvec, β, v, m0, B0, a0, b0, XX, Xy, yy, n) = model
    # Refresh XX and Xy
    for j in 1:J
        XX[j] .= 0.0
        Xy[j] .= 0.0
        yy[j] = 0.0
    end
    @inbounds for i in 1:N
        BLAS.syr!('L', 1.0, Xvec[i], XX[d[i]]) # XX[d[i]] += x[i] * x[i]'
        BLAS.axpy!(y[i], Xvec[i], Xy[d[i]])    # Xy[d[i]] += x[i] * y[i]
        yy[d[i]] += y[i]^2
    end

    # Compute the posterior hyperparameters
    for j in 1:J
        B1 = inv(Symmetric(XX[j], :L) + inv(B0))
        m1 = B1 * (Xy[j] + B0 \ m0)
        a1 = a0 + n[j] / 2
        b1 = b0 + (yy[j] + m0' * (B0 \ m0) - m1' * (B1 \ m1)) / 2
        v[j] = rand(InverseGamma(a1, b1))
        β[j] = rand(MvNormal(m1, v[j] * B1))
    end
    return nothing
end

function sample!(model::Foo; mcmcsize = 4000, burnin = mcmcsize ÷ 2, thin = 1)
    (; K, J, β, v) = model
    chainβ = [zeros(K, J) for _ in 1:(mcmcsize - burnin) ÷ thin]
    chainv = [zeros(J) for _ in 1:(mcmcsize - burnin) ÷ thin]
    for iter in 1:mcmcsize
        step!(model)
        if iter > burnin && iszero((iter - burnin) % thin)
            for j in 1:J
                chainβ[(iter - burnin) ÷ thin][:, j] .= β[j]
                chainv[(iter - burnin) ÷ thin][j] = v[j]
            end
        end
    end
    return chainβ, chainv
end

begin
    Random.seed!(1)
    N = 300
    K = 5
    J = 3
    ϵ = randn(N)
    X = randn(N, K)
    d = rand(1:J, N)
    β = ones(K)
    y = d .* (X * β + ϵ)
    model = Foo(; y, X, d)
    chainβ, chainv = sample!(model; mcmcsize = 10000)
end
mean(chainβ)
mean(chainv)
