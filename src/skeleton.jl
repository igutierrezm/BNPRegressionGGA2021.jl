Base.@kwdef struct Skeleton
    # Data
    y0::Vector{Float64}
    X0::Matrix{Float64}
    y1::Vector{Float64}
    X1::Matrix{Float64}
    # Transformed data
    N0::Int = size(X0, 1)
    N1::Int = size(X1, 1)
    D0::Int = size(X0, 2)
    D1::Int = size(X1, 2)
    # HyperParameters
    m0β::Vector{Float64} = zeros(D0)
    Σ0β::Matrix{Float64} = 2 * I(D0)
    # Parameters
    rmodel::BNB.Sampler = BNB.Sampler(ones(Int, N0), X0)
    r::Vector{Int} = ones(Int, N0)
    d::Vector{Int} = ones(Int, N0)
    β::Vector{Float64} = rmodel.β
    s::Vector{Int} = rmodel.s
    # Transformed parameters
    n::Vector{Int} = [N0]
    rmax::Base.RefValue{Int} = Ref(maximum(r))
    ϕ0::Vector{Float64} = ones(N0) / 2
    ϕ1::Vector{Float64} = ones(N1) / 2
    f::Vector{Float64} = ones(N1)
end

rmax(s::Skeleton) = s.rmax[]
cluster_sizes(s::Skeleton) = s.n
cluster_labels(s::Skeleton) = s.d
