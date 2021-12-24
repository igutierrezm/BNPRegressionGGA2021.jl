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
    n::Vector{Int} = ones(Int, N0)
    r::Vector{Int} = ones(Int, N0)
    d::Vector{Int} = ones(Int, N0)
    β::Vector{Float64} = rmodel.β
    s::Vector{Int} = rmodel.s
    # Transformed parameters
    rmax::Base.RefValue{Int} = Ref(1)
    ϕ::Vector{Float64} = ones(N0) / 2
    f::Vector{Float64} = ones(N1)
end

"""
    rmax(s::Skeleton)
    
Return `maximum(r)`.
"""
rmax(s::Skeleton) = s.rmax[]

"""
    cluster_labels(s::Skeleton)
    
Return the cluster labels of each point.
"""
cluster_labels(s::Skeleton) = s.d
