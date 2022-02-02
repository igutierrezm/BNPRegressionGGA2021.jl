Base.@kwdef struct Skeleton
    # Data
    y0::Vector{Float64}
    X0::Matrix{Float64}
    y1::Vector{Float64}
    X1::Matrix{Float64}
    update_γ::Bool = true
    mapping::Vector{Vector{Int}} = [[i] for i in 1:size(X0, 2)]
    # Transformed data
    N0::Int = size(X0, 1)
    N1::Int = size(X1, 1)
    D0::Int = size(X0, 2)
    D1::Int = size(X1, 2)
    # HyperParameters
    m0β::Vector{Float64} = zeros(D0)
    Σ0β::Matrix{Float64} = 1000 * I(D0)
    # Parameters
    rmodel::BNB.Sampler = BNB.Sampler(ones(Int, N0), X0; mapping, update_γ, s = 5)
    r::Vector{Int} = ones(Int, N0)
    d::Vector{Int} = ones(Int, N0)
    β::Vector{Float64} = rmodel.β
    s::Vector{Int} = rmodel.s
    # Transformed parameters
    n::Vector{Int} = [N0]
    rmax::Base.RefValue{Int} = Ref(maximum(r))
    ϕ0::Vector{Float64} = ones(N0) / 2
    ϕ1::Vector{Float64} = ones(N1) / 2
    f::Vector{Float64} = zeros(N1)
end

ygrid(skl::Skeleton) = skl.y1
Xgrid(skl::Skeleton) = skl.X1
Ngrid(skl::Skeleton) = skl.N1
y(skl::Skeleton) = skl.y0
X(skl::Skeleton) = skl.X0
N(skl::Skeleton) = skl.N0
cluster_sizes(skl::Skeleton) = skl.n
cluster_labels(skl::Skeleton) = skl.d
r(skl::Skeleton) = skl.r
f(skl::Skeleton) = skl.f
rmax(skl::Skeleton) = skl.rmax[]
