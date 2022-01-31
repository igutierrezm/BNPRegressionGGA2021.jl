Base.@kwdef struct DGSBPNormal <: AbstractModel
    # Data
    y0::Vector{Float64}
    X0::Matrix{Float64}
    y1::Vector{Float64}
    X1::Matrix{Float64}
    mapping::Vector{Vector{Int}} = [[i] for i in 1:size(X0, 2)]
    # Hyperparameters
    m_μ0::Float64 = 0.0
    c_μ0::Float64 = 1.0
    a_τ0::Float64 = 1.0
    b_τ0::Float64 = 1.0
    # Parameters
    μ::Vector{Float64} = zeros(1)
    τ::Vector{Float64} = ones(1)
    # Transformed parameters
    ȳ::Vector{Float64} = zeros(1)
    v::Vector{Float64} = zeros(1)
    # Skeleton
    skl::Skeleton = Skeleton(; y0, y1, X0, X1, mapping)
end

function skeleton(m::DGSBPNormal)
    return m.skl
end

function kernel_pdf(m::DGSBPNormal, yi::Float64, j::Int)
    kernel = Normal(m.μ[j], 1 / √m.τ[j])
    return pdf(kernel, yi)
end

function update_atoms!(m::DGSBPNormal)
    (; m_μ0, c_μ0, a_τ0, b_τ0, μ, τ, ȳ, v, skl) = m
    update_suffstats!(m)
    n = cluster_sizes(skl)
    while length(μ) < rmax(skl)
        new_τ = rand(Gamma(a_τ0, 1 / b_τ0))
        new_μ = rand(Normal(m_μ0, √(c_μ0 / new_τ)))
        push!(τ, new_τ)
        push!(μ, new_μ)
    end    
    for j in 1:rmax(skl)
        m_μ1 = (c_μ0 * n[j] * ȳ[j] + m_μ0) / (c_μ0 * n[j] + 1)
        c_μ1 = c_μ0 / (c_μ0 * n[j] + 1)
        a_τ1 = a_τ0 + n[j] / 2
        b_τ1 = b_τ0 + n[j] * (ȳ[j] - m_μ0)^2 / 2 / (c_μ0 * n[j] + 1) + v[j] / 2
        τ[j] = rand(Gamma(a_τ1, 1 / b_τ1))
        μ[j] = rand(Normal(m_μ1, √(c_μ1 / τ[j])))
    end
    return nothing
end

function update_suffstats!(m::DGSBPNormal)
    (; y0, ȳ, v, skl) = m
    d = cluster_labels(skl)
    n = cluster_sizes(skl)
    y = y0
    if length(ȳ) < rmax(skl)
        resize!(ȳ, rmax(skl))
        resize!(v, rmax(skl))
    end
    for j in 1:rmax(skl)
        ȳ[j] = 0.0
        v[j] = 0.0
    end
    for i in 1:length(y)
        ȳ[d[i]] += y[i]
    end
    for j in 1:rmax(skl)
        iszero(n[j]) && continue
        ȳ[j] /= n[j]
    end
    for i in 1:length(y)
        v[d[i]] += (y[i] - ȳ[d[i]])^2
    end
end