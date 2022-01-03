Base.@kwdef struct DGPMBeta <: AbstractModel
    # Data
    y0::Vector{Float64}
    X0::Matrix{Float64}
    y1::Vector{Float64}
    X1::Matrix{Float64}
    # Hyperparameters
    a_v0::Float64 = 1.0
    b_v0::Float64 = 1.0
    # Parameters
    μ::Vector{Float64} = rand(1)
    v::Vector{Float64} = rand(Gamma(a_v0, 1 / b_v0), 1)
    μ_shadow::Vector{Float64} = randexp(1) * 20.0
    v_shadow::Vector{Float64} = randexp(1) * 20.0
    # Transformed parameters
    sumlogy1::Vector{Float64} = zeros(1)
    sumlogy2::Vector{Float64} = zeros(1)
    # Skeleton
    skl::Skeleton = Skeleton(; y0, y1, X0, X1)
end

function skeleton(m::DGPMBeta)
    return m.skl
end

function kernel_pdf(m::DGPMBeta, y0::Float64, j::Int)
    (; μ, v) = m
    kernel = Beta(μ[j] / v[j], (1 - μ[j])/ v[j])
    return pdf(kernel, y0)
    # μv = μ[j] * v[j]
    # return exp(
    #     (μv - 1) * log(y0) +
    #     (μ[j] - μv - 1) * log(1.0 - y0) -
    #     log_beta(μv, v[j] - μv)
    # );
end

function update_atoms!(m::DGPMBeta)
    (; a_v0, b_v0, μ, v, μ_shadow, v_shadow, skl) = m
    update_suffstats!(m)
    n = cluster_sizes(skl)
    while length(μ) < rmax(skl)
        push!(μ, rand())
        push!(v, rand(Gamma(a_v0, 1 / b_v0)))
        push!(μ_shadow, 20 * randexp())
        push!(v_shadow, 20 * randexp())
    end
    s_μ = Walker2020Sampler(20.0, 0.0, 1.0);
    s_v = Walker2020Sampler(20.0, 0.0, Inf);
    for j in 1:rmax(skl)
        if n[j] > 0
            μ[j], μ_shadow[j] = 
                rand(Random.GLOBAL_RNG, s_μ, x -> logpμ(m, x, j), μ[j], μ_shadow[j])
            v[j], v_shadow[j] = 
                rand(Random.GLOBAL_RNG, s_v, x -> logpv(m, x, j), v[j], v_shadow[j])
        else 
            μ[j] = rand()
            v[j] = rand(Gamma(a_v0, 1 / b_v0))
            μ_shadow[j] = 20 * randexp()
            v_shadow[j] = 20 * randexp()
        end
    end
end

function update_suffstats!(m::DGPMBeta)
    (; y0, sumlogy1, sumlogy2, skl) = m
    d = cluster_labels(skl)
    while length(sumlogy1) < rmax(skl)
        push!(sumlogy1, 0.0)
        push!(sumlogy2, 0.0)
    end
    for j in 1:rmax(skl)
        sumlogy1[j] = 0.0
        sumlogy2[j] = 0.0
    end
    for i in 1:length(y0)
        sumlogy1[d[i]] += log(y0[i])
        sumlogy2[d[i]] += log(1.0 - y0[i])
    end
    return nothing
end

function logpμ(m::DGPMBeta, μ0::Float64, j::Int)
    (; sumlogy1, sumlogy2, v, skl) = m
    n = cluster_sizes(skl)
    μv = μ0 * v[j]
    return (
        (μv - 1) * sumlogy1[j] +
        (μ0 - μv - 1) * sumlogy2[j] -
        n[j] * logbeta(BigFloat(μv), BigFloat(v[j] - μv))
    )
end

function logpv(m::DGPMBeta, v0::Float64, j::Int)
    (; sumlogy1, sumlogy2, a_v0, b_v0, μ, skl) = m
    n = cluster_sizes(skl)
    μv = μ[j] * v0
    return (
        (μv - 1) * sumlogy1[j] +
        (v0 - μv - 1) * sumlogy2[j] -
        n[j] * logbeta(μv, v0 - μv) -
        (a_v0 + 1) * log(v0) - 
        b_v0 / v0
    )
end