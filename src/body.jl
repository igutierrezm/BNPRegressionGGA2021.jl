Base.@kwdef struct Data
    # Sample
    y0::Vector{Float64}
    X0::Matrix{Float64}
    # Grid points
    y1::Vector{Float64}
    X1::Matrix{Float64}
    # HyperParameters
    m0β::Vector{Float64} = zeros(size(X0, 1))
    Σ0β::Matrix{Float64} = 3 * I(size(X0, 1))
    m0μ::Float64 = 0.0
    c0μ::Float64 = 2.0
    a0τ::Float64 = 1.0
    b0τ::Float64 = 1.0
    a0θ::Float64 = 1.0
    b0θ::Float64 = 1.0
    s0r::Int = 2
end

Base.@kwdef struct Parameters
    N::Int
    D::Int
    r::Vector{Int} = ones(Int, N)
    d::Vector{Int} = ones(Int, N)
    β::Vector{Float64} = zeros(D)
    μ::Vector{Float64} = zeros(N)
    τ::Vector{Float64} = ones(N)
    θ::Vector{Float64} = ones(N) * 0.5
end

Base.@kwdef struct TransformedParameters
    N::Int
    D::Int
    n::Vector{Int} = zeros(Int, N)
    ȳ::Vector{Float64} = zeros(N)
    s::Vector{Float64} = zeros(N)
end

Base.@kwdef struct GeneratedQuantities
    N::Int
    f::Vector{Float64} = zeros(N)
end

struct Sampler
    data::Data
    pa::Parameters
    tp::TransformedParameters
    gq::GeneratedQuantities
    βsmpl::BayesNegativeBinomial.Sampler
    function Sampler(data::Data)
        N0, D = size(data.X0)
        pa = Parameters(N = N0, D = D)
        tp = TransformedParameters(N = N0, D = D)
        gq = GeneratedQuantities(N = length(data.y1))
        βsmpl = BayesNegativeBinomial.Sampler(ones(Int, N0), data.X0)
        new(data, pa, tp, gq, βsmpl)
    end
end

function update_suffstats!(smpl::Sampler)
    @extract smpl.pa : d
    @extract smpl.tp : ȳ s n
    @extract smpl.data : y0
    for j in 1:m(smpl)
        ȳ[j] = 0.0
        s[j] = 0.0
        n[j] = 0
    end
    for i in 1:length(y0)
        n[d[i]] += 1
        ȳ[d[i]] += y0[i]
    end
    for j in 1:m(smpl)
        iszero(n[j]) && continue
        ȳ[j] /= n[j]
    end
    for i in 1:length(y0)
        s[d[i]] += (y0[i] - ȳ[d[i]])^2
    end
end

function update_χ!(rng::AbstractRNG, smpl::Sampler)
    @extract smpl.pa : μ τ
    @extract smpl.tp : n ȳ s
    @extract smpl.data : m0μ c0μ a0τ b0τ
    update_suffstats!(smpl)
    for j in 1:m(smpl)
        mμ1 = (c0μ * n[j] * ȳ[j] + m0μ) / (c0μ * n[j] + 1)
        cμ1 = c0μ / (c0μ * n[j] + 1)
        aτ1 = a0τ + n[j] / 2
        bτ1 = b0τ + n[j] * (ȳ[j] - m0μ)^2 / 2 / (c0μ * n[j] + 1) + s[j] / 2
        τ[j] = rand(rng, Gamma(aτ1, 1 / bτ1))
        μ[j] = mμ1 + randn(rng) * √(cμ1 / τ[j])
    end
end

function update_r!(rng::AbstractRNG, smpl::Sampler)
    @extract smpl.pa : r d θ
    @extract smpl.data : s0r
    for i in 1:length(r)
        zi = rand(rng, Beta(r[i] - d[i] + 1, d[i]))
        vi = rand(rng, Gamma(r[i] + s0r - 1, 1))
        r[i] = rand(rng, Poisson((1 - θ[i]) * zi * vi)) + d[i]
    end
end

function update_θ!(rng::AbstractRNG, smpl::Sampler)
    # @extract smpl.pa : r θ
    # @extract smpl.data : a0θ b0θ s0r
    # N = length(r)
    # θ0 = rand(rng, Beta(s0r * N + a0θ, sum(r) + b0θ - N))
    # θ .= θ0
    @extract smpl.data : X0
    @extract smpl.pa : θ β
    mul!(θ, X0, β)
    @. θ = 1 / (1 + exp(- θ))
end

# function update_w!(smpl::Sampler)
#     @extract smpl.data : X1 s0r
#     @extract smpl.pa : θ
#     @extract smpl.gq : w
#     for ix in 1:size(X1, 1), j in 1:m(smpl)
#         # logwj = 
#         #     - log(j) +
#         #     logabsbinomial(j + sr0 - 2, j - 1)[1] +
#         #     sr0 * log(θ[]) +
#         #     (j - 1) * log(1 - θ[]) +
#         #     log(_₂F₁(j + sr0 - 1, 1, j + 1, 1 - θ[]))
#         # w[j] = exp(logwj)
#         w[ix][j] = θ[ix] * (1 - θ[ix])^(j - 1)
#     end
# end

function update_d!(rng::AbstractRNG, smpl::Sampler)
    @extract smpl.pa : r d
    @extract smpl.data : y0
    p = zeros(m(smpl))
    for i in 1:length(d)
        resize!(p, r[i])
        for j in 1:r[i]
            p[j] = κ(smpl, y0[i], j)
        end
        p ./= sum(p)
        d[i] = rand(rng, Categorical(p))
    end
    return nothing
end

function update_β!(rng::AbstractRNG, smpl::Sampler)
    @extract smpl : pa βsmpl
    @extract pa : r β
    @. βsmpl.y = r - 1
    BayesNegativeBinomial.step!(rng, βsmpl)
    β .= βsmpl.β
end

function update_f!(smpl::Sampler)
    @extract smpl.data : y1 X1
    @extract smpl.gq : f
    @extract smpl.pa : β
    for i in 1:length(f)
        f[i] = 0.0
        θ0 = 1 / (1 + exp(- X1[i, :] ⋅ β))
        for j in 1:m(smpl)
            wj = θ0 * (1 - θ0)^(j - 1)
            f[i] += wj * κ(smpl, y1[i], j)
        end
    end
end

function m(smpl::Sampler)
    return maximum(smpl.pa.r)
end

function κ(smpl::Sampler, yi, j)
    pdf(Normal(smpl.pa.μ[j], 1 / √smpl.pa.τ[j]), yi)
end

function step!(rng::AbstractRNG, mdl::Sampler)
    update_χ!(rng, mdl)
    update_d!(rng, mdl)
    update_f!(mdl)
    update_r!(rng, mdl)
    update_θ!(rng, mdl)
    update_β!(rng, mdl)
end

function sample(rng::AbstractRNG, s::Sampler; mcmcsize = 10000, burnin = 5000)
    chain = [zeros(length(s.gq.f)) for _ in 1:(mcmcsize - burnin)]
    for iter in 1:mcmcsize
        step!(rng, s)
        if iter > burnin
            chain[iter - burnin] .= s.gq.f
        end
    end
    return chain
end
