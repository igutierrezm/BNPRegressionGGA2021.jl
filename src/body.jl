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
    θ::Vector{Float64} = ones(N) * 0.5
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
    Ny::Int
    Nx::Int
    w::Vector{Vector{Float64}} = [zeros(128) for i in 1:Nx]
    f::Vector{Vector{Float64}} = [zeros(Ny) for i in 1:Nx] 
end

struct Sampler
    data::Data
    pa::Parameters
    hp::HyperParameters
    gq::GeneratedQuantities
    βsubsampler::BayesNegativeBinomial.Sampler
    function Sampler(
        y0::Vector{Float64}, 
        Xtrain::Matrix{Float64}, 
        ypred::Vector{Float64}, 
        Xpred::Matrix{Float64}; 
        hp::HyperParameters = HyperParameters(D = size(Xtrain, 2))
    )
        N1, D = size(Xpred)
        N0, D = size(Xtrain)
        pa = Parameters(N = N0, D = D)
        gq = GeneratedQuantities(N = N1)
        data = Data(y0, Xtrain, ypred, Xpred)
        βsubsampler = BayesNegativeBinomial.Sampler(ones(Int, N0), Xtrain)
        new(data, pa, hp, gq, βsubsampler)
    end
end


function update_suffstats!(smpl::Sampler)
    @extract smpl.pa : d ȳ s n
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
    @extract smpl.pa : n ȳ s μ τ
    @extract smpl.hp : m0μ c0μ a0τ b0τ
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
    @extract smpl.hp : s0r
    for i in 1:length(r)
        zi = rand(rng, Beta(r[i] - d[i] + 1, d[i]))
        vi = rand(rng, Gamma(r[i] + s0r - 1, 1))
        r[i] = rand(rng, Poisson((1 - θ[]) * zi * vi)) + d[i]
    end
end

function update_θ!(rng::AbstractRNG, smpl::Sampler)
    @extract smpl.pa : r θ
    @extract smpl.hp : a0θ b0θ s0r
    N = length(r)
    θ0 = rand(rng, Beta(s0r * N + a0θ, sum(r) + b0θ - N))
    θ .= θ0
end

function update_w!(smpl::Sampler)
    @extract smpl.data : y1
    @extract smpl.pa : w θ
    @extract smpl.hp : s0r
    for i in 1:length(y1), j in 1:m(smpl)
        # logwj = 
        #     - log(j) +
        #     logabsbinomial(j + sr0 - 2, j - 1)[1] +
        #     sr0 * log(θ[]) +
        #     (j - 1) * log(1 - θ[]) +
        #     log(_₂F₁(j + sr0 - 1, 1, j + 1, 1 - θ[]))
        # w[j] = exp(logwj)
        w[i][j] = θ[i] * (1 - θ[i])^(j - 1)
    end
end

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

function update_f!(smpl::Sampler)
    @extract smpl.data : y1
    @extract smpl.pa : w
    @extract smpl.gq : f
    f .= 0.0
    for i in 1:length(f), j in 1:m(smpl)
        f[i] += w[j] * κ(smpl, y1[i], j)
    end
end

function m(smpl::Sampler)
    return maximum(smpl.pa.r)
end

function κ(smpl::Sampler, yi, j)
    pdf(Normal(smpl.pa.μ[j], 1 / √smpl.pa.τ[j]), yi)
end

function step!(rng::AbstractRNG, mdl::Sampler)
    update_χ!(rng, mdl) #
    update_d!(rng, mdl) #
    update_w!(mdl)
    update_f!(mdl)
    update_r!(rng, mdl)
    update_θ!(rng, mdl)
end

function sample(rng::AbstractRNG, s::Sampler; mcmcsize = 10000, burnin = 5000)
    chain = [zeros(size(s.data.X1, 1)) for _ in 1:(mcmcsize - burnin)]
    for iter in 1:mcmcsize
        step!(rng, s)
        if iter > burnin
            chain[iter - burnin] .= s.gq.f
        end
    end
    return chain
end
