Base.@kwdef struct Sampler
    # (Public) Data
    y0::Vector{Float64}
    X0::Matrix{Float64}
    y1::Vector{Float64}
    X1::Matrix{Float64}
    # (Private) Data
    N0::Int = size(X0, 1)
    N1::Int = size(X1, 1)
    D0::Int = size(X0, 2)
    D1::Int = size(X1, 2)
    # (Public) HyperParameters
    m0β::Vector{Float64} = zeros(D0)
    Σ0β::Matrix{Float64} = 3 * I(D0)
    m0μ::Float64 = 0.0
    c0μ::Float64 = 2.0
    a0τ::Float64 = 1.0
    b0τ::Float64 = 1.0
    a0θ::Float64 = 1.0
    b0θ::Float64 = 1.0
    s0r::Int = 2
    # (Private) Parameters
    ϕ::Vector{Float64} = ones(N0) / 2
    τ::Vector{Float64} = ones(N0)
    μ::Vector{Float64} = zeros(N0)
    ȳ::Vector{Float64} = zeros(N0)
    v::Vector{Float64} = zeros(N0)
    n::Vector{Int} = ones(Int, N0)
    r::Vector{Int} = ones(Int, N0)
    d::Vector{Int} = ones(Int, N0)
    f::Vector{Float64} = zeros(N1)
    βsmpl::BayesNegativeBinomial.Sampler = 
        BayesNegativeBinomial.Sampler(ones(Int, N0), X0; r0y = [s0r])
end

function m(sampler::Sampler)
    maximum(sampler.r)
end

function f0(sampler::Sampler, yi, j)
    pdf(Normal(sampler.μ[j], 1 / √sampler.τ[j]), yi)
end

function update_suffstats!(sampler::Sampler)
    (; N0, y0, d, ȳ, v, n) = sampler 
    jmax = m(sampler)
    for j in 1:jmax
        ȳ[j] = 0.0
        v[j] = 0.0
        n[j] = 0
    end
    for i in 1:N0
        n[d[i]] += 1
        ȳ[d[i]] += y0[i]
    end
    for j in 1:jmax
        iszero(n[j]) && continue
        ȳ[j] /= n[j]
    end
    for i in 1:N0
        v[d[i]] += (y0[i] - ȳ[d[i]])^2
    end
end

function update_χ!(rng::AbstractRNG, sampler::Sampler)
    (; μ, τ, n, ȳ, v, m0μ, c0μ, a0τ, b0τ) = sampler
    update_suffstats!(sampler)
    for j in 1:m(sampler)
        mμ1 = (c0μ * n[j] * ȳ[j] + m0μ) / (c0μ * n[j] + 1)
        cμ1 = c0μ / (c0μ * n[j] + 1)
        aτ1 = a0τ + n[j] / 2
        bτ1 = b0τ + n[j] * (ȳ[j] - m0μ)^2 / 2 / (c0μ * n[j] + 1) + v[j] / 2
        τ[j] = rand(rng, Gamma(aτ1, 1 / bτ1))
        μ[j] = mμ1 + randn(rng) * √(cμ1 / τ[j])
    end
end

function update_d!(rng::AbstractRNG, sampler::Sampler)
    (; N0, y0, d, r) = sampler
    p = zeros(m(sampler))
    for i in 1:N0
        resize!(p, r[i])
        for j in 1:r[i]
            p[j] = f0(sampler, y0[i], j)
        end
        p ./= sum(p)
        d[i] = rand(rng, Categorical(p))
    end
    return nothing
end

function update_ϕ!(sampler::Sampler)
    (; X0, βsmpl, ϕ) = sampler
    mul!(ϕ, X0, βsmpl.β)
    @. ϕ = 1.0 / (1.0 + exp(ϕ))
end

function update_r!(rng::AbstractRNG, sampler::Sampler)
    (; N0, d, r, ϕ, s0r) = sampler
    for i in 1:N0
        zi = rand(rng, Beta(r[i] - d[i] + 1, d[i]))
        vi = rand(rng, Gamma(r[i] + s0r - 1, 1))
        r[i] = rand(rng, Poisson((1 - ϕ[i]) * zi * vi)) + d[i]
    end
end

function get_w(θ0, s0r, j)
    exp(
        - log(j) +
        logabsbinomial(j + s0r - 2, j - 1)[1] +
        s0r * log(θ0) +
        (j - 1) * log(1 - θ0) +
        log(_₂F₁(j + s0r - 1, 1, j + 1, 1 - θ0))
    )
end

function update_β!(rng::AbstractRNG, sampler::Sampler)
    (; r, βsmpl) = sampler
    @. βsmpl.y = r - 1
    BayesNegativeBinomial.step!(rng, βsmpl)
end

function update_f!(sampler::Sampler)
    (; N1, y1, X1, βsmpl, f, s0r) = sampler
    for i in 1:N1
        f[i] = 0.0
        ϕ1 = 1 / (1 + exp(X1[i, :] ⋅ βsmpl.β))
        for j in 1:m(sampler)
            wj = get_w(ϕ1, s0r, j)
            f[i] += wj * f0(sampler, y1[i], j)
        end
    end
end

function step!(rng::AbstractRNG, mdl::Sampler)
    update_χ!(rng, mdl)
    update_d!(rng, mdl)
    update_f!(mdl)
    update_ϕ!(mdl)
    update_r!(rng, mdl)
    update_β!(rng, mdl)
end

function sample(rng::AbstractRNG, smpl::Sampler; mcmcsize = 4000, burnin = 2000)
    (; N1, D1, f, βsmpl) = smpl
    chainf = [zeros(N1) for _ in 1:(mcmcsize - burnin)]
    chainβ = [zeros(D1) for _ in 1:(mcmcsize - burnin)]
    for iter in 1:mcmcsize
        step!(rng, smpl)
        if iter > burnin
            chainf[iter - burnin] .= f
            chainβ[iter - burnin] .= βsmpl.β
        end
    end
    return chainf, chainβ
end
