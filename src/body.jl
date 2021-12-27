abstract type Sampler end

Base.@kwdef struct NormalSampler <: Sampler
    # (Public) Data
    rng::MersenneTwister
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
    # (Private) Parameters
    ϕ::Vector{Float64} = ones(N0) / 2
    τ::Vector{Float64} = ones(N0)
    μ::Vector{Float64} = zeros(N0)
    n::Vector{Int} = [N0; zeros(Int, N0 - 1)]
    r::Vector{Int} = ones(Int, N0)
    d::Vector{Int} = ones(Int, N0)
    f::Vector{Float64} = zeros(N1)
    rmodel::BayesNegativeBinomial.Sampler = 
        BayesNegativeBinomial.Sampler(ones(Int, N0), X0)
    β::Vector{Float64} = rmodel.β
    s::Vector{Int} = rmodel.s
    ȳ::Vector{Float64} = zeros(N0)
    v::Vector{Float64} = zeros(N0)
end

Base.@kwdef struct ErlangSampler <: Sampler
    # (Public) Data
    rng::MersenneTwister
    ỹ0::Vector{Float64}
    y0::Vector{Float64} = deepcopy(ỹ0)
    X0::Matrix{Float64}
    y1::Vector{Float64}
    X1::Matrix{Float64}
    event::Vector{Bool} = ones(length(y0))
    # (Private) Data
    N0::Int = size(X0, 1)
    N1::Int = size(X1, 1)
    D0::Int = size(X0, 2)
    D1::Int = size(X1, 2)
    # (Public) HyperParameters
    m0β::Vector{Float64} = zeros(D0)
    Σ0β::Matrix{Float64} = 3 * I(D0)
    a0φ::Float64 = 2.0
    b0φ::Float64 = 0.1
    a0λ::Float64 = 0.1
    b0λ::Float64 = 0.1
    # (Private) Parameters
    ϕ::Vector{Float64} = ones(N0) / 2
    φ::Vector{Float64} = rand(rng, Gamma(a0φ, 1.0 / b0φ), N0)
    λ::Vector{Float64} = rand(rng, Gamma(a0λ, 1.0 / b0λ), 1)
    n::Vector{Int} = [N0; zeros(Int, N0 - 1)]
    r::Vector{Int} = ones(Int, N0)
    d::Vector{Int} = ones(Int, N0)
    f::Vector{Float64} = zeros(N1)
    rmodel::BayesNegativeBinomial.Sampler = 
        BayesNegativeBinomial.Sampler(ones(Int, N0), X0)
    β::Vector{Float64} = rmodel.β
    s::Vector{Int} = rmodel.s
    sumy::Vector{Float64} = [0.0]
    sumlogy::Vector{Float64} = zeros(N0)
end

function y(sampler::Sampler)
    return sampler.y0
end

function m(sampler::Sampler)
    return maximum(sampler.r)
end

function f0(sampler::NormalSampler, yi, j)
    kernel = Normal(sampler.μ[j], 1 / √sampler.τ[j])
    return pdf(kernel, yi)
end

function f0(sampler::ErlangSampler, yi, j)
    kernel = Erlang(ceil(Int, sampler.φ[j]), 1.0 / sampler.λ[])
    return pdf(kernel, yi)
end

function update_suffstats!(sampler::NormalSampler)
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

function update_suffstats!(sampler::ErlangSampler)
    (; y0, sumlogy, sumy, d, n, N0) = sampler
    jmax = m(sampler)
    for j in 1:jmax
        n[j] = 0
        sumlogy[j] = 0.0
    end
    sumy[] = 0.0
    for i in 1:N0
        n[d[i]] += 1
        sumy[] += y0[i]
        sumlogy[d[i]] += log(y0[i])
    end
    return nothing
end

function logpφ(sampler::ErlangSampler, φ0, j)
    (; λ, a0φ, b0φ, sumlogy, n) = sampler
    δ0 = ceil(Int, φ0)
    return (
        n[j] * δ0 * log(λ[]) +
        (δ0 - 1) * sumlogy[j] -
        n[j] * logfactorial(δ0 - 1) +
        (a0φ - 1) * log(φ0) -
        b0φ * φ0
    )
end

function update_φ!(rng::AbstractRNG, sampler::ErlangSampler)
    (; φ, a0φ, b0φ, n) = sampler
    pφ = Gamma(a0φ, 1.0 / b0φ)
    for j in 1:m(sampler)
        if n[j] == 0
            φ[j] = rand(rng, pφ)
        else
            φ0 = φ[j]
            d0 = truncated(Normal(φ0, 0.1), 0, Inf)
            φ1 = rand(rng, d0)
            d1 = truncated(Normal(φ1, 0.1), 0, Inf)
            log_ar = 0.0
            log_ar += (logpφ(sampler, φ1, j) + logpdf(d1, φ0))
            log_ar -= (logpφ(sampler, φ0, j) + logpdf(d0, φ1))
            if rand(rng) < exp(log_ar)
                (φ[j] = φ1)
            end
        end
    end
    return nothing
end

function update_λ!(rng::AbstractRNG, sampler::ErlangSampler)
    (; λ, φ, a0λ, b0λ, d, sumy) = sampler
    a1λ = a0λ
    b1λ = b0λ + sumy[]
     for di in d
        a1λ += ceil(φ[di])
    end
    λ[] = rand(rng, Gamma(a1λ, 1.0 / b1λ))
    return nothing
end

function update_χ!(rng::AbstractRNG, sampler::ErlangSampler)
    update_suffstats!(sampler)
    update_φ!(rng, sampler)
    update_λ!(rng, sampler)
end

function update_χ!(rng::AbstractRNG, sampler::NormalSampler)
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
    update_y!(rng, mdl)
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

function update_n!(sampler::Sampler)
    (; N0, d, n) = sampler 
    jmax = m(sampler)
    for j in 1:jmax
        n[j] = 0
    end
    for i in 1:N0
        n[d[i]] += 1
    end
    return nothing
end

function update_ϕ!(sampler::Sampler)
    (; X0, β, ϕ) = sampler
    mul!(ϕ, X0, β)
    @. ϕ = 1.0 / (1.0 + exp(ϕ))
end

function update_r!(rng::AbstractRNG, sampler::Sampler)
    (; N0, d, r, ϕ, s) = sampler
    for i in 1:N0
        zi = rand(rng, Beta(r[i] - d[i] + 1, d[i]))
        vi = rand(rng, Gamma(r[i] + s[] - 1, 1))
        r[i] = rand(rng, Poisson((1 - ϕ[i]) * zi * vi)) + d[i]
    end
end

function get_w(θ0, s0, j)
    exp(
        - log(j) +
        logabsbinomial(j + s0 - 2, j - 1)[1] +
        s0 * log(θ0) +
        (j - 1) * log(1 - θ0) +
        log(_₂F₁(BigFloat(j + s0 - 1), 1, j + 1, 1 - θ0))
    )
end

function update_β!(rng::AbstractRNG, sampler::Sampler)
    (; r, rmodel) = sampler
    @. rmodel.y = r - 1
    BayesNegativeBinomial.step!(rng, rmodel)
end

function update_f!(sampler::Sampler)
    (; N1, y1, X1, β, f, s) = sampler
    for i in 1:N1
        f[i] = 0.0
        ϕ1 = 1 / (1 + exp(X1[i, :] ⋅ β))
        for j in 1:m(sampler)
            wj = get_w(ϕ1, s[], j)
            f[i] += wj * f0(sampler, y1[i], j)
        end
    end
end

function update_y!(rng::AbstractRNG, mdl::ErlangSampler)
    (; N0, y0, ỹ0, event) = mdl
    for i in 1:N0
        if !event[i]
            pdf = Erlang(ceil(Int, sampler.φ[d[i]]), 1.0 / sampler.λ[])
            tpdf = Truncated(pdf, ỹ0[i], Inf)
            y0[i] = rand(rng, tpdf)
        end
    end
end


function step!(rng::AbstractRNG, mdl::Sampler)
    update_χ!(rng, mdl)
    update_d!(rng, mdl)
    # update_n!(mdl)
    update_f!(mdl)
    update_ϕ!(mdl)
    update_r!(rng, mdl)
    update_β!(rng, mdl)
end

function sample(rng::AbstractRNG, smpl::Sampler; mcmcsize = 4000, burnin = 2000)
    (; N1, D1, f, β) = smpl
    chainf = [zeros(N1) for _ in 1:(mcmcsize - burnin)]
    chainβ = [zeros(D1) for _ in 1:(mcmcsize - burnin)]
    for iter in 1:mcmcsize
        step!(rng, smpl)
        if iter > burnin
            chainf[iter - burnin] .= f
            chainβ[iter - burnin] .= β
        end
    end
    return chainf, chainβ
end
