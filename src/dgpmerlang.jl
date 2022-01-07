Base.@kwdef struct DGPMErlang <: AbstractModel
    # Data
    c0::Vector{Float64}
    y0::Vector{Float64}
    X0::Matrix{Float64}
    y1::Vector{Float64}
    X1::Matrix{Float64}
    # Transformed data
    event::Vector{Bool} = y0 .< c0
    N0::Int = length(y0)
    N1::Int = length(y1)
    # HyperParameters
    a_φ0::Float64 = 2.0
    b_φ0::Float64 = 0.1
    a_λ0::Float64 = 0.1
    b_λ0::Float64 = 0.1
    # Parameters
    φ::Vector{Float64} = rand(Gamma(a_φ0, 1.0 / b_φ0), 1)
    λ::Vector{Float64} = rand(Gamma(a_λ0, 1.0 / b_λ0), 1)
    # Transformed parameters
    sumy::Base.RefValue{Float64} = Ref(0.0)
    sumlogy::Vector{Float64} = zeros(1)
    S::Vector{Float64} = zeros(N1)
    # Skeleton
    skl::Skeleton = Skeleton(; y0, y1, X0, X1)
end

function skeleton(m::DGPMErlang)
    return m.skl
end

function kernel_pdf(m::DGPMErlang, yi::Float64, j::Int)
    kernel = Erlang(ceil(Int, m.φ[j]), 1.0 / m.λ[])
    return pdf(kernel, yi)
end

function kernel_cdf(m::DGPMErlang, yi::Float64, j::Int)
    kernel = Erlang(ceil(Int, m.φ[j]), 1.0 / m.λ[])
    return 1 - cdf(kernel, yi)
end

function update_atoms!(m::DGPMErlang)
    update_suffstats!(m)
    update_λ!(m)
    update_φ!(m)
    update_y!(m)
end

function update_suffstats!(m::DGPMErlang)
    (; y0, sumlogy, sumy, skl) = m
    d = cluster_labels(skl)
    if length(sumlogy) < rmax(skl)
        resize!(sumlogy, rmax(skl))
    end
    for j in 1:rmax(skl)
        sumlogy[j] = 0.0
    end
    sumy[] = 0.0
    for i in 1:length(y0)
        sumy[] += y0[i]
        sumlogy[d[i]] += log(y0[i])
    end
    return nothing
end

function update_λ!(m::DGPMErlang)
    (; λ, φ, a_λ0, b_λ0, sumy, skl) = m
    (; d) = skl
    a_λ1 = a_λ0
    b_λ1 = b_λ0 + sumy[]
     for di in d
        a_λ1 += ceil(φ[di])
    end
    λ[] = rand(Gamma(a_λ1, 1.0 / b_λ1))
    return nothing
end

function update_φ!(m::DGPMErlang)
    (; φ, a_φ0, b_φ0, skl) = m
    pφ = Gamma(a_φ0, 1.0 / b_φ0)
    while length(φ) < rmax(skl)
        push!(φ, rand(pφ))
    end
    for j in 1:rmax(skl)
        φ0 = φ[j]
        d0 = truncated(Normal(φ0, 0.1), 0, Inf)
        φ1 = rand(d0)
        d1 = truncated(Normal(φ1, 0.1), 0, Inf)
        log_ar = 0.0
        log_ar += (logpφ(m, φ1, j) + logpdf(d1, φ0))
        log_ar -= (logpφ(m, φ0, j) + logpdf(d0, φ1))
        if rand() < exp(log_ar)
            (φ[j] = φ1)
        end
    end
    return nothing
end

function logpφ(m::DGPMErlang, φ0, j)
    (; λ, a_φ0, b_φ0, sumlogy, skl) = m
    n = cluster_sizes(skl)
    δ0 = ceil(Int, φ0)
    return (
        n[j] * δ0 * log(λ[]) +
        (δ0 - 1) * sumlogy[j] -
        n[j] * logfactorial(δ0 - 1) +
        (a_φ0 - 1) * log(φ0) -
        b_φ0 * φ0
    )
end

function update_y!(m::DGPMErlang)
    (; y0, c0, event, φ, λ, skl) = m
    d = cluster_labels(skl)
    for i in 1:length(y0)
        if !event[i]
            pdf = Erlang(ceil(Int, φ[d[i]]), 1.0 / λ[])
            tpdf = Truncated(pdf, c0[i], Inf)
            if cdf(pdf, c0[i]) < 0.99
                y0[i] = rand(tpdf)
            end
        end
    end
end

function update_S!(m::DGPMErlang)
    (; S, skl) = m
    (; N1, y1, ϕ1, s, rmax) = skl
    for i in 1:N1
        S[i] = 0.0
        for j in 1:rmax[]
            wj = get_w(ϕ1[i], s[], j)
            S[i] += wj * kernel_cdf(m, y1[i], j)
        end
    end
end

function sample!(m::DGPMErlang; mcmcsize = 4000, burnin = 2000, thin = 1)
    (; S, skl) = m
    (; N1, D1, f, β) = skl
    chainf = [zeros(N1) for _ in 1:(mcmcsize - burnin) ÷ thin]
    chainS = [zeros(N1) for _ in 1:(mcmcsize - burnin) ÷ thin]
    chainβ = [zeros(D1) for _ in 1:(mcmcsize - burnin) ÷ thin]
    chaing = [zeros(Bool, D1) for _ in 1:(mcmcsize - burnin) ÷ thin]
    for iter in 1:mcmcsize
        step!(m)
        update_S!(m)
        if iter > burnin && iszero((iter - burnin) % thin)
            chainf[(iter - burnin) ÷ thin] .= f
            chainS[(iter - burnin) ÷ thin] .= S
            chainβ[(iter - burnin) ÷ thin] .= β
            chaing[(iter - burnin) ÷ thin] .= (β .!= 0.0)
        end
    end
    return chainf, chainS, chainβ, chaing
end
