Base.@kwdef struct DGPMErlang <: AbstractModel
    # Data
    c0::Vector{Float64}
    y0::Vector{Float64}
    X0::Matrix{Float64}
    y1::Vector{Float64}
    X1::Matrix{Float64}
    # Transformed data
    event::Vector{Bool} = y0 .< c0
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

function update_atoms!(m::DGPMErlang)
    update_suffstats!(m)
    update_φ!(m)
    update_λ!(m)
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
    (; y0, c0, event) = m
    for i in 1:length(y0)
        if !event[i]
            pdf = Erlang(ceil(Int, sampler.φ[d[i]]), 1.0 / sampler.λ[])
            tpdf = Truncated(pdf, c0[i], Inf)
            y0[i] = rand(tpdf)
        end
    end
end
