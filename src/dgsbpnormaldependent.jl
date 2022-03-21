Base.@kwdef struct DGSBPNormalDependent <: AbstractModel
    # Data
    y0::Vector{Float64}
    X0::Matrix{Float64}
    y1::Vector{Float64}
    X1::Matrix{Float64}
    mapping::Vector{Vector{Int}} = [[i] for i in 1:size(X0, 2)]
    update_g::Vector{Bool} = zeros(Bool, size(X0, 2))
    # Transformed data
    D0::Int = size(X0, 2)
    ỹ0::Vector{Float64} = deepcopy(y0)
    event0::Vector{Bool} = ones(length(y0))
    # Hyperparameters
    B0_b::Symmetric{Float64, Matrix{Float64}} = Symmetric(9.0 * I(D0))
    m0_b::Vector{Float64} = zeros(D0)
    a0_τ::Float64 = 0.1
    b0_τ::Float64 = 0.1
    # Parameters
    b::Vector{Vector{Float64}} = [zeros(D0)]
    τ::Vector{Float64} = [1.0]
    # Transformed parameters
    XX::Vector{Matrix{Float64}} = [zeros(D0, D0)]
    Xy::Vector{Vector{Float64}} = [zeros(D0)]
    yy::Vector{Float64} = [0.0]    
    # Skeleton
    skl::Skeleton = Skeleton(; y0, y1, X0, X1, mapping, update_g)
end

function skeleton(m::DGSBPNormalDependent)
    return m.skl
end

function kernel_pdf(m::DGSBPNormalDependent, yi::Float64, xi::Vector{Float64}, j::Int)
    kernel = Normal(m.b[j] ⋅ xi, 1 / √m.τ[j])
    return pdf(kernel, yi)
end

function update_atoms!(m::DGSBPNormalDependent)
    (; b, τ, m0_b, B0_b, a0_τ, b0_τ, XX, Xy, yy, skl) = m
    (; mapping, D0, g, n) = skl
    update_suffstats!(m)
    while length(τ) < rmax(skl)
        new_τ = 1.0
        new_b = zeros(D0)
        push!(τ, new_τ)
        push!(b, new_b)
    end
    gexp = zeros(Bool, D0)
    for d in 1:length(mapping)
        gexp[mapping[d]] .= g[d]
    end    
    for j in 1:rmax(skl)
        B1_b = inv((Symmetric(XX[j], :L) + inv(B0_b) + √eps(Float64) * I(D0))[gexp, gexp])
        m1_b = B1_b * (Xy[j] + B0_b \ m0_b)[gexp]
        a1_τ = a0_τ + n[j] / 2
        b1_τ = b0_τ + (yy[j] + m0_b ⋅ (B0_b \ m0_b) - m1_b ⋅ (B1_b \ m1_b)) / 2
        τ[j] = rand(Gamma(a1_τ, 1 / b1_τ))
        b[j][.!gexp] .= 0.0
        b[j][gexp] .= rand(MvNormal(m1_b, Symmetric(B1_b / τ[j])))
    end
    update_y!(m)
    update_g!(m)
    return nothing
end

function update_suffstats!(m::DGSBPNormalDependent)
    (; XX, Xy, yy, skl) = m
    (; y0, X0vec, N0, D0, d, n, rmax) = skl
    for j in 1:rmax[]
        if length(XX) >= j
            XX[j] .= 0.0
            Xy[j] .= 0.0
            yy[j] = 0.0    
            n[j] = 0
        else
            push!(XX, zeros(D0, D0))
            push!(Xy, zeros(D0))
            push!(yy, 0.0)
            push!(n, 0)
        end
    end
    @inbounds for i in 1:N0
        BLAS.syr!('L', 1.0, X0vec[i], XX[d[i]]) # XX[d[i]] += x[i] * x[i]'
        BLAS.axpy!(y0[i], X0vec[i], Xy[d[i]])   # Xy[d[i]] += x[i] * y[i]
        yy[d[i]] += y0[i]^2
        n[d[i]] += 1
    end
    return nothing
end

function update_y!(m::DGSBPNormalDependent)
    (; ỹ0, event0, b, τ, skl) = m
    (; X0vec, N0, y0, d) = skl
    for i in 1:N0
        if !event0[i]
            dist = Normal(b[d[i]] ⋅ X0vec[i], 1 / √τ[d[i]])
            tdist = Truncated(dist, ỹ0[i], Inf)
            y0[i] = rand(tdist)
        end
    end
end

function update_g!(m::DGSBPNormalDependent)
    (; m0_b, B0_b, a0_τ, b0_τ, XX, Xy, yy, update_g, mapping, skl) = m
    (; D0, rmodel, n) = skl
    (; g, μ0β, Σ0β, ζ0g) = rmodel
    gexp = zeros(Bool, D0)
    for d in 1:length(mapping)
        gexp[mapping[d]] .= g[d]
    end
    pg = Womack(length(g), ζ0g)
    update_suffstats!(m)
    for d in 1:length(g)
        update_g[d] || continue        
        logodds = 0.0
        for val in 0:1
            g[d] = val
            gexp[mapping[d]] .= val
            m1, Σ1 = posterior_hyperparameters(sampler)
            # Find the posterior hyperparameters related to the atoms
            B1_b = inv((Symmetric(XX[j], :L) + inv(B0_b) + √eps(Float64) * I(D0))[gexp, gexp])
            m1_b = B1_b * (Xy[j] + B0_b \ m0_b)[gexp]
            a1_τ = a0_τ + n[j] / 2
            b1_τ = b0_τ + (yy[j] + m0_b ⋅ (B0_b \ m0_b) - m1_b ⋅ (B1_b \ m1_b)) / 2
            logQ = (
                0.5 * logdet(B1_b) -
                0.5 * logdet(B0_b) + 
                loggamma(a1_τ / 2) -
                loggamma(a0_τ / 2) + 
                (a0_τ / 2) * log(b0_τ) -
                (a1_τ / 2) * log(b1_τ) -
                (n[j] / 2) * log(2 * π)
            )
            # Compute the contribution to the logodds
            logodds += (-1)^(val + 1) * (
                logQ + 
                logpdf(pg, g) +
                logpdf(MvNormal(μ0β[gexp], Σ0β[gexp, gexp]), zeros(sum(gexp))) -
                logpdf(MvNormal(m1, Σ1), zeros(length(m1)))
            )
        end
        g[d] = rand(rng) < exp(logodds) / (1.0 + exp(logodds))
        gexp[mapping[d]] .= g[d]
    end
    return nothing
end