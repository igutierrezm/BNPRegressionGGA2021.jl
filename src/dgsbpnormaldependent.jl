Base.@kwdef struct DGSBPNormalDependent <: AbstractModel
    # Data
    y0::Vector{Float64}
    X0::Matrix{Float64}
    y1::Vector{Float64}
    X1::Matrix{Float64}
    mapping::Vector{Vector{Int}} = [[i] for i in 1:size(X0, 2)]
    update_g::Vector{Bool} = ones(Bool, length(mapping))
    event0::Vector{Bool} = ones(length(y0))
    # Transformed data
    D0::Int = size(X0, 2)
    ỹ0::Vector{Float64} = deepcopy(y0)
    # Hyperparameters
    B0_b::Symmetric{Float64, Matrix{Float64}} = Symmetric(9.0 * I(D0))
    m0_b::Vector{Float64} = zeros(D0)
    a0_τ::Float64 = 0.1
    b0_τ::Float64 = 0.1
    ζ0g::Float64 = 0.5
    # Parameters
    b::Vector{Vector{Float64}} = [zeros(D0)]
    τ::Vector{Float64} = [1.0]
    # Transformed parameters
    XX::Vector{Matrix{Float64}} = [zeros(D0, D0)]
    Xy::Vector{Vector{Float64}} = [zeros(D0)]
    yy::Vector{Float64} = [0.0]
    # Skeleton
    skl::Skeleton = Skeleton(; y0, y1, X0, X1, ζ0g, mapping, update_g)
end

function skeleton(m::DGSBPNormalDependent)
    return m.skl
end

function kernel_pdf(m::DGSBPNormalDependent, yi::Float64, xi::Vector{Float64}, j::Int)
    kernel = Normal(m.b[j] ⋅ xi, 1 / √m.τ[j])
    return pdf(kernel, yi)
end

function update_atoms!(m::DGSBPNormalDependent)
    (; a0_τ, b, B0_b, b0_τ, m0_b, skl, XX, Xy, yy, τ) = m
    (; D0, g, mapping, n) = skl
    update_suffstats!(m)
    while length(τ) < rmax(skl)
        push!(τ, 1.0)
        push!(b, zeros(D0))
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
    update_g!(m)
    update_y!(m)
    return nothing
end

function update_suffstats!(m::DGSBPNormalDependent)
    (; skl, XX, Xy, yy) = m
    (; d, D0, n, N0, rmax, X0vec, y0) = skl
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
    # (; b, event0, skl, ỹ0, τ) = m
    # (; d, N0, X0vec, y0) = skl
    # m, s = mean_and_std(ỹ0)
    # ub = 2 * maximum(ỹ0)

    # (; f, N1, rmax, s, X1, X1vec, y1, β, ϕ1) = skeleton(m)
    # mul!(ϕ1, X1, β)
    # @. ϕ1 = 1.0 / (1.0 + exp(ϕ1))

    # for i in 1:N0
    #     # Choose a cluster
    #     di = 0
    #     temp = rand()
    #     cumsum_w = 0.0
    #     for j in 1:rmax(skl)
    #         cumsum_w = get_w(θ0, s0, j)
    #     end
    #     if !event0[i]
    #         dist = Normal(b[d[i]] ⋅ X0vec[i], 1 / √τ[d[i]])
    #         tdist = Truncated(dist, ỹ0[i], ub)
    #         y0[i] = rand(tdist)
    #     end
    # end
    (; b, event0, skl, ỹ0, τ) = m
    (; N0, X0, X0vec, β, y0, ϕ0) = skl
    ub = 2 * maximum(ỹ0)    
    mul!(ϕ0, X0, β)
    @. ϕ0 = 1.0 / (1.0 + exp(ϕ0))
    for i in 1:N0
        if !event0[i]
            # Choose a cluster
            di = rmax(skl)
            tmp = rand()
            cumsum_w = 0.0
            for j in 1:(rmax(skl) - 1)
                cumsum_w += get_w(ϕ0[i], 2, j)
                if tmp <= cumsum_w
                    di = j
                    break
                end
            end
            dist = Normal(b[di] ⋅ X0vec[i], 1 / √τ[di])
            tdist = Truncated(dist, ỹ0[i], ub)
            y0[i] = rand(tdist)
        end
    end    
end

function update_g!(m::DGSBPNormalDependent)
    (; a0_τ, B0_b, b0_τ, m0_b, mapping, skl, update_g, XX, Xy, yy) = m
    (; D0, n, rmodel) = skl
    (; g, ζ0g, μ0β, Σ0β) = rmodel
    gexp = zeros(Bool, D0)
    for d in 1:length(mapping)
        gexp[mapping[d]] .= g[d]
    end
    pg = Womack(length(g), ζ0g)
    update_suffstats!(m)
    # Old mechanism
    # for d in 1:length(g)
    #     update_g[d] || continue
    #     logodds = 0.0
    #     for val in 0:1
    #         g[d] = val
    #         gexp[mapping[d]] .= val
    #         # Find the posterior hyperparameters related to the atoms
    #         logQ = 0.0
    #         for j in 1:rmax(skl)
    #             B1_b = inv((Symmetric(XX[j], :L) + inv(B0_b) + √eps(Float64) * I(D0))[gexp, gexp])
    #             m1_b = B1_b * (Xy[j] + B0_b \ m0_b)[gexp]
    #             a1_τ = a0_τ + n[j] / 2
    #             b1_τ = b0_τ + (yy[j] + m0_b ⋅ (B0_b \ m0_b) - m1_b ⋅ (B1_b \ m1_b)) / 2
    #             logQ += (
    #                 0.5 * logdet(B1_b) -
    #                 0.5 * logdet(B0_b) + 
    #                 loggamma(a1_τ / 2) -
    #                 loggamma(a0_τ / 2) + 
    #                 (a0_τ / 2) * log(b0_τ) -
    #                 (a1_τ / 2) * log(b1_τ) -
    #                 (n[j] / 2) * log(2 * π)
    #             )
    #         end
    #         # Compute the contribution to the logodds
    #         m1, Σ1 = BNB.posterior_hyperparameters(rmodel)
    #         logodds += (-1)^(val + 1) * (
    #             logQ + 
    #             logpdf(pg, g) +
    #             logpdf(MvNormal(μ0β[gexp], Σ0β[gexp, gexp]), zeros(sum(gexp))) -
    #             logpdf(MvNormal(m1, Σ1), zeros(length(m1)))
    #         )
    #     end
    #     g[d] = rand() < exp(logodds) / (1.0 + exp(logodds))
    #     gexp[mapping[d]] .= g[d]
    # end
    # return nothing

    # New mechanism ------------------------------

    # Select the component to be updated
    updateable = (1:length(g))[update_g]
    d = rand(updateable)
    gd_old = g[d]

    # Compute the log-odds
    logodds = 0.0
    for val in 0:1
        g[d] = val
        gexp[mapping[d]] .= val
        # Find the posterior hyperparameters related to the atoms
        logQ = 0.0
        for j in 1:rmax(skl)
            B1_b = inv((Symmetric(XX[j], :L) + inv(B0_b) + √eps(Float64) * I(D0))[gexp, gexp])
            m1_b = B1_b * (Xy[j] + B0_b \ m0_b)[gexp]
            a1_τ = a0_τ + n[j] / 2
            b1_τ = b0_τ + (yy[j] + m0_b ⋅ (B0_b \ m0_b) - m1_b ⋅ (B1_b \ m1_b)) / 2
            logQ += (
                0.5 * logdet(B1_b) -
                0.5 * logdet(B0_b) + 
                loggamma(a1_τ / 2) -
                loggamma(a0_τ / 2) + 
                (a0_τ / 2) * log(b0_τ) -
                (a1_τ / 2) * log(b1_τ) -
                (n[j] / 2) * log(2 * π)
            )
        end
        # Compute the contribution to the logodds
        m1, Σ1 = BNB.posterior_hyperparameters(rmodel)
        logodds += (-1)^(val + 1) * (
            logQ + 
            logpdf(pg, g) +
            logpdf(MvNormal(μ0β[gexp], Σ0β[gexp, gexp]), zeros(sum(gexp))) -
            logpdf(MvNormal(m1, Σ1), zeros(length(m1)))
        )
    end

    # Compute the acceptance probability
    ap = gd_old == false ? exp(logodds) : exp(-logodds)

    # Update g[d] using a MH step 
    g[d] = rand() < ap ? !gd_old : gd_old
    gexp[mapping[d]] .= g[d]
    return nothing
end
