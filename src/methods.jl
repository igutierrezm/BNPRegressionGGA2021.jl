function sample!(m::AbstractModel; mcmcsize = 4000, burnin = mcmcsize ÷ 2, thin = 1)
    (; N1, D1, f, β, g) = skeleton(m)
    chainf = [zeros(N1) for _ in 1:(mcmcsize - burnin) ÷ thin]
    chainβ = [zeros(D1) for _ in 1:(mcmcsize - burnin) ÷ thin]
    chaing = [zeros(Bool, length(g)) for _ in 1:(mcmcsize - burnin) ÷ thin]
    for iter in 1:mcmcsize
        step!(m)
        if iter > burnin && iszero((iter - burnin) % thin)
            chainf[(iter - burnin) ÷ thin] .= f
            chainβ[(iter - burnin) ÷ thin] .= β
            chaing[(iter - burnin) ÷ thin] .= g
        end
    end
    return chainf, chainβ, chaing
end

function step!(m::AbstractModel)
    update_f!(m)
    update_d!(m)
    update_r!(m)
    update_n!(m)
    update_β!(m)
    update_atoms!(m) # see interface.jl
end

function update_f!(m::AbstractModel)
    (; N1, y1, X1, X1vec, β, ϕ1, f, s, rmax) = skeleton(m)
    mul!(ϕ1, X1, β)
    @. ϕ1 = 1.0 / (1.0 + exp(ϕ1))
    for i in 1:N1
        f[i] = 0.0
        for j in 1:rmax[]
            wj = get_w(ϕ1[i], s[], j)
            f[i] += wj * kernel_pdf(m, y1[i], X1vec[i], j)
        end
    end
end

function get_w(θ0, s0, j)
    # exp(
    #     - log(j) +
    #     logabsbinomial(j + s0 - 2, j - 1)[1] +
    #     s0 * log(θ0) +
    #     (j - 1) * log(1 - θ0) +
    #     log(_₂F₁(BigFloat(j + s0 - 1), 1, j + 1, 1 - θ0))
    # )
    return θ0 * (1 - θ0)^(j - 1)
end

function update_d!(m::AbstractModel)
    (; N0, y0, X0vec, d, r, rmax) = skeleton(m)
    p = zeros(rmax[])
    for i in 1:N0
        resize!(p, r[i])
        for j in 1:r[i]
            p[j] = eps() + kernel_pdf(m, y0[i], X0vec[i], j)
        end
        p ./= sum(p)
        d[i] = rand(Categorical(p))
    end
    return nothing    
end

function update_r!(m::AbstractModel)
    (; N0, X0, d, r, ϕ0, s, β, rmax) = skeleton(m)
    mul!(ϕ0, X0, β)
    @. ϕ0 = 1.0 / (1.0 + exp(ϕ0))
    for i in 1:N0
        zi = rand(Beta(r[i] - d[i] + 1, d[i]))
        vi = rand(Gamma(r[i] + s[] - 1, 1))
        mi = rand(Poisson((1 - ϕ0[i]) * zi * vi)) # is ϕ0 or 1 - ϕ0? Verify this.
        r[i] = mi + d[i]
    end
    rmax[] = maximum(r)
    return nothing
end

function update_n!(m::AbstractModel)
    (; N0, d, n, rmax) = skeleton(m)
    while length(n) < rmax[] 
        push!(n, 0)
    end
    n .= 0
    for i in 1:N0
        n[d[i]] += 1
    end
    return nothing
end

function update_β!(m::AbstractModel)
    (; r, rmodel) = skeleton(m)
    @. rmodel.y = r - 1
    BNB.step!(Random.GLOBAL_RNG, rmodel)
    return nothing
end

struct Womack <: DiscreteMultivariateDistribution
    D::Int
    ζ::Float64
    p::Vector{Float64}
    punnormalized::Vector{Float64}
    function Womack(D::Int, ζ::Float64)
        p = big.([zeros(D); 1.0])
        for d1 in (D - 1):-1:0
            for d2 in 1:(D - d1)
                p[1 + d1] += (
                    ζ * p[1 + d1 + d2] * binomial(big(d1 + d2), big(d1))
                )
            end
        end
        p /= sum(p)
        punnormalized = copy(p)
        for d1 in 1:D
            p[d1] /= binomial(big(D), big(d1 - 1))
        end
        return new(D, ζ, p, punnormalized)
    end
end

function pdf(d::Womack, g::Vector{Bool})
    return d.p[sum(g) + 1]
end

function logpdf(d::Womack, g::Vector{Bool})
    return log(pdf(d, g))
end