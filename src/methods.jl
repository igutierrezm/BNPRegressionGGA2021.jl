function sample!(m::AbstractModel; mcmcsize = 4000, burnin = 2000)
    (; N1, D1, f, β) = skeleton(m)
    chainf = [zeros(N1) for _ in 1:(mcmcsize - burnin)]
    chainβ = [zeros(D1) for _ in 1:(mcmcsize - burnin)]
    chaing = [zeros(Bool, D1) for _ in 1:(mcmcsize - burnin)]
    for iter in 1:mcmcsize
        step!(m)
        if iter > burnin
            chainf[iter - burnin] .= f
            chainβ[iter - burnin] .= β
            chaing[iter - burnin] .= (β .!= 0.0)
        end
    end
    return chainf, chainβ, chaing
end

function step!(m::AbstractModel)
    update_atoms!(m) # see interface.jl
    update_f!(m)
    update_d!(m)
    update_r!(m)
    update_n!(m)
    update_β!(m)
end

function update_f!(m::AbstractModel)
    (; N1, y1, X1, β, ϕ1, f, s, rmax) = skeleton(m)
    mul!(ϕ1, X1, β)
    @. ϕ1 = 1.0 / (1.0 + exp(ϕ1))
    for i in 1:N1
        f[i] = 0.0
        for j in 1:rmax[]
            wj = get_w(ϕ1[i], s[], j)
            f[i] += wj * kernel_pdf(m, y1[i], j)
        end
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

function update_d!(m::AbstractModel)
    (; N0, y0, d, r, rmax) = skeleton(m)
    p = zeros(rmax[])
    for i in 1:N0
        yi = y0[i]
        resize!(p, r[i])
        for j in 1:r[i]
            p[j] = kernel_pdf(m, yi, j)
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
        r[i] = rand(Poisson((1 - ϕ0[i]) * zi * vi)) + d[i]
    end
    rmax[] = maximum(r)
    return nothing
end

function update_n!(m::AbstractModel)
    (; N0, d, n, rmax) = skeleton(m)
    while length(n) < rmax[] 
        push!(n, 0)
    end
    for j in 1:rmax[]
        n[j] = 0
    end
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
