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

function update_ϕ!(m::AbstractModel)
    (; X0, β, ϕ) = skeleton(m)
    mul!(ϕ, X0, β)
    @. ϕ = 1.0 / (1.0 + exp(ϕ))
end

function update_r!(m::AbstractModel)
    (; N0, d, r, ϕ, s) = skeleton(m)
    for i in 1:N0
        zi = rand(Beta(r[i] - d[i] + 1, d[i]))
        vi = rand(Gamma(r[i] + s[] - 1, 1))
        r[i] = rand(Poisson((1 - ϕ[i]) * zi * vi)) + d[i]
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

function update_β!(m::AbstractModel)
    (; r, rmodel) = skeleton(m)
    @. rmodel.y = r - 1
    BayesNegativeBinomial.step!(rmodel)
end

function update_f!(m::AbstractModel)
    (; N1, y1, X1, β, f, s) = skeleton(m)
    for i in 1:N1
        f[i] = 0.0
        ϕ1 = 1 / (1 + exp(X1[i, :] ⋅ β))
        for j in 1:m(m)
            wj = get_w(ϕ1, s[], j)
            f[i] += wj * f0(m, y1[i], j)
        end
    end
end

function step!(mdl::Sampler)
    update_atoms!(mdl)
    update_d!(mdl)
    update_f!(mdl)
    update_ϕ!(mdl)
    update_r!(mdl)
    update_β!(mdl)
end

function sample(smpl::Sampler; mcmcsize = 4000, burnin = 2000)
    (; N1, D1, f, β) = smpl
    chainf = [zeros(N1) for _ in 1:(mcmcsize - burnin)]
    chainβ = [zeros(D1) for _ in 1:(mcmcsize - burnin)]
    for iter in 1:mcmcsize
        step!(smpl)
        if iter > burnin
            chainf[iter - burnin] .= f
            chainβ[iter - burnin] .= β
        end
    end
    return chainf, chainβ
end
