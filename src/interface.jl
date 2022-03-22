"""
    skeleton(m::AbstractModel)

Return the model skeleton.
"""
skeleton(m::AbstractModel)

"""
    update_atoms!(m::AbstractModel)
Apply 1 Gibbs update to the cluster atoms.
"""
update_atoms!(m::AbstractModel)

"""
    kernel_pdf(m::AbstractModel, y0, x0, j::Int)
    
Return the kernel pdf at `y0`, given a predictor `x0` a cluster label `j`.
"""
kernel_pdf(m::AbstractModel, y0, x0, j::Int)
# ok: 2022-03-22