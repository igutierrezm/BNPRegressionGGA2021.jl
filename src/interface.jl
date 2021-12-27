"""
    skeleton(m::AbstractModel)

Return the model skeleton.
"""
function skeleton(m::AbstractModel)
    error("not implemented")
end

"""
    update_atoms!(m::AbstractModel)
Apply 1 Gibbs update to the cluster atoms.
"""
function update_atoms!(m::AbstractModel)
    error("not implemented.")
end

"""
    kernel_pdf(m::AbstractModel, y0, j::Int)
    
Return the kernel pdf at `y0`, given a cluster label equal to `j`.
"""
function kernel_pdf(m::AbstractModel, y0, j::Int)
    error("not implemented")
end

"""
    y(m::AbstractModel)
    
Return the outcome vector.
"""
function kernel_pdf(m::AbstractModel, y0, j::Int)
    error("not implemented")
end