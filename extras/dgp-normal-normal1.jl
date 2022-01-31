using Base.Iterators
using Distributions
using LinearAlgebra
using Random
using RCall

# Define the DGP
function simulate_sample(N0, N1)
    # Set the conditional density
    function dy(xc) 
        atoms = [(-xc, 0.2), (xc, 0.2)]
        weights = [0.5, 0.5]
        # weights = [exp(-2xc), 1 - exp(-2xc)]
        MixtureModel(Normal, atoms, weights)
    end
    
    # Simulate a sample
    x0 = rand(N0)
    y0 = [rand(dy(xc)) for xc in x0]
    X0 = rcopy(R"cbind(1, splines::bs($x0, df = 6, Boundary.knots = c(0, 1)))")

    # Generate the grid points
    y_grid = LinRange(-3.0, 3.0, N1)
    x_grid = [0.25, 0.5, 0.75]
    y1, x1 = Iterators.product(y_grid, x_grid) |> x -> zip(x...) .|> collect
    X1 = rcopy(R"cbind(1, predict(splines::bs($x0, df = 6, Boundary.knots = c(0, 1)), $x1))")
    return dy, y0, x0, X0, y1, x1, X1
end

# # Not run
# Random.seed!(1);
# N0, N1 = 500, 50;
# dy, y0, x0, X0, y1, x1, X1 = simulate_sample(N0, N1);
