# using Cairo
# using Gadfly
# using DataFrames
# using HypergeometricFunctions
# using SpecialFunctions
# using CategoricalArrays
# using CSV 

# function weight(j, s, xi)
#     exp(
#         logabsbinomial(j + s - 2, j - 1)[1] -
#         log(j) +
#         s * log(xi) +
#         (j - 1) * log(1 - xi) +
#         log(_₂F₁(j + s - 1, 1, j + 1, 1 - xi))
#     )
# end

# df_j = DataFrame(j = 1:20)
# df_xi = DataFrame(xi = [0.2, 0.4, 0.6])
# df_s = DataFrame(s = [1, 2, 3, 4])
# df = crossjoin(df_j, df_xi, df_s)
# df[!, :wj] = weight.(df[!, :j], df[!, :s], df[!, :xi])
# df[!, :s] = CategoricalArray(df[!, :s])
# CSV.write("weights.csv", df)

# import Pkg; 
# Pkg.activate()
# Pkg.add([
#     "AlgebraOfGraphics", 
#     "CategoricalArrays", 
#     "CairoMakie", 
#     "DataFrames", 
#     "HypergeometricFunctions", 
#     "LaTeXStrings",
#     "SpecialFunctions"
# ])
# Pkg.activate("")
using AlgebraOfGraphics, CategoricalArrays, CairoMakie, DataFrames
using LaTeXStrings, HypergeometricFunctions, SpecialFunctions
set_aog_theme!()
update_theme!(font = "Arial", fontsize = 28)
theme_minimal
function weight(j, s, xi)
    exp(
        logabsbinomial(j + s - 2, j - 1)[1] -
        log(j) +
        s * log(xi) +
        (j - 1) * log(1 - xi) +
        log(_₂F₁(j + s - 1, 1, j + 1, 1 - xi))
    )
end

df_j = DataFrame(j = 1:20);
df_ϕ = DataFrame(ϕ = [0.1, 0.3, 0.5]);
df_s = DataFrame(s = [1, 2, 3, 4]);
df = crossjoin(df_j, df_ϕ, df_s);
df[!, :w] = weight.(df[!, :j], df[!, :s], df[!, :ϕ]);
df[!, :s] = string.(df[!, :s]);
df[!, :ϕ] = string.(df[!, :ϕ]);
df[!, :ϕ] = LaTeXString.("φ(x) = " .* df[!, :ϕ]);
plt = data(df) *
    visual(Lines) *
    mapping(:j, :w, col = :ϕ, color = :s)
axis = (xlabel = L"j", ylabel = L"w_j(x)", color = L"s");
fg = draw(plt; axis, figure = (resolution = (900, 450), px_per_unit = 8))
fg
save("fig-01.pdf", fg, px_per_unit = 3)