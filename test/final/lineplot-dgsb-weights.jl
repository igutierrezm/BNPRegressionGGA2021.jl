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
df[!, :ϕ] = LaTeXString.("φ(x_i ⋅ β) = " .* df[!, :ϕ]);
plt = data(df) *
    visual(Lines) *
    mapping(:j, :w, col = :ϕ, color = :s)
axis = (xlabel = L"j", ylabel = L"w_j(x)", color = L"s");
fg = draw(plt; axis, figure = (resolution = (900, 450), px_per_unit = 8))
fg
save("figures/final/lineplot-dgsb-weights.pdf", fg, px_per_unit = 3)