using BNPRegressionGGA2021
using Documenter

DocMeta.setdocmeta!(BNPRegressionGGA2021, :DocTestSetup, :(using BNPRegressionGGA2021); recursive=true)

makedocs(;
    modules=[BNPRegressionGGA2021],
    authors="Iván Gutiérrez <ivangutierrez1988@gmail.com> and contributors",
    repo="https://github.com/igutierrezm/BNPRegressionGGA2021.jl/blob/{commit}{path}#{line}",
    sitename="BNPRegressionGGA2021.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://igutierrezm.github.io/BNPRegressionGGA2021.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/igutierrezm/BNPRegressionGGA2021.jl",
)
