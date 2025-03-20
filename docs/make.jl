using OrdinaryDiffEqOperatorSplitting
using Documenter

DocMeta.setdocmeta!(OrdinaryDiffEqOperatorSplitting, :DocTestSetup, :(using OrdinaryDiffEqOperatorSplitting); recursive=true)

makedocs(;
    modules=[OrdinaryDiffEqOperatorSplitting],
    authors="termi-official <termi-official@users.noreply.github.com> and contributors",
    sitename="OrdinaryDiffEqOperatorSplitting.jl",
    format=Documenter.HTML(;
        canonical="https://termi-official.github.io/OrdinaryDiffEqOperatorSplitting.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/termi-official/OrdinaryDiffEqOperatorSplitting.jl",
    devbranch="main",
)
