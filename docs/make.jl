using OrdinaryDiffEqOperatorSplitting
using Documenter, DocumenterCitations

DocMeta.setdocmeta!(OrdinaryDiffEqOperatorSplitting, :DocTestSetup,
    :(using OrdinaryDiffEqOperatorSplitting); recursive = true)

const is_ci = haskey(ENV, "GITHUB_ACTIONS")

bibtex_plugin = CitationBibliography(
    joinpath(@__DIR__, "src", "assets", "references.bib"),
    style = :numeric
)

# Build documentation.
makedocs(
    format = Documenter.HTML(
        assets = [
            "assets/citations.css",
        # "assets/favicon.ico"
        ],
        # canonical = "https://localhost/",
        collapselevel = 1
    ),
    sitename = "OrdinaryDiffEqOperatorSplitting.jl",
    doctest = false,
    warnonly = true,
    draft = false,
    pages = Any[
        "Home" => "index.md",
        "usage/index.md",
        "Theory Manual" => "topics/time-integration.md",
        "api-reference/index.md",
        "devdocs/index.md",
        "references.md"
    ],
    plugins = [
        bibtex_plugin,
    ]
)

# Deploy built documentation
deploydocs(
    repo = "github.com/SciML/OrdinaryDiffEqOperatorSplitting.jl",
    push_preview = true,
    devbranch = "main",
    versions = [
        "stable" => "v^",
        "dev" => "dev"
    ]
)
