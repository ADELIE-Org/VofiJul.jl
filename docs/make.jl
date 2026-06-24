using Documenter
using VofiJul

makedocs(
    modules = [VofiJul],
    authors = "ADELIE-org contributors",
    sitename = "VofiJul.jl",
    format = Documenter.HTML(
        canonical = "https://ADELIE-org.github.io/VofiJul.jl",
        repolink = "https://github.com/ADELIE-org/VofiJul.jl",
        collapselevel = 2,
    ),
    pages = [
        "Home" => "index.md",
        "API Reference" => "reference.md",
    ],
    pagesonly = true,
    warnonly = true,
    remotes = nothing,
)

# Only deploy docs if running in CI environment
if get(ENV, "CI", "") == "true"
    deploydocs(
        repo = "github.com/ADELIE-org/VofiJul.jl",
        push_preview = true,
    )
end
