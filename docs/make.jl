using Documenter, AdvancedTopics

makedocs(;
    modules=[AdvancedTopics],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/rjww/AdvancedTopics.jl/blob/{commit}{path}#L{line}",
    sitename="AdvancedTopics.jl",
    authors="Robert Woods",
    assets=String[],
)

deploydocs(;
    repo="github.com/rjww/AdvancedTopics.jl",
)
