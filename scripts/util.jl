using Plots

function plot_decision_boundary(elm::T₁,
                                samples::T₂,
                                targets::T₃) where {T₁ <: ELM,
                                                    T₂ <: AbstractMatrix,
                                                    T₃ <: AbstractVector}
    x, y = samples[1,:], samples[2,:]
    colours = [label == 1 ? "#F8766D" : "#00BFC4" for label in targets]
    rng = -10:0.1:10
    plt = scatter(x, y,
            markersize = 4,
            markerstrokewidth = 0,
            markercolor = colours,
            xaxis = false,
            yaxis = false,
            grid = false,
            legend = false)
    plt = contour!(rng, rng, (x, y) -> predict(elm, [x, y]),
            levels = [0],
            linewidth = 1,
            linecolor = :black,
            legend = false)
end
