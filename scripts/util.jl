using Plots

function build_histogram(predictions::T₁,
                         targets::T₂) where {T₁ <: AbstractVector,
                                             T₂ <: AbstractVector}
    y = predictions
    t = targets
    indices = t .== 1
    bins = -2:0.03:2
    hist = histogram(y[indices], bins = bins, fill = (0, 0.5, :red), linecolor = false, legend = false)
    hist = histogram!(y[.!indices], bins = bins, fill = (0, 0.5, :green), linecolor = false, legend = false)
end

function plot_decision_boundary(elm::T₁,
                                samples::T₂,
                                targets::T₃) where {T₁ <: ELM,
                                                    T₂ <: AbstractMatrix,
                                                    T₃ <: AbstractVector}
    X = samples
    t = targets
    indices = t .== 1
    rng = -15:0.5:15
    plt = plot(X[1,indices], X[2,indices], seriestype = :scatter, color = :green, legend = false)
    plt = plot!(X[1,.!indices], X[2,.!indices], seriestype = :scatter, color = :red, legend = false)
    plt = contour!(rng, rng, (x, y) -> predict(elm, [x, y]), levels = [0], color = :black, legend = false)
end

function plot_decision_boundary(sieve::T₁,
                                samples::T₂,
                                targets::T₃) where {T₁ <: Sieve,
                                                    T₂ <: AbstractMatrix,
                                                    T₃ <: AbstractVector}
    X = samples
    t = targets
    indices = t .== 1
    rng = -15:0.5:15
    plt = plot(X[1,indices], X[2,indices], seriestype = :scatter, color = :green, legend = false)
    plt = plot!(X[1,.!indices], X[2,.!indices], seriestype = :scatter, color = :red, legend = false)
    plt = contour!(rng, rng, (x, y) -> predict(sieve, [x, y]), levels = [0], color = :black, legend = false)
end
