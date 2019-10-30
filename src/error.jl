function calculate_error(predictions::T₁,
                         targets::T₂) where {T₁ <: AbstractVector,
                                             T₂ <: AbstractVector}
    y = predictions
    t = targets
    N = length(y)
    misclassified = sum(sign.(y) .!= sign.(t))
    return sum(sign.(y) .!= sign.(t)) / N, misclassified
end
