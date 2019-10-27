function calculate_sample_weights(targets::T) where {T <: AbstractVector}
    t = targets
    N = length(t)
    ψ₀ = sum(t .!= 1) / N
    ψ₁ = sum(t .== 1) / N
    ψ = [q == 1 ? ψ₁ : ψ₀ for q in t]
    LinearAlgebra.Diagonal(ψ)
end

function gaussian_projection_matrix(::Type{T},
                                    n_neurons::Int,
                                    n_features::Int) where {T <: Number}
    randn(T, n_neurons, n_features) ./ sqrt(n_features)
end

function project(samples::T₁,
                 weights::T₂,
                 activation_functions::Vector{T₃}) where {T₁ <: AbstractMatrix,
                                                          T₂ <: AbstractMatrix,
                                                          T₃ <: ActivationFunction}
    X = samples
    W = weights
    fs = activation_functions
    L = length(fs)

    H = W * X
    for l in 1:L
        H[l,:] .= fs[l].(H[l,:])
    end
    
    H
end
