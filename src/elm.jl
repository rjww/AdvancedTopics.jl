struct ELM{T₁ <: Number, T₂ <: ActivationFunction}
    input_weights::Matrix{T₁}
    output_weights::Matrix{T₁}
    activation_functions::Vector{T₂}

    function ELM{T₁}(samples::T₂,
                     targets::T₃,
                     input_weights::T₄,
                     activation_function::T₅) where {T₁ <: Number,
                                                     T₂ <: AbstractMatrix,
                                                     T₃ <: AbstractVector,
                                                     T₄ <: AbstractMatrix,
                                                     T₅ <: NaiveActivationFunction}
        X = samples
        t = targets
        W = input_weights
        L = first(size(W))
        fs = [activation_function for l in 1:L]
        D, N = size(X)

        H = project(X, W, fs)
        H = [H; ones(eltype(H), 1, N)]

        Ψ = calculate_sample_weights(t)
        H = H * Ψ
        T = reshape(t, 1, :) * Ψ

        β = (T * H') * LinearAlgebra.pinv(H * H')

        new{T₁,eltype(fs)}(W, β, fs)
    end

    function ELM{T₁}(samples::T₂,
                     targets::T₃,
                     n_neurons::Int,
                     activation_function::T₄) where {T₁ <: Number,
                                                     T₂ <: AbstractMatrix,
                                                     T₃ <: AbstractVector,
                                                     T₄ <: NaiveActivationFunction}
        X = samples
        t = targets
        L = n_neurons
        f = activation_function
        D = first(size(X))
        W = gaussian_projection_matrix(T₁, L, D)
        ELM{T₁}(X, t, W, f)
    end

    function ELM{T₁}(samples::T₂,
                     targets::T₃,
                     input_weights::T₄,
                     activation_functions::Vector{T₅}) where {T₁ <: Number,
                                                              T₂ <: AbstractMatrix,
                                                              T₃ <: AbstractVector,
                                                              T₄ <: AbstractMatrix,
                                                              T₅ <: TrainedActivationFunction}
        X = samples
        t = targets
        W = input_weights
        fs = activation_functions
        L = length(fs)
        D, N = size(X)

        H = project(X, W, fs)
        H = [H; ones(eltype(H), 1, N)]

        Ψ = calculate_sample_weights(t)
        H = H * Ψ
        T = reshape(t, 1, :) * Ψ

        β = (T * H') * LinearAlgebra.pinv(H * H')

        new{T₁,eltype(fs)}(W, β, fs)
    end
end

function predict(elm::T₁,
                 samples::T₂) where {T₁ <: ELM,
                                     T₂ <: AbstractMatrix}
    X = samples
    W = elm.input_weights
    β = elm.output_weights
    fs = elm.activation_functions
    N = last(size(X))
    H = project(X, W, fs)
    H = [H; ones(eltype(H), 1, N)]
    y = vec(β * H)

end

function predict(elm::T₁,
                 sample::T₂) where {T₁ <: ELM,
                                    T₂ <: AbstractVector}
    y = predict(elm, reshape(sample, :, 1))
    first(y)
end
