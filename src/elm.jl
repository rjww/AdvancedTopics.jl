struct ELM{T₁ <: Number, T₂ <: ActivationFunction}
    input_weights::Matrix{T₁}
    output_weights::Matrix{T₁}
    activation_functions::Vector{T₂}

    function ELM{T₁}(samples::T₂,
                     targets::T₃,
                     input_weights::T₄,
                     activation_functions::Vector{T₅};
                     batch_size::Int = 1000) where {T₁ <: Number,
                                                    T₂ <: AbstractMatrix,
                                                    T₃ <: AbstractVector,
                                                    T₄ <: AbstractMatrix,
                                                    T₅ <: ActivationFunction}
        X = samples
        t = targets
        W = input_weights
        L = first(size(W))
        fs = activation_functions
        B = train_output_weights(T₁, X, t, W, fs, batch_size)
        new{T₁,eltype(fs)}(W, B, fs)
    end

    function ELM{T₁}(samples::T₂,
                     targets::T₃,
                     n_neurons::Int,
                     activation_function::T₄;
                     batch_size::Int = 1000) where {T₁ <: Number,
                                                    T₂ <: AbstractMatrix,
                                                    T₃ <: AbstractVector,
                                                    T₄ <: NaiveActivationFunction}
        X = samples
        t = targets
        L = n_neurons
        f = activation_function
        D = first(size(X))
        W = gaussian_projection_matrix(T₁, L, D)
        ELM{T₁}(X, t, W, f, batch_size = batch_size)
    end
end

function train_output_weights(::Type{T₁},
                              samples::T₂,
                              targets::T₃,
                              input_weights::T₄,
                              activation_functions::Vector{T₅},
                              batch_size::Int) where {T₁ <: Number,
                                                      T₂ <: AbstractMatrix,
                                                      T₃ <: AbstractVector,
                                                      T₄ <: AbstractMatrix,
                                                      T₅ <: ActivationFunction}
    X = samples
    t = targets
    ψ = calculate_sample_weights(t)
    W = input_weights
    fs = activation_functions
    L = length(fs)
    D, N = size(X)

    HH = zeros(T₁, L + 1, L + 1)
    TH = zeros(T₁, 1, L + 1)

    for batch in partition_range(1:N, batch_size)
        X₀ = @view X[:,batch]
        t₀ = @view t[batch]
        ψ₀ = @view ψ[batch]
        train_on_batch!(HH, TH, X₀, t₀, ψ₀, W, fs)
    end

    output_weights = TH * LinearAlgebra.pinv(HH)
end

function train_on_batch!(HH::T₁,
                         TH::T₂,
                         samples::T₃,
                         targets::T₄,
                         sample_weights::T₅,
                         input_weights::T₆,
                         activation_functions::Vector{T₇}) where {T₁ <: AbstractMatrix,
                                                                  T₂ <: AbstractMatrix,
                                                                  T₃ <: AbstractMatrix,
                                                                  T₄ <: AbstractVector,
                                                                  T₅ <: AbstractVector,
                                                                  T₆ <: AbstractMatrix,
                                                                  T₇ <: ActivationFunction}
    X = samples
    T = reshape(targets, 1, :)
    Ψ = LinearAlgebra.Diagonal(sample_weights)
    W = input_weights
    fs = activation_functions
    N = last(size(X))

    H = project(X, W, fs)
    H = [H; ones(eltype(H), 1, N)]

    H = H * Ψ
    T = T * Ψ

    HH .+= (H * H')
    TH .+= (T * H')

    HH, TH
end

function predict(elm::T₁,
                 samples::T₂) where {T₁ <: ELM,
                                     T₂ <: AbstractMatrix}
    X = samples
    W = elm.input_weights
    B = elm.output_weights
    fs = elm.activation_functions
    N = last(size(X))

    H = project(X, W, fs)
    H = [H; ones(eltype(H), 1, N)]

    y = vec(B * H)
end

function predict(elm::T₁,
                 sample::T₂) where {T₁ <: ELM,
                                    T₂ <: AbstractVector}
    y = predict(elm, reshape(sample, :, 1))
    first(y)
end
