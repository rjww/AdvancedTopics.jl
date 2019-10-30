struct SieveLayer{T₁ <: Number, T₂ <: KDEComparator}
    input_weights::Matrix{T₁}
    comparators::Vector{T₂}

    function SieveLayer(input_weights::Matrix{T₁},
                        comparators::Vector{T₂}) where {T₁ <: Number,
                                                        T₂ <: KDEComparator}
        new{T₁,T₂}(input_weights, comparators)
    end
end

abstract type Sieve end

struct DataPassingSieve{T <: Number} <: Sieve
    n_neurons::Int
    layers::Vector{SieveLayer{T}}
    output_weights::Matrix{T}
    consensus_threshold::Real

    function DataPassingSieve{T₁}(samples::T₂,
                                  targets::T₃,
                                  n_neurons::Int;
                                  consensus_threshold::Real = 0.9,
                                  max_layers::Int = 3,
                                  min_split::Int = 2) where {T₁ <: Number,
                                                             T₂ <: AbstractMatrix,
                                                             T₃ <: AbstractVector}
        X = samples
        t = targets
        L = n_neurons
        N = last(size(X))
        H = zeros(T₁, L * max_layers, N)
        layers = Vector{SieveLayer{T₁}}()

        X₀ = X
        t₀ = t
        active_samples = Set{Int}(1:N)
        indices = get_indices(active_samples, N)

        for depth in 1:max_layers
            if length(active_samples) < min_split || all(t .== 1) || all(t .!= 1)
                break
            end

            i₀ = (depth-1) * L + 1
            i₁ = i₀ + L - 1

            H₀ = @view H[i₀:i₁,indices]
            D, N₀ = size(X₀)

            W, fs = train_kde_comparators(T₁, X₀, t₀, L)
            H₀ .= project(X₀, W, fs)
            push!(layers, SieveLayer(W, fs))

            to_remove = Set{Int}()
            for n in active_samples
                if consensus(H[i₀:i₁,n], consensus_threshold)
                    push!(to_remove, n)
                end
            end
            setdiff!(active_samples, to_remove)
            indices = get_indices(active_samples, N)

            X₀ = @view X[:,indices]
            t₀ = @view t[indices]
        end

        H = [H; ones(eltype(H), 1, N)]

        Ψ = calculate_sample_weights(t)
        H = H * Ψ
        T = reshape(t, 1, :) * Ψ

        B = (T * H') * LinearAlgebra.pinv(H * H')

        new{T₁}(L, layers, B, consensus_threshold)
    end
end

function predict(sieve::T₁,
                 samples::T₂) where {T₁ <: DataPassingSieve,
                                     T₂ <: AbstractMatrix}
    param(::DataPassingSieve{T}) where {T} = T

    X = samples
    L = sieve.n_neurons
    N = last(size(X))
    H = zeros(param(sieve), L * length(sieve.layers), N)

    X₀ = X
    active_samples = Set{Int}(1:N)
    indices = get_indices(active_samples, N)

    for (depth, layer) in enumerate(sieve.layers)
        i₀ = (depth-1) * L + 1
        i₁ = i₀ + L - 1

        H₀ = @view H[i₀:i₁,indices]
        D, N₀ = size(X₀)

        W = layer.input_weights
        fs = layer.comparators
        H₀ .= project(X₀, W, fs)

        to_remove = Set{Int}()
        for n in active_samples
            if consensus(H[i₀:i₁,n], sieve.consensus_threshold)
                push!(to_remove, n)
            end
        end
        setdiff!(active_samples, to_remove)
        indices = get_indices(active_samples, N)

        X₀ = @view X[:,indices]
    end

    H = [H; ones(eltype(H), 1, N)]
    B = sieve.output_weights
    y = vec(B * H)
end

function predict(sieve::T₁,
                 sample::T₂) where {T₁ <: DataPassingSieve,
                                    T₂ <: AbstractVector}
    y = predict(sieve, reshape(sample, :, 1))
    first(y)
end

struct ProjectionPassingSieve{T <: Number} <: Sieve
    n_neurons::Int
    layers::Vector{SieveLayer{T}}
    output_weights::Matrix{T}
    consensus_threshold::Real

    function ProjectionPassingSieve{T₁}(samples::T₂,
                                        targets::T₃,
                                        n_neurons::Int;
                                        consensus_threshold::Real = 0.9,
                                        max_layers::Int = 3,
                                        min_split::Int = 2) where {T₁ <: Number,
                                                                   T₂ <: AbstractMatrix,
                                                                   T₃ <: AbstractVector}
        X = samples
        t = targets
        L = n_neurons
        N = last(size(X))
        H = zeros(T₁, L * max_layers, N)
        layers = Vector{SieveLayer{T₁}}()

        X₀ = X
        t₀ = t
        active_samples = Set{Int}(1:N)
        indices = get_indices(active_samples, N)

        for depth in 1:max_layers
            if length(active_samples) < min_split || all(t .== 1) || all(t .!= 1)
                break
            end

            i₀ = (depth-1) * L + 1
            i₁ = i₀ + L - 1

            H₀ = @view H[i₀:i₁,indices]
            D, N₀ = size(X₀)

            W, fs = train_kde_comparators(T₁, X₀, t₀, L)
            H₀ .= project(X₀, W, fs)
            push!(layers, SieveLayer(W, fs))

            to_remove = Set{Int}()
            for n in active_samples
                if consensus(H[i₀:i₁,n], consensus_threshold)
                    push!(to_remove, n)
                end
            end
            setdiff!(active_samples, to_remove)
            indices = get_indices(active_samples, N)

            X₀ = @view H[i₀:i₁,indices]
            t₀ = @view t[indices]
        end

        H = [H; ones(eltype(H), 1, N)]

        Ψ = calculate_sample_weights(t)
        H = H * Ψ
        T = reshape(t, 1, :) * Ψ

        B = (T * H') * LinearAlgebra.pinv(H * H')

        new{T₁}(L, layers, B, consensus_threshold)
    end
end

function predict(sieve::T₁,
                 samples::T₂) where {T₁ <: ProjectionPassingSieve,
                                     T₂ <: AbstractMatrix}
    param(::ProjectionPassingSieve{T}) where {T} = T

    X = samples
    L = sieve.n_neurons
    N = last(size(X))
    H = zeros(param(sieve), L * length(sieve.layers), N)

    X₀ = X
    active_samples = Set{Int}(1:N)
    indices = get_indices(active_samples, N)

    for (depth, layer) in enumerate(sieve.layers)
        i₀ = (depth-1) * L + 1
        i₁ = i₀ + L - 1

        H₀ = @view H[i₀:i₁,indices]
        D, N₀ = size(X₀)

        W = layer.input_weights
        fs = layer.comparators
        H₀ .= project(X₀, W, fs)

        to_remove = Set{Int}()
        for n in active_samples
            if consensus(H[i₀:i₁,n], sieve.consensus_threshold)
                push!(to_remove, n)
            end
        end
        setdiff!(active_samples, to_remove)
        indices = get_indices(active_samples, N)

        X₀ = @view H[i₀:i₁,indices]
    end

    H = [H; ones(eltype(H), 1, N)]
    B = sieve.output_weights
    y = vec(B * H)
end

function predict(sieve::T₁,
                 sample::T₂) where {T₁ <: ProjectionPassingSieve,
                                    T₂ <: AbstractVector}
    y = predict(sieve, reshape(sample, :, 1))
    first(y)
end

function consensus(samples::T₁,
                   consensus_threshold::Real) where {T₁ <: AbstractVector}
    y = sign.(samples)
    th = consensus_threshold
    f(x) = x == 0 || x == 1
    g(x) = x == 0 || x == -1

    all(f.(y)) || all(g.(y))
end

function get_indices(active_samples::Set{Int}, N::Int)
    [n in active_samples for n in 1:N]
end
