# Returns a partitioning of `range` into sub-ranges of length `partition_size`.
# If `range` is not divisible by `partition_size`, the last sub-range will be
# of length `range` mod `partition_size`.
function partition_range(range::UnitRange, partition_size::Int)
    n_partitions = ceil(Int, last(range) / partition_size)
    partitions = Vector{UnitRange{Int}}(undef, n_partitions)

    for p in 1:n_partitions
        l::Int = first(range) + (p-1) * partition_size
        r::Int = min(l - 1 + partition_size, last(range))
        partitions[p] = l:r
    end

    return partitions
end

function calculate_sample_weights(targets::T) where {T <: AbstractVector}
    t = targets
    N = length(t)
    ψ₀ = sum(t .!= 1) / N
    ψ₁ = sum(t .== 1) / N
    ψ = [q == 1 ? ψ₁ : ψ₀ for q in t]
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
