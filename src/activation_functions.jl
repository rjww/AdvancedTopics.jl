abstract type ActivationFunction end
abstract type NaiveActivationFunction <: ActivationFunction end
abstract type TrainedActivationFunction <: ActivationFunction end

struct Cube <: NaiveActivationFunction end
(::Cube)(x::T) where {T <: Number} = x^3.0

struct Linear <: NaiveActivationFunction end
(::Linear)(x::T) where {T <: Number} = x

struct ReLU <: NaiveActivationFunction end
(::ReLU)(x::T) where {T <: Number} = max(zero(x), x)

struct Sigmoid <: NaiveActivationFunction end
(::Sigmoid)(x::T) where {T <: Number} = 1.0 / (1.0 + exp(-x))

struct Square <: NaiveActivationFunction end
(::Square)(x::T) where {T <: Number} = x^2.0

struct Tanh <: NaiveActivationFunction end
(::Tanh)(x::T) where {T <: Number} = tanh(x)

struct KDEComparator{T₁ <: KernelDensity.UnivariateKDE,
                     T₂ <: KernelDensity.UnivariateKDE,
                     T₃ <: AbstractArray} <: TrainedActivationFunction
    pdf₁::KernelDensity.InterpKDE{T₁}
    pdf₂::KernelDensity.InterpKDE{T₂}
    training_activations::T₃
    minval::Float64
    maxval::Float64


    function KDEComparator(samples::T₁,
                           targets::T₂,
                           weights::T₃,
                           importance::T₄;
                           boundary_offset::Int = 1) where {T₁ <: AbstractMatrix,
                                                            T₂ <: AbstractVector,
                                                            T₃ <: AbstractVector,
                                                            T₄ <: AbstractVector}
        X = samples
        t = targets
        w = weights

        # Map the projections to the interval [-1 1].
        h = vec(w' * X)
        minval = minimum(h)
        maxval = maximum(h)
        map!(x -> linear_stretch(x, minval, maxval, -1, 1), h, h)

        h₀ = h[t .!= 1]
        h₁ = h[t .== 1]

        importance₀ = importance[t .!= 1]
        importance₁ = importance[t .== 1]

        boundary = (min(minimum(h₀), minimum(h₁)) - boundary_offset,
                    max(maximum(h₀), maximum(h₁)) + boundary_offset)

        # weights=Weights(ones(X)/length(X))
        kde₀ = KernelDensity.kde_lscv(h₀, npoints = 1000, boundary = boundary, weights = importance₀ / sum(importance₀))
        kde₁ = KernelDensity.kde_lscv(h₁, npoints = 1000, boundary = boundary, weights = importance₁ / sum(importance₁))
        # kde₀ = KernelDensity.kde_lscv(h₀, npoints = 1000, boundary = boundary)
        # kde₁ = KernelDensity.kde_lscv(h₁, npoints = 1000, boundary = boundary)

        pdf₀ = KernelDensity.InterpKDE(kde₀)
        pdf₁ = KernelDensity.InterpKDE(kde₁)

        training_activations = [Distributions.pdf(pdf₀, h[n]) - Distributions.pdf(pdf₁, h[n]) for n = 1:length(h)]

        param(::KernelDensity.InterpKDE{T}) where {T} = T
        new{param(pdf₀),param(pdf₁), Vector{<: eltype(training_activations)}}(pdf₀, pdf₁, training_activations, minval, maxval)
    end
end

function (c::KDEComparator)(x::T) where {T <: Number}
    x′ = linear_stretch(x, c.minval, c.maxval, -1, 1)
    -(Distributions.pdf(c.pdf₁, x′) - Distributions.pdf(c.pdf₂, x′))
end

function train_kde_comparators(::Type{T₁},
                               samples::T₂,
                               targets::T₃,
                               n_neurons::Int;
                               boundary_offset::Int = 1) where {T₁ <: Number,
                                                                T₂ <: AbstractMatrix,
                                                                T₃ <: AbstractVector}
    X = samples
    t = targets
    L = n_neurons
    D = first(size(X))
    W = gaussian_projection_matrix(T₁, L, D)
    importance = ones(length(t))
    fs = [KDEComparator(X, t, W[1,:], importance, boundary_offset = boundary_offset)]
    #for l = 2:L
    #    estimator = fs[l-1]
    #    map(estimator)
    #end
    fs = [KDEComparator(X, t, W[l,:], importance, boundary_offset = boundary_offset) for l in 1:L]
    W, fs
end

# Map values in the range [A,B] to the range to [a,b]
function linear_stretch(x, A, B, a, b)
    return (x-A) * ((b-a)/(B-A)) + a
end
