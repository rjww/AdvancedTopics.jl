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
                     T₂ <: KernelDensity.UnivariateKDE} <: TrainedActivationFunction
    pdf₁::KernelDensity.InterpKDE{T₁}
    pdf₂::KernelDensity.InterpKDE{T₂}

    function KDEComparator(samples::T₁,
                           targets::T₂,
                           weights::T₃) where {T₁ <: AbstractMatrix,
                                               T₂ <: AbstractVector,
                                               T₃ <: AbstractVector}
        X = samples
        t = targets
        w = weights

        h = vec(w' * X)
        h₀ = h[t .!= 1]
        h₁ = h[t .== 1]

        boundary = (min(minimum(h₀), minimum(h₁)) - 1,
                    max(maximum(h₀), maximum(h₁)) + 1)

        kde₀ = KernelDensity.kde_lscv(h₀, npoints = 100, boundary = boundary)
        kde₁ = KernelDensity.kde_lscv(h₁, npoints = 100, boundary = boundary)

        pdf₀ = KernelDensity.InterpKDE(kde₀)
        pdf₁ = KernelDensity.InterpKDE(kde₁)

        param(::KernelDensity.InterpKDE{T}) where {T} = T
        new{param(pdf₀),param(pdf₁)}(pdf₀, pdf₁)
    end
end

function (c::KDEComparator)(x::T) where {T <: Number}
    Distributions.pdf(c.pdf₁, x) - Distributions.pdf(c.pdf₂, x)
end

function train_kde_comparators(::Type{T₁},
                               samples::T₂,
                               targets::T₃,
                               n_neurons::Int) where {T₁ <: Number,
                                                      T₂ <: AbstractMatrix,
                                                      T₃ <: AbstractVector}
    X = samples
    t = targets
    L = n_neurons
    D = first(size(X))
    W = gaussian_projection_matrix(T₁, L, D)
    fs = [KDEComparator(X, t, W[l,:]) for l in 1:L]
    W, fs
end
