using AdvancedTopics
using Plots

include("util.jl")

function perform_tests(training_samples::T₁,
                       training_targets::T₂,
                       testing_samples::T₃,
                       testing_targets::T₄,
                       n_neurons::Int) where {T₁ <: AbstractMatrix,
                                              T₂ <: AbstractVector,
                                              T₃ <: AbstractMatrix,
                                              T₄ <: AbstractVector}
    X₀ = training_samples
    t₀ = training_targets
    X₁ = testing_samples
    t₁ = testing_targets
    L = n_neurons

    W, fs = train_kde_comparators(Float64, X₀, t₀, 3 * L)

    elm_sig = ELM{Float64}(X₀, t₀, W, Sigmoid())
    elm_kde = ELM{Float64}(X₀, t₀, W, fs)
    sieve = Sieve{Float64}(X₀, t₀, L)

    y_sig = predict(elm_sig, X₁)
    y_kde = predict(elm_kde, X₁)
    y_sieve = predict(sieve, X₁)

    hist_sig = build_histogram(y_sig, t₁)
    hist_kde = build_histogram(y_kde, t₁)
    hist_sieve = build_histogram(y_sieve, t₁)

    error_sig = calculate_error(y_sig, t₁)
    error_kde = calculate_error(y_kde, t₁)
    error_sieve = calculate_error(y_sieve, t₁)

    (hist_sig, error_sig,
     hist_kde, error_kde,
     hist_sieve, error_sieve)
end

X₀, t₀ = mnist_training_data()
X₁, t₁ = mnist_testing_data()
n_neurons = 50

(hist_sig, error_sig,
 hist_kde, error_kde,
 hist_sieve, error_sieve) = perform_tests(X₀, t₀, X₁, t₁, n_neurons)

hist_sieve
