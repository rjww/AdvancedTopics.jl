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
    max_layers = 3

    W, fs = train_kde_comparators(Float64, X₀, t₀, max_layers * L)

    elm_sig = ELM{Float64}(X₀, t₀, W, Sigmoid())
    elm_kde = ELM{Float64}(X₀, t₀, W, fs)
    sieve_d = DataPassingSieve{Float64}(X₀, t₀, L, max_layers = max_layers)
    sieve_p = ProjectionPassingSieve{Float64}(X₀, t₀, L, max_layers = max_layers)

    y_sig = predict(elm_sig, X₁)
    y_kde = predict(elm_kde, X₁)
    y_sieve_d = predict(sieve_d, X₁)
    y_sieve_p = predict(sieve_p, X₁)

    hist_sig = build_histogram(y_sig, t₁)
    hist_kde = build_histogram(y_kde, t₁)
    hist_sieve_d = build_histogram(y_sieve_d, t₁)
    hist_sieve_p = build_histogram(y_sieve_p, t₁)

    error_sig = calculate_error(y_sig, t₁)
    error_kde = calculate_error(y_kde, t₁)
    error_sieve_d = calculate_error(y_sieve_d, t₁)
    error_sieve_p = calculate_error(y_sieve_p, t₁)

    (hist_sig, error_sig,
     hist_kde, error_kde,
     hist_sieve_d, error_sieve_d,
     hist_sieve_p, error_sieve_p)
end

X₀, t₀ = mnist_training_data()
X₁, t₁ = mnist_testing_data()
n_neurons = 50

(hist_sig, error_sig,
 hist_kde, error_kde,
 hist_sieve_d, error_sieve_d,
 hist_sieve_p, error_sieve_p) = perform_tests(X₀, t₀, X₁, t₁, n_neurons)
