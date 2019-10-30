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
    max_layers = 10

    W, fs = train_kde_comparators(Float64, X₀, t₀, max_layers * L, boundary_offset = 1)

    elm_sig = ELM{Float64}(X₀, t₀, W, Sigmoid())
    elm_kde = ELM{Float64}(X₀, t₀, W, fs)
    sieve_d = DataPassingSieve{Float64}(X₀, t₀, L, max_layers = max_layers, consensus_threshold = 0.9,  initial_boundary_offset = 1, subsequent_boundary_offset = 1)
    sieve_p = sieve_d
    #sieve_p = ProjectionPassingSieve{Float64}(X₀, t₀, L, initial_boundary_offset = 1, max_layers = max_layers)
    #sieve_d = sieve_p

    y_sig₀ = predict(elm_sig, X₀)
    y_kde₀ = predict(elm_kde, X₀)
    y_sieve_d₀= predict(sieve_d, X₀)
    y_sieve_p₀ = predict(sieve_p, X₀)

    hist_sig₀ = build_histogram(y_sig₀, t₀)
    hist_kde₀ = build_histogram(y_kde₀, t₀)
    hist_sieve_d₀ = build_histogram(y_sieve_d₀, t₀)
    hist_sieve_p₀ = build_histogram(y_sieve_p₀, t₀)

    error_sig₀ = calculate_error(y_sig₀, t₀)
    error_kde₀ = calculate_error(y_kde₀, t₀)
    error_sieve_d₀ = calculate_error(y_sieve_d₀, t₀)
    error_sieve_p₀ = calculate_error(y_sieve_p₀, t₀)


    y_sig₁ = predict(elm_sig, X₁)
    y_kde₁ = predict(elm_kde, X₁)
    y_sieve_d₁ = predict(sieve_d, X₁)
    y_sieve_p₁ = predict(sieve_p, X₁)

    hist_sig₁ = build_histogram(y_sig₁, t₁)
    hist_kde₁ = build_histogram(y_kde₁, t₁)
    hist_sieve_d₁ = build_histogram(y_sieve_d₁, t₁)
    hist_sieve_p₁ = build_histogram(y_sieve_p₁, t₁)

    error_sig₁ = calculate_error(y_sig₁, t₁)
    error_kde₁ = calculate_error(y_kde₁, t₁)
    error_sieve_d₁ = calculate_error(y_sieve_d₁, t₁)
    error_sieve_p₁ = calculate_error(y_sieve_p₁, t₁)

    @show size(W)

    (hist_sig₀, error_sig₀,
     hist_kde₀, error_kde₀,
     hist_sieve_d₀, error_sieve_d₀,
     hist_sieve_p₀, error_sieve_p₀,
     hist_sig₁, error_sig₁,
      hist_kde₁, error_kde₁,
      hist_sieve_d₁, error_sieve_d₁,
      hist_sieve_p₁, error_sieve_p₁)
end

X₀, t₀ = mnist_training_data()
X₁, t₁ = mnist_testing_data()
n_neurons = 5

# (hist_sig, error_sig,
#  hist_kde, error_kde,
#  hist_sieve_d, error_sieve_d,
#  hist_sieve_p, error_sieve_p) = perform_tests(X₀, t₀, X₁, t₁, n_neurons)

(hist_sig₀, error_sig₀,
 hist_kde₀, error_kde₀,
 hist_sieve_d₀, error_sieve_d₀,
 hist_sieve_p₀, error_sieve_p₀,
 hist_sig₁, error_sig₁,
  hist_kde₁, error_kde₁,
  hist_sieve_d₁, error_sieve_d₁,
  hist_sieve_p₁, error_sieve_p₁) = perform_tests(X₀, t₀, X₁, t₁, n_neurons)

hist_sig₀
hist_kde₀
hist_sieve_d₀

display("Training Error")
@show error_sig₀
@show error_kde₀
@show error_sieve_d₀

display("Testing Error")
@show error_sig₁
@show error_kde₁
@show error_sieve_d₁
