using AdvancedTopics

include("util.jl")

function perform_tests(samples::T₁,
                       targets::T₂,
                       n_neurons::Int) where {T₁ <: AbstractMatrix,
                                              T₂ <: AbstractVector}
      X = samples
      t = targets
      L = n_neurons
      max_layers = 3

      W, fs = train_kde_comparators(Float64, X, t, max_layers * L, boundary_offset = 100)

      elm_sig = ELM{Float64}(X, t, W, Sigmoid())
      elm_kde = ELM{Float64}(X, t, W, fs)
      sieve_d = DataPassingSieve{Float64}(X, t, L, max_layers = max_layers)
      sieve_p = ProjectionPassingSieve{Float64}(X, t, L, max_layers = max_layers)

      y_sig = predict(elm_sig, X)
      y_kde = predict(elm_kde, X)
      y_sieve_d = predict(sieve_d, X)
      y_sieve_p = predict(sieve_p, X)

      plt_sig = plot_decision_boundary(elm_sig, X, t)
      plt_kde = plot_decision_boundary(elm_kde, X, t)
      plt_sieve_d = plot_decision_boundary(sieve_d, X, t)
      plt_sieve_p = plot_decision_boundary(sieve_p, X, t)

      error_sig = calculate_error(y_sig, t)
      error_kde = calculate_error(y_kde, t)
      error_sieve_d = calculate_error(y_sieve_d, t)
      error_sieve_p = calculate_error(y_sieve_p, t)

      (plt_sig, error_sig,
       plt_kde, error_kde,
       plt_sieve_d, error_sieve_d,
       plt_sieve_p, error_sieve_p)
end

N = 200
degrees = 250.0
start = 200.0
σ = 1.2

X, t = synthetic_spiral_data(N, degrees, start, σ)
n_neurons = 5

(plt_sig, error_sig,
 plt_kde, error_kde,
 plt_sieve_d, error_sieve_d,
 plt_sieve_p, error_sieve_p) = perform_tests(X, t, n_neurons)
